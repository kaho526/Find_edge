"""Unit tests for the Volatility Prediction Engine.

These tests pin numerical behavior so future refactors of signals.py
can't silently shift the sizing decisions downstream in backtest.py.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from find_edge.providers.yahoo import _bsm_greeks
from find_edge.signals import (
    TRADING_DAYS_PER_YEAR,
    calc_iv_hv_spread,
    forecast_ewma_vol,
    forecast_garch_vol,
    proxy_dealer_gex,
)


@pytest.fixture(scope="module")
def synthetic_returns() -> pd.Series:
    """500 days of ~40% annualized vol Normal returns — deterministic seed."""
    rng = np.random.default_rng(seed=42)
    daily_sigma = 0.40 / np.sqrt(TRADING_DAYS_PER_YEAR)
    returns = rng.normal(loc=0.0, scale=daily_sigma, size=500)
    index = pd.date_range("2022-01-01", periods=500, freq="B")
    return pd.Series(returns, index=index, name="return")


# --- forecast_ewma_vol -----------------------------------------------------


def test_ewma_recovers_generating_vol(synthetic_returns: pd.Series) -> None:
    """EWMA on 500d of 40%-vol noise should estimate within ±5 vol pts."""
    vol = forecast_ewma_vol(synthetic_returns, lambda_=0.94)
    assert 0.30 < vol < 0.50


def test_ewma_matches_hand_calc() -> None:
    """Closed-form check on a 3-point series."""
    r = pd.Series([0.01, -0.02, 0.015])
    lam = 0.94
    # var_0 = 0.01**2 = 1e-4
    # var_1 = 0.94*1e-4 + 0.06*(-0.02)**2 = 9.4e-5 + 2.4e-5 = 1.18e-4
    # var_2 = 0.94*1.18e-4 + 0.06*0.015**2 = 1.1092e-4 + 1.35e-5 = 1.2442e-4
    expected = float(np.sqrt(1.2442e-4 * TRADING_DAYS_PER_YEAR))
    got = forecast_ewma_vol(r, lambda_=lam)
    assert got == pytest.approx(expected, rel=1e-6)


def test_ewma_rejects_invalid_lambda() -> None:
    r = pd.Series([0.01, -0.02, 0.015])
    with pytest.raises(ValueError):
        forecast_ewma_vol(r, lambda_=1.0)
    with pytest.raises(ValueError):
        forecast_ewma_vol(r, lambda_=0.0)


def test_ewma_empty_series_raises() -> None:
    with pytest.raises(ValueError):
        forecast_ewma_vol(pd.Series([], dtype=float))


# --- forecast_garch_vol ----------------------------------------------------


def test_garch_recovers_generating_vol(synthetic_returns: pd.Series) -> None:
    """GARCH on stationary 40%-vol noise — sanity bracket, not point estimate."""
    vol = forecast_garch_vol(synthetic_returns, horizon=14)
    assert 0.25 < vol < 0.55


def test_garch_output_is_positive(synthetic_returns: pd.Series) -> None:
    assert forecast_garch_vol(synthetic_returns, horizon=5) > 0.0


def test_garch_empty_series_raises() -> None:
    with pytest.raises(ValueError):
        forecast_garch_vol(pd.Series([], dtype=float))


# --- calc_iv_hv_spread -----------------------------------------------------


def test_spread_sign_and_magnitude() -> None:
    assert calc_iv_hv_spread(0.95, 0.60) == pytest.approx(0.35)
    assert calc_iv_hv_spread(0.40, 0.55) == pytest.approx(-0.15)
    assert calc_iv_hv_spread(0.50, 0.50) == pytest.approx(0.0)


# --- proxy_dealer_gex ------------------------------------------------------


def _chain(call_rows: list[tuple[float, int]], put_rows: list[tuple[float, int]]) -> pd.DataFrame:
    """Build a minimal chain with (gamma, open_interest) rows."""
    rows = [
        {"option_type": "call", "gamma": g, "open_interest": oi}
        for g, oi in call_rows
    ] + [
        {"option_type": "put", "gamma": g, "open_interest": oi}
        for g, oi in put_rows
    ]
    return pd.DataFrame(rows)


def test_gex_raw_hand_calc() -> None:
    """Raw (unscaled) GEX matches Σ(call γ·OI) − Σ(put γ·OI) exactly."""
    df = _chain(
        call_rows=[(0.02, 1000), (0.01, 500)],   # 20 + 5 = 25
        put_rows=[(0.015, 800)],                  # 12
    )
    gex = proxy_dealer_gex(df, spot_price=100.0, scale_to_dollars=False)
    assert gex == pytest.approx(25.0 - 12.0)


def test_gex_dollar_scaling() -> None:
    """Dollar GEX = raw × S² (per-1%-move convention)."""
    df = _chain(call_rows=[(0.02, 1000)], put_rows=[])
    raw = proxy_dealer_gex(df, spot_price=100.0, scale_to_dollars=False)
    dollar = proxy_dealer_gex(df, spot_price=100.0, scale_to_dollars=True)
    assert dollar == pytest.approx(raw * 100.0**2)


def test_gex_positive_when_calls_dominate() -> None:
    df = _chain(call_rows=[(0.03, 10_000)], put_rows=[(0.01, 500)])
    assert proxy_dealer_gex(df, spot_price=50.0) > 0


def test_gex_negative_when_puts_dominate() -> None:
    df = _chain(call_rows=[(0.01, 500)], put_rows=[(0.03, 10_000)])
    assert proxy_dealer_gex(df, spot_price=50.0) < 0


def test_gex_missing_columns_raises() -> None:
    bad = pd.DataFrame({"option_type": ["call"], "gamma": [0.02]})
    with pytest.raises(ValueError):
        proxy_dealer_gex(bad, spot_price=100.0)


def test_gex_bad_spot_raises() -> None:
    df = _chain(call_rows=[(0.02, 1000)], put_rows=[])
    with pytest.raises(ValueError):
        proxy_dealer_gex(df, spot_price=0.0)


def test_gex_handles_nan_rows() -> None:
    df = pd.DataFrame(
        [
            {"option_type": "call", "gamma": 0.02, "open_interest": 1000},
            {"option_type": "call", "gamma": np.nan, "open_interest": 500},
            {"option_type": "put", "gamma": 0.01, "open_interest": None},
        ]
    )
    gex = proxy_dealer_gex(df, spot_price=100.0, scale_to_dollars=False)
    assert gex == pytest.approx(20.0)


def test_bsm_greeks_textbook_case() -> None:
    """S=100, K=100, T=1, σ=0.20, r=0.05 → Δ≈0.6368, γ≈0.01876."""
    delta, gamma = _bsm_greeks(
        spot=100.0, strike=100.0, t_years=1.0, iv=0.20,
        option_type="call", r=0.05,
    )
    assert delta == pytest.approx(0.6368, abs=1e-3)
    assert gamma == pytest.approx(0.01876, abs=1e-4)


def test_bsm_greeks_put_delta() -> None:
    """Put delta = call delta − 1 (put-call parity for Δ)."""
    d_call, _ = _bsm_greeks(100.0, 100.0, 1.0, 0.20, "call", 0.05)
    d_put, _ = _bsm_greeks(100.0, 100.0, 1.0, 0.20, "put", 0.05)
    assert d_put == pytest.approx(d_call - 1.0, abs=1e-9)


def test_bsm_greeks_gamma_symmetric() -> None:
    """γ(call) == γ(put) at matched strike (identical second derivative)."""
    _, g_call = _bsm_greeks(100.0, 105.0, 0.5, 0.30, "call", 0.04)
    _, g_put = _bsm_greeks(100.0, 105.0, 0.5, 0.30, "put", 0.04)
    assert g_call == pytest.approx(g_put, abs=1e-12)


def test_bsm_greeks_invalid_returns_nan() -> None:
    import math as _m
    d, g = _bsm_greeks(100.0, 100.0, 0.0, 0.20, "call")
    assert _m.isnan(d) and _m.isnan(g)
    d, g = _bsm_greeks(100.0, 100.0, 1.0, 0.0, "call")
    assert _m.isnan(d) and _m.isnan(g)


def test_gex_case_insensitive_option_type() -> None:
    df = pd.DataFrame(
        [
            {"option_type": "CALL", "gamma": 0.02, "open_interest": 1000},
            {"option_type": "Put", "gamma": 0.01, "open_interest": 500},
        ]
    )
    gex = proxy_dealer_gex(df, spot_price=100.0, scale_to_dollars=False)
    assert gex == pytest.approx(20.0 - 5.0)
