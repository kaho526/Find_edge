"""Volatility Prediction Engine.

Quantitative signal layer for the retail options framework. Forecasts
realized volatility with two independent models (GARCH(1,1) and EWMA),
compares the forecast against market-implied IV to surface mispricings,
and proxies dealer gamma exposure (GEX) to gate entries on forced-hedge
pressure.

The trade thesis this module supports: institutions are often price-
insensitive (mandate-driven hedging, VaR-driven liquidation). Retail
quants can extract +EV only when a statistically verifiable gap exists
between the market-implied vol (IV) and the model-forecasted realized
vol (HV), and the dealer positioning makes the subsequent price path
mechanically predictable.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from arch import arch_model

TRADING_DAYS_PER_YEAR: int = 252
RISKMETRICS_LAMBDA: float = 0.94


def forecast_garch_vol(
    returns: pd.Series,
    horizon: int = 14,
    dist: str = "t",
) -> float:
    """Forecast average annualized volatility over ``horizon`` trading days.

    Fits a GARCH(1,1) to daily simple returns and returns the square root
    of the mean conditional variance over the forecast horizon, annualized
    by sqrt(252). GARCH is used (vs. a rolling HV window) because it models
    volatility clustering and mean-reversion — the two empirical features
    that let us argue a spot IV reading is mispriced relative to the
    vol regime the stock will actually realize.

    Student-t innovations are the default: equity returns are fat-tailed,
    and a Normal GARCH systematically under-estimates tail vol, which is
    precisely the regime where the IV/HV dislocation trade loses money.

    Parameters
    ----------
    returns : pd.Series
        Daily simple returns in decimal form (e.g. 0.012 for +1.2%).
    horizon : int
        Forecast horizon in trading days. 14 is a standard earnings-window
        look-ahead; shorten for weekly structures, lengthen for monthlies.
    dist : str
        Innovation distribution passed to ``arch_model`` ('t', 'normal',
        'skewt', 'ged').

    Returns
    -------
    float
        Forecasted annualized volatility over the horizon (e.g. 0.60 = 60%).
    """
    clean = returns.dropna()
    if clean.empty:
        raise ValueError("returns series is empty after dropna")

    # arch_model converges better on percent-scaled returns; undo after.
    scaled = clean.to_numpy() * 100.0
    model = arch_model(scaled, vol="Garch", p=1, q=1, dist=dist, rescale=False)
    fit = model.fit(disp="off")

    forecast = fit.forecast(horizon=horizon, reindex=False)
    daily_var_pct2 = forecast.variance.to_numpy()[-1]
    avg_daily_var = float(np.mean(daily_var_pct2)) / 10_000.0
    return float(np.sqrt(avg_daily_var * TRADING_DAYS_PER_YEAR))


def forecast_ewma_vol(
    returns: pd.Series,
    lambda_: float = RISKMETRICS_LAMBDA,
) -> float:
    """Forecast annualized volatility via RiskMetrics EWMA.

    Recursion: σ²_t = λ·σ²_{t-1} + (1 − λ)·r²_{t-1}.

    Role in the stack: a secondary confirmation of the GARCH forecast.
    EWMA has no long-run mean-reversion term, so it reacts faster than
    GARCH to regime changes but over-shoots on isolated shocks. When
    EWMA and GARCH agree on a vol regime, confidence in the IV/HV signal
    is materially higher; when they disagree, treat the signal as noisy
    and size down.

    Parameters
    ----------
    returns : pd.Series
        Daily simple returns in decimal form.
    lambda_ : float
        Decay factor. 0.94 is the RiskMetrics daily default; 0.97 is the
        monthly convention. Must be in (0, 1).

    Returns
    -------
    float
        Annualized EWMA volatility estimate (decimal).
    """
    if not 0.0 < lambda_ < 1.0:
        raise ValueError("lambda_ must lie in (0, 1)")

    r = returns.dropna().to_numpy()
    if r.size == 0:
        raise ValueError("returns series is empty after dropna")

    var = float(r[0] ** 2)
    for ret in r[1:]:
        var = lambda_ * var + (1.0 - lambda_) * float(ret) ** 2
    return float(np.sqrt(var * TRADING_DAYS_PER_YEAR))


def calc_iv_hv_spread(current_iv: float, forecasted_hv: float) -> float:
    """Return IV − HV spread in vol points.

    The core +EV gate. A large positive spread means the market is pricing
    more uncertainty than our models forecast will realize — premium is
    rich, and short-vol structures (with tail hedges) carry positive
    expectancy. A negative spread flips the trade to long-vol / long-wings.

    Interpretation rough cuts (per the mega-cap equity regime):
        spread >  0.10  — IV materially rich, investigate short-vol
        spread <  -0.05 — IV materially cheap, investigate long-vol
        |spread| < 0.03 — no edge, stand down

    Parameters
    ----------
    current_iv : float
        Market-implied volatility, annualized decimal (e.g. 0.95).
    forecasted_hv : float
        Model-forecasted realized volatility, annualized decimal.

    Returns
    -------
    float
        IV minus HV in vol-point decimals. 0.35 == 35 vol points rich.
    """
    return float(current_iv) - float(forecasted_hv)


def proxy_dealer_gex(
    option_chain_df: pd.DataFrame,
    spot_price: float,
    scale_to_dollars: bool = True,
) -> float:
    """Estimate dealer net gamma exposure for retail-heavy names.

    Baseline dealer-flow assumption: for meme / retail-concentrated names
    dealers are net short calls (retail lifts OTM calls) and net long puts
    (retail writes puts into rallies, dealers warehouse the flow). Under
    that prior, dealer-facing gamma is:

        GEX ≈ Σ(call_γ × OI_call) − Σ(put_γ × OI_put)

    Sign interpretation:
        GEX > 0  — dealers net short gamma: must BUY rallies / SELL dips.
                   Expect trend amplification, rising vol-of-vol.
        GEX < 0  — dealers net long gamma: they dampen moves (pinning /
                   mean-reversion around large strikes).

    With ``scale_to_dollars=True`` the result is expressed as dollar-gamma
    per 1% spot move (γ × OI × S²), the convention used on SpotGamma /
    Menthor Q dashboards. The 100-shares-per-contract and 1%-move scalers
    cancel, leaving S² as the unit-conversion term.

    Parameters
    ----------
    option_chain_df : pd.DataFrame
        Must contain columns: ``option_type`` ({'call', 'put'}, case-
        insensitive), ``gamma`` (per-share gamma), ``open_interest``
        (contracts).
    spot_price : float
        Underlying spot price; required when ``scale_to_dollars=True``.
    scale_to_dollars : bool
        Scale raw gamma notional to dollar-gamma per 1% spot move.

    Returns
    -------
    float
        Net dealer gamma under the retail-baseline assumption. Positive
        = destabilizing (short-gamma dealers); negative = stabilizing.
    """
    required = {"option_type", "gamma", "open_interest"}
    missing = required - set(option_chain_df.columns)
    if missing:
        raise ValueError(f"option_chain_df missing columns: {sorted(missing)}")
    if spot_price <= 0:
        raise ValueError("spot_price must be positive")

    df = option_chain_df.dropna(subset=["gamma", "open_interest"])
    is_call = df["option_type"].astype(str).str.lower().eq("call")

    call_gamma_oi = float(
        (df.loc[is_call, "gamma"] * df.loc[is_call, "open_interest"]).sum()
    )
    put_gamma_oi = float(
        (df.loc[~is_call, "gamma"] * df.loc[~is_call, "open_interest"]).sum()
    )

    gex = call_gamma_oi - put_gamma_oi
    if scale_to_dollars:
        gex *= spot_price ** 2
    return gex
