"""Volatility-dislocation report for a single ticker.

Usage:
    python3 scripts/scan_ticker.py TSLA
    python3 scripts/scan_ticker.py PLTR --horizon 14

Emits a plain-text institutional-style report:
    - GARCH(1,1) + EWMA forward-vol forecasts
    - Front-month ATM implied vol (median of near-ATM strikes)
    - IV − HV spread with trade gate (short-vol / long-vol / stand down)
    - Net dealer GEX ($ per 1% move) with hedge-regime classification
"""

from __future__ import annotations

import argparse
import datetime as dt
import importlib
import sys
from pathlib import Path
from types import ModuleType

import pandas as pd

# Allow running from repo root without installing the package.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from find_edge.signals import (  # noqa: E402
    calc_iv_hv_spread,
    forecast_ewma_vol,
    forecast_garch_vol,
    proxy_dealer_gex,
)

PROVIDERS = ("polygon", "tradier", "yahoo")


def _load_provider(name: str) -> ModuleType:
    """Dynamic import so unused providers don't demand their keys upfront."""
    if name not in PROVIDERS:
        raise ValueError(f"unknown provider {name!r}; expected one of {PROVIDERS}")
    return importlib.import_module(f"find_edge.providers.{name}")

SHORT_VOL_GATE: float = 0.10
LONG_VOL_GATE: float = -0.05


def _front_month_atm_iv(
    chain: pd.DataFrame, spot: float, today: dt.date | None = None
) -> float:
    """Median IV across strikes within ±5% of spot on the nearest expiry.

    Contracts expiring today (0DTE) or earlier are skipped: their IV prices
    the last few hours of the session, not the forecast horizon, and on an
    expiry day it would otherwise be the "nearest" expiry.
    """
    if chain.empty or "expiry" not in chain:
        raise RuntimeError("chain is empty or missing expiry column")
    today = today or dt.date.today()
    expiries = chain["expiry"].dropna()
    expiries = expiries[expiries > today]
    if expiries.empty:
        raise RuntimeError("no expiries after today in the chain")
    nearest = expiries.min()
    window = chain[
        (chain["expiry"] == nearest)
        & chain["iv"].notna()
        & (chain["strike"].between(spot * 0.95, spot * 1.05))
    ]
    if window.empty:
        raise RuntimeError("no ATM contracts with IV on the nearest expiry")
    return float(window["iv"].median())


def _spread_verdict(spread: float) -> str:
    if spread > SHORT_VOL_GATE:
        return "IV RICH — investigate short-vol (calendars, broken-wing flies)"
    if spread < LONG_VOL_GATE:
        return "IV CHEAP — investigate long-vol (straddles, long wings)"
    return "NO EDGE — stand down"


def _gex_verdict(gex: float) -> str:
    if gex > 0:
        return "dealers NET SHORT GAMMA — trend-amplifying, vol-of-vol up"
    if gex < 0:
        return "dealers NET LONG GAMMA — pinning / mean-reversion regime"
    return "dealer gamma neutral"


def run_report(ticker: str, provider: str = "polygon", horizon: int = 14) -> None:
    print(f"\n=== Volatility Dislocation Report — {ticker.upper()} "
          f"[provider: {provider}] ===\n")
    dp = _load_provider(provider)

    bars = dp.fetch_daily_bars(ticker, lookback_days=400)
    print(f"bars: {len(bars)} daily rows, last close {bars['close'].iloc[-1]:.2f} "
          f"on {bars.index[-1]}")

    returns = bars["return"]
    garch_hv = forecast_garch_vol(returns, horizon=horizon)
    ewma_hv = forecast_ewma_vol(returns, lambda_=0.94)
    blend_hv = 0.5 * garch_hv + 0.5 * ewma_hv

    print(f"\nForecasted HV (annualized, {horizon}d lookahead):")
    print(f"  GARCH(1,1) t-dist : {garch_hv:.2%}")
    print(f"  EWMA λ=0.94       : {ewma_hv:.2%}")
    print(f"  blend (50/50)     : {blend_hv:.2%}")

    try:
        spot = dp.fetch_spot_price(ticker)
        chain = dp.fetch_option_chain_snapshot(ticker)
    except RuntimeError as err:
        print(f"\n[!] option data unavailable via {provider}: {err}")
        return

    print(f"\nspot: {spot:.2f} | chain rows: {len(chain)}")

    iv = _front_month_atm_iv(chain, spot)
    spread = calc_iv_hv_spread(iv, blend_hv)

    print(f"\nFront-month ATM IV  : {iv:.2%}")
    print(f"IV − HV spread      : {spread:+.2%}")
    print(f"Verdict             : {_spread_verdict(spread)}")

    gex = proxy_dealer_gex(chain, spot_price=spot, scale_to_dollars=True)
    print(f"\nDealer GEX proxy    : ${gex:,.0f} per 1% move")
    print(f"Regime              : {_gex_verdict(gex)}")

    print()


def main() -> None:
    parser = argparse.ArgumentParser(description="Vol-dislocation scan")
    parser.add_argument("ticker", help="e.g. TSLA, PLTR, NVDA")
    parser.add_argument("--provider", choices=PROVIDERS, default="yahoo",
                        help="data provider (default: yahoo — free, no account)")
    parser.add_argument("--horizon", type=int, default=14,
                        help="forward-vol horizon in trading days")
    args = parser.parse_args()
    run_report(args.ticker, provider=args.provider, horizon=args.horizon)


if __name__ == "__main__":
    main()
