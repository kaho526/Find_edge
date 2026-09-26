"""Tradier REST provider (sandbox or production).

Tradier offers a free sandbox with full option-chain greeks + OI, which
unblocks the GEX computation without paying for a Polygon options tier.
Sandbox data is 15-minute delayed — fine for analysis, not for execution.

Endpoints used
--------------
    /v1/markets/history      — daily OHLCV bars
    /v1/markets/quotes       — latest quote (spot)
    /v1/markets/options/expirations  — listing of expiries
    /v1/markets/options/chains       — per-expiry chain with greeks

Public API mirrors ``find_edge.providers.polygon`` so ``scan_ticker.py``
can swap providers with a CLI flag and nothing else changes.

Auth
----
Token resolved via ``TRADIER_TOKEN`` env var or ``key.md`` entry
``TRADIER_TOKEN="..."``. Sandbox base URL is the default; set
``TRADIER_ENV=production`` to hit the live endpoint (production tokens
only).
"""

from __future__ import annotations

import datetime as dt
import os
from typing import Any

import pandas as pd
import requests

from ._keyfile import load_key

SANDBOX_URL: str = "https://sandbox.tradier.com/v1"
PRODUCTION_URL: str = "https://api.tradier.com/v1"
DEFAULT_TIMEOUT: float = 15.0
MAX_EXPIRIES_DEFAULT: int = 4


def _base_url() -> str:
    env = os.environ.get("TRADIER_ENV", "sandbox").lower()
    return PRODUCTION_URL if env == "production" else SANDBOX_URL


def _headers() -> dict[str, str]:
    return {
        "Authorization": f"Bearer {load_key('TRADIER_TOKEN')}",
        "Accept": "application/json",
    }


def _get(path: str, params: dict[str, Any] | None = None) -> dict[str, Any]:
    """Authenticated GET. Raises on non-200 with a readable message."""
    resp = requests.get(
        f"{_base_url()}{path}",
        params=params or {},
        headers=_headers(),
        timeout=DEFAULT_TIMEOUT,
    )
    if resp.status_code == 401:
        raise RuntimeError("Tradier 401: invalid TRADIER_TOKEN or wrong environment")
    if resp.status_code == 429:
        raise RuntimeError("Tradier rate-limited (429)")
    resp.raise_for_status()
    return resp.json()


def fetch_daily_bars(
    ticker: str,
    lookback_days: int = 400,
    end: dt.date | None = None,
) -> pd.DataFrame:
    """Daily bars + simple returns. Same frame contract as polygon.py."""
    end = end or dt.date.today()
    start = end - dt.timedelta(days=lookback_days)
    data = _get(
        "/markets/history",
        {
            "symbol": ticker.upper(),
            "interval": "daily",
            "start": start.isoformat(),
            "end": end.isoformat(),
            "session_filter": "all",
        },
    )

    history = (data.get("history") or {}).get("day") or []
    if not history:
        raise RuntimeError(f"no bars returned for {ticker} [{start}..{end}]")
    if isinstance(history, dict):
        history = [history]

    df = pd.DataFrame(history)
    df["date"] = pd.to_datetime(df["date"]).dt.date
    df = df.set_index("date").sort_index()
    df["return"] = df["close"].pct_change()
    return df[["open", "high", "low", "close", "volume", "return"]].dropna()


def fetch_spot_price(ticker: str) -> float:
    """Latest quote 'last' as spot. Sandbox is 15-min delayed."""
    data = _get("/markets/quotes", {"symbols": ticker.upper()})
    quote = (data.get("quotes") or {}).get("quote")
    if not quote:
        raise RuntimeError(f"no quote for {ticker}")
    if isinstance(quote, list):
        quote = quote[0]
    last = quote.get("last") or quote.get("close")
    if last is None:
        raise RuntimeError(f"quote for {ticker} has no last/close")
    return float(last)


def _list_expirations(ticker: str) -> list[str]:
    data = _get(
        "/markets/options/expirations",
        {"symbol": ticker.upper(), "includeAllRoots": "true"},
    )
    dates = (data.get("expirations") or {}).get("date") or []
    if isinstance(dates, str):
        dates = [dates]
    return list(dates)


def _fetch_chain_for_expiry(ticker: str, expiry: str) -> list[dict[str, Any]]:
    data = _get(
        "/markets/options/chains",
        {"symbol": ticker.upper(), "expiration": expiry, "greeks": "true"},
    )
    options = (data.get("options") or {}).get("option") or []
    if isinstance(options, dict):
        options = [options]
    return options


def fetch_option_chain_snapshot(
    ticker: str,
    max_expiries: int = MAX_EXPIRIES_DEFAULT,
) -> pd.DataFrame:
    """Chain snapshot with greeks + OI across the nearest ``max_expiries``.

    Tradier's chain endpoint is expiry-scoped, so we list expirations and
    loop the first N. N=4 covers front/next weekly + front/next monthly,
    which is where >90% of GEX concentration lives.

    Output columns (same as polygon provider):
        ticker_symbol, option_type, strike, expiry, gamma, delta, theta,
        vega, iv, open_interest, volume, last_price, bid, ask.
    """
    expiries = _list_expirations(ticker)
    if not expiries:
        raise RuntimeError(f"no expirations listed for {ticker}")

    rows: list[dict[str, Any]] = []
    for expiry in expiries[:max_expiries]:
        for opt in _fetch_chain_for_expiry(ticker, expiry):
            greeks = opt.get("greeks") or {}
            rows.append(
                {
                    "ticker_symbol": opt.get("symbol"),
                    "option_type": opt.get("option_type"),
                    "strike": opt.get("strike"),
                    "expiry": opt.get("expiration_date"),
                    "gamma": greeks.get("gamma"),
                    "delta": greeks.get("delta"),
                    "theta": greeks.get("theta"),
                    "vega": greeks.get("vega"),
                    "iv": greeks.get("mid_iv"),
                    "open_interest": opt.get("open_interest"),
                    "volume": opt.get("volume"),
                    "last_price": opt.get("last"),
                    "bid": opt.get("bid"),
                    "ask": opt.get("ask"),
                }
            )

    if not rows:
        raise RuntimeError(f"option-chain snapshot empty for {ticker}")
    df = pd.DataFrame(rows)
    df["expiry"] = pd.to_datetime(df["expiry"], errors="coerce").dt.date
    return df
