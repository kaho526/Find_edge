"""Polygon.io REST provider.

Thin client for the three data cuts the signal engine needs:
    - historical daily bars (for GARCH / EWMA return series)
    - option chain snapshot with greeks + OI (for the GEX proxy)
    - spot price (for GEX dollar-scaling and moneyness)

Design notes
------------
The API key is read from the ``MASSIVEKEY`` env var OR from ``key.md`` in
the repo root (``MASSIVEKEY="..."`` form). ``key.md`` is gitignored; it
is never hardcoded in this module. If neither source resolves, the
loader raises loudly rather than silently using an empty string.

The public-facing functions all return plain ``pd.DataFrame`` / scalar
types that match the contract ``signals.py`` expects — in particular,
the option chain frame exposes ``option_type``, ``gamma``, and
``open_interest`` columns.
"""

from __future__ import annotations

import datetime as dt
from typing import Any

import pandas as pd
import requests

from ._keyfile import load_key

BASE_URL: str = "https://api.polygon.io"
DEFAULT_TIMEOUT: float = 15.0


def load_api_key() -> str:
    """Resolve the Polygon API key (alias: ``MASSIVEKEY``) from env or key.md."""
    return load_key("MASSIVEKEY")


def _get(path: str, params: dict[str, Any] | None = None) -> dict[str, Any]:
    """Signed GET against Polygon. Raises on non-200."""
    params = dict(params or {})
    params["apiKey"] = load_api_key()
    url = f"{BASE_URL}{path}"
    resp = requests.get(url, params=params, timeout=DEFAULT_TIMEOUT)
    if resp.status_code == 429:
        raise RuntimeError("Polygon rate-limited (429). Back off or upgrade tier.")
    if resp.status_code == 403:
        raise RuntimeError(
            "Polygon 403: endpoint requires a higher plan tier than this key "
            f"has. Endpoint: {path}"
        )
    resp.raise_for_status()
    return resp.json()


def fetch_daily_bars(
    ticker: str,
    lookback_days: int = 400,
    end: dt.date | None = None,
) -> pd.DataFrame:
    """Pull daily OHLCV bars and return a frame with a ``return`` column.

    ``lookback_days`` is in calendar days; GARCH fitting wants at least
    ~250 observations for stable parameter estimates, so default 400 cal
    days comfortably clears that. Returns are simple (close-to-close);
    signals.py handles scaling internally.

    Parameters
    ----------
    ticker : str
        Underlying symbol (e.g. 'TSLA').
    lookback_days : int
        Calendar-day window ending ``end``.
    end : datetime.date | None
        End date (inclusive). Defaults to today (local).

    Returns
    -------
    pd.DataFrame
        Indexed by date, columns: ``open, high, low, close, volume,
        return``. Rows with NaN returns (first bar) are dropped.
    """
    end = end or dt.date.today()
    start = end - dt.timedelta(days=lookback_days)
    path = (
        f"/v2/aggs/ticker/{ticker.upper()}/range/1/day/"
        f"{start.isoformat()}/{end.isoformat()}"
    )
    data = _get(path, {"adjusted": "true", "sort": "asc", "limit": 50_000})
    results = data.get("results") or []
    if not results:
        raise RuntimeError(f"no bars returned for {ticker} [{start}..{end}]")

    df = pd.DataFrame(results).rename(
        columns={"o": "open", "h": "high", "l": "low", "c": "close", "v": "volume"}
    )
    df["date"] = pd.to_datetime(df["t"], unit="ms").dt.date
    df = df.set_index("date").sort_index()
    df["return"] = df["close"].pct_change()
    return df[["open", "high", "low", "close", "volume", "return"]].dropna()


def fetch_spot_price(ticker: str) -> float:
    """Return the most recent close as the spot proxy.

    Uses the prev-close aggregate (free-tier friendly). For intraday spot
    we would use ``/v2/last/trade`` but that endpoint requires a higher
    tier and the prev-close is close enough for GEX dollar-scaling.
    """
    path = f"/v2/aggs/ticker/{ticker.upper()}/prev"
    data = _get(path, {"adjusted": "true"})
    results = data.get("results") or []
    if not results:
        raise RuntimeError(f"no prev-close for {ticker}")
    return float(results[0]["c"])


def fetch_option_chain_snapshot(
    ticker: str,
    max_pages: int = 10,
    contract_type: str | None = None,
) -> pd.DataFrame:
    """Pull a full option-chain snapshot with greeks and OI.

    Endpoint: ``/v3/snapshot/options/{underlying}`` (paginated). Each
    entry carries ``details`` (strike, expiry, type), ``greeks``
    (delta, gamma, theta, vega), ``open_interest``, ``implied_volatility``,
    and last-quote/last-trade blocks. We flatten into a single frame
    that ``signals.proxy_dealer_gex`` consumes directly.

    Parameters
    ----------
    ticker : str
        Underlying symbol.
    max_pages : int
        Hard cap on pagination walks; TSLA-style chains fit in ~5 pages
        of 250 contracts.
    contract_type : str | None
        Optional filter ('call' or 'put').

    Returns
    -------
    pd.DataFrame
        Columns: ``ticker_symbol, option_type, strike, expiry, gamma,
        delta, theta, vega, iv, open_interest, volume, last_price,
        bid, ask``. Rows missing greeks or OI are NOT dropped here —
        downstream code handles that.
    """
    path = f"/v3/snapshot/options/{ticker.upper()}"
    params: dict[str, Any] = {"limit": 250}
    if contract_type:
        params["contract_type"] = contract_type.lower()

    rows: list[dict[str, Any]] = []
    next_url: str | None = None
    pages = 0

    while True:
        if next_url is None:
            data = _get(path, params)
        else:
            resp = requests.get(
                next_url,
                params={"apiKey": load_api_key()},
                timeout=DEFAULT_TIMEOUT,
            )
            resp.raise_for_status()
            data = resp.json()

        for entry in data.get("results") or []:
            details = entry.get("details") or {}
            greeks = entry.get("greeks") or {}
            last_quote = entry.get("last_quote") or {}
            last_trade = entry.get("last_trade") or {}
            day = entry.get("day") or {}
            rows.append(
                {
                    "ticker_symbol": details.get("ticker"),
                    "option_type": details.get("contract_type"),
                    "strike": details.get("strike_price"),
                    "expiry": details.get("expiration_date"),
                    "gamma": greeks.get("gamma"),
                    "delta": greeks.get("delta"),
                    "theta": greeks.get("theta"),
                    "vega": greeks.get("vega"),
                    "iv": entry.get("implied_volatility"),
                    "open_interest": entry.get("open_interest"),
                    "volume": day.get("volume"),
                    "last_price": last_trade.get("price"),
                    "bid": last_quote.get("bid"),
                    "ask": last_quote.get("ask"),
                }
            )

        next_url = data.get("next_url")
        pages += 1
        if not next_url or pages >= max_pages:
            break

    if not rows:
        raise RuntimeError(f"option-chain snapshot empty for {ticker}")
    df = pd.DataFrame(rows)
    df["expiry"] = pd.to_datetime(df["expiry"], errors="coerce").dt.date
    return df
