"""Yahoo Finance provider via ``yfinance``.

No account, no API key, no quota. Covers bars + spot + full option
chain. Yahoo does not publish greeks, so gamma/delta are computed
locally via Black-Scholes using Yahoo's IV — this is consistent
because Yahoo derives its IV by inverting the same BSM model.

Caveats
-------
- Yahoo IV is mid-quote derived and becomes stale / thin on illiquid
  strikes. Deep-OTM wing greeks can carry more error than a broker
  feed; the GEX dollar figure will be approximately right but the
  wings contribute noise.
- Intraday spot is delayed ~15 minutes on Yahoo's public endpoint.
- ``yfinance`` changes with Yahoo's markup; if calls start failing,
  upgrade the package first before suspecting our code.
"""

from __future__ import annotations

import datetime as dt
import math

import pandas as pd
import yfinance as yf

RISK_FREE_RATE: float = 0.045
MAX_EXPIRIES_DEFAULT: int = 4


def _norm_pdf(x: float) -> float:
    return math.exp(-0.5 * x * x) / math.sqrt(2.0 * math.pi)


def _norm_cdf(x: float) -> float:
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def _bsm_greeks(
    spot: float,
    strike: float,
    t_years: float,
    iv: float,
    option_type: str,
    r: float = RISK_FREE_RATE,
) -> tuple[float, float]:
    """Black-Scholes (delta, gamma). Returns (NaN, NaN) if inputs invalid.

    Gamma is identical for calls and puts (put-call symmetry of γ).
    Delta differs: call N(d1), put N(d1) − 1.
    """
    if not (spot > 0 and strike > 0 and t_years > 0 and iv > 0):
        return float("nan"), float("nan")
    sigma_sqrt_t = iv * math.sqrt(t_years)
    d1 = (math.log(spot / strike) + (r + 0.5 * iv * iv) * t_years) / sigma_sqrt_t
    gamma = _norm_pdf(d1) / (spot * sigma_sqrt_t)
    if option_type.lower() == "call":
        delta = _norm_cdf(d1)
    else:
        delta = _norm_cdf(d1) - 1.0
    return float(delta), float(gamma)


def fetch_daily_bars(
    ticker: str,
    lookback_days: int = 400,
    end: dt.date | None = None,
) -> pd.DataFrame:
    """Daily bars + simple-return column. Same contract as other providers."""
    end = end or dt.date.today()
    start = end - dt.timedelta(days=lookback_days)
    hist = yf.Ticker(ticker.upper()).history(
        start=start,
        end=end + dt.timedelta(days=1),
        interval="1d",
        auto_adjust=True,
    )
    if hist.empty:
        raise RuntimeError(f"no bars returned for {ticker} [{start}..{end}]")

    df = hist.rename(columns=str.lower)[["open", "high", "low", "close", "volume"]].copy()
    df.index = pd.to_datetime(df.index).date
    df.index.name = "date"
    df["return"] = df["close"].pct_change()
    return df.dropna()


def fetch_spot_price(ticker: str) -> float:
    """Most recent price via ``fast_info``; falls back to last close."""
    t = yf.Ticker(ticker.upper())
    try:
        price = float(t.fast_info["last_price"])
        if price > 0:
            return price
    except (KeyError, TypeError, ValueError):
        pass
    hist = t.history(period="2d", auto_adjust=True)
    if hist.empty:
        raise RuntimeError(f"no spot for {ticker}")
    return float(hist["Close"].iloc[-1])


def fetch_option_chain_snapshot(
    ticker: str,
    max_expiries: int = MAX_EXPIRIES_DEFAULT,
) -> pd.DataFrame:
    """Flatten nearest ``max_expiries`` chains with locally-computed greeks.

    Output columns: ticker_symbol, option_type, strike, expiry, gamma,
    delta, iv, open_interest, volume, last_price, bid, ask. Matches the
    shape the ``polygon`` and ``tradier`` providers emit, so
    ``signals.proxy_dealer_gex`` accepts it without adaptation.
    """
    t = yf.Ticker(ticker.upper())
    expiries = list(t.options)
    if not expiries:
        raise RuntimeError(f"no option expiries listed for {ticker}")

    spot = fetch_spot_price(ticker)
    today = dt.date.today()
    rows: list[dict[str, object]] = []

    for exp_str in expiries[:max_expiries]:
        try:
            exp = dt.date.fromisoformat(exp_str)
        except ValueError:
            continue
        t_years = max((exp - today).days / 365.0, 1.0 / 365.0)
        try:
            chain = t.option_chain(exp_str)
        except Exception:  # noqa: BLE001 — yfinance wraps many failure modes
            continue

        for side_df, side_name in ((chain.calls, "call"), (chain.puts, "put")):
            for _, row in side_df.iterrows():
                iv = row.get("impliedVolatility")
                strike = row.get("strike")
                delta, gamma = _bsm_greeks(
                    spot=spot,
                    strike=float(strike) if strike else 0.0,
                    t_years=t_years,
                    iv=float(iv) if iv else 0.0,
                    option_type=side_name,
                )
                rows.append(
                    {
                        "ticker_symbol": row.get("contractSymbol"),
                        "option_type": side_name,
                        "strike": strike,
                        "expiry": exp,
                        "gamma": gamma,
                        "delta": delta,
                        "iv": iv,
                        "open_interest": row.get("openInterest"),
                        "volume": row.get("volume"),
                        "last_price": row.get("lastPrice"),
                        "bid": row.get("bid"),
                        "ask": row.get("ask"),
                    }
                )

    if not rows:
        raise RuntimeError(f"option-chain snapshot empty for {ticker}")
    return pd.DataFrame(rows)
