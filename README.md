# Volatility Dislocation Engine

Compares a stock's option-implied volatility with a model forecast of realised volatility, and reports
which way the gap points. It also estimates dealer gamma exposure from the option chain.

```
=== Volatility Dislocation Report — TSLA [provider: yahoo] ===

Forecasted HV (annualized, 14d lookahead):
  GARCH(1,1) t-dist : ...
  EWMA λ=0.94       : ...
  blend (50/50)     : ...

Front-month ATM IV  : ...
IV − HV spread      : ...
Verdict             : IV RICH / IV CHEAP / NO EDGE

Dealer GEX proxy    : $... per 1% move
Regime              : dealers NET SHORT GAMMA / NET LONG GAMMA
```

## How it works

`find_edge/signals.py` holds the signal functions. They are pure and can be tested without network access.

| Function | What it computes |
|---|---|
| `forecast_garch_vol` | Fits GARCH(1,1) with Student-t innovations to daily returns (`arch`). Returns the annualised average conditional vol over the horizon (default 14 trading days). |
| `forecast_ewma_vol` | RiskMetrics EWMA, σ²ₜ = λσ²ₜ₋₁ + (1−λ)r²ₜ₋₁, with λ = 0.94. No mean reversion, so it reacts faster than GARCH. |
| `calc_iv_hv_spread` | Implied vol minus forecast vol, in vol points. |
| `proxy_dealer_gex` | Σ(call γ × OI) − Σ(put γ × OI), optionally × S² for dollar gamma per 1% move. Assumes dealers are short calls and long puts, which is the usual retail-flow prior for retail-heavy names. |

`scripts/scan_ticker.py` builds the report from those functions:

1. Pulls ~400 calendar days of daily bars and forecasts vol with GARCH and EWMA, then blends them 50/50.
2. Takes implied vol as the median IV of strikes within ±5% of spot, on the nearest expiry **after today**.
   Same-day (0DTE) contracts are skipped because their IV prices the last hours of the session, not the
   forecast horizon.
3. Gates the spread:

   | Spread (IV − HV) | Verdict |
   |---|---|
   | > +0.10 | IV rich: investigate short-vol |
   | < −0.05 | IV cheap: investigate long-vol |
   | otherwise | no edge: stand down |

4. Classifies the sign of the GEX proxy as trend-amplifying (dealers net short gamma) or pinning (net long).

## Data providers

All three return the same frame shapes, so `--provider` is the only thing that changes.

| Provider | Key | Greeks |
|---|---|---|
| `yahoo` (default) | none | Computed locally with Black-Scholes from Yahoo's IV |
| `polygon` | `MASSIVEKEY` | From the snapshot endpoint |
| `tradier` | `TRADIER_TOKEN` (sandbox by default; `TRADIER_ENV=production` for live) | From the chains endpoint |

Keys are read from an environment variable of that name, or from a gitignored `key.md` in the repo root
(`NAME="value"`). A missing key raises an error instead of sending an unauthenticated request.

## Usage

```bash
python3 -m venv .venv
.venv/bin/pip install arch numpy pandas requests yfinance pytest

.venv/bin/python scripts/scan_ticker.py TSLA
.venv/bin/python scripts/scan_ticker.py PLTR --provider tradier --horizon 10
```

## Tests

```bash
.venv/bin/python -m pytest -q
```

The 22 tests pin down the EWMA recursion, GARCH output bounds, the spread, the GEX sign and dollar
scaling, the Black-Scholes greeks, and the 0DTE expiry filter.
