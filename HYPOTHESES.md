# Pre-registered hypotheses: volatility study v1

**Written:** 2026-10-02, before any evaluation code was run. The commit timestamp of this file is the record.

**Rule:** this file is not edited once results exist. Hypotheses for later versions are added in a new, separately dated section.

## Scope

- **Underlying:** S&P 500 index (^GSPC), daily OHLC from Yahoo Finance, from 2000
- **Implied volatility:** Cboe VIX, official daily close history
- **Forecast origin:** the close of day t. Models use returns up to and including day t.
- **Target:** realised volatility of daily log returns from day t+1 to t+21, annualised as sqrt(252 × mean squared return)
- **Out-of-sample period:** forecast origins from 2005-01-01 to the latest date with a complete 21-day target
- **Models:** naive, EWMA (λ = 0.94), GARCH(1,1) with Student-t errors, HAR. All use past prices only; no model uses implied volatility.

## Pre-specified choices

| Choice | Value |
|---|---|
| Primary loss | QLIKE on variance |
| Secondary loss | MSE on variance |
| Significance level | 0.05 |
| Models vs naive | Bonferroni over 3 comparisons (p < 0.0167) |
| Standard errors | Newey-West, 20 lags, because 21-day targets overlap |
| Refitting | GARCH and HAR refitted monthly on past data only |
| Subperiods | 2005–2012, 2013–2019, 2020–latest |
| Crisis windows | Sep 2008–Mar 2009, Feb 2018, Feb–Apr 2020. A date t is in a crisis window if any day of its 21-day target window falls inside one. |
| Engine thresholds | spread > +0.10 rich, < −0.05 cheap, unchanged from the engine |
| Forecast used in Block 2 | HAR's out-of-sample forecast, fixed in advance; EWMA and GARCH reported as robustness only |

## Block 1: forecast accuracy

**H1. At least one model beats the naive forecast.**

- Naive forecast: realised volatility of the 21 trading days up to and including day t
- H0: E[QLIKE(model) − QLIKE(naive)] = 0. H1: < 0.
- Test: one-sided Diebold-Mariano test for each of EWMA, GARCH and HAR
- **Supported** if at least one model has lower QLIKE with Bonferroni-adjusted p < 0.0167. Otherwise **not supported**.

**H2. At a 21-day horizon, mean-reverting models beat EWMA.**

- This would reverse my capstone finding, at a 30-second intraday horizon, that EWMA outperformed GARCH.
- Reason: volatility tends to fall back after spikes. EWMA has no mean reversion, so it carries spikes forward and over-forecasts.
- Test: one-sided DM tests of GARCH vs EWMA and HAR vs EWMA
- **Supported** if both have lower QLIKE than EWMA with p < 0.05. **Contradicted** if EWMA has lower QLIKE than both. Otherwise **inconclusive**.

Also reported, without a hypothesis: Mincer-Zarnowitz regressions (an unbiased forecast has a = 0 and b = 1) and every result above by subperiod.

## Block 2: implied vs realised volatility

**H3. A variance risk premium exists.**

- H0: E[VIX_t − RV(t+1, t+21)] = 0. H1: > 0.
- Test: one-sided t-test on the mean gap, Newey-West standard errors
- **Supported** if p < 0.05. Otherwise **not supported**.

**H4. My forecast contains information that VIX does not.**

- Regression: RV(t+1, t+21) = a + b·VIX_t + c·F_t + e, where F is the HAR forecast made at the close of t
- H0: c = 0. H1: c > 0. Newey-West standard errors.
- **Supported** if c > 0 with p < 0.05. Otherwise **not supported**.
- If not supported: VIX already reflects what the forecast knows. The engine's spread then mostly measures the level of VIX, and the premium is compensation for risk rather than something this forecast can time.

**H5. The premium has a fat left tail.**

- The distribution of VIX − RV is negatively skewed, and most of its worst 1% of days fall inside the crisis windows.
- Descriptive: sample skewness, and the share of the worst 1% of days inside crisis windows

Also reported, descriptive only: for each engine threshold bucket, the share of days on which realised volatility came in below VIX. Because the spread (VIX − F) and the outcome share VIX, this is not evidence of predictive power; H4 is the formal test.

## Known limitations of v1

- Daily close-to-close realised volatility is a noisy target, which lowers the power of every test.
- VIX is a model-free variance measure across all strikes, not at-the-money implied volatility, which is what the engine uses.
- VIX covers 30 calendar days; the target covers 21 trading days. These are close but not identical.
- This compares volatilities. It is not an options trading backtest: there are no option prices, transaction costs or P&L.

## Planned for v2 (to be pre-registered separately)

- OptionMetrics 30-day at-the-money implied volatility in place of VIX, with a 30-calendar-day target
- Realised volatility from intraday data, with overnight returns handled
- Model Confidence Set in place of Bonferroni, and a Giacomini-White test of H2's mechanism
- HARQ and Realized GARCH
