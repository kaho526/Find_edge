"""Block 2 of the study pre-registered in HYPOTHESES.md: implied vs realised volatility.

Usage (from the repo root, after research/block1_forecasts.py):
    .venv/bin/python research/block2_vrp.py              # full run
    .venv/bin/python research/block2_vrp.py --download   # re-download VIX first
    .venv/bin/python research/block2_vrp.py --report-only  # figures + RESULTS.md from saved tables

Uses Block 1's saved out-of-sample forecasts (results/block1_forecasts.csv) and
refits nothing. This compares volatilities; it is not an options backtest.

Stages:
    1. data      Cboe VIX daily closes into data/, manifest entry "vix"
    2. panel     Block 1 forecasts joined to VIX by origin date t; crisis flags
    3. evaluate  H3, H4, H5, engine buckets, crisis-excluded robustness
    4. report    figures and the Block 2 section of RESULTS.md, from saved tables
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "research"), str(ROOT / "scripts")]

import block1_forecasts as b1  # noqa: E402
from scan_ticker import LONG_VOL_GATE, SHORT_VOL_GATE  # noqa: E402

VIX_URL = "https://cdn.cboe.com/api/global/us_indices/daily_prices/VIX_History.csv"
VIX_FILE = ROOT / "data" / "vix_history.csv"
H3_CSV = b1.RESULTS / "block2_h3.csv"
H4_CSV = b1.RESULTS / "block2_h4.csv"
H5_CSV = b1.RESULTS / "block2_h5.csv"
ENGINE_CSV = b1.RESULTS / "block2_engine.csv"

# Pre-specified in HYPOTHESES.md.
CRISIS_WINDOWS = (
    ("2008-09-01", "2009-03-31"),
    ("2018-02-01", "2018-02-28"),
    ("2020-02-01", "2020-04-30"),
)
WORST_SHARE = 0.01
FORECASTS = ("har", "ewma", "garch")  # HAR is the pre-registered forecast; others robustness
SAMPLES = ("all", "ex-crisis")


# --- 1. data ----------------------------------------------------------------


def download_vix() -> None:
    import requests

    resp = requests.get(VIX_URL, timeout=30)
    resp.raise_for_status()
    VIX_FILE.parent.mkdir(exist_ok=True)
    VIX_FILE.write_bytes(resp.content)
    vix = load_vix()
    manifest = json.loads(b1.MANIFEST.read_text())
    manifest["vix"] = {
        "source": "Cboe Global Markets, official VIX daily price history",
        "url": VIX_URL,
        "fields": ["DATE", "OPEN", "HIGH", "LOW", "CLOSE"],
        "used": "CLOSE, divided by 100 (volatility points to decimal)",
        "file": str(VIX_FILE.relative_to(ROOT)),
        "download_date": dt.date.today().isoformat(),
        "first_date": vix.index[0].date().isoformat(),
        "last_date": vix.index[-1].date().isoformat(),
        "rows": len(vix),
        "sha256": hashlib.sha256(VIX_FILE.read_bytes()).hexdigest(),
    }
    b1.MANIFEST.write_text(json.dumps(manifest, indent=2) + "\n")


def load_vix() -> pd.Series:
    """VIX close as a decimal (19.5 points -> 0.195), indexed by date."""
    d = pd.read_csv(VIX_FILE)
    idx = pd.to_datetime(d["DATE"], format="%m/%d/%Y")
    return (d["CLOSE"] / 100.0).set_axis(idx).sort_index().rename("vix")


# --- 2. panel ---------------------------------------------------------------


def crisis_flags(origins: pd.DatetimeIndex, trading_days: pd.DatetimeIndex) -> np.ndarray:
    """True where any day of the target window t+1..t+21 falls in a crisis window."""
    pos = trading_days.get_indexer(origins)
    if (pos < 0).any():
        raise ValueError("origin missing from the trading-day calendar")
    first, last = trading_days[pos + 1], trading_days[pos + b1.HORIZON]
    flag = np.zeros(len(origins), dtype=bool)
    for start, end in CRISIS_WINDOWS:
        flag |= (first <= pd.Timestamp(end)) & (last >= pd.Timestamp(start))
    return flag


def build_panel() -> tuple[pd.DataFrame, int]:
    """Block 1 forecasts joined to VIX_t by origin date. Returns (panel, n Block 1 origins)."""
    fc = pd.read_csv(b1.FORECASTS_CSV, index_col="date", parse_dates=True)
    panel = fc.join(load_vix(), how="inner")
    trading_days = pd.read_csv(b1.DATA_FILE, index_col="Date", parse_dates=True).index
    panel["crisis"] = crisis_flags(panel.index, trading_days)
    panel["gap"] = panel["vix"] - panel["rv"]
    return panel, len(fc)


def _sample(panel: pd.DataFrame, name: str) -> pd.DataFrame:
    return panel if name == "all" else panel[~panel["crisis"]]


# --- 3. evaluate ------------------------------------------------------------


def h3_row(p: pd.DataFrame) -> dict[str, float]:
    """Mean VIX - RV gap, one-sided H1: mean > 0, Newey-West SE."""
    from scipy.stats import norm

    mean, t = b1._nw_mean_test(p["gap"])
    return {"mean_gap": mean, "median_gap": float(p["gap"].median()),
            "share_vix_above_rv": float((p["vix"] > p["rv"]).mean()),
            "nw_t": t, "p_one_sided": float(1 - norm.cdf(t))}


def h4_row(p: pd.DataFrame, forecast: str) -> dict[str, float]:
    """RV = a + b·VIX + c·F, Newey-West SE; one-sided H1: c > 0."""
    import statsmodels.api as sm
    from scipy.stats import norm

    fit = sm.OLS(p["rv"], sm.add_constant(p[["vix", forecast]])).fit(
        cov_type="HAC", cov_kwds={"maxlags": b1.NW_LAGS}
    )
    return {"a": fit.params["const"], "a_se": fit.bse["const"],
            "b": fit.params["vix"], "b_se": fit.bse["vix"],
            "c": fit.params[forecast], "c_se": fit.bse[forecast], "c_t": fit.tvalues[forecast],
            "p_one_sided": float(1 - norm.cdf(fit.tvalues[forecast])), "r2": fit.rsquared}


def evaluate() -> None:
    panel, n_block1 = build_panel()
    h3, h4 = [], []
    for name in SAMPLES:
        p = _sample(panel, name)
        span = {"sample": name, "n": len(p), "first_origin": p.index[0].date(),
                "last_origin": p.index[-1].date()}
        h3.append(dict(span, n_block1_origins=n_block1, n_crisis=int(panel["crisis"].sum()),
                       **h3_row(p)))
        for f in FORECASTS:
            h4.append(dict(span, forecast=f, role="primary" if f == "har" else "robustness",
                           **h4_row(p, f)))

    threshold = panel["gap"].quantile(WORST_SHARE)
    worst = panel[panel["gap"] <= threshold]
    lowest = panel["gap"].idxmin()
    h5 = {"n": len(panel), "skewness": float(panel["gap"].skew()),
          "worst_share": WORST_SHARE, "worst_threshold": threshold, "n_worst": len(worst),
          "share_worst_in_crisis": float(worst["crisis"].mean()),
          "share_all_in_crisis": float(panel["crisis"].mean()),
          "min_gap": float(panel["gap"].min()), "min_gap_origin": lowest.date(),
          "min_gap_in_crisis": bool(panel.loc[lowest, "crisis"])}

    spread = panel["vix"] - panel["har"]
    buckets = {
        f"rich (spread > {SHORT_VOL_GATE:+.2f})": spread > SHORT_VOL_GATE,
        f"between ({LONG_VOL_GATE:+.2f} to {SHORT_VOL_GATE:+.2f})":
            (spread >= LONG_VOL_GATE) & (spread <= SHORT_VOL_GATE),
        f"cheap (spread < {LONG_VOL_GATE:+.2f})": spread < LONG_VOL_GATE,
    }
    engine = [{"bucket": k, "n": int(m.sum()), "share_of_days": float(m.mean()),
               "share_rv_below_vix": float((panel.loc[m, "rv"] < panel.loc[m, "vix"]).mean())
               if m.any() else np.nan}
              for k, m in buckets.items()]

    pd.DataFrame(h3).to_csv(H3_CSV, index=False, float_format="%.6g")
    pd.DataFrame(h4).to_csv(H4_CSV, index=False, float_format="%.6g")
    pd.DataFrame([h5]).to_csv(H5_CSV, index=False, float_format="%.6g")
    pd.DataFrame(engine).to_csv(ENGINE_CSV, index=False, float_format="%.6g")


# --- 4. report --------------------------------------------------------------


def _footnote(manifest: dict, first: str, last: str) -> str:
    return (f"Sources: {manifest['vix']['source']}; {manifest['source']}, S&P 500 "
            f"({manifest['ticker']}).\nForecast origins {first} to {last}; "
            f"realised volatility covers t+1 to t+21.")


def make_figures() -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    b1.FIGURES.mkdir(parents=True, exist_ok=True)
    manifest = json.loads(b1.MANIFEST.read_text())
    h3 = pd.read_csv(H3_CSV).set_index("sample").loc["all"]
    h5 = pd.read_csv(H5_CSV).iloc[0]
    panel, _ = build_panel()
    foot = _footnote(manifest, h3.first_origin, h3.last_origin)

    # (1) VIX against the realised volatility that followed.
    fig, ax = plt.subplots(figsize=(10, 5))
    for start, end in CRISIS_WINDOWS:
        ax.axvspan(pd.Timestamp(start), pd.Timestamp(end), color="#f4cccc", lw=0)
    ax.plot(panel.index, panel["rv"] * 100, color="black", lw=0.8,
            label="Realised volatility, t+1 to t+21")
    ax.plot(panel.index, panel["vix"] * 100, color="#b2182b", lw=0.8, alpha=0.85,
            label="VIX at the close of t")
    ax.plot([], [], color="#f4cccc", lw=8, label="Pre-registered crisis windows")
    ax.set_ylabel("Annualised volatility (%)")
    ax.set_xlabel("Forecast origin date t")
    where = "inside" if h5.min_gap_in_crisis else "outside"
    ax.set_title(
        f"VIX was above the volatility that followed on {h3.share_vix_above_rv * 100:.0f}% of days, "
        f"by a median {h3.median_gap * 100:.1f} vol points\n"
        f"The largest shortfall, {-h5.min_gap * 100:.0f} points at origin {h5.min_gap_origin}, "
        f"came {where} a crisis window",
        fontsize=10.5,
    )
    ax.legend(frameon=False, loc="upper right")
    fig.text(0.01, 0.01, foot, fontsize=7, color="#444444")
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    fig.savefig(b1.FIGURES / "vix_vs_realised.png", dpi=150)
    plt.close(fig)

    # (2) Distribution of VIX - RV with the worst 1% marked.
    gap = panel["gap"] * 100
    cut = h5.worst_threshold * 100
    bins = np.arange(np.floor(gap.min()), np.ceil(gap.max()) + 1, 1.0)
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.hist(gap[gap > cut], bins=bins, color="#4d4d4d", label="Other days")
    ax.hist(gap[gap <= cut], bins=bins, color="#b2182b")
    # One or two days per bin are invisible at this scale, so tick each worst day too.
    ax.plot(gap[gap <= cut], np.zeros((gap <= cut).sum()), "|", color="#b2182b", ms=14, mew=1.2,
            label=f"Worst {h5.worst_share:.0%}, one tick per day (n = {int(h5.n_worst)})")
    ax.axvline(cut, color="#b2182b", ls="--", lw=1)
    ax.axvline(0, color="black", lw=0.8)
    ax.set_xlabel("VIX − subsequent realised volatility (vol points; negative = VIX too low)")
    ax.set_ylabel("Number of forecast origins")
    sign = "negatively" if h5.skewness < 0 else "not negatively"
    ax.set_title(
        f"The gap is {sign} skewed (skewness {h5.skewness:.2f}): its worst {h5.worst_share:.0%} of "
        f"days, below {cut:.0f} points,\nfall {h5.share_worst_in_crisis * 100:.0f}% inside crisis "
        f"windows, against {h5.share_all_in_crisis * 100:.0f}% of all days",
        fontsize=10.5,
    )
    ax.legend(frameon=False)
    fig.text(0.01, 0.01, foot, fontsize=7, color="#444444")
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    fig.savefig(b1.FIGURES / "vrp_histogram.png", dpi=150)
    plt.close(fig)


def h3_verdict(h3: pd.DataFrame) -> str:
    r = h3.set_index("sample").loc["all"]
    return "supported" if r.mean_gap > 0 and r.p_one_sided < b1.ALPHA else "not supported"


def h4_verdict(h4: pd.DataFrame) -> str:
    r = h4[(h4["sample"] == "all") & (h4.forecast == "har")].iloc[0]
    return "supported" if r.c > 0 and r.p_one_sided < b1.ALPHA else "not supported"


def h5_verdict(h5: pd.DataFrame) -> str:
    r = h5.iloc[0]
    return "supported" if r.skewness < 0 and r.share_worst_in_crisis > 0.5 else "not supported"


def write_results_md() -> None:
    manifest = json.loads(b1.MANIFEST.read_text())
    vm = manifest["vix"]
    h3 = pd.read_csv(H3_CSV)
    h4 = pd.read_csv(H4_CSV)
    h5 = pd.read_csv(H5_CSV)
    engine = pd.read_csv(ENGINE_CSV)
    a = h3.set_index("sample").loc["all"]
    r5 = h5.iloc[0]
    _p = b1._p

    def vp(x: float) -> str:  # decimal vol to signed vol points
        return f"{x * 100:+.2f}"

    h3_tab = ["| Sample | Days | Mean VIX − RV (vol pts) | Median (vol pts) | Share VIX > RV "
              "| NW t | One-sided p |", "|---|---|---|---|---|---|---|"]
    for _, r in h3.iterrows():
        h3_tab.append(f"| {r['sample']} | {r.n} | {vp(r.mean_gap)} | {vp(r.median_gap)} | "
                      f"{r.share_vix_above_rv:.1%} | {r.nw_t:.2f} | {_p(r.p_one_sided)} |")

    h4_tab = ["| Sample | Forecast F | Role | a (SE) | b on VIX (SE) | c on F (SE) | One-sided p, c > 0 "
              "| R² |", "|---|---|---|---|---|---|---|---|"]
    for _, r in h4.iterrows():
        h4_tab.append(f"| {r['sample']} | {r.forecast.upper()} | {r.role} | {r.a:.4f} ({r.a_se:.4f}) "
                      f"| {r.b:.3f} ({r.b_se:.3f}) | {r.c:.3f} ({r.c_se:.3f}) | "
                      f"{_p(r.p_one_sided)} | {r.r2:.3f} |")

    eng_tab = ["| Bucket | Days | Share of days | Share with RV below VIX |", "|---|---|---|---|"]
    for _, r in engine.iterrows():
        share = "n/a" if pd.isna(r.share_rv_below_vix) else f"{r.share_rv_below_vix:.1%}"
        eng_tab.append(f"| {r.bucket} | {r.n} | {r.share_of_days:.1%} | {share} |")

    v3, v4, v5 = h3_verdict(h3), h4_verdict(h4), h5_verdict(h5)
    L = [
        b1.BLOCK2_MARKER,
        "",
        "# Block 2 results: implied vs realised volatility",
        "",
        "_Generated by `research/block2_vrp.py` from the `block2_*.csv` tables in `results/`. "
        "Do not edit by hand._",
        "",
        "> **This is not an options trading backtest.** It compares volatility levels: VIX "
        "against the realised volatility that followed. There are no option prices, transaction "
        "costs, margin or P&L, and a positive average gap does not mean a short-volatility "
        "strategy would have made money after costs and tail losses.",
        "",
        f"- **VIX:** {vm['source']}, daily close, {vm['first_date']} to {vm['last_date']} "
        f"({vm['rows']} rows), downloaded {vm['download_date']}. Converted from volatility points "
        "to decimals; tables show vol points.",
        f"- **Sample:** {a.n} of Block 1's {a.n_block1_origins} out-of-sample origins have a VIX "
        f"close on the same date ({a.first_origin} to {a.last_origin}). {a.n_crisis} of them are "
        "in a crisis window, meaning some day of their 21-day target window falls inside one.",
        "- **Forecasts:** Block 1's saved out-of-sample forecasts. No model was refitted.",
        f"- **Tests:** Newey-West standard errors with {b1.NW_LAGS} lags, p-values from the normal "
        "approximation.",
        "",
        "## How it was done",
        "",
        "1. **Data.** Cboe's official VIX closing values, matched to each forecast date t. VIX_t is "
        "the market's price for volatility over roughly the next month, set at the close of t.",
        "2. **The gap.** For each day, VIX_t minus the realised volatility of the next 21 trading "
        "days. If option buyers pay a premium for protection, the gap is positive on average.",
        "3. **Does the forecast add anything?** Realised volatility is regressed on VIX and the "
        "HAR forecast together. If the forecast's coefficient is positive and significant, it "
        "knows something VIX did not already price.",
        "4. **The tail.** How skewed the gap is, and whether its worst days cluster in the "
        "pre-registered crisis windows.",
        "5. **Robustness.** H3 and H4 repeated without crisis-window days, and H4 repeated with "
        "the EWMA and GARCH forecasts.",
        "",
        "## H3. A variance risk premium exists",
        "",
        "> H0: E[VIX_t − RV(t+1, t+21)] = 0. H1: > 0. One-sided t-test on the mean gap, "
        "Newey-West standard errors. **Supported** if p < 0.05. Otherwise **not supported**.",
        "",
        *h3_tab,
        "",
        f"**Verdict (all days): {v3}.** The ex-crisis row is robustness only.",
        "",
        "## H4. The forecast contains information that VIX does not",
        "",
        "> RV(t+1, t+21) = a + b·VIX_t + c·F_t + e, where F is the HAR forecast made at the close "
        "of t. H0: c = 0. H1: c > 0. Newey-West standard errors. **Supported** if c > 0 with "
        "p < 0.05. Otherwise **not supported**.",
        "",
        *h4_tab,
        "",
        f"**Verdict (HAR, all days): {v4}.** EWMA, GARCH and ex-crisis rows are robustness only.",
    ]
    if v4 == "not supported":
        L += ["", "As pre-registered for this outcome: VIX already reflects what the forecast "
              "knows. The engine's spread then mostly measures the level of VIX, and the premium "
              "is compensation for risk rather than something this forecast can time."]
    L += [
        "",
        "## H5. The premium has a fat left tail",
        "",
        "> The distribution of VIX − RV is negatively skewed, and most of its worst 1% of days "
        "fall inside the crisis windows. Descriptive: sample skewness, and the share of the "
        "worst 1% of days inside crisis windows.",
        "",
        "| Days | Skewness of VIX − RV | Worst 1%: days | Worst 1%: threshold (vol pts) "
        "| Worst 1% inside crisis windows | All days inside crisis windows |",
        "|---|---|---|---|---|---|",
        f"| {r5.n} | {r5.skewness:.2f} | {r5.n_worst} | {vp(r5.worst_threshold)} | "
        f"{r5.share_worst_in_crisis:.1%} | {r5.share_all_in_crisis:.1%} |",
        "",
        f"**Verdict: {v5}.** HYPOTHESES.md gives H5 as a descriptive claim with no significance "
        "test, so this verdict reads the claim literally: supported only if both parts hold "
        "(skewness below 0, and more than half of the worst 1% of days inside crisis windows).",
        "",
        "## Engine thresholds: descriptive check",
        "",
        f"Spread = VIX_t − HAR forecast, bucketed at the engine's unchanged thresholds "
        f"({SHORT_VOL_GATE:+.2f} rich, {LONG_VOL_GATE:+.2f} cheap). For each bucket, the share of "
        "days on which realised volatility came in below VIX.",
        "",
        *eng_tab,
        "",
        "**Descriptive only, not evidence of predictive power.** The spread and the outcome both "
        "contain VIX: a high VIX makes the spread large and also makes 'RV below VIX' more likely, "
        "whatever the forecast knows. H4 is the formal test.",
        "",
        "## Figures",
        "",
        "![VIX against subsequent realised volatility](figures/vix_vs_realised.png)",
        "",
        "![Distribution of VIX minus realised volatility](figures/vrp_histogram.png)",
        "",
        "## Known limitations (from HYPOTHESES.md)",
        "",
        "- Daily close-to-close realised volatility is a noisy target, which lowers the power of "
        "every test.",
        "- VIX is a model-free variance measure across all strikes, not at-the-money implied "
        "volatility, which is what the engine uses.",
        "- VIX covers 30 calendar days; the target covers 21 trading days.",
        "- This compares volatilities. It is not an options trading backtest.",
        "",
        "## Files",
        "",
        "- `block2_h3.csv`, `block2_h4.csv`, `block2_h5.csv`, `block2_engine.csv`: the tables "
        "above. Raw VIX values stay in `data/` and are not committed.",
        "",
    ]
    old = b1.RESULTS_MD.read_text()
    head = old[: old.find(b1.BLOCK2_MARKER)] if b1.BLOCK2_MARKER in old else old
    b1.RESULTS_MD.write_text(head.rstrip("\n") + "\n\n" + "\n".join(L))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--download", action="store_true", help="re-download VIX")
    ap.add_argument("--report-only", action="store_true",
                    help="rebuild figures and the Block 2 section from saved tables")
    args = ap.parse_args()
    if not args.report_only:
        manifest = json.loads(b1.MANIFEST.read_text())
        if args.download or not VIX_FILE.exists() or "vix" not in manifest:
            download_vix()
        evaluate()
    make_figures()
    write_results_md()


if __name__ == "__main__":
    main()
