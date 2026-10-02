"""No-lookahead and alignment tests for the Block 1 study."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "research"))

from block1_forecasts import MODELS, make_forecasts, realised_vol_target  # noqa: E402


def _simulate_garch(n: int, seed: int = 7) -> np.ndarray:
    """GARCH(1,1) daily returns, so the data has the clustering the models expect."""
    rng = np.random.default_rng(seed)
    omega, alpha, beta = 2e-6, 0.08, 0.90
    var, out = omega / (1 - alpha - beta), np.empty(n)
    for i in range(n):
        out[i] = np.sqrt(var) * rng.standard_normal()
        var = omega + alpha * out[i] ** 2 + beta * var
    return out


@pytest.fixture(scope="module")
def runs() -> tuple[pd.DataFrame, pd.DataFrame, pd.Timestamp]:
    """Forecasts on the same returns, before and after scaling every return after t.

    Origins span three calendar months and t sits mid-month, so a GARCH/HAR
    refit happens after t on the perturbed data.
    """
    idx = pd.bdate_range("2020-01-01", periods=700)
    r = pd.Series(_simulate_garch(len(idx)), index=idx)
    origins = idx[(idx >= "2022-03-01") & (idx < "2022-06-01")]
    t = pd.Timestamp("2022-04-13")  # a Wednesday, so t+1 is a trading day

    perturbed = r.copy()
    perturbed[perturbed.index > t] *= 3.0
    return make_forecasts(r, origins), make_forecasts(perturbed, origins), t


@pytest.mark.parametrize("model", MODELS)
def test_forecast_at_t_ignores_data_after_t(runs, model: str) -> None:
    base, perturbed, t = runs
    pd.testing.assert_series_equal(base.loc[:t, model], perturbed.loc[:t, model])


@pytest.mark.parametrize("model", MODELS)
def test_perturbation_reaches_later_forecasts(runs, model: str) -> None:
    """Guards the test above: if nothing changed after t it would prove nothing."""
    base, perturbed, t = runs
    assert not np.allclose(base.loc[base.index > t, model],
                           perturbed.loc[perturbed.index > t, model])


def test_target_covers_t_plus_1_to_t_plus_21() -> None:
    r = pd.Series(np.arange(1, 61) / 1000.0, index=pd.bdate_range("2024-01-01", periods=60))
    rv = realised_vol_target(r)
    expected = np.sqrt(252 * np.mean(r.iloc[6:27] ** 2))  # origin index 5
    assert rv.iloc[5] == pytest.approx(expected)
    assert rv.iloc[-21:].isna().all() and rv.iloc[-22:-21].notna().all()
