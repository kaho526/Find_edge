"""Realised-volatility forecast models.

Each model is ``f(returns_up_to_t, horizon=21) -> annualised vol``: it sees
daily log returns up to and including the forecast origin t and forecasts
the volatility of returns t+1..t+horizon. Past prices only, never implied
volatility.

GARCH and HAR also take fitted parameters, so a study can refit them monthly
and forecast daily with the parameters held fixed in between. Without
parameters they fit on the returns they are given.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from find_edge.signals import (
    TRADING_DAYS_PER_YEAR,
    fit_garch,
    forecast_ewma_vol,
    forecast_garch_vol,
)


def naive(returns: pd.Series, horizon: int = 21) -> float:
    """Realised vol of the last ``horizon`` days, t included."""
    r = returns.dropna().to_numpy()[-horizon:]
    return float(np.sqrt(TRADING_DAYS_PER_YEAR * np.mean(r**2)))


def ewma(returns: pd.Series, horizon: int = 21) -> float:
    """RiskMetrics EWMA (λ = 0.94). Flat term structure, so ``horizon`` is unused."""
    return forecast_ewma_vol(returns)


def garch(returns: pd.Series, horizon: int = 21, params: np.ndarray | None = None) -> float:
    """GARCH(1,1), Student-t errors, average vol over the horizon."""
    return forecast_garch_vol(returns, horizon=horizon, params=params)


garch_fit = fit_garch


def _har_design(r2: pd.Series) -> np.ndarray:
    """[1, 1-day, 5-day, 22-day mean squared return] for each day."""
    return np.column_stack(
        [np.ones(len(r2)), r2, r2.rolling(5).mean(), r2.rolling(22).mean()]
    )


def har_fit(returns: pd.Series, horizon: int = 21) -> np.ndarray:
    """OLS of next-``horizon``-day mean squared return on the HAR regressors.

    A row s is used only when its target r(s+1..s+horizon) lies inside
    ``returns``, so nothing after the last observation enters the fit.
    """
    r2 = returns.dropna().pow(2).reset_index(drop=True)
    X = _har_design(r2)
    y = r2.rolling(horizon).mean().shift(-horizon).to_numpy()
    ok = ~np.isnan(X).any(axis=1) & ~np.isnan(y)
    return np.linalg.lstsq(X[ok], y[ok], rcond=None)[0]


def har(returns: pd.Series, horizon: int = 21, coefs: np.ndarray | None = None) -> float:
    """HAR forecast of average daily variance over the horizon, annualised."""
    if coefs is None:
        coefs = har_fit(returns, horizon)
    r2 = returns.dropna().pow(2).reset_index(drop=True)
    var = float(_har_design(r2)[-1] @ coefs)
    if var <= 0:
        raise ValueError(f"HAR forecast non-positive variance {var:.3g}")
    return float(np.sqrt(TRADING_DAYS_PER_YEAR * var))
