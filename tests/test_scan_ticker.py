"""Tests for the report helpers in scripts/scan_ticker.py."""

from __future__ import annotations

import datetime as dt
import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from scan_ticker import _front_month_atm_iv  # noqa: E402

TODAY = dt.date(2026, 9, 25)


def _chain() -> pd.DataFrame:
    """0DTE contract with a blown-out IV next to a normal weekly."""
    return pd.DataFrame(
        {
            "expiry": [TODAY, TODAY, TODAY + dt.timedelta(days=7)],
            "strike": [100.0, 101.0, 100.0],
            "iv": [3.50, 3.10, 0.45],
        }
    )


def test_atm_iv_skips_0dte() -> None:
    assert _front_month_atm_iv(_chain(), spot=100.0, today=TODAY) == pytest.approx(0.45)


def test_atm_iv_raises_when_only_0dte_left() -> None:
    chain = _chain().iloc[:2]
    with pytest.raises(RuntimeError, match="no expiries after today"):
        _front_month_atm_iv(chain, spot=100.0, today=TODAY)
