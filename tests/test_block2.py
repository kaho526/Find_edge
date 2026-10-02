"""Crisis-window membership for Block 2: t counts if any of t+1..t+21 is inside a window."""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "research"))

from block2_vrp import crisis_flags  # noqa: E402


def test_crisis_flag_uses_the_target_window_not_the_origin() -> None:
    days = pd.bdate_range("2017-12-01", "2018-04-30")
    first_in = days[days.get_loc(pd.Timestamp("2018-02-01")) - 21]  # t+21 is 1 Feb 2018
    just_out = days[days.get_loc(first_in) - 1]  # t+21 is 31 Jan 2018
    last_in = pd.Timestamp("2018-02-27")  # t+1 is 28 Feb 2018
    after = pd.Timestamp("2018-02-28")  # t+1 is 1 Mar 2018
    flags = crisis_flags(pd.DatetimeIndex([just_out, first_in, last_in, after]), days)
    assert flags.tolist() == [False, True, True, False]
