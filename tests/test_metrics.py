from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

from runtime.metrics import (
    calc_calmar_ratio,
    calc_max_drawdown,
    calc_sharpe_ratio,
    parse_percentage,
    safe_parse_date,
)


def test_safe_parse_date_handles_compact_and_iso_dates() -> None:
    parsed = safe_parse_date(pd.Series(["20251208", "2025-12-09", "invalid"]))

    assert parsed.iloc[0] == pd.Timestamp("2025-12-08")
    assert parsed.iloc[1] == pd.Timestamp("2025-12-09")
    assert pd.isna(parsed.iloc[2])
    assert safe_parse_date("20251231") == pd.Timestamp("2025-12-31")
    assert pd.isna(safe_parse_date("bad-date"))


def test_max_drawdown_reports_peak_and_trough() -> None:
    nav = pd.Series([100.0, 120.0, 90.0, 110.0], index=[1, 2, 3, 4])

    max_dd, peak_idx, trough_idx = calc_max_drawdown(nav)

    assert max_dd == pytest.approx(-0.25)
    assert peak_idx == 2
    assert trough_idx == 3


def test_max_drawdown_handles_flat_or_invalid_series() -> None:
    assert calc_max_drawdown([100.0, 100.0]) == (0.0, None, None)
    assert calc_max_drawdown([0.0, 1.0]) == (0.0, None, None)
    assert calc_max_drawdown([100.0]) == (0.0, None, None)
    assert calc_max_drawdown([100.0, math.nan, 90.0])[0] == pytest.approx(-0.1)


def test_sharpe_ratio_is_zero_for_short_or_constant_returns() -> None:
    assert calc_sharpe_ratio([0.01]) == 0.0
    assert calc_sharpe_ratio([0.01, 0.01, 0.01]) == 0.0


def test_sharpe_ratio_matches_manual_calculation() -> None:
    returns = np.array([0.01, 0.02, -0.01, 0.03])
    rf_weekly = (1.015 ** (1 / 52)) - 1
    excess = returns - rf_weekly
    expected = excess.mean() / excess.std(ddof=1) * np.sqrt(52)

    assert calc_sharpe_ratio(returns) == pytest.approx(expected)


def test_calmar_ratio_handles_zero_drawdown() -> None:
    assert calc_calmar_ratio(0.12, -0.06) == pytest.approx(2.0)
    assert calc_calmar_ratio(0.12, 0.0) == 0.0
    assert calc_calmar_ratio(float("nan"), -0.06) == 0.0


def test_parse_percentage() -> None:
    assert parse_percentage("5.66%") == pytest.approx(0.0566)
    assert parse_percentage("0.973") == pytest.approx(0.973)
    assert math.isnan(parse_percentage("-"))
