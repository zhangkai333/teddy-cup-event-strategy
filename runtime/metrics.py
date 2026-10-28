from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd


def safe_parse_date(series_or_value: Any) -> pd.Series | pd.Timestamp | None:
    """Parse common YYYYMMDD, ISO-like, and datetime inputs without raising."""
    if isinstance(series_or_value, pd.Series):
        series = series_or_value.copy()
        numeric = pd.to_numeric(series, errors="coerce")
        is_compact_date = (
            numeric.notna()
            & (numeric >= 19000101)
            & (numeric <= 29991231)
            & (numeric % 1 == 0)
        )
        if is_compact_date.any():
            parsed = pd.to_datetime(
                numeric.where(is_compact_date).astype("Int64").astype(str),
                format="%Y%m%d",
                errors="coerce",
            )
            remaining = series[~is_compact_date]
            if not remaining.empty:
                parsed.loc[remaining.index] = pd.to_datetime(
                    remaining.astype(str),
                    errors="coerce",
                    format="mixed",
                )
            return parsed
        return pd.to_datetime(series.astype(str), errors="coerce", format="mixed")

    if series_or_value is None or pd.isna(series_or_value):
        return None
    if isinstance(series_or_value, (pd.Timestamp, np.datetime64)):
        return pd.Timestamp(series_or_value)

    text = str(series_or_value).strip()
    if not text:
        return None
    if text.isdigit() and len(text) == 8:
        return pd.to_datetime(text, format="%Y%m%d", errors="coerce")
    return pd.to_datetime(text, errors="coerce")


def calc_max_drawdown(
    nav_series: pd.Series | list[float] | np.ndarray,
) -> tuple[float, Any, Any]:
    """Return max drawdown, peak index, and trough index using a running high."""
    nav = pd.Series(nav_series, copy=True).apply(pd.to_numeric, errors="coerce")
    nav = nav.replace([np.inf, -np.inf], np.nan).dropna()
    if len(nav) < 2 or (nav <= 0).any():
        return 0.0, None, None

    drawdown = nav / nav.cummax() - 1.0
    max_dd = float(drawdown.min())
    if max_dd >= 0:
        return 0.0, None, None

    trough_idx = drawdown.idxmin()
    before_trough = nav.loc[:trough_idx].iloc[:-1]
    peak_idx = before_trough.idxmax() if not before_trough.empty else nav.index[0]
    return max_dd, peak_idx, trough_idx


def calc_sharpe_ratio(
    periodic_returns: pd.Series | list[float] | np.ndarray,
    risk_free_rate_annual: float = 0.015,
    periods_per_year: int = 52,
) -> float:
    """Calculate an annualized Sharpe ratio from periodic simple returns."""
    returns = np.asarray(periodic_returns, dtype=float)
    returns = returns[np.isfinite(returns)]
    if returns.size < 2:
        return 0.0

    rf_periodic = (1.0 + risk_free_rate_annual) ** (1.0 / periods_per_year) - 1.0
    excess_returns = returns - rf_periodic
    volatility = excess_returns.std(ddof=1)
    if not np.isfinite(volatility) or volatility < 1e-10:
        return 0.0
    return float(excess_returns.mean() / volatility * np.sqrt(periods_per_year))


def calc_calmar_ratio(annual_return: float, max_drawdown: float) -> float:
    """Calculate annual return divided by absolute max drawdown."""
    if not np.isfinite(annual_return) or not np.isfinite(max_drawdown):
        return 0.0
    if abs(max_drawdown) < 1e-10:
        return 0.0
    return float(annual_return / abs(max_drawdown))


def parse_percentage(value: Any) -> float:
    """Convert strings such as '5.66%' to the decimal value 0.0566."""
    if value is None or (not isinstance(value, str) and pd.isna(value)):
        return float("nan")
    text = str(value).strip()
    if not text or text == "-":
        return float("nan")
    if text.endswith("%"):
        return float(text[:-1]) / 100.0
    return float(text)
