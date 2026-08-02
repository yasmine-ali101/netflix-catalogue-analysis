"""Reusable dataset profiling helpers.

Extracted from three separate EDA notebooks that each re-implemented the same
histogram/boxplot/barplot functions by copy-paste. That duplication is what let a
bug survive in the Netflix notebook: its numeric-plotting cell still carried the
FIFA column list (`['Age', 'Overall', 'Potential', 'Value', 'Wage', ...]`), so
the cell either raised a `KeyError` or silently plotted nothing.

Column lists are now derived from the dataframe rather than pasted in, which
makes that class of error impossible.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def numeric_columns(frame: pd.DataFrame, max_unique_for_categorical: int = 0) -> list[str]:
    """Numeric columns, derived from dtypes rather than hardcoded."""
    columns = frame.select_dtypes(include=[np.number]).columns.tolist()
    if max_unique_for_categorical:
        columns = [c for c in columns if frame[c].nunique() > max_unique_for_categorical]
    return columns


def categorical_columns(frame: pd.DataFrame, max_unique: int = 50) -> list[str]:
    """Object/categorical columns with a tractable number of levels.

    The cap matters: `cast` in the Netflix data has tens of thousands of unique
    values, and a bar chart of it is unreadable rather than merely large.
    """
    candidates = frame.select_dtypes(include=["object", "category", "bool"]).columns
    return [c for c in candidates if frame[c].nunique() <= max_unique]


def missing_report(frame: pd.DataFrame) -> pd.DataFrame:
    """Per-column missing counts and percentages, worst first."""
    missing = frame.isna().sum()
    report = pd.DataFrame(
        {
            "missing": missing,
            "missing_pct": (missing / len(frame) * 100).round(2),
            "dtype": frame.dtypes.astype(str),
            "unique": frame.nunique(),
        }
    )
    return report.sort_values("missing", ascending=False)


def outlier_bounds(series: pd.Series, factor: float = 1.5) -> tuple[float, float]:
    """Tukey fences."""
    q1, q3 = series.quantile(0.25), series.quantile(0.75)
    iqr = q3 - q1
    return q1 - factor * iqr, q3 + factor * iqr


def outlier_summary(frame: pd.DataFrame, columns: list[str] | None = None) -> pd.DataFrame:
    """Count IQR outliers per numeric column."""
    columns = columns or numeric_columns(frame)
    rows = []
    for column in columns:
        series = frame[column].dropna()
        if series.empty:
            continue
        lower, upper = outlier_bounds(series)
        mask = (series < lower) | (series > upper)
        rows.append(
            {
                "column": column,
                "outliers": int(mask.sum()),
                "outlier_pct": round(mask.mean() * 100, 2),
                "lower_bound": round(float(lower), 3),
                "upper_bound": round(float(upper), 3),
                "skew": round(float(series.skew()), 3),
            }
        )
    return pd.DataFrame(rows).sort_values("outlier_pct", ascending=False)


def explode_multi_value(frame: pd.DataFrame, column: str, separator: str = ",") -> pd.Series:
    """Split a delimited column (`listed_in`, `cast`, `country`) into one row per value.

    Returns a Series rather than mutating the frame. The notebooks did this with
    `df_copy = df`, which is a reference, not a copy, so each "exploded copy"
    silently overwrote the original dataframe's column, and every later cell ran
    against mutated data.
    """
    return (
        frame[column]
        .dropna()
        .astype(str)
        .str.split(separator)
        .explode()
        .str.strip()
        .replace("", np.nan)
        .dropna()
    )


def top_values(frame: pd.DataFrame, column: str, n: int = 10, separator: str | None = None) -> pd.Series:
    """Most frequent values, handling delimited multi-value columns."""
    series = explode_multi_value(frame, column, separator) if separator else frame[column].dropna()
    return series.value_counts().head(n)
