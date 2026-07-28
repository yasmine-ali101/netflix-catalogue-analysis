"""Shared EDA tooling used across the three studies in this repository."""

from .profiling import (
    categorical_columns,
    explode_multi_value,
    missing_report,
    numeric_columns,
    outlier_bounds,
    outlier_summary,
    top_values,
)

__version__ = "0.1.0"
__all__ = [
    "numeric_columns",
    "categorical_columns",
    "missing_report",
    "outlier_bounds",
    "outlier_summary",
    "explode_multi_value",
    "top_values",
]
