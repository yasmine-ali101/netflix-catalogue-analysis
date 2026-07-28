"""Tests for the shared profiling library.

Two of these are regression tests for bugs that existed in the original
notebooks — see the README.
"""

import numpy as np
import pandas as pd
import pytest

from edatools import profiling


@pytest.fixture
def mixed_frame():
    return pd.DataFrame(
        {
            "age": [20, 30, 40, 50],
            "score": [1.5, 2.5, 3.5, 4.5],
            "name": ["a", "b", "c", "d"],
            "grade": ["A", "B", "A", "B"],
            "flag": [True, False, True, False],
        }
    )


def test_numeric_columns_are_derived_from_dtypes_not_hardcoded():
    """The fix for the Netflix notebook's dead plotting cell.

    It carried FIFA's column list, so every column it named was absent.
    Deriving from dtypes makes that impossible.
    """
    frame = pd.DataFrame({"runtime": [90, 100], "title": ["x", "y"]})

    assert profiling.numeric_columns(frame) == ["runtime"]
    # None of FIFA's columns leak in.
    assert "Overall" not in profiling.numeric_columns(frame)


def test_categorical_columns_exclude_high_cardinality_ones():
    """`cast` has tens of thousands of levels; a bar chart of it is unreadable."""
    frame = pd.DataFrame(
        {"grade": ["A", "B"] * 50, "cast": [f"actor_{i}" for i in range(100)]}
    )

    columns = profiling.categorical_columns(frame, max_unique=50)

    assert "grade" in columns
    assert "cast" not in columns


def test_missing_report_counts_and_ranks_by_severity(mixed_frame):
    frame = mixed_frame.copy()
    frame.loc[0:2, "score"] = np.nan
    frame.loc[0, "name"] = np.nan

    report = profiling.missing_report(frame)

    assert report.index[0] == "score"
    assert report.loc["score", "missing"] == 3
    assert report.loc["score", "missing_pct"] == 75.0
    assert report.loc["age", "missing"] == 0


def test_outlier_bounds_are_tukey_fences():
    series = pd.Series([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])

    lower, upper = profiling.outlier_bounds(series)

    q1, q3 = series.quantile(0.25), series.quantile(0.75)
    assert lower == pytest.approx(q1 - 1.5 * (q3 - q1))
    assert upper == pytest.approx(q3 + 1.5 * (q3 - q1))


def test_outlier_summary_flags_an_extreme_value():
    frame = pd.DataFrame({"value": [10, 11, 12, 11, 10, 12, 11, 5000]})

    summary = profiling.outlier_summary(frame, ["value"])

    assert summary.loc[0, "outliers"] == 1
    assert summary.loc[0, "skew"] > 2


def test_explode_multi_value_splits_and_strips():
    frame = pd.DataFrame({"listed_in": ["Dramas, Comedies", "Documentaries"]})

    result = profiling.explode_multi_value(frame, "listed_in")

    assert sorted(result.tolist()) == ["Comedies", "Documentaries", "Dramas"]


def test_explode_multi_value_does_not_mutate_the_source_frame():
    """Regression test for the notebook's `df_copy = df` bug.

    That was a second reference to the same object, so each 'exploded copy'
    overwrote the original column and every later cell saw mutated data.
    """
    frame = pd.DataFrame({"listed_in": ["Dramas, Comedies", "Documentaries"]})
    before = frame["listed_in"].tolist()

    profiling.explode_multi_value(frame, "listed_in")

    assert frame["listed_in"].tolist() == before
    assert isinstance(frame["listed_in"].iloc[0], str)  # still a string, not a list


def test_explode_multi_value_drops_blanks_and_nulls():
    frame = pd.DataFrame({"country": ["USA, , UK", None, "France"]})

    result = profiling.explode_multi_value(frame, "country")

    assert sorted(result.tolist()) == ["France", "UK", "USA"]


def test_top_values_handles_delimited_columns():
    frame = pd.DataFrame(
        {"listed_in": ["Dramas, Comedies", "Dramas", "Dramas, Documentaries"]}
    )

    result = profiling.top_values(frame, "listed_in", n=2, separator=",")

    assert result.index[0] == "Dramas"
    assert result.iloc[0] == 3


def test_top_values_without_separator_is_a_plain_value_count():
    frame = pd.DataFrame({"type": ["Movie", "Movie", "TV Show"]})

    result = profiling.top_values(frame, "type", n=2)

    assert result.to_dict() == {"Movie": 2, "TV Show": 1}
