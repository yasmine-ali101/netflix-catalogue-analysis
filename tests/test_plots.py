"""Tests for the plotting helpers.

These check that figures are built with the right shape and that the helpers
fail loudly on bad input, rather than checking pixels.
"""

import matplotlib
import pandas as pd
import pytest

matplotlib.use("Agg")

from edatools import plots


@pytest.fixture
def frame():
    return pd.DataFrame(
        {
            "a": [1.0, 2.0, 3.0, 4.0, 5.0],
            "b": [5.0, 4.0, 3.0, 2.0, 1.0],
            "c": [1.0, 3.0, 2.0, 5.0, 4.0],
            "cat": ["x", "y", "x", "y", "x"],
        }
    )


def test_histogram_grid_makes_one_axis_per_numeric_column(frame):
    fig = plots.histogram_grid(frame, ncols=2)

    visible = [ax for ax in fig.axes if ax.get_visible()]
    assert len(visible) == 3  # a, b, c, not the categorical column


def test_histogram_grid_hides_unused_axes_in_the_grid(frame):
    fig = plots.histogram_grid(frame, columns=["a"], ncols=3)

    assert len([ax for ax in fig.axes if ax.get_visible()]) == 1


def test_histogram_grid_raises_when_there_is_nothing_numeric():
    frame = pd.DataFrame({"only": ["text", "here"]})

    with pytest.raises(ValueError, match="No numeric columns"):
        plots.histogram_grid(frame)


def test_categorical_grid_raises_when_there_is_nothing_categorical():
    frame = pd.DataFrame({"a": [1.0, 2.0]})

    with pytest.raises(ValueError, match="No categorical columns"):
        plots.categorical_grid(frame)


def test_correlation_heatmap_is_square_over_numeric_columns(frame):
    fig = plots.correlation_heatmap(frame)

    assert fig.axes  # heatmap + colourbar
    assert fig.axes[0].get_title() == "Correlation matrix"


def test_missing_bar_handles_a_frame_with_no_missing_values(frame):
    """Must produce a readable figure rather than an empty or broken one."""
    fig = plots.missing_bar(frame)

    texts = [t.get_text() for ax in fig.axes for t in ax.texts]
    assert "No missing values" in texts
