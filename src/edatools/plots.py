"""Plotting helpers shared across the three EDA studies.

Every function takes an explicit `ax` or returns a figure, so plots can be
composed into grids instead of being emitted one-per-cell as the notebooks did.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from .profiling import categorical_columns, numeric_columns

PALETTE = ["#2563eb", "#059669", "#f59e0b", "#dc2626", "#7c3aed", "#0891b2"]


def apply_style() -> None:
    plt.style.use("seaborn-v0_8-whitegrid")
    sns.set_palette(PALETTE)


def _axes_list(axes) -> list:
    """Flatten whatever `plt.subplots` returned into a flat list of axes.

    `subplots` returns a bare Axes for a 1x1 grid, a 1-D array for a single
    row/column, and a 2-D array otherwise. Special-casing on the *number of
    columns being plotted* gets this wrong whenever the grid is larger than the
    data (one column with ncols=3 yields three axes, not one).
    """
    return list(np.atleast_1d(np.asarray(axes, dtype=object)).ravel())


def histogram_grid(
    frame: pd.DataFrame, columns: list[str] | None = None, ncols: int = 3, bins: int = 40
):
    """Distribution of every numeric column in one figure."""
    columns = columns or numeric_columns(frame)
    if not columns:
        raise ValueError("No numeric columns to plot.")

    nrows = -(-len(columns) // ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 3.4 * nrows))
    axes = _axes_list(axes)

    for ax, column in zip(axes, columns):
        sns.histplot(frame[column].dropna(), bins=bins, kde=True, ax=ax,
                     color=PALETTE[0], edgecolor="none")
        ax.set_title(column)
        ax.set_xlabel("")
    for ax in axes[len(columns):]:
        ax.set_visible(False)

    fig.tight_layout()
    return fig


def boxplot_grid(frame: pd.DataFrame, columns: list[str] | None = None, ncols: int = 3):
    """Outlier view for every numeric column."""
    columns = columns or numeric_columns(frame)
    nrows = -(-len(columns) // ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 2.6 * nrows))
    axes = _axes_list(axes)

    for ax, column in zip(axes, columns):
        sns.boxplot(x=frame[column].dropna(), ax=ax, color=PALETTE[2])
        ax.set_title(column)
        ax.set_xlabel("")
    for ax in axes[len(columns):]:
        ax.set_visible(False)

    fig.tight_layout()
    return fig


def categorical_grid(
    frame: pd.DataFrame, columns: list[str] | None = None, ncols: int = 2, top_n: int = 12
):
    """Value counts for categorical columns, capped at `top_n` levels each."""
    columns = columns or categorical_columns(frame)
    if not columns:
        raise ValueError("No categorical columns to plot.")

    nrows = -(-len(columns) // ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(7 * ncols, 3.6 * nrows))
    axes = _axes_list(axes)

    for ax, column in zip(axes, columns):
        counts = frame[column].value_counts().head(top_n)
        sns.barplot(x=counts.values, y=counts.index, ax=ax, color=PALETTE[1])
        ax.set_title(column)
        ax.set_xlabel("count")
    for ax in axes[len(columns):]:
        ax.set_visible(False)

    fig.tight_layout()
    return fig


def correlation_heatmap(frame: pd.DataFrame, columns: list[str] | None = None):
    """Lower-triangle correlation heatmap."""
    columns = columns or numeric_columns(frame)
    corr = frame[columns].corr()
    mask = np.triu(np.ones_like(corr, dtype=bool), k=1)

    fig, ax = plt.subplots(figsize=(1.1 * len(columns) + 3, 0.9 * len(columns) + 2))
    sns.heatmap(corr, mask=mask, annot=True, fmt=".2f", cmap="RdBu_r", center=0,
                square=True, linewidths=0.5, cbar_kws={"shrink": 0.7}, ax=ax)
    ax.set_title("Correlation matrix")
    fig.tight_layout()
    return fig


def missing_bar(frame: pd.DataFrame):
    """Missing-value percentage per column."""
    from .profiling import missing_report

    report = missing_report(frame)
    report = report[report["missing"] > 0]
    if report.empty:
        fig, ax = plt.subplots(figsize=(6, 2))
        ax.text(0.5, 0.5, "No missing values", ha="center", va="center", fontsize=13)
        ax.axis("off")
        return fig

    fig, ax = plt.subplots(figsize=(8, 0.45 * len(report) + 1.5))
    sns.barplot(x=report["missing_pct"], y=report.index, ax=ax, color=PALETTE[3])
    ax.set_xlabel("% missing")
    ax.set_title("Missing values by column")
    fig.tight_layout()
    return fig
