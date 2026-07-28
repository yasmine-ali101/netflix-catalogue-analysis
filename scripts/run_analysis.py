"""Run all three EDA studies and write findings + figures.

    python scripts/run_analysis.py

Produces results/<study>/ with figures and a findings.json of computed statistics.
Every claim in the README comes from here.
"""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from edatools import plots, profiling  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger("analysis")

DATA = ROOT / "data"
RESULTS = ROOT / "results"


def _save(fig, study: str, name: str) -> None:
    directory = RESULTS / study
    directory.mkdir(parents=True, exist_ok=True)
    fig.savefig(directory / f"{name}.png", dpi=130, bbox_inches="tight")
    import matplotlib.pyplot as plt

    plt.close(fig)


# --------------------------------------------------------------------------
# Study 1 — Supermarket sales
# --------------------------------------------------------------------------

def supermarket() -> dict:
    frame = pd.read_csv(DATA / "supermarket_sales.csv")
    frame.columns = [c.strip() for c in frame.columns]

    # Mirrors of this dataset disagree on column naming ("Gross income" vs
    # "gross income", "Customer stratification rating" vs "Rating"), so normalise
    # rather than assume one variant.
    renames = {
        "Cost of goods sold": "cogs",
        "Gross income": "gross income",
        "Customer stratification rating": "Rating",
    }
    frame = frame.rename(columns={k: v for k, v in renames.items() if k in frame.columns})
    frame["Date"] = pd.to_datetime(frame["Date"], format="mixed")

    numeric = ["Unit price", "Quantity", "Tax 5%", "Total", "cogs", "gross income", "Rating"]
    numeric = [c for c in numeric if c in frame.columns]

    findings: dict = {
        "rows": len(frame),
        "columns": len(frame.columns),
        "missing_total": int(frame.isna().sum().sum()),
        "duplicates": int(frame.duplicated().sum()),
    }

    # The accounting identity the notebook hypothesised — now actually verified.
    if {"Total", "cogs", "gross income"} <= set(frame.columns):
        residual = (frame["Total"] - frame["cogs"] - frame["gross income"]).abs()
        findings["identity_total_equals_cogs_plus_gross_income"] = {
            "max_absolute_residual": round(float(residual.max()), 6),
            "holds": bool(residual.max() < 0.01),
        }
    if {"gross income", "Tax 5%"} <= set(frame.columns):
        diff = (frame["gross income"] - frame["Tax 5%"]).abs()
        findings["identity_gross_income_equals_tax"] = {
            "max_absolute_residual": round(float(diff.max()), 6),
            "holds": bool(diff.max() < 0.01),
        }

    findings["revenue_by_product_line"] = (
        frame.groupby("Product line")["Total"].sum().round(2).sort_values(ascending=False).to_dict()
    )
    findings["revenue_by_branch"] = (
        frame.groupby("Branch")["Total"].sum().round(2).to_dict()
    )
    findings["payment_share_pct"] = (
        (frame["Payment"].value_counts(normalize=True) * 100).round(2).to_dict()
    )
    findings["mean_rating_by_branch"] = (
        frame.groupby("Branch")["Rating"].mean().round(3).to_dict()
    )

    # Does gender actually differ by product line, or does it just look that way?
    contingency = pd.crosstab(frame["Product line"], frame["Gender"])
    from scipy.stats import chi2_contingency

    chi2, p_value, dof, _ = chi2_contingency(contingency)
    findings["gender_vs_product_line_chi2"] = {
        "chi2": round(float(chi2), 4),
        "p_value": round(float(p_value), 4),
        "dof": int(dof),
        "significant_at_0.05": bool(p_value < 0.05),
    }

    findings["outliers"] = profiling.outlier_summary(frame, numeric).to_dict("records")

    plots.apply_style()
    _save(plots.histogram_grid(frame, numeric), "supermarket", "distributions")
    _save(plots.boxplot_grid(frame, numeric), "supermarket", "outliers")
    _save(plots.correlation_heatmap(frame, numeric), "supermarket", "correlation")
    _save(plots.categorical_grid(frame, ["Branch", "Payment", "Product line", "Customer type"]),
          "supermarket", "categoricals")
    return findings


# --------------------------------------------------------------------------
# Study 2 — FIFA 19 players
# --------------------------------------------------------------------------

def _parse_money(series: pd.Series) -> pd.Series:
    """Parse '€110.5M' / '€565K' into numeric euros."""
    cleaned = series.astype(str).str.replace("€", "", regex=False).str.strip()
    multiplier = np.where(cleaned.str.endswith("M"), 1e6,
                          np.where(cleaned.str.endswith("K"), 1e3, 1.0))
    numeric = pd.to_numeric(cleaned.str.rstrip("MK"), errors="coerce")
    return numeric * multiplier


def fifa() -> dict:
    frame = pd.read_csv(DATA / "fifa_players.csv", low_memory=False)

    for column in ("Value", "Wage", "Release Clause"):
        if column in frame.columns:
            frame[column] = _parse_money(frame[column])

    numeric = [c for c in ("Age", "Overall", "Potential", "Value", "Wage", "Release Clause")
               if c in frame.columns]

    findings: dict = {
        "rows": len(frame),
        "columns": len(frame.columns),
        "duplicates": int(frame.duplicated().sum()),
        "missing_by_column": profiling.missing_report(frame).head(12).to_dict("index"),
    }

    findings["value_distribution"] = {
        "mean": round(float(frame["Value"].mean()), 2),
        "median": round(float(frame["Value"].median()), 2),
        "skew": round(float(frame["Value"].skew()), 3),
        "top_1pct_share_of_total_value": round(
            float(frame["Value"].nlargest(max(1, len(frame) // 100)).sum() / frame["Value"].sum() * 100), 2
        ),
    }

    # Potential minus Overall: how much room is left, and how it decays with age.
    if {"Potential", "Overall", "Age"} <= set(frame.columns):
        frame["growth_headroom"] = frame["Potential"] - frame["Overall"]
        by_age = frame.groupby("Age")["growth_headroom"].mean().round(3)
        findings["growth_headroom_by_age"] = {
            int(k): float(v) for k, v in by_age.items() if 16 <= k <= 40
        }
        findings["age_headroom_correlation"] = round(
            float(frame[["Age", "growth_headroom"]].corr().iloc[0, 1]), 4
        )

    if {"Overall", "Value"} <= set(frame.columns):
        subset = frame[["Overall", "Value"]].dropna()
        subset = subset[subset["Value"] > 0]
        findings["overall_vs_value"] = {
            "pearson_linear": round(float(subset.corr().iloc[0, 1]), 4),
            "pearson_log_value": round(
                float(np.corrcoef(subset["Overall"], np.log(subset["Value"]))[0, 1]), 4
            ),
        }

    if "Preferred Foot" in frame.columns:
        findings["preferred_foot_share_pct"] = (
            (frame["Preferred Foot"].value_counts(normalize=True) * 100).round(2).to_dict()
        )
        findings["mean_overall_by_foot"] = (
            frame.groupby("Preferred Foot")["Overall"].mean().round(3).to_dict()
        )

    findings["outliers"] = profiling.outlier_summary(frame, numeric).to_dict("records")

    plots.apply_style()
    _save(plots.histogram_grid(frame, numeric), "fifa", "distributions")
    _save(plots.boxplot_grid(frame, numeric), "fifa", "outliers")
    _save(plots.correlation_heatmap(frame, numeric), "fifa", "correlation")
    _save(plots.missing_bar(frame), "fifa", "missing")
    return findings


# --------------------------------------------------------------------------
# Study 3 — Netflix catalogue
# --------------------------------------------------------------------------

def netflix() -> dict:
    frame = pd.read_csv(DATA / "netflix_titles.csv")
    frame["date_added"] = pd.to_datetime(frame["date_added"].astype(str).str.strip(),
                                         format="mixed", errors="coerce")

    findings: dict = {
        "rows": len(frame),
        "columns": len(frame.columns),
        "duplicates": int(frame.duplicated().sum()),
        "missing_by_column": profiling.missing_report(frame).to_dict("index"),
    }

    # The data-entry bug the original notebook caught: runtimes filed under `rating`.
    malformed = frame["rating"].astype(str).str.contains("min", case=False, na=False)
    findings["ratings_containing_runtime"] = {
        "count": int(malformed.sum()),
        "values": frame.loc[malformed, "rating"].dropna().unique().tolist(),
    }
    frame.loc[malformed, "rating"] = np.nan

    # duration mixes minutes (movies) and seasons (TV) in one column.
    frame["movie_runtime_mins"] = pd.to_numeric(
        frame["duration"].str.extract(r"(\d+)\s*min", expand=False), errors="coerce"
    )
    frame["tv_seasons"] = pd.to_numeric(
        frame["duration"].str.extract(r"(\d+)\s*Season", expand=False), errors="coerce"
    )
    findings["type_counts"] = frame["type"].value_counts().to_dict()
    findings["movie_runtime"] = {
        "count": int(frame["movie_runtime_mins"].notna().sum()),
        "mean": round(float(frame["movie_runtime_mins"].mean()), 2),
        "median": round(float(frame["movie_runtime_mins"].median()), 2),
    }
    findings["tv_seasons"] = {
        "count": int(frame["tv_seasons"].notna().sum()),
        "single_season_pct": round(
            float((frame["tv_seasons"] == 1).sum() / frame["tv_seasons"].notna().sum() * 100), 2
        ),
    }

    # Multi-value columns handled without mutating the source frame.
    findings["top_genres"] = profiling.top_values(frame, "listed_in", 10, separator=",").to_dict()
    findings["top_countries"] = profiling.top_values(frame, "country", 10, separator=",").to_dict()
    findings["top_directors"] = profiling.top_values(frame, "director", 10, separator=",").to_dict()

    frame["year_added"] = frame["date_added"].dt.year
    additions = frame["year_added"].value_counts().sort_index()
    findings["titles_added_per_year"] = {
        int(k): int(v) for k, v in additions.items() if not pd.isna(k)
    }

    # Content age at the moment it was added — has Netflix shifted toward new titles?
    frame["age_at_add"] = frame["year_added"] - frame["release_year"]
    valid = frame[frame["age_at_add"].between(0, 80)]
    findings["median_content_age_at_add_by_year"] = {
        int(k): float(v)
        for k, v in valid.groupby("year_added")["age_at_add"].median().items()
    }

    plots.apply_style()
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(14, 4.5))
    additions.plot(kind="bar", ax=axes[0], color=plots.PALETTE[0])
    axes[0].set_title("Titles added to Netflix per year")
    axes[0].set_xlabel("year added")
    genres = pd.Series(findings["top_genres"]).sort_values()
    genres.plot(kind="barh", ax=axes[1], color=plots.PALETTE[1])
    axes[1].set_title("Top 10 genres")
    fig.tight_layout()
    _save(fig, "netflix", "catalogue_growth")

    _save(plots.missing_bar(frame), "netflix", "missing")

    fig, ax = plt.subplots(figsize=(11, 4.5))
    frame[frame["type"] == "Movie"]["movie_runtime_mins"].dropna().plot(
        kind="hist", bins=50, ax=ax, color=plots.PALETTE[2], edgecolor="none"
    )
    ax.set_title("Movie runtime distribution (minutes)")
    ax.set_xlabel("minutes")
    fig.tight_layout()
    _save(fig, "netflix", "runtime_distribution")
    return findings


def main() -> None:
    RESULTS.mkdir(exist_ok=True)
    all_findings = {}
    for name, function in (("supermarket", supermarket), ("fifa", fifa), ("netflix", netflix)):
        logger.info("Running %s study", name)
        all_findings[name] = function()

    (RESULTS / "findings.json").write_text(
        json.dumps(all_findings, indent=2, ensure_ascii=False, default=str), encoding="utf-8"
    )
    logger.info("Wrote findings to %s", RESULTS / "findings.json")


if __name__ == "__main__":
    main()
