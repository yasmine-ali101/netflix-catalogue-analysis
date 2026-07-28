"""Fetch the three public datasets these studies run on.

    python scripts/download_data.py

All three are public and freely redistributable, but they are not committed here
— a data directory in git is a maintenance liability, and these are one HTTP
request away. Each source below is a stable raw-file mirror.
"""

from __future__ import annotations

import socket
import sys
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"

SOURCES = {
    "supermarket_sales.csv": (
        "https://raw.githubusercontent.com/plotly/datasets/master/supermarket_Sales.csv",
        "Supermarket sales — 1,000 transactions across 3 branches (Kaggle: aungpyaeap/supermarket-sales)",
    ),
    "netflix_titles.csv": (
        "https://raw.githubusercontent.com/rfordatascience/tidytuesday/master/"
        "data/2021/2021-04-20/netflix_titles.csv",
        "Netflix catalogue — 8,800+ titles (Kaggle: shivamb/netflix-shows, via TidyTuesday)",
    ),
    "fifa_players.csv": (
        "https://raw.githubusercontent.com/amanthedorkknight/"
        "fifa18-all-player-statistics/master/2019/data.csv",
        "FIFA 19 player attributes — 18,000+ players",
    ),
}


def download(name: str, url: str, description: str, timeout: int = 60) -> Path:
    DATA.mkdir(parents=True, exist_ok=True)
    destination = DATA / name

    if destination.exists():
        print(f"  [skip] {name} already present ({destination.stat().st_size / 1e6:.1f} MB)")
        return destination

    print(f"  [get ] {name} — {description}")
    request = urllib.request.Request(url, headers={"User-Agent": "eda-portfolio/0.1"})
    with urllib.request.urlopen(request, timeout=timeout) as response:
        payload = response.read()
    destination.write_bytes(payload)
    print(f"  [ok  ] {name} ({len(payload) / 1e6:.1f} MB)")
    return destination


def main() -> int:
    socket.setdefaulttimeout(60)
    print(f"Downloading datasets into {DATA}\n")

    failures = []
    for name, (url, description) in SOURCES.items():
        try:
            download(name, url, description)
        except Exception as exc:  # noqa: BLE001 - report and continue
            print(f"  [FAIL] {name}: {type(exc).__name__}: {exc}")
            failures.append(name)

    if failures:
        print(
            f"\n{len(failures)} download(s) failed: {', '.join(failures)}\n"
            "These are public Kaggle datasets — you can download them manually and drop "
            f"them into {DATA} under the same filenames."
        )
        return 1

    print("\nAll datasets ready.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
