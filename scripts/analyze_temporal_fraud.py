from __future__ import annotations

from pathlib import Path
import sys


PROJECT_ROOT = (
    Path(__file__).resolve().parents[1]
)

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(
        0,
        str(PROJECT_ROOT),
    )


import pandas as pd

from src.data.temporal_analysis import (
    build_concentration_analysis,
    build_exposure_threshold_analysis,
    build_peak_analysis,
    build_temporal_correlations,
    build_temporal_profile,
    build_temporal_summary,
)

CANONICAL_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "canonical"
    / "finsight_canonical.parquet"
)

REPORT_DIR = (
    PROJECT_ROOT
    / "reports"
    / "statistical"
    / "temporal"
)


def print_header(title: str) -> None:

    print()
    print("=" * 100)
    print(title)
    print("=" * 100)


def main() -> None:

    print_header(
        "FINSIGHT AI — TEMPORAL FRAUD ANALYSIS"
    )

    print(
        f"Canonical dataset:\n"
        f"{CANONICAL_PATH}"
    )

    print(
        "\nLoading canonical dataset..."
    )

    df = pd.read_parquet(
        CANONICAL_PATH
    )

    print(
        f"Rows    : {len(df):,}"
    )

    print(
        f"Columns : {len(df.columns)}"
    )

    print_header(
        "BUILDING TEMPORAL PROFILE"
    )

    profile = build_temporal_profile(
        df
    )

    print(
        f"Observed steps: {len(profile)}"
    )

    print(
        "\nFirst 10 temporal observations:"
    )

    print(
        profile.head(10).to_string(
            index=False
        )
    )

    print_header(
        "TEMPORAL SUMMARY"
    )

    summary = build_temporal_summary(
        profile
    )

    print_header(
        "EXPOSURE-AWARE FRAUD RATE ANALYSIS"
    )

    exposure_analysis = (
        build_exposure_threshold_analysis(
            profile
        )
    )

    print(
        exposure_analysis.to_string(
            index=False
        )
    )

    print(
        summary.to_string(
            index=False
        )
    )

    print_header(
        "VOLUME VS FRAUD CORRELATIONS"
    )

    correlations = (
        build_temporal_correlations(
            profile
        )
    )

    print(
        correlations.to_string(
            index=False
        )
    )

    print_header(
        "TOP 10% TEMPORAL CONCENTRATION"
    )

    concentration = (
        build_concentration_analysis(
            profile
        )
    )

    print(
        concentration.to_string(
            index=False
        )
    )

    print_header(
        "PEAK PERIODS"
    )

    peaks = build_peak_analysis(
        profile,
        top_n=10,
    )

    print(
        peaks.to_string(
            index=False
        )
    )

    REPORT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    paths = {
        "profile": (
            REPORT_DIR
            / "temporal_profile.csv"
        ),
        "summary": (
            REPORT_DIR
            / "temporal_summary.csv"
        ),
        "correlations": (
            REPORT_DIR
            / "temporal_correlations.csv"
        ),
        "concentration": (
            REPORT_DIR
            / "temporal_concentration.csv"
        ),
        "peaks": (
            REPORT_DIR
            / "temporal_peaks.csv"
        ),
        "exposure_thresholds": (
            REPORT_DIR
            / "temporal_exposure_thresholds.csv"
        ),
    }

    profile.to_csv(
        paths["profile"],
        index=False,
    )

    summary.to_csv(
        paths["summary"],
        index=False,
    )

    correlations.to_csv(
        paths["correlations"],
        index=False,
    )

    concentration.to_csv(
        paths["concentration"],
        index=False,
    )

    peaks.to_csv(
        paths["peaks"],
        index=False,
    )

    exposure_analysis.to_csv(
        paths["exposure_thresholds"],
        index=False,
    )

    print_header(
        "TEMPORAL FRAUD ANALYSIS COMPLETE"
    )

    for path in paths.values():
        print(f"Saved: {path}")

    print(
        "\nCanonical dataset was not modified."
    )

    print(
        "No statistical transformations were applied."
    )


if __name__ == "__main__":
    main()