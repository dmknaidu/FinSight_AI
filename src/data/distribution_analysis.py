from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy import stats


CONTINUOUS_COLUMNS = [
    "amount",
    "oldbalanceOrg",
    "newbalanceOrig",
    "oldbalanceDest",
    "newbalanceDest",
]

NORMALITY_SAMPLE_SIZE = 10_000
NORMALITY_RANDOM_SEED = 42


class DistributionAnalysisError(Exception):
    """Raised when distribution analysis fails."""


@dataclass(frozen=True)
class DistributionAnalysisConfig:
    sample_size: int = NORMALITY_SAMPLE_SIZE
    random_seed: int = NORMALITY_RANDOM_SEED


def validate_columns(
    df: pd.DataFrame,
) -> None:

    missing = [
        column
        for column in CONTINUOUS_COLUMNS
        if column not in df.columns
    ]

    if missing:
        raise DistributionAnalysisError(
            "Missing continuous columns: "
            + ", ".join(missing)
        )


def calculate_distribution_profile(
    df: pd.DataFrame,
) -> pd.DataFrame:

    validate_columns(df)

    records = []

    for column in CONTINUOUS_COLUMNS:

        series = df[column].dropna()

        q1 = float(
            series.quantile(0.25)
        )

        median = float(
            series.quantile(0.50)
        )

        q3 = float(
            series.quantile(0.75)
        )

        iqr = q3 - q1

        lower_bound = (
            q1 - 1.5 * iqr
        )

        upper_bound = (
            q3 + 1.5 * iqr
        )

        outlier_mask = (
            (series < lower_bound)
            | (series > upper_bound)
        )

        zero_count = int(
            (series == 0).sum()
        )

        records.append(
            {
                "column": column,
                "count": int(series.count()),
                "missing_count": int(
                    df[column].isna().sum()
                ),
                "min": float(series.min()),
                "q25": q1,
                "median": median,
                "mean": float(series.mean()),
                "q75": q3,
                "max": float(series.max()),
                "std": float(series.std()),
                "variance": float(series.var()),
                "iqr": iqr,
                "skewness": float(series.skew()),
                "kurtosis": float(series.kurtosis()),
                "zero_count": zero_count,
                "zero_percentage": (
                    zero_count
                    / len(series)
                    * 100
                ),
                "iqr_outlier_count": int(
                    outlier_mask.sum()
                ),
                "iqr_outlier_percentage": (
                    outlier_mask.mean()
                    * 100
                ),
                "iqr_lower_bound": lower_bound,
                "iqr_upper_bound": upper_bound,
            }
        )

    return pd.DataFrame(records)


def calculate_normality_profile(
    df: pd.DataFrame,
    config: DistributionAnalysisConfig | None = None,
) -> pd.DataFrame:

    validate_columns(df)

    if config is None:
        config = DistributionAnalysisConfig()

    records = []

    for column in CONTINUOUS_COLUMNS:

        series = df[column].dropna()

        sample_size = min(
            config.sample_size,
            len(series),
        )

        if sample_size < 8:
            raise DistributionAnalysisError(
                f"Insufficient observations for "
                f"normality analysis: {column}"
            )

        sample = series.sample(
            n=sample_size,
            random_state=config.random_seed,
        )

        # Shapiro-Wilk is intentionally bounded to a small,
        # deterministic sample because the full dataset contains
        # more than six million observations.
        statistic, p_value = (
            stats.shapiro(sample)
        )

        records.append(
            {
                "column": column,
                "test": "Shapiro-Wilk",
                "sample_size": sample_size,
                "random_seed": config.random_seed,
                "statistic": float(statistic),
                "p_value": float(p_value),
                "normality_rejected_alpha_0_05": (
                    bool(p_value < 0.05)
                ),
            }
        )

    return pd.DataFrame(records)


def merge_distribution_results(
    distribution_profile: pd.DataFrame,
    normality_profile: pd.DataFrame,
) -> pd.DataFrame:

    return distribution_profile.merge(
        normality_profile,
        on="column",
        how="left",
        validate="one_to_one",
    )


def interpret_distribution(
    profile: pd.DataFrame,
) -> pd.DataFrame:

    results = profile.copy()

    results["distribution_assessment"] = np.select(
        [
            results["skewness"].abs() >= 1.0,
            results["skewness"].abs() >= 0.5,
        ],
        [
            "strongly_skewed",
            "moderately_skewed",
        ],
        default="approximately_symmetric",
    )

    results["normality_assessment"] = np.where(
        results[
            "normality_rejected_alpha_0_05"
        ],
        "normality_rejected",
        "normality_not_rejected",
    )

    results["recommended_statistical_family"] = np.select(
        [
            results["skewness"].abs() >= 1.0,
            results["skewness"].abs() >= 0.5,
        ],
        [
            "robust_or_nonparametric",
            "robust_or_distribution_aware",
        ],
        default="parametric_methods_may_be_considered",
    )

    return results