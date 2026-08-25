from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd


def calculate_memory_usage(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Calculate memory consumption for every column.

    Memory usage includes the underlying object/string data.
    """

    memory_bytes = df.memory_usage(
        index=False,
        deep=True,
    )

    total_memory = memory_bytes.sum()

    result = pd.DataFrame(
        {
            "column": df.columns,
            "dtype": [
                str(dtype)
                for dtype in df.dtypes
            ],
            "memory_bytes": memory_bytes.values,
            "memory_mb": (
                memory_bytes.values
                / (1024 ** 2)
            ),
            "memory_percentage": (
                memory_bytes.values
                / total_memory
                * 100
            ),
            "unique_count": [
                df[column].nunique(
                    dropna=False
                )
                for column in df.columns
            ],
            "missing_count": [
                int(df[column].isna().sum())
                for column in df.columns
            ],
        }
    )

    return result.sort_values(
        "memory_bytes",
        ascending=False,
    ).reset_index(drop=True)


def get_numeric_ranges(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Capture numeric min/max/range information.

    This is used to determine whether smaller integer
    dtypes may be safe later.
    """

    numeric_columns = df.select_dtypes(
        include=np.number
    ).columns

    rows: list[dict[str, Any]] = []

    for column in numeric_columns:

        series = df[column]

        rows.append(
            {
                "column": column,
                "dtype": str(series.dtype),
                "min": series.min(),
                "max": series.max(),
                "unique_count": series.nunique(),
                "has_negative": bool(
                    (series < 0).any()
                ),
            }
        )

    return pd.DataFrame(rows)


def get_string_cardinality(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Analyze object/string columns to determine their
    cardinality and potential suitability for categorical
    encoding.
    """

    string_columns = df.select_dtypes(
        include=["object", "string", "category"]
    ).columns

    rows: list[dict[str, Any]] = []

    for column in string_columns:

        series = df[column]

        total = len(series)

        unique_count = series.nunique(
            dropna=False
        )

        cardinality_ratio = (
            unique_count / total
            if total
            else 0.0
        )

        rows.append(
            {
                "column": column,
                "dtype": str(series.dtype),
                "rows": total,
                "unique_count": unique_count,
                "cardinality_ratio": cardinality_ratio,
                "memory_mb": (
                    series.memory_usage(
                        index=False,
                        deep=True,
                    )
                    / (1024 ** 2)
                ),
            }
        )

    return pd.DataFrame(rows)


def estimate_integer_downcasts(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Determine the smallest safe integer dtype for each
    integer column based on its observed value range.

    This function only recommends a dtype.

    It does NOT modify the DataFrame.
    """

    integer_columns = df.select_dtypes(
        include=["int8", "int16", "int32", "int64",
                 "uint8", "uint16", "uint32", "uint64"]
    ).columns

    rows: list[dict[str, Any]] = []

    for column in integer_columns:

        series = df[column]

        minimum = series.min()
        maximum = series.max()

        if minimum >= 0:

            if maximum <= np.iinfo(np.uint8).max:
                candidate = "uint8"

            elif maximum <= np.iinfo(np.uint16).max:
                candidate = "uint16"

            elif maximum <= np.iinfo(np.uint32).max:
                candidate = "uint32"

            else:
                candidate = "uint64"

        else:

            if (
                minimum >= np.iinfo(np.int8).min
                and maximum <= np.iinfo(np.int8).max
            ):
                candidate = "int8"

            elif (
                minimum >= np.iinfo(np.int16).min
                and maximum <= np.iinfo(np.int16).max
            ):
                candidate = "int16"

            elif (
                minimum >= np.iinfo(np.int32).min
                and maximum <= np.iinfo(np.int32).max
            ):
                candidate = "int32"

            else:
                candidate = "int64"

        current_memory = (
            series.memory_usage(
                index=False,
                deep=True,
            )
        )

        candidate_dtype = np.dtype(
            candidate
        )

        estimated_memory = (
            len(series)
            * candidate_dtype.itemsize
        )

        rows.append(
            {
                "column": column,
                "current_dtype": str(
                    series.dtype
                ),
                "min": minimum,
                "max": maximum,
                "candidate_dtype": candidate,
                "current_memory_mb": (
                    current_memory
                    / (1024 ** 2)
                ),
                "estimated_memory_mb": (
                    estimated_memory
                    / (1024 ** 2)
                ),
                "estimated_savings_mb": (
                    current_memory
                    - estimated_memory
                )
                / (1024 ** 2),
            }
        )

    return pd.DataFrame(rows)


def estimate_float_downcasts(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Evaluate float columns for potential float32 conversion.

    This function does NOT perform the conversion.

    It compares the original values against float32
    representations to quantify the resulting precision
    difference.
    """

    float_columns = df.select_dtypes(
        include=["float16", "float32", "float64"]
    ).columns

    rows: list[dict[str, Any]] = []

    for column in float_columns:

        series = df[column]

        original = series.to_numpy(
            dtype=np.float64
        )

        float32_values = series.to_numpy(
            dtype=np.float32
        ).astype(np.float64)

        absolute_difference = np.abs(
            original - float32_values
        )

        nonzero_mask = original != 0

        if nonzero_mask.any():

            relative_difference = (
                absolute_difference[
                    nonzero_mask
                ]
                / np.abs(
                    original[
                        nonzero_mask
                    ]
                )
            )

            max_relative_error = float(
                relative_difference.max()
            )

            mean_relative_error = float(
                relative_difference.mean()
            )

        else:

            max_relative_error = 0.0
            mean_relative_error = 0.0

        current_memory = (
            series.memory_usage(
                index=False,
                deep=True,
            )
        )

        estimated_memory = (
            len(series)
            * np.dtype("float32").itemsize
        )

        rows.append(
            {
                "column": column,
                "current_dtype": str(
                    series.dtype
                ),
                "min": series.min(),
                "max": series.max(),
                "current_memory_mb": (
                    current_memory
                    / (1024 ** 2)
                ),
                "estimated_float32_memory_mb": (
                    estimated_memory
                    / (1024 ** 2)
                ),
                "estimated_savings_mb": (
                    current_memory
                    - estimated_memory
                )
                / (1024 ** 2),
                "max_absolute_error": float(
                    absolute_difference.max()
                ),
                "mean_absolute_error": float(
                    absolute_difference.mean()
                ),
                "max_relative_error": (
                    max_relative_error
                ),
                "mean_relative_error": (
                    mean_relative_error
                ),
            }
        )

    return pd.DataFrame(rows)


def build_memory_summary(
    df: pd.DataFrame,
) -> dict[str, Any]:
    """
    Produce a high-level memory summary.
    """

    memory_bytes = df.memory_usage(
        index=False,
        deep=True,
    ).sum()

    return {
        "rows": len(df),
        "columns": len(df.columns),
        "memory_bytes": int(
            memory_bytes
        ),
        "memory_mb": (
            memory_bytes / (1024 ** 2)
        ),
        "memory_gb": (
            memory_bytes / (1024 ** 3)
        ),
    }