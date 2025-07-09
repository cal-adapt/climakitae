"""Tests to see that GWL times csv's are formatted correctly"""

import pandas as pd
import pytest

from climakitae.core.constants import WARMING_LEVELS
from climakitae.core.paths import GWL_1850_1900_FILE, GWL_1981_2010_FILE
from climakitae.util.utils import read_csv_file

# Load the DataFrames
gwl_times_1850_1900 = read_csv_file(GWL_1850_1900_FILE)
gwl_times_1981_2010 = read_csv_file(GWL_1981_2010_FILE)

# Parameterized DataFrames for testing
gwl_dfs = [("1850-1900", gwl_times_1850_1900), ("1981-2010", gwl_times_1981_2010)]


@pytest.mark.parametrize("name, df", gwl_dfs)
def test_file_structure(name, df):
    """
    Test that each DataFrame has the required structure.

    - Ensures the DataFrame is not empty.
    - Validates that all required columns ('GCM', 'run', 'scenario', and `warming_levels` values) are present.
    """
    required_columns = ["GCM", "run", "scenario"] + [str(wl) for wl in WARMING_LEVELS]
    assert not df.empty, f"{name} DataFrame is empty."
    assert all(
        col in df.columns for col in required_columns
    ), f"{name} is missing required columns."


@pytest.mark.parametrize("name, df", gwl_dfs)
def test_unique_combinations(name, df):
    """
    Test that the combination of 'GCM', 'run', and 'scenario' columns is unique.

    - Ensures there are no duplicate rows based on these columns.
    """
    assert (
        df.duplicated(subset=["GCM", "run", "scenario"]).sum() == 0
    ), f"Duplicate combinations of 'GCM', 'run', and 'scenario' found in DataFrame '{name}'."


@pytest.mark.parametrize("name, df", gwl_dfs)
def test_column_data_types(name, df):
    """
    Test that columns have the correct data types.

    - Checks that warming level columns contain only valid datetime strings or NaN.
    - Ensures that 'GCM', 'run', and 'scenario' columns are string types.
    """
    # Check that warming level columns are valid datetimes or NaN
    for wl in WARMING_LEVELS:
        datetime_check = pd.to_datetime(df[str(wl)], errors="coerce")
        is_valid_or_nan = datetime_check.notna() | df[str(wl)].isna()
        assert (
            is_valid_or_nan.all()
        ), f"Column {str(wl)} in {name} contains values that are neither valid datetimes nor NaN."

    # Check that 'GCM', 'run', and 'scenario' columns are string dtype
    for col in ["GCM", "run", "scenario"]:
        assert pd.api.types.is_string_dtype(
            df[col]
        ), f"Column '{col}' in {name} is not of string dtype."


@pytest.mark.parametrize("name, df", gwl_dfs)
def test_check_columns_sorted(name, df):
    """
    Test that warming level columns are sorted in ascending order.

    - Validates that the columns appear in sorted order by their numeric values.
    """
    column_names = [col for col in df.columns if col.replace(".", "").isdigit()]
    assert column_names == sorted(
        column_names, key=float
    ), f"Dynamic columns are not sorted in {name}."
