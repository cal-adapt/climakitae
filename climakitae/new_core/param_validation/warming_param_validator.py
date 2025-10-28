"""
Validator for parameters provided to Warming Level Processor.
"""

from __future__ import annotations

import warnings
from typing import Any

from climakitae.core.constants import UNSET
from climakitae.core.paths import GWL_1850_1900_TIMEIDX_FILE
from climakitae.new_core.data_access.data_access import DataCatalog
from climakitae.new_core.param_validation.abc_param_validation import (
    register_processor_validator,
)
from climakitae.util.utils import read_csv_file


@register_processor_validator("warming_level")
def validate_warming_level_param(
    value: dict[str, Any],
    **kwargs: Any,
) -> bool:
    """
    Validate the parameters provided to the Warming Level Processor.

    This function checks the value provided to the Warming Level Processor and ensures that it
    meets the expected criteria. Will raise a user warning and return false if the value
    is not valid.

    Parameters
    ----------
    value : Union[str, list, dict[str, Any]]
        The configuration dictionary to validate. Expected keys:
        - warming_levels : list[float]
            List of global warming levels in degrees C (e.g., [1.5, 2.0])
        - warming_level_months : list[int], optional
            List of months to include (1-12). Default: all months
        - warming_level_window : int, optional
            Number of years before and after the central year. Default: 15

    Returns
    -------
    bool
        True if all parameters are valid, False otherwise
    """
    if not _check_input_types(value):
        return False

    # now we have to check some more serious stuff
    query = kwargs.get("query", UNSET)
    if query is UNSET:
        warnings.warn(
            "\n\nWarming Level Processor requires a 'query' parameter. "
            "\nPlease check the configuration."
        )
        return False

    # validate query
    if not _check_query(query):
        return False

    if not _check_wl_values(value, query):
        return False

    return True


def _check_input_types(
    value: dict[str, Any],
) -> bool:
    """
    Validates the input dictionary for warming level parameters.

    This function checks the types and values of the keys in the input dictionary
    to ensure they conform to the expected structure and constraints for warming
    level processing.

    Parameters:
        value (dict[str, Any]): A dictionary containing the parameters to validate.

    Returns:
        bool: True if the input dictionary is valid, False otherwise.

    Validation Rules:
        - The input must be a dictionary.
        - The "warming_levels" key, if present, must be a list of integers or floats.
        - The "warming_level_months" key, if present, must be a list of integers
          representing months (1-12).
        - The "warming_level_window" key, if present, must be a non-negative integer.

    Warnings:
        - Issues a warning if the input dictionary or any of its keys do not meet
          the expected criteria.
    """
    if not isinstance(value, dict):
        warnings.warn(
            "\n\nWarming Level Processor expects a dictionary of parameters. "
            "\nPlease check the configuration."
        )
        return False

    wl = value.get("warming_levels", UNSET)
    if (
        wl is UNSET
        or not isinstance(wl, list)
        or not all(isinstance(x, (int, float)) for x in wl)
    ):
        warnings.warn(
            "\n\nInvalid 'warming_levels' parameter. "
            "\nExpected a list of global warming levels (e.g., [1.5, 2.0])."
        )
        return False

    wl_months = value.get("warming_level_months", UNSET)
    if wl_months is not UNSET:
        if not isinstance(wl_months, list) or not all(
            isinstance(x, int) and 1 <= x <= 12 for x in wl_months
        ):
            warnings.warn(
                "\n\nInvalid 'warming_level_months' parameter. "
                "\nExpected a list of months (1-12). Default: all months."
            )
            return False

    wl_window = value.get("warming_level_window", UNSET)
    if wl_window is not UNSET:
        if not isinstance(wl_window, int) or wl_window < 0:
            warnings.warn(
                "\n\nInvalid 'warming_level_window' parameter. "
                "\nExpected a non-negative integer (default: 15)."
            )
            return False

    return True


def _check_query(query: Any) -> bool:
    """
    Warming level approach requires we check activity_id and experiment_id

    Activity_id needs to be "WRF" or "LOCA2" or UNSET

    experiment_id needs to be "UNSET".

    """
    if not isinstance(query, dict):
        return False

    activity_id = query.get("activity_id", UNSET)
    experiment_id = query.get("experiment_id", UNSET)

    if activity_id not in ["WRF", "LOCA2", UNSET]:
        warnings.warn(
            "\n\nInvalid 'activity_id' parameter. "
            "\nExpected 'WRF', 'LOCA2', or not passed (UNSET)."
        )
        # force the user to fix this. Cannot assume intention here
        return False

    if experiment_id is not UNSET:
        warnings.warn(
            "\n\nWarming level approach requires 'experiment_id' to be UNSET. "
            "\nModify the query accordingly."
        )
        return False

    time_slice = query.get("processes", {}).get("time_slice", UNSET)
    if time_slice is not UNSET:
        warnings.warn(
            "\n\nWarming level approach does not support 'time_slice' in the query. "
            "\nIt will be ignored."
        )
        del query["processes"]["time_slice"]

    return True


def _check_wl_values(value, query: dict[str, Any] = {}) -> bool:
    """
    Validates that requested warming levels are within the available ranges in climate model trajectories.
    This function checks if the warming levels specified in the query are within the
    minimum and maximum values of the available climate model trajectories, after filtering
    based on activity_id if provided.

    Parameters
    ----------
    query : dict[str, Any]
        A dictionary containing query parameters. Expected keys include:
        - 'warming_levels': list of warming level values to check
        - 'activity_id': (optional) activity ID to filter the climate model catalog

    Returns
    -------
    bool
        True if all requested warming levels are within the range of available trajectories,
        False otherwise. Issues a warning if any warming level is outside the valid range.

    Notes
    -----
    The function reads trajectory data from a global warming level file and the model
    catalog from DataCatalog. It filters trajectories based on the query parameters
    before checking the warming level values.
    """

    catalog_df = DataCatalog().catalog_df.copy()
    trajectories = read_csv_file(
        GWL_1850_1900_TIMEIDX_FILE, index_col="time", parse_dates=True
    )

    # Filter catalog based on activity_id if provided
    columns_to_keep = []
    activity_id = query.get("activity_id", UNSET)
    if activity_id is not UNSET:
        catalog_df = catalog_df[catalog_df["activity_id"] == activity_id]

    # Filter catalog based on experiment_id if provided
    for _, row in catalog_df.iterrows():
        pattern = f"{row['source_id']}_{row['member_id']}_{row['experiment_id']}"
        matches = [col for col in trajectories.columns if pattern in col]
        columns_to_keep.extend(matches)

    trajectories = trajectories[columns_to_keep]

    max_trajectory = round(trajectories.max().max(), 2)
    min_trajectory = round(trajectories.min().min(), 2)

    for wl in value.get("warming_levels", []):
        if not (min_trajectory <= wl <= max_trajectory):
            warnings.warn(
                f"\n\nWarming level {wl} is outside the range of available trajectories."
                f"({min_trajectory} to {max_trajectory}). "
                "\nPlease check the configuration."
            )
            return False

    return True
