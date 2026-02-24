"""
Validator for parameters provided to Warming Level Processor.
"""

from __future__ import annotations

import logging
from typing import Any

from climakitae.core.constants import UNSET
from climakitae.core.paths import GWL_1850_1900_TIMEIDX_FILE
from climakitae.new_core.data_access.data_access import DataCatalog
from climakitae.new_core.param_validation.abc_param_validation import (
    register_processor_validator,
)
from climakitae.util.utils import read_csv_file

# Module logger
logger = logging.getLogger(__name__)


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
        - add_dummy_time: bool, optional
            Default: False
            If True, replace the [hours/days/months]_from_center or time_delta dimension
                in a DataArray returned from WarmingLevels with a dummy time index for
                calculations with tools that require a time dimension.

    Returns
    -------
    bool
        True if all parameters are valid, False otherwise
    """
    logger.debug(
        "validate_warming_level_param called with value=%s kwargs=%s", value, kwargs
    )

    if not _check_input_types(value):
        return False

    # now we have to check some more serious stuff
    query = kwargs.get("query", UNSET)
    if query is UNSET:
        msg = "Warming Level Processor requires a 'query' parameter. Please check the configuration."
        logger.warning(msg)
        return False

    # check that catalog is "cadcat"
    if not _check_catalog(query):
        return False

    # validate query
    if not _check_query(query):
        return False

    if not _check_wl_values(value, query):
        return False

    return True


def _check_catalog(query: Any) -> bool:
    """
    Validates that the catalog is "cadcat" (warming levels only supported for cadcat).

    Parameters
    ----------
    query : Any
        The query dictionary that may contain a 'catalog' key.

    Returns
    -------
    bool
        True if the catalog is "cadcat", False otherwise.
    """
    logger.debug("_check_catalog called with query: %s", query)

    if not isinstance(query, dict):
        return False

    catalog = query.get("catalog", UNSET)
    if catalog is UNSET:
        # If catalog is not specified, assume it will default to cadcat
        return True

    if catalog != "cadcat":
        msg = (
            f"Warming level processor is not supported for '{catalog}' catalog. "
            "Warming levels are only available for 'cadcat' (Cal-Adapt climate data). "
            "Please use time_slice processor instead to specify a time range."
        )
        logger.warning(msg)
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
        - The "add_dummy_time" key, if present, must be a boolean

    Warnings:
        - Issues a warning if the input dictionary or any of its keys do not meet
          the expected criteria.
    """
    logger.debug("_check_input_types called with value: %s", value)

    if not isinstance(value, dict):
        msg = "Warming Level Processor expects a dictionary of parameters. Please check the configuration."
        logger.warning(msg)
        return False

    wl = value.get("warming_levels", UNSET)
    if (
        wl is UNSET
        or not isinstance(wl, list)
        or not all(isinstance(x, (int, float)) for x in wl)
    ):
        msg = "Invalid 'warming_levels' parameter. Expected a list of global warming levels (e.g., [1.5, 2.0])."
        logger.warning(msg)
        return False

    wl_months = value.get("warming_level_months", UNSET)
    if wl_months is not UNSET:
        if not isinstance(wl_months, list) or not all(
            isinstance(x, int) and 1 <= x <= 12 for x in wl_months
        ):
            msg = "Invalid 'warming_level_months' parameter. Expected a list of months (1-12). Default: all months."
            logger.warning(msg)
            return False

    wl_window = value.get("warming_level_window", UNSET)
    if wl_window is not UNSET:
        if not isinstance(wl_window, int) or wl_window < 0:
            msg = "Invalid 'warming_level_window' parameter. Expected a non-negative integer (default: 15)."
            logger.warning(msg)
            return False

    add_dummy_time = value.get("add_dummy_time", UNSET)
    if add_dummy_time is not UNSET:
        if not isinstance(add_dummy_time, bool):
            msg = "Invalid 'add_dummy_time' parameter. Expected a boolean (default: False)."
            logger.warning(msg)
            return False

    return True


def _check_query(query: Any) -> bool:
    """
    Warming level approach requires we check activity_id and experiment_id

    Activity_id needs to be "WRF" or "LOCA2" or UNSET

    experiment_id needs to be "UNSET".

    """
    logger.debug("_check_query called with query: %s", query)

    if not isinstance(query, dict):
        return False

    activity_id = query.get("activity_id", UNSET)
    experiment_id = query.get("experiment_id", UNSET)

    if activity_id not in ["WRF", "LOCA2", UNSET]:
        msg = "Invalid 'activity_id' parameter. Expected 'WRF', 'LOCA2', or not passed (UNSET)."
        logger.warning(msg)
        # force the user to fix this. Cannot assume intention here
        return False

    if experiment_id is not UNSET:
        msg = "Warming level approach requires 'experiment_id' to be UNSET. Modify the query accordingly."
        logger.warning(msg)
        return False

    time_slice = query.get("processes", {}).get("time_slice", UNSET)
    if time_slice is not UNSET:
        msg = "The warming_level and time_slice processors cannot be used concurrently."
        logger.error(msg)
        return False

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

    logger.debug("_check_wl_values called with value: %s, query: %s", value, query)

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
    logger.debug("Available trajectories min=%s max=%s", min_trajectory, max_trajectory)

    for wl in value.get("warming_levels", []):
        if not (min_trajectory <= wl <= max_trajectory):
            msg = f"Warming level {wl} is outside the range of available trajectories. ({min_trajectory} to {max_trajectory}). Please check the configuration."
            logger.warning(msg)
            return False

    return True
