"""Parameter validator for StationBiasCorrection processor.

This module provides validation for parameters used with the StationBiasCorrection
processor, which applies Quantile Delta Mapping (QDM) bias correction to gridded
climate data using historical weather station observations.

The validator ensures:
- Valid station selection from available HadISD stations
- Proper time slice specification for bias correction
- Valid QDM parameters (window, nquantiles, group, kind)
- Station metadata availability
- Compatibility between selected stations and data variables

Functions
---------
validate_station_bias_correction_param
    Main validation function for station bias correction parameters.

Examples
--------
>>> # Valid station bias correction parameters
>>> params = {
...     "stations": ["Sacramento (KSAC)", "San Francisco (KSFO)"],
...     "time_slice": (2030, 2060),
...     "window": 90,
...     "nquantiles": 20
... }
>>> validate_station_bias_correction_param(params)
True

>>> # Invalid station name
>>> params = {"stations": ["InvalidStation"], "time_slice": (2030, 2060)}
>>> validate_station_bias_correction_param(params)
False

Notes
-----
- Station observational data is available through 2014-08-31
- Bias correction requires historical period (1980-2014) in input data
- Currently only supports temperature (tas/tasmax/tasmin) bias correction
"""

import logging
from typing import Any, Dict

import pandas as pd

from climakitae.core.constants import UNSET
from climakitae.core.paths import STATIONS_CSV_PATH
from climakitae.new_core.param_validation.abc_param_validation import (
    register_processor_validator,
)
from climakitae.util.utils import read_csv_file

# Module logger
logger = logging.getLogger(__name__)

# Load station metadata once at module level
_STATION_METADATA = None


def _get_station_metadata() -> pd.DataFrame:
    """Load and cache HadISD station metadata.

    Returns
    -------
    pd.DataFrame
        DataFrame with station information including 'station', 'station id',
        'latitude', 'longitude', and 'elevation' columns.
    """
    global _STATION_METADATA  # noqa: PLW0603
    if _STATION_METADATA is None:
        _STATION_METADATA = read_csv_file(STATIONS_CSV_PATH)
    return _STATION_METADATA


@register_processor_validator("station_bias_correction")
def validate_station_bias_correction_param(
    value: Any,
    query: Dict[str, Any] | None = None,
    **kwargs: Any,  # noqa: ARG001
) -> bool:
    """Validate parameters for StationBiasCorrection processor.

    This function validates all parameters required for station bias correction:
    - Station selection (must exist in HadISD dataset)
    - Time slice (must be valid years, properly ordered)
    - QDM parameters (window, nquantiles, group, kind)
    - Variable compatibility (currently only temperature variables supported)

    Parameters
    ----------
    value : Any
        Dictionary containing station bias correction parameters. Expected keys:
        - stations: list[str] - Station names or codes
        - time_slice: tuple[int, int] - Start and end years
        - window: int, optional - Seasonal grouping window (default: 90)
        - nquantiles: int, optional - Number of quantiles (default: 20)
        - group: str, optional - Temporal grouping (default: "time.dayofyear")
        - kind: str, optional - Adjustment kind (default: "+")
    query : Dict[str, Any], optional
        Full query dictionary for cross-validation with other parameters.
        Used to check variable compatibility.
    **kwargs : Any
        Additional keyword arguments (unused, for interface compatibility).

    Returns
    -------
    bool
        True if parameters are valid, False otherwise.

    Raises
    ------
    ValueError
        If parameters are invalid with specific error messages.
    TypeError
        If parameter types are incorrect.

    Examples
    --------
    >>> params = {
    ...     "stations": ["Sacramento (KSAC)"],
    ...     "time_slice": (2030, 2060),
    ...     "window": 90
    ... }
    >>> validate_station_bias_correction_param(params)
    True
    """
    logger.debug("validate_station_bias_correction_param called with value: %s", value)

    # Handle None or UNSET values
    if value is None or value is UNSET:
        msg = (
            "Station bias correction parameters cannot be None. "
            "Please provide a dictionary with 'stations' and 'time_slice' keys."
        )
        logger.warning(msg)
        return False

    # Validate it's a dictionary
    if not isinstance(value, dict):
        msg = (
            f"Station bias correction parameters must be a dictionary, "
            f"got {type(value).__name__}. "
            f"Example: {{'stations': ['Sacramento (KSAC)'], 'time_slice': (2030, 2060)}}"
        )
        logger.warning(msg)
        return False

    # Validate required keys
    required_keys = ["stations", "time_slice"]
    missing_keys = [key for key in required_keys if key not in value]
    if missing_keys:
        msg = (
            f"Missing required parameter(s): {', '.join(missing_keys)}. "
            f"Station bias correction requires 'stations' (list) and "
            f"'time_slice' (tuple of years)."
        )
        logger.warning(msg)
        return False

    # Validate stations parameter
    if not _validate_stations(value["stations"]):
        return False

    # Validate time_slice parameter
    if not _validate_time_slice(value["time_slice"]):
        return False

    # Validate optional QDM parameters if provided
    if "window" in value and not _validate_window(value["window"]):
        return False

    if "nquantiles" in value and not _validate_nquantiles(value["nquantiles"]):
        return False

    if "group" in value and not _validate_group(value["group"]):
        return False

    if "kind" in value and not _validate_kind(value["kind"]):
        return False

    # Cross-validate with query if provided
    if query is not None:
        if not _validate_variable_compatibility(query):
            return False

    logger.info(
        "Station bias correction parameters validated successfully for %d station(s)",
        len(value["stations"]),
    )
    return True


def _validate_stations(stations: Any) -> bool:
    """Validate station selection parameter.

    Parameters
    ----------
    stations : Any
        Station names or codes to validate.

    Returns
    -------
    bool
        True if stations are valid, False otherwise.
    """
    # Check type
    if not isinstance(stations, list):
        msg = (
            f"'stations' must be a list of station names, got {type(stations).__name__}. "
            f"Example: ['Sacramento (KSAC)', 'San Francisco (KSFO)']"
        )
        logger.warning(msg)
        return False

    # Check not empty
    if not stations:
        msg = "Station list cannot be empty. Please select at least one station."
        logger.warning(msg)
        return False

    # Check all elements are strings
    if not all(isinstance(s, str) for s in stations):
        msg = "All station names must be strings."
        logger.warning(msg)
        return False

    # Load station metadata
    station_metadata = _get_station_metadata()
    available_stations = set(station_metadata["station"].values)

    # Validate each station exists
    invalid_stations = [s for s in stations if s not in available_stations]
    if invalid_stations:
        msg = (
            f"Invalid station(s): {', '.join(invalid_stations)}. "
            f"Please choose from available HadISD stations. "
            f"Use show_stations_options() to see available stations."
        )
        logger.warning(msg)

        # Provide helpful suggestions for close matches
        if len(invalid_stations) == 1:
            station_name = invalid_stations[0]
            # Try to find close matches
            close_matches = [
                s
                for s in available_stations
                if station_name.lower() in s.lower()
                or s.lower() in station_name.lower()
            ]
            if close_matches:
                suggestions = ", ".join(close_matches[:5])
                logger.info("Did you mean one of these? %s", suggestions)

        return False

    logger.debug("Station validation passed for %d station(s)", len(stations))
    return True


def _validate_time_slice(time_slice: Any) -> bool:
    """Validate time slice parameter.

    Parameters
    ----------
    time_slice : Any
        Time slice tuple to validate.

    Returns
    -------
    bool
        True if time slice is valid, False otherwise.
    """
    # Check type
    if not isinstance(time_slice, tuple):
        msg = (
            f"'time_slice' must be a tuple of (start_year, end_year), "
            f"got {type(time_slice).__name__}. "
            f"Example: (2030, 2060)"
        )
        logger.warning(msg)
        return False

    # Check length
    if len(time_slice) != 2:
        msg = (
            f"'time_slice' must contain exactly 2 elements (start_year, end_year), "
            f"got {len(time_slice)} elements."
        )
        logger.warning(msg)
        return False

    start_year, end_year = time_slice

    # Check both are integers
    if not isinstance(start_year, int) or not isinstance(end_year, int):
        msg = (
            f"Time slice years must be integers, "
            f"got start_year={type(start_year).__name__}, "
            f"end_year={type(end_year).__name__}"
        )
        logger.warning(msg)
        return False

    # Check proper ordering
    if start_year >= end_year:
        msg = (
            f"Start year ({start_year}) must be less than end year ({end_year}). "
            f"Please provide a valid time range."
        )
        logger.warning(msg)
        return False

    # Check reasonable year range (HadISD data available through 2014)
    if end_year < 1980:
        msg = (
            f"End year ({end_year}) is before HadISD observational period starts (1980). "
            f"Station bias correction requires overlap with historical observations."
        )
        logger.warning(msg)
        return False

    # Warn if start year is in historical period (but don't fail)
    if start_year < 2015:
        logger.info(
            "Note: Time slice includes historical period (%d-%d). "
            "Bias correction will be applied to all years in the range.",
            start_year,
            end_year,
        )

    logger.debug("Time slice validation passed: %s", time_slice)
    return True


def _validate_window(window: Any) -> bool:
    """Validate window parameter for seasonal grouping.

    Parameters
    ----------
    window : Any
        Window size in days.

    Returns
    -------
    bool
        True if window is valid, False otherwise.
    """
    if not isinstance(window, int):
        msg = f"'window' must be an integer (days), got {type(window).__name__}"
        logger.warning(msg)
        return False

    if window <= 0:
        msg = f"'window' must be positive, got {window}"
        logger.warning(msg)
        return False

    if window > 365:
        msg = (
            f"'window' ({window} days) exceeds one year. "
            f"Typical values are 30-90 days for seasonal grouping."
        )
        logger.warning(msg)
        return False

    logger.debug("Window validation passed: %d days", window)
    return True


def _validate_nquantiles(nquantiles: Any) -> bool:
    """Validate nquantiles parameter for QDM.

    Parameters
    ----------
    nquantiles : Any
        Number of quantiles for QDM training.

    Returns
    -------
    bool
        True if nquantiles is valid, False otherwise.
    """
    if not isinstance(nquantiles, int):
        msg = f"'nquantiles' must be an integer, got {type(nquantiles).__name__}"
        logger.warning(msg)
        return False

    if nquantiles < 2:
        msg = f"'nquantiles' must be at least 2, got {nquantiles}"
        logger.warning(msg)
        return False

    if nquantiles > 100:
        logger.warning(
            "'nquantiles' (%d) is very high. This may increase computation time. "
            "Typical values are 10-30.",
            nquantiles,
        )

    logger.debug("Nquantiles validation passed: %d", nquantiles)
    return True


def _validate_group(group: Any) -> bool:
    """Validate group parameter for temporal grouping strategy.

    Parameters
    ----------
    group : Any
        Temporal grouping string.

    Returns
    -------
    bool
        True if group is valid, False otherwise.
    """
    if not isinstance(group, str):
        msg = f"'group' must be a string, got {type(group).__name__}"
        logger.warning(msg)
        return False

    valid_groups = ["time.dayofyear", "time.month", "time.season"]
    if group not in valid_groups:
        msg = (
            f"'group' must be one of {valid_groups}, got '{group}'. "
            f"'time.dayofyear' is recommended for daily data."
        )
        logger.warning(msg)
        return False

    logger.debug("Group validation passed: %s", group)
    return True


def _validate_kind(kind: Any) -> bool:
    """Validate kind parameter for adjustment type.

    Parameters
    ----------
    kind : Any
        Adjustment kind ("+" for additive, "*" for multiplicative).

    Returns
    -------
    bool
        True if kind is valid, False otherwise.
    """
    if not isinstance(kind, str):
        msg = f"'kind' must be a string, got {type(kind).__name__}"
        logger.warning(msg)
        return False

    valid_kinds = ["+", "*"]
    if kind not in valid_kinds:
        msg = (
            f"'kind' must be '+' (additive) or '*' (multiplicative), got '{kind}'. "
            f"Use '+' for temperature, '*' for precipitation."
        )
        logger.warning(msg)
        return False

    logger.debug("Kind validation passed: %s", kind)
    return True


def _validate_variable_compatibility(query: Dict[str, Any]) -> bool:
    """Validate that selected variable is compatible with station bias correction.

    Parameters
    ----------
    query : Dict[str, Any]
        Full query dictionary containing variable selection.

    Returns
    -------
    bool
        True if variable is compatible, False otherwise.
    """
    # Currently, station bias correction only supports temperature variables
    # HadISD dataset contains temperature (tas) observations
    supported_variables = ["tas", "tasmax", "tasmin", "t2"]

    variable_id = query.get("variable_id", None)
    if variable_id is None:
        # Can't validate without variable info, assume OK
        return True

    # Handle both string and list inputs
    if isinstance(variable_id, str):
        variable_ids = [variable_id]
    elif isinstance(variable_id, list):
        variable_ids = variable_id
    else:
        return True  # Unknown format, skip validation

    # Check if any selected variable is unsupported
    unsupported = [v for v in variable_ids if v not in supported_variables]
    if unsupported:
        msg = (
            f"Station bias correction currently only supports temperature variables "
            f"(tas, tasmax, tasmin, t2), but got: {', '.join(unsupported)}. "
            f"HadISD station data contains temperature observations only."
        )
        logger.warning(msg)
        return False

    logger.debug("Variable compatibility check passed")
    return True
