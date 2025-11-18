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


@register_processor_validator("bias_correct_station_data")
def validate_station_bias_correction_param(
    value: Any,
    query: Dict[str, Any] | None = None,
    **kwargs: Any,  # noqa: ARG001
) -> bool:
    """Validate parameters for StationBiasCorrection processor.

    This function validates all parameters required for station bias correction:
    - Station selection (must exist in HadISD dataset)
    - Historical slice (optional, must be valid years if provided)
    - QDM parameters (window, nquantiles, group, kind)
    - Variable compatibility (currently only temperature variables supported)

    Parameters
    ----------
    value : Any
        Dictionary containing station bias correction parameters. Expected keys:
        - stations: list[str] - Station names or codes (REQUIRED)
        - historical_slice: tuple[int, int], optional - Historical training period (default: (1980, 2014))
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
            "Please provide a dictionary with 'stations' key."
        )
        logger.warning(msg)
        return False

    # Validate it's a dictionary
    if not isinstance(value, dict):
        msg = (
            f"Station bias correction parameters must be a dictionary, "
            f"got {type(value).__name__}. "
            f"Example: {{'stations': ['Sacramento (KSAC)']}}"
        )
        logger.warning(msg)
        return False

    # Validate required keys
    required_keys = ["stations"]
    missing_keys = [key for key in required_keys if key not in value]
    if missing_keys:
        msg = (
            f"Missing required parameter(s): {', '.join(missing_keys)}. "
            f"Station bias correction requires 'stations' (list of station names)."
        )
        logger.warning(msg)
        return False

    # Validate stations parameter
    if not _validate_stations(value["stations"]):
        return False

    # Validate historical_slice parameter if provided
    if "historical_slice" in value and not _validate_historical_slice(
        value["historical_slice"]
    ):
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
        if not _validate_timescale_requirement(query):
            return False
        if not _validate_downscaling_method_requirement(query):
            return False
        if not _validate_resolution_requirement(query):
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


def _validate_historical_slice(historical_slice: Any) -> bool:
    """Validate historical slice parameter for bias correction training period.

    Parameters
    ----------
    historical_slice : Any
        Historical training period tuple to validate.

    Returns
    -------
    bool
        True if historical slice is valid, False otherwise.
    """
    # Check type
    if not isinstance(historical_slice, tuple):
        msg = (
            f"'historical_slice' must be a tuple of (start_year, end_year), "
            f"got {type(historical_slice).__name__}. "
            f"Example: (1980, 2014)"
        )
        logger.warning(msg)
        return False

    # Check length
    if len(historical_slice) != 2:
        msg = (
            f"'historical_slice' must contain exactly 2 elements (start_year, end_year), "
            f"got {len(historical_slice)} elements."
        )
        logger.warning(msg)
        return False

    start_year, end_year = historical_slice

    # Check both are integers
    if not isinstance(start_year, int) or not isinstance(end_year, int):
        msg = (
            f"Historical slice years must be integers, "
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

    # Check reasonable year range (HadISD data available 1980-2014)
    if start_year < 1980:
        msg = (
            f"Start year ({start_year}) is before HadISD observational period starts (1980). "
            f"Please use a start year >= 1980."
        )
        logger.warning(msg)
        return False

    if end_year > 2014:
        msg = (
            f"End year ({end_year}) is after HadISD observational period ends (2014). "
            f"Please use an end year <= 2014."
        )
        logger.warning(msg)
        return False

    logger.debug("Historical slice validation passed: %s", historical_slice)
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


def _validate_timescale_requirement(query: Dict[str, Any]) -> bool:
    """Validate that timescale is set to hourly for station bias correction.

    Station bias correction requires hourly data to match HadISD observational
    data resolution. This is a legacy constraint from the original implementation.

    Parameters
    ----------
    query : Dict[str, Any]
        Full query dictionary containing timescale selection.

    Returns
    -------
    bool
        True if timescale is valid, False otherwise.
    """
    table_id = query.get("table_id", None)

    if table_id is None:
        # Can't validate without table_id info, assume OK
        return True

    # Check if table_id is hourly (1hr or hr)
    if table_id not in ["1hr", "hr"]:
        msg = (
            f"Station bias correction requires hourly data (table_id='1hr'), "
            f"but got table_id='{table_id}'. HadISD observations are hourly and cannot "
            f"be resampled to match other temporal resolutions. Please set "
            f"table_id='1hr' or use .table_id('1hr') in your query."
        )
        logger.warning(msg)
        return False

    logger.debug("Timescale requirement validation passed: hourly data")
    return True


def _validate_downscaling_method_requirement(query: Dict[str, Any]) -> bool:
    """Validate that downscaling method is Dynamical (WRF) for station bias correction.

    Station bias correction only supports WRF dynamical downscaling because bias
    correction parameters were calibrated specifically for WRF data. This is a
    legacy constraint from the original implementation.

    Parameters
    ----------
    query : Dict[str, Any]
        Full query dictionary containing downscaling method selection.

    Returns
    -------
    bool
        True if downscaling method is valid, False otherwise.
    """
    activity_id = query.get("activity_id", None)

    if activity_id is None:
        # Can't validate without activity_id info, assume OK
        return True

    # Check if activity_id is WRF (Dynamical downscaling)
    if activity_id != "WRF":
        msg = (
            f"Station bias correction only supports WRF dynamical downscaling "
            f"(activity_id='WRF'), but got activity_id='{activity_id}'. Bias correction "
            f"parameters are calibrated for WRF data only. Please set activity_id='WRF' "
            f"or use .activity_id('WRF') in your query."
        )
        logger.warning(msg)
        return False

    logger.debug("Downscaling method requirement validation passed: WRF")
    return True


def _validate_resolution_requirement(query: Dict[str, Any]) -> bool:
    """Validate that resolution is not 3km for station bias correction.

    Station bias correction does not support 3km resolution due to limitations
    in the original calibration. Only 9km and 45km resolutions are supported.
    This is a legacy constraint from the original implementation.

    Parameters
    ----------
    query : Dict[str, Any]
        Full query dictionary containing resolution selection.

    Returns
    -------
    bool
        True if resolution is valid, False otherwise.
    """
    grid_label = query.get("grid_label", None)

    if grid_label is None:
        # Can't validate without grid_label info, assume OK
        return True

    # Check if grid_label is 3km (d03)
    if grid_label == "d03":
        msg = (
            "Station bias correction does not support 3km resolution (grid_label='d03'). "
            "Only 9km (grid_label='d02') and 45km (grid_label='d01') resolutions are "
            "supported. Please use grid_label='d02' (9km) or grid_label='d01' (45km)."
        )
        logger.warning(msg)
        return False

    logger.debug("Resolution requirement validation passed: %s", grid_label)
    return True
