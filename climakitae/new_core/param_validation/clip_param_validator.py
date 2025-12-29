"""
Validator for parameters provided to Clip Processor.
"""

from __future__ import annotations

import logging
import os
import pprint
from typing import Any, List, Tuple, Union

from climakitae.core.constants import UNSET
from climakitae.new_core.data_access.data_access import DataCatalog
from climakitae.new_core.param_validation.abc_param_validation import (
    register_processor_validator,
)
from climakitae.new_core.param_validation.param_validation_tools import (
    _get_closest_options,
)
from climakitae.new_core.processors.processor_utils import (
    find_station_match,
    is_station_identifier,
)

# Module logger
logger = logging.getLogger(__name__)


@register_processor_validator("clip")
def validate_clip_param(
    value: Any,
    **kwargs: Any,
) -> bool:
    """
    Validate parameter passed to Clip Processor.

    This function validates and normalizes parameters for the Clip processor,
    addressing common input validation issues:

    1. **Input Validation**: Rejects None values and mixed types
    2. **Empty/Whitespace Handling**: Filters out empty/whitespace strings
    3. **Case Sensitivity**: Provides consistent handling and warnings for case mismatches
    4. **Duplicate Handling**: Deduplicates boundary lists while preserving order
    5. **Coordinate Validation**: Validates lat/lon coordinate bounds

    Parameters
    ----------
    value : Any
        Parameter value to validate. This can be a string, list of strings,
        or a tuple of coordinate bounds.

    Returns
    -------
    bool:
        True if the parameter is valid, otherwise raises an exception.

    Raises
    ------
    ValueError
        If the input is invalid and cannot be corrected
    TypeError
        If the input type is not supported
    """

    logger.debug("validate_clip_param called with value: %s", value)

    # Handle None values early
    if value is None or value is UNSET:
        msg = "Clip parameter cannot be None. Please provide a valid boundary key, list of keys, file path, or coordinate bounds."
        logger.warning(msg)
        return False

    match value:
        case str():
            return _validate_string_param(value)
        case list():
            return _validate_list_param(value)
        case tuple():
            return _validate_tuple_param(value)
        case dict():
            return _validate_dict_param(value)
        case _:
            logger.warning(
                f"\n\nInvalid parameter type for Clip processor. "
                f"\nExpected str, list, tuple, or dict, but got {type(value).__name__}. "
                f"\nValid examples: 'CA', ['CA', 'OR'], ((32.0, 42.0), (-125.0, -114.0)), "
                f"or {{'boundaries': ['CA', 'OR'], 'separated': True}}"
            )

    return False


def _validate_string_param(value: str) -> bool:
    """
    Validate a string parameter for the Clip processor.

    This function tries multiple validation paths to provide helpful feedback:
    1. Check if it's a file path (highest priority - most specific)
    2. Check if it's a station identifier (specific format)
    3. Check if it's a boundary key (most general)

    If a value fails one validation but might match another type, we provide
    helpful suggestions across all categories.

    Parameters
    ----------
    value : str
        String value to validate

    Returns
    -------
    bool
        True if the parameter is valid, False otherwise

    Raises
    ------
    ValueError
        If the string is empty, whitespace-only, or invalid
    """
    logger.debug("_validate_string_param called with value: %s", value)

    # Check for empty or whitespace-only strings
    if not value or not value.strip():
        msg = (
            "Empty or whitespace-only strings are not valid clip parameters. "
            "Please provide a valid boundary key (e.g., 'CA'), file path, station code (e.g., 'KSAC'), or coordinate bounds."
        )
        logger.warning(msg)
        return False

    # Clean the string
    cleaned_value = value.strip()

    # Priority 1: Check if it looks like a file path (most specific check)
    if _is_file_path_like(cleaned_value):
        if os.path.exists(cleaned_value):
            logger.info("Clip parameter is a valid file path: %s", cleaned_value)
            return True
        else:
            msg = f"File path '{cleaned_value}' does not exist. Please provide a valid file path to a shapefile or other geospatial data."
            logger.warning(msg)
            return False

    # Priority 2: Check if it strongly looks like a station identifier
    # Only consider it a station if it matches the strict format
    is_station_like = is_station_identifier(cleaned_value)

    # If it looks like a station, validate it as such first
    if is_station_like:
        logger.debug("Value appears station-like: %s", cleaned_value)
        station_result = _validate_station_identifier(cleaned_value)
        if station_result:
            logger.info("Station identifier validated: %s", cleaned_value)
            return True

        # Station validation failed - check if it might be a boundary instead
        # We temporarily raise the log level to suppress warnings during
        # boundary validation since we may issue our own more specific warning
        clip_logger = logging.getLogger(
            "climakitae.new_core.param_validation.clip_param_validator"
        )
        original_level = clip_logger.level
        clip_logger.setLevel(logging.ERROR)
        try:
            boundary_result = _validate_boundary_key_string(cleaned_value)
        finally:
            clip_logger.setLevel(original_level)

        if boundary_result:
            # Provide helpful message that it's not a station but could be a boundary
            msg = (
                f"'{cleaned_value}' looks like a station identifier but was not found in the station database. "
                f"However, it appears to match a boundary key. If you intended to clip to a station, please check the station code. "
                f"If you intended to clip to a boundary, the value will be accepted."
            )
            logger.info(
                "String matches a boundary key instead of station: %s", cleaned_value
            )
            logger.warning(msg)
            return True

        # Failed both validations - station warning already issued
        return False

    # Priority 3: Not station-like, validate as boundary key (most common case)
    return _validate_boundary_key_string(cleaned_value)


def _validate_list_param(value: List[Any]) -> Union[List[str], None]:
    """
    Validate a list parameter for the Clip processor.

    Parameters
    ----------
    value : List[Any]
        List value to validate

    Returns
    -------
    Union[List[str], None]
        Validated and cleaned list or None if invalid

    Raises
    ------
    ValueError
        If the list is empty, contains invalid types, or all items are invalid
    """
    logger.debug(
        "_validate_list_param called with list of length %d",
        len(value) if value is not None else 0,
    )
    if not value or value is UNSET:
        msg = "Empty list is not valid for clip parameters. Please provide a list of boundary keys (e.g., ['CA', 'OR', 'WA'])."
        logger.warning(msg)
        return False

    # Check for mixed types (should all be the same type)
    mixed_types = [item for item in value if not isinstance(item, type(value[0]))]
    if mixed_types:
        unique_types = set([type(item).__name__ for item in mixed_types])
        msg = f"All items in clip parameter list must be the same type. Found {len(unique_types)} different types: {', '.join(set(unique_types))}."
        logger.warning(msg)
        return False

    # Check if all items are station identifiers
    all_stations = all(
        is_station_identifier(item) if isinstance(item, str) else False
        for item in value
    )

    if all_stations:
        # Validate all station identifiers
        for item in value:
            if not _validate_station_identifier(item):
                return False
        return True

    # Filter out empty/whitespace strings and validate each item
    valid_items = []
    invalid_items = []
    match value[0]:
        case str():
            for item in value:
                try:
                    cleaned_item = item.strip()
                    if not cleaned_item:
                        invalid_items.append(f"'{item}' (empty/whitespace)")
                        continue

                    # For boundary keys, provide basic validation
                    validated_item = _validate_boundary_key_string(
                        cleaned_item, raise_on_invalid=True
                    )
                    if validated_item is not None:
                        valid_items.append(validated_item)
                    else:
                        invalid_items.append(f"'{item}'")

                except Exception:
                    invalid_items.append(f"'{item}'")

            # Report invalid items as warnings rather than errors
            if invalid_items:
                msg = f"Found {len(invalid_items)} invalid items in clip parameter list: {', '.join(invalid_items)}."
                logger.warning(msg)
                return False

            if not valid_items:
                logger.warning(
                    "\n\nNo valid boundary keys found in the provided list. "
                    "\nPlease provide valid boundary keys such as ['CA', 'OR', 'WA'] or "
                    "\n['Los Angeles County', 'Orange County']."
                )
                return False
        case tuple():
            for item in value:
                if isinstance(item, (tuple, list)):
                    if _validate_tuple_param(item):
                        valid_items.append(item)
                    else:
                        invalid_items.append(f"{item} (invalid tuple)")
                else:
                    invalid_items.append(f"{item} (not a tuple/list)")

            if invalid_items:
                msg = f"Found {len(invalid_items)} invalid items in clip parameter list: {', '.join(invalid_items)}."
                logger.warning(msg)
                return False

    # Check for duplicates
    unique = set(value)

    if len(unique) != len(value):
        msg = f"Duplicate boundary keys found in the list. Original list: {value} Unique list: {unique}. Please remove duplicates."
        logger.warning(msg)
        return False

    return True


def _validate_single_point(point: Any) -> bool:
    """
    Validate a single (lat, lon) point tuple.

    Parameters
    ----------
    point : Any
        Point to validate. Expected to be a tuple/list of 2 numeric values.

    Returns
    -------
    bool
        True if the point is valid, False otherwise
    """
    # Check structure
    if not isinstance(point, (tuple, list)):
        logger.warning(
            f"Point must be a tuple or list, got {type(point).__name__}. "
            f"Example: (37.7749, -122.4194)"
        )
        return False

    if len(point) != 2:
        logger.warning(
            f"Point must have exactly 2 elements (lat, lon), got {len(point)}. "
            f"Example: (37.7749, -122.4194)"
        )
        return False

    # Validate numeric values
    try:
        lat, lon = float(point[0]), float(point[1])
    except (ValueError, TypeError):
        logger.warning(
            f"Point coordinates must be numeric. Got: {point}. "
            f"Example: (37.7749, -122.4194)"
        )
        return False

    # Validate coordinate ranges
    if not (-90.0 <= lat <= 90.0):
        logger.warning(f"Latitude must be between -90 and 90 degrees. Got: {lat}")
        return False

    if not (-180.0 <= lon <= 180.0):
        logger.warning(f"Longitude must be between -180 and 180 degrees. Got: {lon}")
        return False

    return True


def _validate_points_list(points: Any) -> bool:
    """
    Validate a list of (lat, lon) point tuples.

    Parameters
    ----------
    points : Any
        List of points to validate.

    Returns
    -------
    bool
        True if all points are valid, False otherwise
    """
    if not isinstance(points, list):
        logger.warning(
            f"'points' must be a list of (lat, lon) tuples, got {type(points).__name__}. "
            f"Example: {{'points': [(37.7749, -122.4194), (34.0522, -118.2437)]}}"
        )
        return False

    if len(points) == 0:
        logger.warning(
            "Empty points list is not valid. Please provide at least one (lat, lon) tuple. "
            f"Example: {{'points': [(37.7749, -122.4194)]}}"
        )
        return False

    # Validate each point
    invalid_points = []
    for i, point in enumerate(points):
        if not _validate_single_point(point):
            invalid_points.append(f"index {i}: {point}")

    if invalid_points:
        logger.warning(
            f"Found {len(invalid_points)} invalid points: {', '.join(invalid_points)}"
        )
        return False

    return True


def _validate_dict_param(value: dict) -> bool:
    """
    Validate a dict parameter for the Clip processor.

    Dict parameters enable advanced clipping features like separated mode
    for both boundaries and points.

    Parameters
    ----------
    value : dict
        Dictionary with clipping configuration. Must contain either:
        - {"boundaries": [...], "separated": bool}
        - {"points": [(lat, lon), ...], "separated": bool}

    Returns
    -------
    bool
        True if the parameter is valid, False otherwise

    Examples
    --------
    >>> _validate_dict_param({"boundaries": ["CA", "OR"], "separated": True})
    True
    >>> _validate_dict_param({"boundaries": ["CA"]})  # separated defaults to False
    True
    >>> _validate_dict_param({"points": [(37.7749, -122.4194)], "separated": True})
    True
    """
    logger.debug("_validate_dict_param called with: %s", value)

    # Check for required key - either 'boundaries' or 'points'
    has_boundaries = "boundaries" in value
    has_points = "points" in value

    if not has_boundaries and not has_points:
        msg = (
            "Dict parameter for Clip must contain 'boundaries' or 'points' key. "
            "Examples:\n"
            "  {'boundaries': ['CA', 'OR', 'WA'], 'separated': True}\n"
            "  {'points': [(37.7749, -122.4194), (34.0522, -118.2437)], 'separated': True}"
        )
        logger.warning(msg)
        return False

    if has_boundaries and has_points:
        msg = (
            "Dict parameter for Clip cannot contain both 'boundaries' and 'points' keys. "
            "Please use only one."
        )
        logger.warning(msg)
        return False

    # Validate 'separated' if present (common to both modes)
    if "separated" in value:
        separated = value["separated"]
        if not isinstance(separated, bool):
            msg = f"'separated' must be a boolean (True or False), got {type(separated).__name__}."
            logger.warning(msg)
            return False

    # Handle 'points' mode
    if has_points:
        points = value["points"]

        # Validate points list
        if not _validate_points_list(points):
            return False

        # Warn if only one point is provided with separated=True
        if value.get("separated", False) and len(points) == 1:
            msg = (
                "Using 'separated': True with a single point has no effect. "
                "Consider removing the 'separated' option or adding more points."
            )
            logger.warning(msg)
            # Still valid, just a warning

        # Check for unknown keys
        known_keys = {"points", "separated", "persist"}
        unknown_keys = set(value.keys()) - known_keys
        if unknown_keys:
            msg = (
                f"Unknown keys in clip dict parameter: {unknown_keys}. "
                f"Valid keys for points mode are: {known_keys}"
            )
            logger.warning(msg)
            # Still valid, just a warning

        logger.debug("Points dict parameter validated successfully")
        return True

    # Handle 'boundaries' mode
    boundaries = value["boundaries"]

    # Validate boundaries is a list
    if not isinstance(boundaries, list):
        msg = (
            f"'boundaries' must be a list, got {type(boundaries).__name__}. "
            "Example: {'boundaries': ['CA', 'OR', 'WA'], 'separated': True}"
        )
        logger.warning(msg)
        return False

    # Validate the boundaries list
    if not _validate_list_param(boundaries):
        return False

    # Validate 'separated' if present
    if "separated" in value:
        separated = value["separated"]
        if not isinstance(separated, bool):
            msg = (
                f"'separated' must be a boolean (True or False), got {type(separated).__name__}. "
                "Example: {'boundaries': ['CA', 'OR'], 'separated': True}"
            )
            logger.warning(msg)
            return False

    # Warn if only one boundary is provided with separated=True
    if value.get("separated", False) and len(boundaries) == 1:
        msg = (
            "Using 'separated': True with a single boundary has no effect. "
            "Consider removing the 'separated' option or adding more boundaries."
        )
        logger.warning(msg)
        # Still valid, just a warning

    # Check for unknown keys
    known_keys = {"boundaries", "separated", "persist"}
    unknown_keys = set(value.keys()) - known_keys
    if unknown_keys:
        msg = (
            f"Unknown keys in clip dict parameter: {unknown_keys}. "
            f"Valid keys are: {known_keys}"
        )
        logger.warning(msg)
        # Still valid, just a warning

    logger.debug("Dict parameter validated successfully")
    return True


def _validate_tuple_param(
    value: Tuple[Any, ...],
) -> bool:
    """
    Validate a tuple parameter for coordinate bounds.

    Parameters
    ----------
    value : Tuple[Any, ...]
        Tuple value to validate

    Returns
    -------
    Union[Tuple[Tuple[float, float], Tuple[float, float]], None]
        Validated coordinate bounds tuple or None if invalid

    Raises
    ------
    ValueError
        If the tuple structure or coordinate values are invalid
    """
    logger.debug("_validate_tuple_param called with value: %s", value)

    # Check tuple structure: should be ((lat_min, lat_max), (lon_min, lon_max))
    if len(value) != 2:
        msg = f"Coordinate bounds tuple must have exactly 2 elements: ((lat_min, lat_max), (lon_min, lon_max)). Got {len(value)} elements."
        logger.warning(msg)
        return False

    lat_bounds, lon_bounds = value

    # Validate that each bound is a tuple/list with 2 numeric values
    for bounds, name in [(lat_bounds, "latitude"), (lon_bounds, "longitude")]:
        # Check if bounds is a tuple/list with wrong length
        if isinstance(bounds, (tuple, list)) and len(bounds) != 2:
            logger.warning(
                f"\n\nLat/Lon clipping must either be a tuple of two numeric values"
                f"\nor a tuple of tuples/lists with two numeric values each. "
                f"\nPoint Example: (35.0, -120.0) "
                f"\nBounds Example: ((32.0, 42.0), (-125.0, -114.0))"
                f"\nGot {name} bounds: {bounds} (type: {type(bounds).__name__})"
            )
            return False
        # Check if bounds is an invalid type (not tuple, list, float, or int)
        elif not isinstance(bounds, (tuple, list, float, int)):
            logger.warning(
                f"\n\nLat/Lon clipping must be a tuple of two numeric values "
                f"or a tuple of tuples/lists with two numeric values each. "
                f"\nPoint Example: (35.0, -120.0) "
                f"\nBounds Example: ((32.0, 42.0), (-125.0, -114.0))"
                f"\nGot {name} bounds: {bounds} (type: {type(bounds).__name__})"
            )
            return False

        try:
            min_val, max_val = None, None
            if isinstance(bounds, (tuple, list)):
                min_val, max_val = float(bounds[0]), float(bounds[1])
            elif isinstance(bounds, (float, int)):
                min_val, max_val = float(bounds), float(bounds)
        except (ValueError, TypeError):
            logger.warning(
                f"\n\nCoordinate bounds must be numeric. Invalid {name} bounds: {bounds}. "
                f"\nBoth values must be convertible to float."
            )
            return False
        finally:
            if min_val is None or max_val is None:
                logger.warning(
                    f"\n\nCoordinate bounds must be numeric. Invalid {name} bounds: {bounds}. "
                    f"\nBoth values must be provided."
                )
                return False

        # Validate coordinate ranges
        if name == "latitude":
            if not (-90.0 <= min_val <= 90.0) or not (-90.0 <= max_val <= 90.0):
                logger.warning(
                    f"\n\nLatitude values must be between -90 and 90 degrees. "
                    f"\nGot latitude bounds: ({min_val}, {max_val})"
                )
                return False
        else:  # longitude
            if not (-180.0 <= min_val <= 180.0) or not (-180.0 <= max_val <= 180.0):
                logger.warning(
                    f"\n\nLongitude values must be between -180 and 180 degrees. "
                    f"\nGot longitude bounds: ({min_val}, {max_val})"
                )
                return False

        # Validate that min < max
        if min_val > max_val:
            logger.warning(
                f"\n\nMinimum {name} must be less than or equal to maximum {name}. "
                f"\nGot {name} bounds: ({min_val}, {max_val})"
            )
            return False

    return True


def _validate_station_identifier(value: str) -> bool:
    """
    Validate a station identifier string.

    Parameters
    ----------
    value : str
        Station identifier to validate

    Returns
    -------
    bool
        True if the station identifier is valid
    """
    logger.debug("Validating station identifier: %s", value)
    try:
        catalog = DataCatalog()
        stations_df = catalog.get("stations")

        if stations_df is None or len(stations_df) == 0:
            msg = "Station data is not available. Cannot validate station identifier."
            logger.warning(msg)
            return False

        # Use the generalized matching logic
        match = find_station_match(value, stations_df)

        if len(match) == 0:
            # Station not found - provide suggestions from both stations and boundaries
            all_stations = stations_df["ID"].tolist() + stations_df["station"].tolist()
            station_suggestions = _get_closest_options(value, all_stations, cutoff=0.5)

            # Also check if this might be a boundary key
            boundary_dict = DataCatalog().list_clip_boundaries()
            boundary_list = []
            for _, v in boundary_dict.items():
                boundary_list.extend(v)
            boundary_suggestions = _get_closest_options(
                value, boundary_list, cutoff=0.6
            )

            error_msg = f"Station '{value}' not found in station database."
            logger.warning(error_msg)

            if station_suggestions:
                error_msg += (
                    "\n\nDid you mean one of these stations?\n  - "
                    + "\n  - ".join(station_suggestions[:5])
                )

            if boundary_suggestions:
                error_msg += (
                    "\n\nOr did you mean one of these boundaries?\n  - "
                    + "\n  - ".join(boundary_suggestions[:5])
                )

            error_msg += (
                "\n\nTo see all available stations, use: cd.show_station_options()"
            )

            logger.warning(error_msg)
            return False

        if len(match) > 1:
            # Multiple matches found - show options
            station_list = (
                match[["ID", "station", "city", "state"]].head(5).to_string(index=False)
            )
            msg = (
                f"Multiple stations match '{value}':\n{station_list}\n"
                f"{'... and more' if len(match) > 5 else ''}\n\nPlease use a more specific identifier (4-character code like 'KSAC')."
            )
            logger.warning(msg)
            return False

        # Valid station found
        logger.info("Station identifier validated: %s", value)
        return True

    except Exception as e:
        logger.error("Error validating station identifier: %s", e, exc_info=True)
        return False


def _is_file_path_like(value: str) -> bool:
    """
    Check if a string looks like a file path.

    Parameters
    ----------
    value : str
        String to check

    Returns
    -------
    bool
        True if the string looks like a file path
    """
    # Simple heuristics for file path detection
    file_indicators = [
        value.endswith(
            (".shp", ".geojson", ".gpkg", ".kml", ".kmz", ".gdb", ".zip")
        ),  # Common geospatial formats
        "/" in value or "\\" in value,  # Path separators
        value.startswith(("./", "../", "~/", "/"))
        or (len(value) > 2 and value[1] == ":"),  # Relative/absolute paths
    ]
    return any(file_indicators)


def _validate_boundary_key_string(value: str, raise_on_invalid: bool = True) -> bool:
    """
    Validate a boundary key string with case sensitivity handling.

    Parameters
    ----------
    value : str
        Boundary key to validate
    raise_on_invalid : bool, default True
        Whether to raise an exception on invalid keys

    Returns
    -------
    Union[str, None]
        Validated boundary key or None if invalid

    Raises
    ------
    ValueError
        If raise_on_invalid is True and the key appears invalid
    """
    logger.debug("Validating boundary key string: %s", value)
    # Basic validation - check for obviously invalid patterns
    if len(value) > 200:  # Unreasonably long boundary key
        logger.warning(
            f"\n\nBoundary key is too long ({len(value)} characters). "
            f"\nExpected keys like 'CA', 'Los Angeles County', etc."
        )
        return False

    # Check for invalid characters that would never be in boundary names
    invalid_chars = ["<", ">", "|", '"', "?", "*", "\n", "\r", "\t"]
    if any(char in value for char in invalid_chars):
        logger.warning(
            f"\n\nBoundary key contains invalid characters: '{value}'. "
            f"\nExpected keys like 'CA', 'Los Angeles County', 'PG&E', etc."
        )
        return False

    # For common case variations, provide helpful warnings
    result = _warn_about_case_sensitivity(value)
    logger.info("Boundary key '%s' validation result: %s", value, result)
    return result


def _warn_about_case_sensitivity(value: str) -> bool:
    """
    Provide warnings about potential case sensitivity issues.

    Parameters
    ----------
    value : str
        The boundary key value to check

    Returns
    -------
    bool
        True if the value is valid, False otherwise with warning
    """
    # Common case variations that users might try
    boundary_dict = DataCatalog().list_clip_boundaries()
    boundary_list = []
    for _, v in boundary_dict.items():
        boundary_list.extend(v)
    suggestions = _get_closest_options(value, boundary_list)

    # Handle case where suggestions is None
    if suggestions is None:
        suggestions = []

    if value in suggestions:
        # If the value is already in the suggestions, it is valid
        return True

    if not suggestions:
        msg = (
            f"Boundary key '{value}' does not match any known boundary keys. Please check the spelling or case of the boundary key. Boundary keys are typically case-sensitive."
            f"\n\n{pprint.pformat(boundary_list, indent=4, width=80)}"
        )
        logger.warning(msg)
        return False

    if suggestions:
        # Provide additional hints for common case issues
        hint = ""
        if value.islower() and len(value) <= 3:  # Probably a state abbreviation
            hint = f"\nNote: '{value}' is all lowercase. If this is a state abbreviation, it should probably be uppercase (e.g., '{value.upper()}')."
        elif "county" in value.lower() and not value.endswith("County"):
            hint = "\nNote: County names typically end with 'County' (e.g., 'Los Angeles County')."

        msg = f"Boundary key '{value}' may have case sensitivity issues. Did you mean any of the following options: {suggestions}. Boundary keys are typically case-sensitive. {hint}"
        logger.warning(msg)
        return False

    return False
