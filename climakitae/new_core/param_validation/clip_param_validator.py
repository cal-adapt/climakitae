"""
Validator for parameters provided to Clip Processor.
"""

from __future__ import annotations

import os
import pprint
import warnings
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

    # Handle None values early
    if value is None or value is UNSET:
        warnings.warn(
            "Clip parameter cannot be None. Please provide a valid boundary key, list of keys, file path, or coordinate bounds.",
            UserWarning,
            stacklevel=999,
        )
        return False

    match value:
        case str():
            return _validate_string_param(value)
        case list():
            return _validate_list_param(value)
        case tuple():
            return _validate_tuple_param(value)
        case _:
            warnings.warn(
                f"\n\nInvalid parameter type for Clip processor. "
                f"\nExpected str, list, or tuple, but got {type(value).__name__}. "
                f"\nValid examples: 'CA', ['CA', 'OR'], or ((32.0, 42.0), (-125.0, -114.0))",
                UserWarning,
                stacklevel=999,
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
    # Check for empty or whitespace-only strings
    if not value or not value.strip():
        warnings.warn(
            "\n\nEmpty or whitespace-only strings are not valid clip parameters. "
            "\nPlease provide a valid boundary key (e.g., 'CA'), file path, station code (e.g., 'KSAC'), or coordinate bounds.",
            UserWarning,
            stacklevel=999,
        )
        return False

    # Clean the string
    cleaned_value = value.strip()

    # Priority 1: Check if it looks like a file path (most specific check)
    if _is_file_path_like(cleaned_value):
        if os.path.exists(cleaned_value):
            return True
        else:
            warnings.warn(
                f"\n\nFile path '{cleaned_value}' does not exist. "
                f"\nPlease provide a valid file path to a shapefile or other geospatial data.",
                UserWarning,
                stacklevel=999,
            )
            return False

    # Priority 2: Check if it strongly looks like a station identifier
    # Only consider it a station if it matches the strict format
    is_station_like = is_station_identifier(cleaned_value)

    # If it looks like a station, validate it as such first
    if is_station_like:
        station_result = _validate_station_identifier(cleaned_value)
        if station_result:
            return True

        # Station validation failed - check if it might be a boundary instead
        # Suppress warnings from boundary validation since we already warned about station
        import contextlib

        with contextlib.redirect_stderr(None):
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                boundary_result = _validate_boundary_key_string(cleaned_value)

        if boundary_result:
            # Provide helpful message that it's not a station but could be a boundary
            warnings.warn(
                f"\n\n'{cleaned_value}' looks like a station identifier but was not found in the station database. "
                f"\nHowever, it appears to match a boundary key. "
                f"\nIf you intended to clip to a station, please check the station code. "
                f"\nIf you intended to clip to a boundary, the value will be accepted.",
                UserWarning,
                stacklevel=999,
            )
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
    if not value or value is UNSET:
        warnings.warn(
            "Empty list is not valid for clip parameters. "
            "Please provide a list of boundary keys (e.g., ['CA', 'OR', 'WA']).",
            UserWarning,
            stacklevel=999,
        )
        return False

    # Check for mixed types (should all be the same type)
    mixed_types = [item for item in value if not isinstance(item, type(value[0]))]
    if mixed_types:
        unique_types = set([type(item).__name__ for item in mixed_types])
        warnings.warn(
            f"\n\nAll items in clip parameter list must be the same type. "
            f"\nFound {len(unique_types)} different types: {', '.join(set(unique_types))}. "
            f"\nExample of valid list: ['CA', 'OR', 'WA'] or [(32.0, 42.0), (-125.0, -114.0)]",
            UserWarning,
            stacklevel=999,
        )
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
                warnings.warn(
                    f"\n\nFound {len(invalid_items)} invalid items in clip parameter list: {', '.join(invalid_items)}. ",
                    UserWarning,
                    stacklevel=999,
                )
                return False

            if not valid_items:
                warnings.warn(
                    "\n\nNo valid boundary keys found in the provided list. "
                    "\nPlease provide valid boundary keys such as ['CA', 'OR', 'WA'] or "
                    "\n['Los Angeles County', 'Orange County'].",
                    UserWarning,
                    stacklevel=999,
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
                warnings.warn(
                    f"\n\nFound {len(invalid_items)} invalid items in clip parameter list: {', '.join(invalid_items)}. ",
                    UserWarning,
                    stacklevel=999,
                )
                return False

    # Check for duplicates
    unique = set(value)

    if len(unique) != len(value):
        warnings.warn(
            f"\n\nDuplicate boundary keys found in the list. "
            f"\nOriginal list: {value} "
            f"\nUnique list: {unique} "
            f"\n\nPlease remove duplicates.",
            UserWarning,
            stacklevel=999,
        )
        return False

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
    # Check tuple structure: should be ((lat_min, lat_max), (lon_min, lon_max))
    if len(value) != 2:
        warnings.warn(
            f"Coordinate bounds tuple must have exactly 2 elements: ((lat_min, lat_max), (lon_min, lon_max)). "
            f"Got {len(value)} elements. Example: ((32.0, 42.0), (-125.0, -114.0))",
            UserWarning,
            stacklevel=999,
        )
        return False

    lat_bounds, lon_bounds = value

    # Validate that each bound is a tuple/list with 2 numeric values
    for bounds, name in [(lat_bounds, "latitude"), (lon_bounds, "longitude")]:
        # Check if bounds is a tuple/list with wrong length
        if isinstance(bounds, (tuple, list)) and len(bounds) != 2:
            warnings.warn(
                f"\n\nLat/Lon clipping must either be a tuple of two numeric values"
                f"\nor a tuple of tuples/lists with two numeric values each. "
                f"\nPoint Example: (35.0, -120.0) "
                f"\nBounds Example: ((32.0, 42.0), (-125.0, -114.0))"
                f"\nGot {name} bounds: {bounds} (type: {type(bounds).__name__})",
                UserWarning,
                stacklevel=999,
            )
            return False
        # Check if bounds is an invalid type (not tuple, list, float, or int)
        elif not isinstance(bounds, (tuple, list, float, int)):
            warnings.warn(
                f"\n\nLat/Lon clipping must be a tuple of two numeric values "
                f"or a tuple of tuples/lists with two numeric values each. "
                f"\nPoint Example: (35.0, -120.0) "
                f"\nBounds Example: ((32.0, 42.0), (-125.0, -114.0))"
                f"\nGot {name} bounds: {bounds} (type: {type(bounds).__name__})",
                UserWarning,
                stacklevel=999,
            )
            return False

        try:
            min_val, max_val = None, None
            if isinstance(bounds, (tuple, list)):
                min_val, max_val = float(bounds[0]), float(bounds[1])
            elif isinstance(bounds, (float, int)):
                min_val, max_val = float(bounds), float(bounds)
        except (ValueError, TypeError):
            warnings.warn(
                f"\n\nCoordinate bounds must be numeric. Invalid {name} bounds: {bounds}. "
                f"\nBoth values must be convertible to float.",
                UserWarning,
                stacklevel=999,
            )
            return False
        finally:
            if min_val is None or max_val is None:
                warnings.warn(
                    f"\n\nCoordinate bounds must be numeric. Invalid {name} bounds: {bounds}. "
                    f"\nBoth values must be provided.",
                    UserWarning,
                    stacklevel=999,
                )
                return False

        # Validate coordinate ranges
        if name == "latitude":
            if not (-90.0 <= min_val <= 90.0) or not (-90.0 <= max_val <= 90.0):
                warnings.warn(
                    f"\n\nLatitude values must be between -90 and 90 degrees. "
                    f"\nGot latitude bounds: ({min_val}, {max_val})",
                    UserWarning,
                    stacklevel=999,
                )
                return False
        else:  # longitude
            if not (-180.0 <= min_val <= 180.0) or not (-180.0 <= max_val <= 180.0):
                warnings.warn(
                    f"\n\nLongitude values must be between -180 and 180 degrees. "
                    f"\nGot longitude bounds: ({min_val}, {max_val})",
                    UserWarning,
                    stacklevel=999,
                )
                return False

        # Validate that min < max
        if min_val > max_val:
            warnings.warn(
                f"\n\nMinimum {name} must be less than or equal to maximum {name}. "
                f"\nGot {name} bounds: ({min_val}, {max_val})",
                UserWarning,
                stacklevel=999,
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
    try:
        catalog = DataCatalog()
        stations_df = catalog.get("stations")

        if stations_df is None or len(stations_df) == 0:
            warnings.warn(
                "\n\nStation data is not available. "
                "\nCannot validate station identifier.",
                UserWarning,
                stacklevel=999,
            )
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

            error_msg = f"\n\nStation '{value}' not found in station database."

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

            warnings.warn(error_msg, UserWarning, stacklevel=999)
            return False

        if len(match) > 1:
            # Multiple matches found - show options
            station_list = (
                match[["ID", "station", "city", "state"]].head(5).to_string(index=False)
            )
            warnings.warn(
                f"\n\nMultiple stations match '{value}':\n{station_list}\n"
                f"{'... and more' if len(match) > 5 else ''}\n\n"
                f"Please use a more specific identifier (4-character code like 'KSAC').",
                UserWarning,
                stacklevel=999,
            )
            return False

        # Valid station found
        return True

    except Exception as e:
        warnings.warn(
            f"\n\nError validating station identifier: {str(e)}",
            UserWarning,
            stacklevel=999,
        )
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
    # Basic validation - check for obviously invalid patterns
    if len(value) > 200:  # Unreasonably long boundary key
        warnings.warn(
            f"\n\nBoundary key is too long ({len(value)} characters). "
            f"\nExpected keys like 'CA', 'Los Angeles County', etc.",
            UserWarning,
            stacklevel=999,
        )
        return False

    # Check for invalid characters that would never be in boundary names
    invalid_chars = ["<", ">", "|", '"', "?", "*", "\n", "\r", "\t"]
    if any(char in value for char in invalid_chars):
        warnings.warn(
            f"\n\nBoundary key contains invalid characters: '{value}'. "
            f"\nExpected keys like 'CA', 'Los Angeles County', 'PG&E', etc.",
            UserWarning,
            stacklevel=999,
        )
        return False

    # For common case variations, provide helpful warnings
    return _warn_about_case_sensitivity(value)


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
        warnings.warn(
            f"\n\nBoundary key '{value}' does not match any known boundary keys. "
            f"\nPlease check the spelling or case of the boundary key. "
            f"\nBoundary keys are typically case-sensitive."
            f"\n\n{pprint.pformat(boundary_list, indent=4, width=80)}",
            UserWarning,
            stacklevel=999,
        )
        return False

    if suggestions:
        # Provide additional hints for common case issues
        hint = ""
        if value.islower() and len(value) <= 3:  # Probably a state abbreviation
            hint = f"\nNote: '{value}' is all lowercase. If this is a state abbreviation, it should probably be uppercase (e.g., '{value.upper()}')."
        elif "county" in value.lower() and not value.endswith("County"):
            hint = "\nNote: County names typically end with 'County' (e.g., 'Los Angeles County')."

        warnings.warn(
            f"\n\nBoundary key '{value}' may have case sensitivity issues. "
            f"\nDid you mean any of the following options: "
            f"\n{suggestions}"
            f"\nBoundary keys are typically case-sensitive."
            f"{hint}",
            UserWarning,
            stacklevel=999,
        )
        return False

    return False
