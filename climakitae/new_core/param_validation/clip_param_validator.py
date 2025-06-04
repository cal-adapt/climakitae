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


@register_processor_validator("clip")
def validate_clip_param(
    value,
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
            "Clip parameter cannot be None. Please provide a valid boundary key, list of keys, file path, or coordinate bounds."
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
            )

    return False


def _validate_string_param(value: str) -> bool:
    """
    Validate a string parameter for the Clip processor.

    Parameters
    ----------
    value : str
        String value to validate

    Returns
    -------
    Union[str, None]
        Validated string or None if invalid

    Raises
    ------
    ValueError
        If the string is empty, whitespace-only, or invalid
    """
    # Check for empty or whitespace-only strings
    if not value or not value.strip():
        warnings.warn(
            "\n\nEmpty or whitespace-only strings are not valid clip parameters. "
            "\nPlease provide a valid boundary key (e.g., 'CA'), file path, or coordinate bounds."
        )
        return False

    # Clean the string
    cleaned_value = value.strip()

    # Check if it looks like a file path
    if _is_file_path_like(cleaned_value):
        if os.path.exists(cleaned_value):
            return True
        else:
            warnings.warn(
                f"\n\nFile path '{cleaned_value}' does not exist. "
                f"\nPlease provide a valid file path to a shapefile or other geospatial data."
            )
            return False

    # For boundary keys, provide basic validation and case sensitivity warnings
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
            "Please provide a list of boundary keys (e.g., ['CA', 'OR', 'WA'])."
        )
        return False

    # Check for mixed types (should all be strings)
    non_string_items = [item for item in value if not isinstance(item, str)]
    if non_string_items:
        non_string_types = [type(item).__name__ for item in non_string_items]
        warnings.warn(
            f"\n\nAll items in clip parameter list must be strings. "
            f"\nFound {len(non_string_items)} non-string items with types: {', '.join(set(non_string_types))}. "
            f"\nExample of valid list: ['CA', 'OR', 'WA']"
        )
        return False

    # Filter out empty/whitespace strings and validate each item
    valid_items = []
    invalid_items = []

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
            f"\n\nFound {len(invalid_items)} invalid items in clip parameter list: {', '.join(invalid_items)}. "
        )
        return False

    if not valid_items:
        warnings.warn(
            "\n\nNo valid boundary keys found in the provided list. "
            "\nPlease provide valid boundary keys such as ['CA', 'OR', 'WA'] or "
            "\n['Los Angeles County', 'Orange County']."
        )
        return False

    # Check for duplicates
    unique = set(value)

    if len(unique) != len(value):
        warnings.warn(
            f"\n\nDuplicate boundary keys found in the list. "
            f"\nOriginal list: {value} "
            f"\nUnique list: {unique} "
            f"\n\nPlease remove duplicates."
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
            f"Got {len(value)} elements. Example: ((32.0, 42.0), (-125.0, -114.0))"
        )
        return False

    lat_bounds, lon_bounds = value

    # Validate that each bound is a tuple/list with 2 numeric values
    for bounds, name in [(lat_bounds, "latitude"), (lon_bounds, "longitude")]:
        if not isinstance(bounds, (tuple, list)) or len(bounds) != 2:
            warnings.warn(
                f"\n\nEach coordinate bound must be a tuple/list with 2 values (min, max). "
                f"\nInvalid {name} bounds: {bounds}. "
                f"\nExample: ((32.0, 42.0), (-125.0, -114.0))"
            )
            return False

        try:
            min_val, max_val = float(bounds[0]), float(bounds[1])
        except (ValueError, TypeError):
            warnings.warn(
                f"\n\nCoordinate bounds must be numeric. Invalid {name} bounds: {bounds}. "
                f"\nBoth values must be convertible to float."
            )
            return False

        # Validate coordinate ranges
        if name == "latitude":
            if not (-90.0 <= min_val <= 90.0) or not (-90.0 <= max_val <= 90.0):
                warnings.warn(
                    f"\n\nLatitude values must be between -90 and 90 degrees. "
                    f"\nGot latitude bounds: ({min_val}, {max_val})"
                )
                return False
        else:  # longitude
            if not (-180.0 <= min_val <= 180.0) or not (-180.0 <= max_val <= 180.0):
                warnings.warn(
                    f"\n\nLongitude values must be between -180 and 180 degrees. "
                    f"\nGot longitude bounds: ({min_val}, {max_val})"
                )
                return False

        # Validate that min < max
        if min_val >= max_val:
            warnings.warn(
                f"\n\nMinimum {name} must be less than maximum {name}. "
                f"\nGot {name} bounds: ({min_val}, {max_val})"
            )
            return False

    return True


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
            f"\nExpected keys like 'CA', 'Los Angeles County', etc."
        )
        return False

    # Check for invalid characters that would never be in boundary names
    invalid_chars = ["<", ">", "|", '"', "?", "*", "\n", "\r", "\t"]
    if any(char in value for char in invalid_chars):
        warnings.warn(
            f"\n\nBoundary key contains invalid characters: '{value}'. "
            f"\nExpected keys like 'CA', 'Los Angeles County', 'PG&E', etc."
        )
        return False

    # For common case variations, provide helpful warnings
    return _warn_about_case_sensitivity(value)


def _warn_about_case_sensitivity(value: str) -> bool:
    """
    Provide warnings about potential case sensitivity issues.

    Parameters
    ----------
    bool :
        true if the value is valid, otherwise raises a warning
    """
    # Common case variations that users might try
    boundary_dict = DataCatalog().list_clip_boundaries()
    boundary_list = []
    for _, v in boundary_dict.items():
        boundary_list.extend(v)
    suggestions = _get_closest_options(value, boundary_list)

    if value in suggestions:
        # If the value is already in the suggestions, it is valid
        return True

    if not suggestions:
        warnings.warn(
            f"\n\nBoundary key '{value}' does not match any known boundary keys. "
            f"\nPlease check the spelling or case of the boundary key. "
            f"\nBoundary keys are typically case-sensitive."
            f"\n\n{pprint.pformat(boundary_list, indent=4, width=80)}",
        )

    if suggestions:
        warnings.warn(
            f"\n\nBoundary key '{value}' may have case sensitivity issues. "
            f"\nDid you mean any of the following options: "
            f"\n{suggestions}"
            f"\nBoundary keys are typically case-sensitive.",
            UserWarning,
        )

    # General case warnings for patterns that look like they might have case issues
    elif value.islower() and len(value) <= 3:  # Probably a state abbreviation
        warnings.warn(
            f"Boundary key '{value}' is all lowercase. "
            f"If this is a state abbreviation, it should probably be uppercase (e.g., '{value.upper()}'). "
            f"Boundary keys are typically case-sensitive.",
            UserWarning,
        )
    elif "county" in value.lower() and not value.endswith("County"):
        warnings.warn(
            f"\n\nCounty name '{value}' may have incorrect capitalization. "
            f"\nCounty names typically end with 'County' (e.g., 'Los Angeles County'). "
            f"\nBoundary keys are typically case-sensitive.",
            UserWarning,
        )

    return False
