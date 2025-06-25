"""
Validator for parameters provided to Export Processor.
"""

from __future__ import annotations

import glob
import os
import warnings
from typing import Any, Dict

from climakitae.core.constants import UNSET
from climakitae.new_core.param_validation.abc_param_validation import (
    register_processor_validator,
)
from climakitae.new_core.param_validation.param_validation_tools import (
    _get_closest_options,
)


@register_processor_validator("export")
def validate_export_param(
    value: Any,
    **_kwargs: Any,
) -> bool:
    """
    Validate parameters passed to Export Processor.

    This function validates and normalizes parameters for the Export processor,
    addressing common input validation issues:

    1. **File Path Validation**: Checks if output files already exist
    2. **Parameter Type Validation**: Ensures correct types for all parameters
    3. **Format/Mode Compatibility**: Validates S3 requires Zarr format
    4. **Template Validation**: Validates filename template placeholders
    5. **File Conflict Detection**: Warns about existing files with similar names

    Parameters
    ----------
    value : Any
        Export configuration parameters (expected to be a dictionary)

    Returns
    -------
    bool
        True if parameters are valid, False otherwise

    Raises
    ------
    ValueError
        If parameters are invalid and cannot be corrected
    TypeError
        If parameter types are incorrect
    """

    if value is None or value is UNSET:
        warnings.warn(
            "Export parameters cannot be None. Using default export configuration."
        )
        return True  # Will use defaults

    if not isinstance(value, dict):
        warnings.warn(
            f"Export parameters must be a dictionary, got {type(value).__name__}. "
            f"Using default export configuration."
        )
        return False

    # Validate individual parameters
    try:
        _validate_filename_param(value)
        _validate_file_format_param(value)
        _validate_mode_param(value)
        _validate_export_method_param(value)
        _validate_boolean_params(value)
        _validate_filename_template_param(value)
        _validate_format_mode_compatibility(value)
        _check_file_conflicts(value)

    except (ValueError, TypeError) as e:
        warnings.warn(f"Export parameter validation failed: {str(e)}")
        return False

    return True


def _validate_filename_param(params: Dict[str, Any]) -> None:
    """
    Validate the filename parameter.

    Parameters
    ----------
    params : Dict[str, Any]
        Export parameters dictionary

    Raises
    ------
    ValueError
        If filename is invalid
    """
    filename = params.get("filename", "dataexport")

    if not isinstance(filename, str):
        raise ValueError(f"filename must be a string, got {type(filename).__name__}")

    if not filename.strip():
        raise ValueError("filename cannot be empty or whitespace-only")

    # Check for invalid filename characters
    invalid_chars = ["<", ">", ":", '"', "|", "?", "*"]
    if any(char in filename for char in invalid_chars):
        raise ValueError(
            f"filename contains invalid characters. "
            f"Avoid: {', '.join(invalid_chars)}"
        )

    # Warn about potential path separators
    if "/" in filename or "\\" in filename:
        warnings.warn(
            "filename appears to contain path separators. "
            "Only the filename portion will be used for the output file."
        )


def _validate_file_format_param(params: Dict[str, Any]) -> None:
    """
    Validate the file_format parameter.

    Parameters
    ----------
    params : Dict[str, Any]
        Export parameters dictionary

    Raises
    ------
    ValueError
        If file_format is invalid
    """
    file_format = params.get("file_format", "NetCDF")
    valid_formats = ["netcdf", "zarr", "csv"]

    if not isinstance(file_format, str):
        raise ValueError(
            f"file_format must be a string, got {type(file_format).__name__}"
        )

    # Case-insensitive validation with suggestion
    if file_format.lower() not in valid_formats:
        closest_options = _get_closest_options(file_format.lower(), valid_formats)
        error_msg = (
            f'file_format "{file_format}" is not valid. Valid options: {valid_formats}'
        )

        if closest_options:
            error_msg += f". Did you mean: {closest_options[0]}?"

        raise ValueError(error_msg)


def _validate_mode_param(params: Dict[str, Any]) -> None:
    """
    Validate the mode parameter.

    Parameters
    ----------
    params : Dict[str, Any]
        Export parameters dictionary

    Raises
    ------
    ValueError
        If mode is invalid
    """
    mode = params.get("mode", "local")
    valid_modes = ["local", "s3"]

    if not isinstance(mode, str):
        raise ValueError(f"mode must be a string, got {type(mode).__name__}")

    if mode.lower() not in valid_modes:
        closest_options = _get_closest_options(mode.lower(), valid_modes)
        error_msg = f'mode "{mode}" is not valid. Valid options: {valid_modes}'

        if closest_options:
            error_msg += f". Did you mean: {closest_options[0]}?"

        raise ValueError(error_msg)


def _validate_export_method_param(params: Dict[str, Any]) -> None:
    """
    Validate the export_method parameter.

    Parameters
    ----------
    params : Dict[str, Any]
        Export parameters dictionary

    Raises
    ------
    ValueError
        If export_method is invalid
    """
    export_method = params.get("export_method", "data")
    valid_methods = ["data", "skip_existing", "none"]

    if not isinstance(export_method, str):
        raise ValueError(
            f"export_method must be a string, got {type(export_method).__name__}"
        )

    if export_method.lower() not in valid_methods:
        closest_options = _get_closest_options(export_method.lower(), valid_methods)
        error_msg = f'export_method "{export_method}" is not valid. Valid options: {valid_methods}'

        if closest_options:
            error_msg += f". Did you mean: {closest_options[0]}?"

        raise ValueError(error_msg)


def _validate_boolean_params(params: Dict[str, Any]) -> None:
    """
    Validate boolean parameters.

    Parameters
    ----------
    params : Dict[str, Any]
        Export parameters dictionary

    Raises
    ------
    ValueError
        If boolean parameters are invalid
    """
    boolean_params = ["separated", "location_based_naming", "fail_on_error"]

    for param_name in boolean_params:
        value = params.get(param_name)
        if value is not None and not isinstance(value, bool):
            raise ValueError(
                f"{param_name} must be a boolean, got {type(value).__name__}"
            )


def _validate_filename_template_param(params: Dict[str, Any]) -> None:
    """
    Validate the filename_template parameter.

    Parameters
    ----------
    params : Dict[str, Any]
        Export parameters dictionary

    Raises
    ------
    ValueError
        If filename_template is invalid
    """
    template = params.get("filename_template")

    if template is None:
        return  # Optional parameter

    if not isinstance(template, str):
        raise ValueError(
            f"filename_template must be a string, got {type(template).__name__}"
        )

    if not template.strip():
        raise ValueError("filename_template cannot be empty or whitespace-only")

    # Check for valid template placeholders
    valid_placeholders = ["{filename}", "{name}", "{lat}", "{lon}"]

    # Look for invalid placeholder patterns
    import re

    all_placeholders = re.findall(r"\{[^}]*\}", template)
    invalid_placeholders = [p for p in all_placeholders if p not in valid_placeholders]

    if invalid_placeholders:
        warnings.warn(
            f"filename_template contains unrecognized placeholders: {invalid_placeholders}. "
            f"Valid placeholders are: {valid_placeholders}"
        )


def _validate_format_mode_compatibility(params: Dict[str, Any]) -> None:
    """
    Validate compatibility between file_format and mode parameters.

    Parameters
    ----------
    params : Dict[str, Any]
        Export parameters dictionary

    Raises
    ------
    ValueError
        If format and mode are incompatible
    """
    file_format = params.get("file_format", "NetCDF").lower()
    mode = params.get("mode", "local").lower()

    # S3 export only works with Zarr
    if mode == "s3" and file_format != "zarr":
        raise ValueError(
            'S3 export (mode="s3") is only supported with Zarr format. '
            'Use file_format="Zarr" or mode="local".'
        )


def _check_file_conflicts(params: Dict[str, Any]) -> None:
    """
    Check for existing files that might be overwritten.

    Parameters
    ----------
    params : Dict[str, Any]
        Export parameters dictionary
    """
    filename = params.get("filename", "dataexport")
    file_format = params.get("file_format", "NetCDF").lower()
    export_method = params.get("export_method", "data").lower()

    # Skip checks for certain export methods
    if export_method in ["none", "skip_existing"]:
        return

    # Determine file extension
    extension_map = {"zarr": ".zarr", "netcdf": ".nc", "csv": ".csv.gz"}
    extension = extension_map.get(file_format, ".nc")

    # Check if exact file exists
    full_filename = f"{filename}{extension}"
    if os.path.exists(full_filename):
        warnings.warn(
            f"File '{full_filename}' already exists and will be overwritten. "
            f'Use export_method="skip_existing" to avoid overwriting existing files.'
        )

    # Check for similar files that might be confused
    _warn_about_similar_files(filename, extension)


def _warn_about_similar_files(base_filename: str, target_extension: str) -> None:
    """
    Warn about existing files with similar names.

    Parameters
    ----------
    base_filename : str
        Base filename without extension
    target_extension : str
        Target file extension
    """
    # Look for files with same base name but different extensions
    similar_patterns = [
        f"{base_filename}.*",
        f"{base_filename}_*",
        f"*_{base_filename}.*",
    ]

    similar_files = []
    for pattern in similar_patterns:
        similar_files.extend(glob.glob(pattern))

    # Remove the exact target file from the list
    target_file = f"{base_filename}{target_extension}"
    similar_files = [f for f in similar_files if f != target_file]

    if similar_files:
        # Limit to first few files to avoid spam
        display_files = similar_files[:5]
        file_list = ", ".join(display_files)
        if len(similar_files) > 5:
            file_list += f" (and {len(similar_files) - 5} more)"

        warnings.warn(
            f"Found existing files with similar names: {file_list}. "
            f"Please verify you're using the correct filename to avoid confusion."
        )


def _is_path_safe(filename: str) -> bool:
    """
    Check if a filename is safe for file creation.

    Parameters
    ----------
    filename : str
        Filename to check

    Returns
    -------
    bool
        True if filename is safe
    """
    # Check for directory traversal attempts
    dangerous_patterns = ["..", "/", "\\", "~"]
    return not any(pattern in filename for pattern in dangerous_patterns)


def validate_export_output_path(
    filename: str, file_format: str = "NetCDF", check_permissions: bool = True
) -> Dict[str, Any]:
    """
    Comprehensive validation of export output path.

    Parameters
    ----------
    filename : str
        Filename for export
    file_format : str
        File format for extension mapping
    check_permissions : bool
        Whether to check write permissions

    Returns
    -------
    Dict[str, Any]
        Validation results with warnings and recommendations
    """
    results = {"is_valid": True, "warnings": [], "errors": [], "full_path": None}

    try:
        # Determine full path
        extension_map = {"zarr": ".zarr", "netcdf": ".nc", "csv": ".csv.gz"}
        extension = extension_map.get(file_format.lower(), ".nc")
        full_path = f"{filename}{extension}"
        results["full_path"] = full_path

        # Check if file exists
        if os.path.exists(full_path):
            results["warnings"].append(f"File '{full_path}' will be overwritten")

        # Check write permissions for directory
        if check_permissions:
            directory = os.path.dirname(full_path) or "."
            if not os.access(directory, os.W_OK):
                results["errors"].append(
                    f"No write permission for directory '{directory}'"
                )
                results["is_valid"] = False

        # Check for path safety
        if not _is_path_safe(filename):
            results["errors"].append(
                f"Filename '{filename}' contains unsafe characters"
            )
            results["is_valid"] = False

    except (OSError, PermissionError, ValueError) as e:
        results["errors"].append(f"Path validation error: {str(e)}")
        results["is_valid"] = False

    return results
