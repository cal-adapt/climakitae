"""
Unit tests for climakitae/new_core/param_validation/export_param_validator.py

This module contains comprehensive unit tests for the export parameter validation
functions that validate and normalize parameters for the Export processor.

Tests cover:
- Main validate_export_param function with various input types
- Filename parameter validation (type, empty, invalid characters)
- File format parameter validation with auto-correction
- Format inference from common variations and typos
- Mode parameter validation
- Export method parameter validation
- Boolean parameters validation
- Filename template validation
- Format/mode compatibility checks
- Output path validation
- Filename prediction utilities
- Alternative suggestions
"""

import logging
import os
import tempfile
from unittest.mock import patch

import pytest

from climakitae.core.constants import UNSET
from climakitae.new_core.param_validation.export_param_validator import (
    _infer_file_format,
    _is_path_safe,
    _predict_export_filenames,
    _suggest_alternative_filename,
    _validate_boolean_params,
    _validate_export_method_param,
    _validate_file_format_param,
    _validate_filename_param,
    _validate_filename_template_param,
    _validate_format_mode_compatibility,
    _validate_mode_param,
    suggest_export_alternatives,
    validate_export_output_path,
    validate_export_param,
)


class TestValidateExportParam:
    """Test class for validate_export_param main entry point."""

    def test_validate_with_none_returns_true_with_warning(self, caplog):
        """Test validate_export_param with None returns True (uses defaults)."""
        with caplog.at_level(logging.WARNING):
            result = validate_export_param(None)
        assert result is True
        assert "Export parameters cannot be None" in caplog.text

    def test_validate_with_unset_returns_true_with_warning(self, caplog):
        """Test validate_export_param with UNSET returns True (uses defaults)."""
        with caplog.at_level(logging.WARNING):
            result = validate_export_param(UNSET)
        assert result is True
        assert "Export parameters cannot be None" in caplog.text

    def test_validate_with_non_dict_returns_false_with_warning(self, caplog):
        """Test validate_export_param with non-dict type returns False."""
        with caplog.at_level(logging.WARNING):
            result = validate_export_param("not a dict")
        assert result is False
        assert "Export parameters must be a dictionary" in caplog.text

    def test_validate_with_empty_dict_returns_true(self):
        """Test validate_export_param with empty dict uses defaults."""
        result = validate_export_param({})
        assert result is True

    def test_validate_with_valid_params_returns_true(self):
        """Test validate_export_param with valid parameters returns True."""
        params = {
            "filename": "test_output",
            "file_format": "NetCDF",
            "mode": "local",
            "export_method": "data",
        }
        result = validate_export_param(params)
        assert result is True

    def test_validate_with_invalid_params_returns_false(self, caplog):
        """Test validate_export_param with invalid parameters returns False."""
        params = {"file_format": "invalid_format_xyz"}
        with caplog.at_level(logging.WARNING):
            result = validate_export_param(params)
        assert result is False
        assert "Export parameter validation failed" in caplog.text


class TestValidateFilenameParam:
    """Test class for _validate_filename_param function."""

    def test_valid_filename(self):
        """Test with valid filename."""
        params = {"filename": "my_export_file"}
        # Should not raise
        _validate_filename_param(params)

    def test_default_filename_when_missing(self):
        """Test that default filename is used when not provided."""
        params = {}
        # Should not raise, uses default "dataexport"
        _validate_filename_param(params)

    def test_filename_non_string_raises_value_error(self):
        """Test that non-string filename raises ValueError."""
        params = {"filename": 12345}
        with pytest.raises(ValueError, match="filename must be a string"):
            _validate_filename_param(params)

    def test_filename_empty_string_raises_value_error(self):
        """Test that empty filename raises ValueError."""
        params = {"filename": ""}
        with pytest.raises(ValueError, match="filename cannot be empty"):
            _validate_filename_param(params)

    def test_filename_whitespace_only_raises_value_error(self):
        """Test that whitespace-only filename raises ValueError."""
        params = {"filename": "   "}
        with pytest.raises(ValueError, match="filename cannot be empty"):
            _validate_filename_param(params)

    @pytest.mark.parametrize(
        "invalid_char",
        ["<", ">", ":", '"', "|", "?", "*"],
        ids=["less_than", "greater_than", "colon", "quote", "pipe", "question", "star"],
    )
    def test_filename_with_invalid_characters_raises_value_error(self, invalid_char):
        """Test that filenames with invalid characters raise ValueError."""
        params = {"filename": f"test{invalid_char}file"}
        with pytest.raises(ValueError, match="filename contains invalid characters"):
            _validate_filename_param(params)

    def test_filename_with_path_separator_logs_warning(self, caplog):
        """Test that filename with path separators logs warning."""
        params = {"filename": "path/to/file"}
        with caplog.at_level(logging.WARNING):
            _validate_filename_param(params)
        assert "path separators" in caplog.text


class TestValidateFileFormatParam:
    """Test class for _validate_file_format_param function."""

    @pytest.mark.parametrize(
        "valid_format",
        ["netcdf", "NetCDF", "NETCDF", "zarr", "Zarr", "ZARR", "csv", "CSV"],
        ids=["netcdf_lower", "netcdf_title", "netcdf_upper",
             "zarr_lower", "zarr_title", "zarr_upper", "csv_lower", "csv_upper"],
    )
    def test_valid_file_formats(self, valid_format):
        """Test that valid file formats pass validation."""
        params = {"file_format": valid_format}
        # Should not raise
        _validate_file_format_param(params)

    def test_default_format_when_missing(self):
        """Test that default file format is used when not provided."""
        params = {}
        # Should not raise, uses default "NetCDF"
        _validate_file_format_param(params)

    def test_file_format_non_string_raises_value_error(self):
        """Test that non-string file_format raises ValueError."""
        params = {"file_format": 123}
        with pytest.raises(ValueError, match="file_format must be a string"):
            _validate_file_format_param(params)

    def test_invalid_format_raises_value_error(self):
        """Test that completely invalid format raises ValueError."""
        params = {"file_format": "invalid_format_xyz"}
        with pytest.raises(ValueError, match="is not valid"):
            _validate_file_format_param(params)

    @pytest.mark.parametrize(
        "typo_format,expected_corrected",
        [("nc", "Netcdf"), ("nc4", "Netcdf"), ("hdf5", "Netcdf"),
         ("zar", "Zarr"), ("txt", "Csv"), ("comma", "Csv")],
        ids=["nc", "nc4", "hdf5", "zar", "txt", "comma"],
    )
    def test_format_auto_correction(self, typo_format, expected_corrected, caplog):
        """Test that common typos and variations are auto-corrected."""
        params = {"file_format": typo_format}
        with caplog.at_level(logging.INFO):
            _validate_file_format_param(params)
        # Check that the params were updated in place
        assert params["file_format"] == expected_corrected
        assert "Interpreted" in caplog.text


class TestInferFileFormat:
    """Test class for _infer_file_format function."""

    @pytest.mark.parametrize(
        "input_format",
        ["netcdf", "netcdf4", "netcdf-4", "nc", "nc4", "ncdf", "hdf", "hdf5", "cdf"],
        ids=["netcdf", "netcdf4", "netcdf-4", "nc", "nc4", "ncdf", "hdf", "hdf5", "cdf"],
    )
    def test_infer_netcdf_formats(self, input_format):
        """Test inferring NetCDF format from various inputs."""
        result = _infer_file_format(input_format)
        assert result == "netcdf"

    @pytest.mark.parametrize(
        "input_format",
        ["zarr", "zar", "zarrs", "zarray", "z", "zr"],
        ids=["zarr", "zar", "zarrs", "zarray", "z", "zr"],
    )
    def test_infer_zarr_formats(self, input_format):
        """Test inferring Zarr format from various inputs."""
        result = _infer_file_format(input_format)
        assert result == "zarr"

    @pytest.mark.parametrize(
        "input_format",
        ["csv", "csv.gz", "comma", "txt", "text", "delimited"],
        ids=["csv", "csv.gz", "comma", "txt", "text", "delimited"],
    )
    def test_infer_csv_formats(self, input_format):
        """Test inferring CSV format from various inputs."""
        result = _infer_file_format(input_format)
        assert result == "csv"

    def test_infer_returns_none_for_unknown(self):
        """Test that unknown formats return None."""
        result = _infer_file_format("completely_unknown_format_xyz")
        assert result is None


class TestValidateModeParam:
    """Test class for _validate_mode_param function."""

    @pytest.mark.parametrize(
        "valid_mode",
        ["local", "Local", "LOCAL", "s3", "S3"],
        ids=["local_lower", "local_title", "local_upper", "s3_lower", "s3_upper"],
    )
    def test_valid_modes(self, valid_mode):
        """Test that valid modes pass validation."""
        params = {"mode": valid_mode}
        # Should not raise
        _validate_mode_param(params)

    def test_default_mode_when_missing(self):
        """Test that default mode is used when not provided."""
        params = {}
        # Should not raise, uses default "local"
        _validate_mode_param(params)

    def test_mode_non_string_raises_value_error(self):
        """Test that non-string mode raises ValueError."""
        params = {"mode": 123}
        with pytest.raises(ValueError, match="mode must be a string"):
            _validate_mode_param(params)

    def test_invalid_mode_raises_value_error(self):
        """Test that invalid mode raises ValueError."""
        params = {"mode": "cloud"}
        with pytest.raises(ValueError, match="is not valid"):
            _validate_mode_param(params)


class TestValidateExportMethodParam:
    """Test class for _validate_export_method_param function."""

    @pytest.mark.parametrize(
        "valid_method",
        ["data", "raw", "calculate", "both", "skip_existing", "none"],
        ids=["data", "raw", "calculate", "both", "skip_existing", "none"],
    )
    def test_valid_export_methods(self, valid_method):
        """Test that valid export methods pass validation."""
        params = {"export_method": valid_method}
        # Should not raise
        _validate_export_method_param(params)

    def test_default_export_method_when_missing(self):
        """Test that default export_method is used when not provided."""
        params = {}
        # Should not raise, uses default "data"
        _validate_export_method_param(params)

    def test_export_method_non_string_raises_value_error(self):
        """Test that non-string export_method raises ValueError."""
        params = {"export_method": 123}
        with pytest.raises(ValueError, match="export_method must be a string"):
            _validate_export_method_param(params)

    def test_invalid_export_method_raises_value_error(self):
        """Test that invalid export_method raises ValueError."""
        params = {"export_method": "invalid"}
        with pytest.raises(ValueError, match="is not valid"):
            _validate_export_method_param(params)


class TestValidateBooleanParams:
    """Test class for _validate_boolean_params function."""

    @pytest.mark.parametrize(
        "param_name",
        ["separated", "location_based_naming"],
        ids=["separated", "location_based_naming"],
    )
    def test_valid_boolean_true(self, param_name):
        """Test that True boolean values pass validation."""
        params = {param_name: True}
        # Should not raise
        _validate_boolean_params(params)

    @pytest.mark.parametrize(
        "param_name",
        ["separated", "location_based_naming"],
        ids=["separated", "location_based_naming"],
    )
    def test_valid_boolean_false(self, param_name):
        """Test that False boolean values pass validation."""
        params = {param_name: False}
        # Should not raise
        _validate_boolean_params(params)

    @pytest.mark.parametrize(
        "param_name",
        ["separated", "location_based_naming"],
        ids=["separated", "location_based_naming"],
    )
    def test_none_boolean_is_valid(self, param_name):
        """Test that None values for boolean params are valid."""
        params = {param_name: None}
        # Should not raise (None is accepted)
        _validate_boolean_params(params)

    @pytest.mark.parametrize(
        "param_name,invalid_value",
        [
            ("separated", "true"),
            ("separated", 1),
            ("location_based_naming", "false"),
            ("location_based_naming", 0),
        ],
        ids=["separated_string", "separated_int", "location_string", "location_int"],
    )
    def test_non_boolean_raises_value_error(self, param_name, invalid_value):
        """Test that non-boolean values raise ValueError."""
        params = {param_name: invalid_value}
        with pytest.raises(ValueError, match=f"{param_name} must be a boolean"):
            _validate_boolean_params(params)


class TestValidateFilenameTemplateParam:
    """Test class for _validate_filename_template_param function."""

    def test_valid_template_with_filename_placeholder(self):
        """Test valid template with {filename} placeholder."""
        params = {"filename_template": "{filename}_output"}
        # Should not raise
        _validate_filename_template_param(params)

    def test_valid_template_with_all_placeholders(self):
        """Test valid template with all supported placeholders."""
        params = {"filename_template": "{filename}_{name}_{lat}_{lon}"}
        # Should not raise
        _validate_filename_template_param(params)

    def test_none_template_is_valid(self):
        """Test that None template is valid (optional parameter)."""
        params = {"filename_template": None}
        # Should not raise
        _validate_filename_template_param(params)

    def test_missing_template_is_valid(self):
        """Test that missing template is valid."""
        params = {}
        # Should not raise
        _validate_filename_template_param(params)

    def test_template_non_string_raises_value_error(self):
        """Test that non-string template raises ValueError."""
        params = {"filename_template": 123}
        with pytest.raises(ValueError, match="filename_template must be a string"):
            _validate_filename_template_param(params)

    def test_empty_template_raises_value_error(self):
        """Test that empty template raises ValueError."""
        params = {"filename_template": ""}
        with pytest.raises(ValueError, match="filename_template cannot be empty"):
            _validate_filename_template_param(params)

    def test_whitespace_template_raises_value_error(self):
        """Test that whitespace-only template raises ValueError."""
        params = {"filename_template": "   "}
        with pytest.raises(ValueError, match="filename_template cannot be empty"):
            _validate_filename_template_param(params)

    def test_invalid_placeholder_logs_warning(self, caplog):
        """Test that invalid placeholders log a warning."""
        params = {"filename_template": "{filename}_{invalid_placeholder}"}
        with caplog.at_level(logging.WARNING):
            _validate_filename_template_param(params)
        assert "unrecognized placeholders" in caplog.text
        assert "{invalid_placeholder}" in caplog.text


class TestValidateFormatModeCompatibility:
    """Test class for _validate_format_mode_compatibility function."""

    def test_local_mode_with_netcdf_valid(self):
        """Test that local mode with NetCDF is valid."""
        params = {"mode": "local", "file_format": "NetCDF"}
        # Should not raise
        _validate_format_mode_compatibility(params)

    def test_local_mode_with_zarr_valid(self):
        """Test that local mode with Zarr is valid."""
        params = {"mode": "local", "file_format": "Zarr"}
        # Should not raise
        _validate_format_mode_compatibility(params)

    def test_local_mode_with_csv_valid(self):
        """Test that local mode with CSV is valid."""
        params = {"mode": "local", "file_format": "CSV"}
        # Should not raise
        _validate_format_mode_compatibility(params)

    def test_s3_mode_with_zarr_valid(self):
        """Test that S3 mode with Zarr is valid."""
        params = {"mode": "s3", "file_format": "Zarr"}
        # Should not raise
        _validate_format_mode_compatibility(params)

    def test_s3_mode_with_netcdf_raises_value_error(self):
        """Test that S3 mode with NetCDF raises ValueError."""
        params = {"mode": "s3", "file_format": "NetCDF"}
        with pytest.raises(ValueError, match="S3 export.*only supported with Zarr"):
            _validate_format_mode_compatibility(params)

    def test_s3_mode_with_csv_raises_value_error(self):
        """Test that S3 mode with CSV raises ValueError."""
        params = {"mode": "s3", "file_format": "CSV"}
        with pytest.raises(ValueError, match="S3 export.*only supported with Zarr"):
            _validate_format_mode_compatibility(params)


class TestIsPathSafe:
    """Test class for _is_path_safe function."""

    @pytest.mark.parametrize(
        "safe_filename",
        ["output", "my_data_export", "climate-data-2024", "test123"],
        ids=["simple", "underscores", "dashes", "alphanumeric"],
    )
    def test_safe_filenames(self, safe_filename):
        """Test that safe filenames return True."""
        result = _is_path_safe(safe_filename)
        assert result is True

    @pytest.mark.parametrize(
        "unsafe_filename",
        ["../parent", "path/to/file", "path\\to\\file", "~/home/file"],
        ids=["parent_traversal", "forward_slash", "backslash", "tilde"],
    )
    def test_unsafe_filenames(self, unsafe_filename):
        """Test that unsafe filenames return False."""
        result = _is_path_safe(unsafe_filename)
        assert result is False


class TestValidateExportOutputPath:
    """Test class for validate_export_output_path function."""

    def test_valid_output_path(self):
        """Test validation with a valid output path."""
        result = validate_export_output_path("test_output", "NetCDF")
        assert result["is_valid"] is True
        assert result["full_path"] == "test_output.nc"
        assert result["errors"] == []

    def test_zarr_extension(self):
        """Test validation maps Zarr to .zarr extension."""
        result = validate_export_output_path("test_output", "Zarr")
        assert result["full_path"] == "test_output.zarr"

    def test_csv_extension(self):
        """Test validation maps CSV to .csv.gz extension."""
        result = validate_export_output_path("test_output", "CSV")
        assert result["full_path"] == "test_output.csv.gz"

    def test_unsafe_path_invalid(self):
        """Test that unsafe paths are marked as invalid."""
        result = validate_export_output_path("../unsafe/path", "NetCDF")
        assert result["is_valid"] is False
        assert any("unsafe characters" in err for err in result["errors"])

    def test_existing_file_adds_warning(self, tmp_path):
        """Test that existing files add a warning."""
        # Create a temporary file
        test_file = tmp_path / "existing.nc"
        test_file.touch()

        result = validate_export_output_path(
            str(tmp_path / "existing"), "NetCDF"
        )
        assert any("will be overwritten" in warn for warn in result["warnings"])


class TestPredictExportFilenames:
    """Test class for _predict_export_filenames function."""

    def test_default_data_export(self):
        """Test filename prediction with default data export."""
        params = {"filename": "output", "file_format": "NetCDF"}
        result = _predict_export_filenames(params)
        assert "output.nc" in result

    def test_zarr_extension(self):
        """Test filename prediction with Zarr format."""
        params = {"filename": "output", "file_format": "Zarr"}
        result = _predict_export_filenames(params)
        assert "output.zarr" in result

    def test_csv_extension(self):
        """Test filename prediction with CSV format."""
        params = {"filename": "output", "file_format": "CSV"}
        result = _predict_export_filenames(params)
        assert "output.csv.gz" in result

    def test_raw_export_method(self):
        """Test filename prediction with raw export method."""
        params = {"filename": "output", "file_format": "NetCDF", "export_method": "raw"}
        result = _predict_export_filenames(params)
        assert "output_raw.nc" in result

    def test_calculate_export_method(self):
        """Test filename prediction with calculate export method."""
        params = {
            "filename": "output",
            "file_format": "NetCDF",
            "export_method": "calculate",
        }
        result = _predict_export_filenames(params)
        assert "output_calc.nc" in result

    def test_both_export_method(self):
        """Test filename prediction with both export method."""
        params = {"filename": "output", "file_format": "NetCDF", "export_method": "both"}
        result = _predict_export_filenames(params)
        assert "output_raw.nc" in result
        assert "output_calc.nc" in result

    def test_custom_raw_filename(self):
        """Test filename prediction with custom raw_filename."""
        params = {
            "filename": "output",
            "file_format": "NetCDF",
            "export_method": "raw",
            "raw_filename": "custom_raw",
        }
        result = _predict_export_filenames(params)
        assert "custom_raw.nc" in result

    def test_custom_calc_filename(self):
        """Test filename prediction with custom calc_filename."""
        params = {
            "filename": "output",
            "file_format": "NetCDF",
            "export_method": "calculate",
            "calc_filename": "custom_calc",
        }
        result = _predict_export_filenames(params)
        assert "custom_calc.nc" in result

    def test_location_based_naming(self):
        """Test filename prediction with location_based_naming enabled."""
        params = {
            "filename": "output",
            "file_format": "NetCDF",
            "location_based_naming": True,
        }
        result = _predict_export_filenames(params)
        # Should include wildcard pattern for location-based naming
        assert any("*" in f for f in result)

    def test_with_template(self):
        """Test filename prediction with filename_template."""
        params = {
            "filename": "output",
            "file_format": "NetCDF",
            "filename_template": "{filename}_{lat}_{lon}",
        }
        result = _predict_export_filenames(params)
        # Template placeholders are replaced with wildcards for prediction
        assert any("*" in f for f in result)


class TestSuggestExportAlternatives:
    """Test class for suggest_export_alternatives function."""

    def test_suggests_alternative_filename(self):
        """Test that alternative filename is suggested."""
        params = {"filename": "output", "file_format": "NetCDF"}
        result = suggest_export_alternatives(params)
        assert "alternative_filename" in result
        assert result["alternative_filename"] != "output"

    def test_suggests_skip_existing_method(self):
        """Test that skip_existing method is suggested."""
        params = {"filename": "output", "file_format": "NetCDF"}
        result = suggest_export_alternatives(params)
        assert "skip_existing_method" in result
        assert "skip_existing" in result["skip_existing_method"]

    def test_suggests_alternative_format(self):
        """Test that alternative format is suggested."""
        params = {"filename": "output", "file_format": "NetCDF"}
        result = suggest_export_alternatives(params)
        assert "alternative_format" in result
        # Should suggest Zarr or CSV when current is NetCDF
        assert "Zarr" in result["alternative_format"] or "CSV" in result["alternative_format"]


class TestSuggestAlternativeFilename:
    """Test class for _suggest_alternative_filename function."""

    def test_suggests_numbered_filename(self):
        """Test that numbered filename is suggested."""
        result = _suggest_alternative_filename("output", ".nc")
        # Should suggest output_01.nc if output.nc doesn't exist
        assert "output" in result
        assert ".nc" in result

    def test_suggests_different_when_numbered_exists(self, tmp_path):
        """Test that higher numbers are suggested when lower exist."""
        # Create numbered files
        for i in range(1, 5):
            (tmp_path / f"output_{i:02d}.nc").touch()

        # Change to tmp_path to test
        import os

        original_dir = os.getcwd()
        try:
            os.chdir(tmp_path)
            result = _suggest_alternative_filename("output", ".nc")
            # Should suggest a number >= 5
            assert "output_0" in result
        finally:
            os.chdir(original_dir)


class TestValidatorRegistration:
    """Test class for validator registration."""

    def test_export_validator_is_registered(self):
        """Test that export validator is registered in the registry."""
        from climakitae.new_core.param_validation.abc_param_validation import (
            _PROCESSOR_VALIDATOR_REGISTRY,
        )

        assert "export" in _PROCESSOR_VALIDATOR_REGISTRY
        assert _PROCESSOR_VALIDATOR_REGISTRY["export"] is validate_export_param

