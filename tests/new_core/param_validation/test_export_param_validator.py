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

