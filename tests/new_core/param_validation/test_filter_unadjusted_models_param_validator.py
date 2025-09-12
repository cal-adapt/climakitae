"""
Unit tests for climakitae/new_core/param_validation/filter_unadjusted_models_param_validator.py

This module contains comprehensive unit tests for the FilterUnadjustedModels processor
parameter validation functionality.
"""

import warnings

import pytest

from climakitae.new_core.param_validation.filter_unadjusted_models_param_validator import (
    validate_filter_unadjusted_models_param,
)

# Suppress known external warnings that are not relevant to our tests
warnings.filterwarnings(
    "ignore",
    message="The 'shapely.geos' module is deprecated",
    category=DeprecationWarning,
)
warnings.filterwarnings(
    "ignore", message="pkg_resources is deprecated", category=DeprecationWarning
)


class TestValidateFilterUnadjustedModelsParam:
    """Test class for validate_filter_unadjusted_models_param function."""

    @pytest.mark.parametrize(
        "value,expected",
        [
            ("yes", True),
            ("no", True),
        ],
        ids=["yes", "no"],
    )
    def test_validate_filter_unadjusted_models_param_valid_values(
        self, value, expected
    ):
        """Test validate_filter_unadjusted_models_param with valid values.
        
        Tests validation with supported string values "yes" and "no".
        Both should return True.
        """
        result = validate_filter_unadjusted_models_param(value)
        assert result == expected