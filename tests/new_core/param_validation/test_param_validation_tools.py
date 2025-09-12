"""
Unit tests for climakitae/new_core/param_validation/param_validation_tools.py

This module contains comprehensive unit tests for the parameter validation
tools including closest option matching, experiment ID validation, and date coercion.
"""

import warnings

from climakitae.new_core.param_validation.param_validation_tools import (
    _get_closest_options,
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


class TestGetClosestOptions:
    """Test class for _get_closest_options function."""

    def test_get_closest_options_capitalization_match(self):
        """Test _get_closest_options with case-insensitive matches.
        
        Tests that the function correctly identifies options that differ
        only in capitalization.
        """
        valid_options = ["historical", "ssp245", "ssp370", "ssp585"]
        
        # Test exact match with different case
        result = _get_closest_options("HISTORICAL", valid_options)
        assert result == ["historical"]
        
        # Test mixed case
        result = _get_closest_options("Ssp245", valid_options)
        assert result == ["ssp245"]

    def test_get_closest_options_substring_match(self):
        """Test _get_closest_options with substring matches.
        
        Tests that the function correctly identifies options where the
        input is a substring of valid options.
        """
        valid_options = ["historical", "ssp245", "ssp370", "ssp585"]
        
        # Test substring that matches one option
        result = _get_closest_options("hist", valid_options)
        assert result == ["historical"]
        
        # Test substring that matches multiple options
        result = _get_closest_options("ssp", valid_options)
        assert result == ["ssp245", "ssp370", "ssp585"]
        
        # Test case-insensitive substring
        result = _get_closest_options("SSP", valid_options)
        assert result == ["ssp245", "ssp370", "ssp585"]