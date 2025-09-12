"""
Unit tests for climakitae/new_core/param_validation/param_validation_tools.py

This module contains comprehensive unit tests for the parameter validation
tools including closest option matching, experiment ID validation, and date coercion.
"""

import warnings

from climakitae.new_core.param_validation.param_validation_tools import (
    _get_closest_options,
    _validate_experimental_id_param,
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

    def test_get_closest_options_difflib_match(self):
        """Test _get_closest_options with difflib fuzzy matching.
        
        Tests that the function uses difflib to find close matches
        when capitalization and substring matching don't work.
        """
        valid_options = ["historical", "ssp245", "ssp370", "ssp585"]
        
        # Test typo that should match with difflib
        result = _get_closest_options("historcal", valid_options)
        assert result == ["historical"]
        
        # Test another typo
        result = _get_closest_options("ssp24", valid_options)
        assert result == ["ssp245"]
        
        # Test with custom cutoff - should still find match
        result = _get_closest_options("historicl", valid_options, cutoff=0.6)
        assert result == ["historical"]

    def test_get_closest_options_no_match(self):
        """Test _get_closest_options when no close matches are found.
        
        Tests that the function returns None when no matches are found
        using any of the matching strategies.
        """
        valid_options = ["historical", "ssp245", "ssp370", "ssp585"]
        
        # Test completely unrelated input
        result = _get_closest_options("banana", valid_options)
        assert result is None
        
        # Test with high cutoff that prevents matches
        result = _get_closest_options("hist", valid_options, cutoff=0.9)
        assert result == ["historical"]  # Should still match as substring
        
        # Test truly no match case with high cutoff and no substring
        result = _get_closest_options("xyz123", valid_options, cutoff=0.9)
        assert result is None


class TestValidateExperimentalIdParam:
    """Test class for _validate_experimental_id_param function."""

    def test_validate_experimental_id_param_none_input(self):
        """Test _validate_experimental_id_param with None input.
        
        Tests that the function returns False when None is provided
        as the experiment ID parameter.
        """
        valid_experiment_ids = ["historical", "ssp245", "ssp370", "ssp585"]
        
        result = _validate_experimental_id_param(None, valid_experiment_ids)
        assert result is False

    def test_validate_experimental_id_param_single_valid(self):
        """Test _validate_experimental_id_param with single valid string.
        
        Tests that the function returns True when a valid single
        experiment ID string is provided.
        """
        valid_experiment_ids = ["historical", "ssp245", "ssp370", "ssp585"]
        
        # Test with valid single string
        result = _validate_experimental_id_param("historical", valid_experiment_ids)
        assert result is True
        
        result = _validate_experimental_id_param("ssp245", valid_experiment_ids)
        assert result is True

    def test_validate_experimental_id_param_single_partial_match(self):
        """Test _validate_experimental_id_param with partial match expansion.
        
        Tests that the function expands partial matches to all matching
        experiment IDs (e.g., 'ssp' matches all SSP scenarios).
        """
        valid_experiment_ids = ["historical", "ssp245", "ssp370", "ssp585"]
        
        # Create a mutable list to test the in-place modification
        value = ["ssp"]
        result = _validate_experimental_id_param(value, valid_experiment_ids)
        
        # Should return True and modify the list in place
        assert result is True
        # The original partial match should be replaced with all matching IDs
        assert "ssp" not in value
        assert "ssp245" in value
        assert "ssp370" in value
        assert "ssp585" in value