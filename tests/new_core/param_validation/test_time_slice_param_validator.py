"""
Unit tests for climakitae/new_core/processors/time_slice_param_validator.py

This module contains comprehensive unit tests for the TimeSlice processor
parameter validation functionality.
"""

import logging
import warnings

import pandas as pd
import pytest

from climakitae.new_core.param_validation.time_slice_param_validator import (
    validate_time_slice_param,
)


class TestValidateTimeSliceParam:
    """Test class for validate_time_slice_param function."""

    @pytest.mark.parametrize(
        "value,expected",
        [
            (("2000-01-01", "2000-12-31"), True),
            (("1990-06-15", "1991-06-15"), True),
            ((1981, 1900), True),
            ((2050.0, 2052.0), True),
            ((pd.Timestamp("2010-05-01"), pd.Timestamp("2015-05-01")), True),
            (
                (pd.DatetimeIndex(["2000-01-01"]), pd.DatetimeIndex(["2005-01-01"])),
                True,
            ),
        ],
        ids=[
            "valid_dates_1",
            "valid_dates_2",
            "valid_int_years",
            "valid_float_years",
            "valid_pd_timestamps",
            "valid_pd_datetimeindex",
        ],
    )
    def test_validate_time_slice_param_valid_values(self, value, expected):
        """Test validate_time_slice_param with valid date-like tuples."""
        result = validate_time_slice_param(value)
        assert result == expected

    @pytest.mark.parametrize(
        "value",
        [
            "2000-01-01 to 2000-12-31",
            {"February 1, 2021", "March 1, 2021"},
            ("2000-01-01", "2000-12-31", "2001-01-01"),
        ],
        ids=[
            "string_instead_of_tuple",
            "set_instead_of_tuple",
            "tuple_with_three_elements",
        ],
    )
    def test_validate_time_slice_param_invalid_type_or_length(self, value):
        """Test validate_time_slice_param with invalid values."""
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            result = validate_time_slice_param(value)

        assert result is False
        assert len(w) == 1  # Ensure a warning was raised

    @pytest.mark.parametrize(
        "value",
        [
            ("invalid-date", "2000-12-31"),
            ("2000-01-01", "not-a-date"),
            ("2000-01-01", None),
        ],
        ids=[
            "invalid_date_string",
            "one_valid_one_invalid_date",
            "one_valid_one_none",
        ],
    )
    def test_validate_time_slice_param_invalid_values(self, value):
        """Test validate_time_slice_param with invalid values."""
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            result = validate_time_slice_param(value)

        assert result is False
        assert len(w) == 1  # Ensure a warning was raised

    def test_validate_time_slice_param_with_seasons(self):
        """Test validate_time_slice_param with seasons parameter."""
        value = {
            "dates": ("2000-01-01", "2000-12-31"),
            "seasons": ["DJF", "MAM"],
        }
        result = validate_time_slice_param(value)
        assert result is True

    def test_validate_time_slice_param_with_single_season(self):
        """Test validate_time_slice_param with a single season string."""
        value = {
            "dates": ("2000-01-01", "2000-12-31"),
            "seasons": "JJA",
        }
        result = validate_time_slice_param(value)
        assert result is True

    def test_validate_time_slice_param_with_invalid_seasons(self, caplog):
        """Test validate_time_slice_param with invalid seasons parameter."""
        value = {
            "dates": ("2000-01-01", "2000-12-31"),
            "seasons": ["INVALID_SEASON", "AMJ"],
        }
        with caplog.at_level(logging.WARNING):
            result = validate_time_slice_param(value)
            assert "'seasons' parameter must be a list of season names" in caplog.text

        assert result is False
