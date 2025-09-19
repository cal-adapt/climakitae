"""
Unit tests for climakitae/new_core/param_validation/param_validation_tools.py

This module contains comprehensive unit tests for the parameter validation
tools including closest option matching, experiment ID validation, and date coercion.
"""

import datetime
import warnings

import pandas as pd
import pytest

from climakitae.new_core.param_validation.param_validation_tools import (
    _coerce_to_dates,
    _get_closest_options,
    _validate_experimental_id_param,
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

    def test_validate_experimental_id_param_single_invalid(self):
        """Test _validate_experimental_id_param with invalid single string.

        Tests that the function returns False and issues warnings
        for invalid experiment IDs.
        """
        valid_experiment_ids = ["historical", "ssp245", "ssp370", "ssp585"]

        # Test invalid ID with close match
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            result = _validate_experimental_id_param("historicl", valid_experiment_ids)

            assert result is False
            assert len(w) == 1
            assert "Experiment ID 'historicl' not found" in str(w[0].message)
            assert "Did you mean any of the following 'historical'" in str(w[0].message)

        # Test invalid ID with no close match
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            result = _validate_experimental_id_param("invalid123", valid_experiment_ids)

            assert result is False
            assert len(w) == 1
            assert "Experiment ID 'invalid123' not found" in str(w[0].message)
            assert "Please check the available experiment IDs" in str(w[0].message)

    def test_validate_experimental_id_param_multiple_all_valid(self):
        """Test _validate_experimental_id_param with multiple valid experiment IDs.

        Tests that the function returns True when all provided experiment IDs
        are valid in a list format.
        """
        valid_experiment_ids = ["historical", "ssp245", "ssp370", "ssp585"]

        # Test with multiple valid IDs
        result = _validate_experimental_id_param(
            ["historical", "ssp245"], valid_experiment_ids
        )
        assert result is True

        # Test with all valid IDs
        result = _validate_experimental_id_param(
            ["historical", "ssp245", "ssp370", "ssp585"], valid_experiment_ids
        )
        assert result is True

    def test_validate_experimental_id_param_multiple_some_invalid(self):
        """Test _validate_experimental_id_param with mixed valid/invalid IDs.

        Tests that the function returns False and issues warnings for each
        invalid experiment ID when some are valid and some are invalid.
        """
        valid_experiment_ids = ["historical", "ssp245", "ssp370", "ssp585"]

        # Test with one valid, one invalid with close match
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            result = _validate_experimental_id_param(
                ["historical", "ssp24"], valid_experiment_ids
            )

            assert result is False
            assert len(w) == 1
            assert "Experiment ID 'ssp24' not found" in str(w[0].message)
            assert "Did you mean 'ssp245'" in str(w[0].message)

    def test_validate_experimental_id_param_multiple_all_invalid(self):
        """Test _validate_experimental_id_param with multiple invalid IDs.

        Tests that the function returns False and issues warnings only for
        invalid experiment IDs that have close matches.
        """
        valid_experiment_ids = ["historical", "ssp245", "ssp370", "ssp585"]

        # Test with multiple invalid IDs, some with close matches
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            result = _validate_experimental_id_param(
                ["histrical", "ssp999", "invalid"], valid_experiment_ids
            )

            assert result is False
            # Only warnings for IDs with close matches are issued
            # 'histrical' should have a close match to 'historical'
            # 'ssp999' and 'invalid' may not have close matches
            assert len(w) >= 1  # At least one warning for close match
            # Check that warning is issued for the ID with close match
            warning_messages = [str(warning.message) for warning in w]
            assert any("histrical" in msg for msg in warning_messages)

    def test_validate_experimental_id_param_empty_list(self):
        """Test _validate_experimental_id_param with empty list.

        Tests that the function returns False when an empty list
        is provided as the experiment ID parameter.
        """
        valid_experiment_ids = ["historical", "ssp245", "ssp370", "ssp585"]

        result = _validate_experimental_id_param([], valid_experiment_ids)
        assert result is False


class TestCoerceToDates:
    """Test class for _coerce_to_dates function.

    This class tests the date coercion functionality including valid inputs,
    invalid types, length validation, and edge cases with comprehensive
    parameterized testing approaches.
    """

    @pytest.mark.parametrize(
        "input_data,expected",
        [
            # String inputs
            (
                ["2020-01-01", "2021-12-31"],
                (pd.Timestamp("2020-01-01"), pd.Timestamp("2021-12-31")),
            ),
            (
                ["2020/01/01", "2021/12/31"],
                (pd.Timestamp("2020-01-01"), pd.Timestamp("2021-12-31")),
            ),
            (
                ["Jan 1, 2020", "Dec 31, 2021"],
                (pd.Timestamp("2020-01-01"), pd.Timestamp("2021-12-31")),
            ),
            (
                ["2020-01-01 12:30:45", "2021-12-31 23:59:59"],
                (
                    pd.Timestamp("2020-01-01 12:30:45"),
                    pd.Timestamp("2021-12-31 23:59:59"),
                ),
            ),
            # Integer inputs (year integers)
            (
                [2020, 2021],
                (pd.Timestamp("2020-01-01"), pd.Timestamp("2021-12-31")),
            ),
            # Float inputs (year floats)
            (
                [2020.0, 2021.0],
                (pd.Timestamp("2020-01-01"), pd.Timestamp("2021-12-31")),
            ),
            # datetime.date inputs
            (
                [datetime.date(2020, 1, 1), datetime.date(2021, 12, 31)],
                (pd.Timestamp("2020-01-01"), pd.Timestamp("2021-12-31")),
            ),
            # datetime.datetime inputs
            (
                [
                    datetime.datetime(2020, 1, 1, 12, 30),
                    datetime.datetime(2021, 12, 31, 23, 59),
                ],
                (
                    pd.Timestamp("2020-01-01 12:30:00"),
                    pd.Timestamp("2021-12-31 23:59:00"),
                ),
            ),
            # pd.Timestamp inputs
            (
                [pd.Timestamp("2020-01-01"), pd.Timestamp("2021-12-31")],
                (pd.Timestamp("2020-01-01"), pd.Timestamp("2021-12-31")),
            ),
            # pd.DatetimeIndex inputs
            (
                [
                    pd.DatetimeIndex(["2020-01-01"])[0],
                    pd.DatetimeIndex(["2021-12-31"])[0],
                ],
                (pd.Timestamp("2020-01-01"), pd.Timestamp("2021-12-31")),
            ),
            # Mixed valid types
            (
                [datetime.date(2020, 1, 1), "2021-12-31"],
                (pd.Timestamp("2020-01-01"), pd.Timestamp("2021-12-31")),
            ),
            (
                [pd.Timestamp("2020-01-01"), datetime.datetime(2021, 12, 31)],
                (pd.Timestamp("2020-01-01"), pd.Timestamp("2021-12-31")),
            ),
            (
                ["2020-01-01", 2021],
                (pd.Timestamp("2020-01-01"), pd.Timestamp("2021-12-31")),
            ),
        ],
        ids=[
            "iso_strings",
            "slash_strings",
            "natural_strings",
            "datetime_strings",
            "year_ints",
            "year_floats",
            "date_objects",
            "datetime_objects",
            "timestamps",
            "datetime_index",
            "mixed_date_string",
            "mixed_timestamp_datetime",
            "mixed_string_year",
        ],
    )
    def test_coerce_to_dates_valid_inputs(self, input_data, expected):
        """Test _coerce_to_dates with various valid input combinations.

        Parameters
        ----------
        input_data : list
            Input data to coerce to dates.
        expected : tuple
            Expected output tuple of pd.Timestamp objects.
        """
        result = _coerce_to_dates(input_data)

        assert isinstance(result, tuple)
        assert len(result) == 2
        assert isinstance(result[0], pd.Timestamp)
        assert isinstance(result[1], pd.Timestamp)
        assert result[0] == expected[0]
        assert result[1] == expected[1]

    @pytest.mark.parametrize(
        "invalid_input,expected_error_message",
        [
            # Invalid types
            ([{"not": "a_date"}, "2021-01-01"], "is not a date-like object"),
            (["2020-01-01", [1, 2, 3]], "is not a date-like object"),
            ([None, "2021-01-01"], "Cannot coerce value"),
            (["2020-01-01", object()], "is not a date-like object"),
            # Invalid date strings
            (["invalid-date", "2021-01-01"], "Cannot coerce value"),
            (["2020-13-01", "2021-01-01"], "Cannot coerce value"),  # Invalid month
            (["2020-01-32", "2021-01-01"], "Cannot coerce value"),  # Invalid day
            # Empty DatetimeIndex
            ([pd.DatetimeIndex([]), "2021-01-01"], "Empty DatetimeIndex"),
            (["2020-01-01", pd.DatetimeIndex([])], "Empty DatetimeIndex"),
        ],
        ids=[
            "dict_input",
            "list_input",
            "none_input",
            "object_input",
            "invalid_string",
            "invalid_month",
            "invalid_day",
            "empty_datetime_index_first",
            "empty_datetime_index_second",
        ],
    )
    def test_coerce_to_dates_invalid_inputs(
        self, invalid_input, expected_error_message
    ):
        """Test _coerce_to_dates with various invalid input types.

        Parameters
        ----------
        invalid_input : list
            Invalid input data that should raise ValueError.
        expected_error_message : str
            Expected error message substring.
        """
        with pytest.raises(ValueError, match=expected_error_message):
            _coerce_to_dates(invalid_input)

    @pytest.mark.parametrize(
        "input_length,input_data",
        [
            (0, []),
            (1, ["2020-01-01"]),
            (3, ["2020-01-01", "2021-01-01", "2022-01-01"]),
            (5, ["2020-01-01", "2021-01-01", "2022-01-01", "2023-01-01", "2024-01-01"]),
        ],
        ids=["empty", "single", "triple", "quintuple"],
    )
    def test_coerce_to_dates_invalid_length(self, input_length, input_data):
        """Test _coerce_to_dates with invalid input lengths.

        Parameters
        ----------
        input_length : int
            Length of input data (for test ID purposes).
        input_data : list
            Input data with invalid length.
        """
        with pytest.raises(
            ValueError, match=f"Expected exactly 2 date-like values, got {input_length}"
        ):
            _coerce_to_dates(input_data)

    @pytest.mark.parametrize(
        "iterable_type",
        [
            tuple,
            list,
            lambda x: iter(x),  # iterator
            lambda x: (i for i in x),  # generator
        ],
        ids=["tuple", "list", "iterator", "generator"],
    )
    def test_coerce_to_dates_iterable_types(self, iterable_type):
        """Test _coerce_to_dates with different iterable types.

        Parameters
        ----------
        iterable_type : callable
            Function to convert list to different iterable type.
        """
        input_data = ["2020-01-01", "2021-12-31"]
        iterable_input = iterable_type(input_data)

        result = _coerce_to_dates(iterable_input)

        assert isinstance(result, tuple)
        assert len(result) == 2
        assert result[0] == pd.Timestamp("2020-01-01")
        assert result[1] == pd.Timestamp("2021-12-31")

    def test_coerce_to_dates_edge_cases_datetime_index(self):
        """Test _coerce_to_dates with DatetimeIndex edge cases."""
        # Single element DatetimeIndex
        single_dt_index = pd.DatetimeIndex(["2020-01-01"])
        result = _coerce_to_dates([single_dt_index, "2021-12-31"])

        assert result[0] == pd.Timestamp("2020-01-01")
        assert result[1] == pd.Timestamp("2021-12-31")

        # Multi-element DatetimeIndex (should use first element)
        multi_dt_index = pd.DatetimeIndex(["2020-01-01", "2020-02-01", "2020-03-01"])
        result = _coerce_to_dates([multi_dt_index, "2021-12-31"])

        assert result[0] == pd.Timestamp("2020-01-01")
        assert result[1] == pd.Timestamp("2021-12-31")

    def test_coerce_to_dates_timezone_handling(self):
        """Test _coerce_to_dates with timezone-aware datetime objects."""
        # Timezone-aware datetime
        tz_datetime = datetime.datetime(
            2020, 1, 1, 12, 0, 0, tzinfo=datetime.timezone.utc
        )

        result = _coerce_to_dates([tz_datetime, "2021-12-31"])

        assert isinstance(result[0], pd.Timestamp)
        assert result[0].year == 2020
        assert result[0].month == 1
        assert result[0].day == 1

    def test_coerce_to_dates_boundary_dates(self):
        """Test _coerce_to_dates with boundary date values."""
        # Test very early and late dates
        early_date = "1900-01-01"
        late_date = "2100-12-31"

        result = _coerce_to_dates([early_date, late_date])

        assert result[0] == pd.Timestamp("1900-01-01")
        assert result[1] == pd.Timestamp("2100-12-31")

    def test_coerce_to_dates_unix_timestamps(self):
        """Test _coerce_to_dates with Unix timestamp values.

        Note: Unix timestamps need to be handled carefully as pandas
        may interpret large integers as nanoseconds by default.
        """
        # Test with Unix timestamps converted to datetime strings first
        # This shows the expected behavior for timestamp-like values
        unix_ts_1 = 1577836800  # 2020-01-01 00:00:00 UTC
        unix_ts_2 = 1640995199  # 2021-12-31 23:59:59 UTC

        # Convert to datetime objects first, then test
        dt1 = datetime.datetime.fromtimestamp(unix_ts_1)
        dt2 = datetime.datetime.fromtimestamp(unix_ts_2)

        result = _coerce_to_dates([dt1, dt2])

        assert isinstance(result, tuple)
        assert len(result) == 2
        assert result[0] == pd.Timestamp(dt1)
        assert result[1] == pd.Timestamp(dt2)

    def test_coerce_to_dates_set_order_independent(self):
        """Test _coerce_to_dates with set input (order-independent check).

        Since sets don't preserve order, we check that both expected
        timestamps are present but don't enforce a specific order.
        """
        input_data = ["2020-01-01", "2021-12-31"]
        set_input = set(input_data)

        result = _coerce_to_dates(set_input)

        assert isinstance(result, tuple)
        assert len(result) == 2
        assert isinstance(result[0], pd.Timestamp)
        assert isinstance(result[1], pd.Timestamp)

        # Check that both expected dates are present (order-independent)
        result_set = {result[0], result[1]}
        expected_set = {pd.Timestamp("2020-01-01"), pd.Timestamp("2021-12-31")}
        assert result_set == expected_set
