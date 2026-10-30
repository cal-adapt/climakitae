"""
Unit tests for climakitae/new_core/param_validation/station_qa_validation.py

This module contains unit tests for the station QA sheet validation used to
cross-check `bias_adjust_model_to_station` requests against a per-station/
variable/year data-quality sheet.
"""

from unittest.mock import patch

import pandas as pd
import pytest

from climakitae.new_core.param_validation.station_qa_validation import (
    _describe_test,
    _load_qa_table,
    _nearest_clean_years,
    _requested_years,
    _year_qa_status,
    resolve_station_to_source_id,
    validate_station_qa_for_requested_years,
)

MODULE = "climakitae.new_core.param_validation.station_qa_validation"


def _synthetic_qa_df() -> pd.DataFrame:
    """A small synthetic QA sheet, independent of the real shipped sheet.

    - 2000: everything clean.
    - 2001: a warning-tier test fails (month_01_nans).
    - 2002: an error-tier test fails (total-time-window).
    """
    return pd.DataFrame(
        [
            {"source-id": "KTES", "variable": "tas", "test": "total-time-window", 2000: 0, 2001: 0, 2002: 1},
            {"source-id": "KTES", "variable": "tas", "test": "window_missing_nans", 2000: 0, 2001: 0, 2002: 0},
            {"source-id": "KTES", "variable": "tas", "test": "month_01_nans", 2000: 0, 2001: 1, 2002: 0},
            {"source-id": "KTES", "variable": "tas", "test": "diurnal_day_nans", 2000: 0, 2001: 0, 2002: 0},
        ]
    )


def _all_error_qa_df() -> pd.DataFrame:
    """A sheet where every year fails the error tier -- no clean year at all."""
    return pd.DataFrame(
        [
            {"source-id": "KBAD", "variable": "tdps", "test": "window_missing_nans", 2000: 1, 2001: 1, 2002: 1},
            {"source-id": "KBAD", "variable": "tdps", "test": "total-time-window", 2000: 0, 2001: 0, 2002: 0},
        ]
    )


@pytest.fixture(autouse=True)
def _clear_qa_table_cache():
    """Ensure the lru_cache doesn't leak a stale table between tests."""
    _load_qa_table.cache_clear()
    yield
    _load_qa_table.cache_clear()


@pytest.fixture
def legacy_stations_df():
    """Mock legacy airport-code lookup table."""
    return pd.DataFrame(
        {
            "ID": ["KTES", "KBAD"],
            "station": ["Test Station (KTES)", "Bad Station (KBAD)"],
            "station id": [72483023225, 72494023234],
        }
    )


class TestYearQaStatus:
    """Test class for _year_qa_status."""

    @patch(f"{MODULE}.read_csv_file")
    def test_clean_year(self, mock_read_csv):
        mock_read_csv.return_value = _synthetic_qa_df()
        assert _year_qa_status("KTES", "tas", 2000) == (False, [])

    @patch(f"{MODULE}.read_csv_file")
    def test_warning_tier_only(self, mock_read_csv):
        mock_read_csv.return_value = _synthetic_qa_df()
        assert _year_qa_status("KTES", "tas", 2001) == (False, ["month_01_nans"])

    @patch(f"{MODULE}.read_csv_file")
    def test_error_tier(self, mock_read_csv):
        mock_read_csv.return_value = _synthetic_qa_df()
        has_error, warn_tests = _year_qa_status("KTES", "tas", 2002)
        assert has_error is True
        assert warn_tests == []

    @patch(f"{MODULE}.read_csv_file")
    def test_unknown_station_variable_year_passes(self, mock_read_csv):
        mock_read_csv.return_value = _synthetic_qa_df()
        assert _year_qa_status("KUNKNOWN", "tas", 2000) == (False, [])
        assert _year_qa_status("KTES", "tas", 1999) == (False, [])
        assert _year_qa_status("KTES", "tdps", 2000) == (False, [])


class TestNearestCleanYears:
    """Test class for _nearest_clean_years."""

    @patch(f"{MODULE}.read_csv_file")
    def test_finds_nearest_before_and_after(self, mock_read_csv):
        df = pd.DataFrame(
            [
                {"source-id": "KTES", "variable": "tas", "test": "total-time-window", 1998: 0, 1999: 1, 2000: 1, 2001: 0},
                {"source-id": "KTES", "variable": "tas", "test": "window_missing_nans", 1998: 0, 1999: 0, 2000: 0, 2001: 0},
            ]
        )
        mock_read_csv.return_value = df
        assert _nearest_clean_years("KTES", "tas", 1999) == [1998, 2001]

    @patch(f"{MODULE}.read_csv_file")
    def test_no_clean_year_returns_empty(self, mock_read_csv):
        mock_read_csv.return_value = _all_error_qa_df()
        assert _nearest_clean_years("KBAD", "tdps", 2001) == []


class TestDescribeTest:
    """Test class for _describe_test."""

    def test_month_test(self):
        assert _describe_test("month_07_nans") == "July"

    def test_diurnal_test(self):
        assert _describe_test("diurnal_day_nans") == "daytime coverage"
        assert _describe_test("diurnal_night_nans") == "nighttime coverage"

    def test_unknown_test_passthrough(self):
        assert _describe_test("some_other_test") == "some_other_test"


class TestResolveStationToSourceId:
    """Test class for resolve_station_to_source_id."""

    def test_airport_code(self, legacy_stations_df):
        assert resolve_station_to_source_id("KTES", legacy_stations_df) == "KTES"

    def test_hdp_station_id(self, legacy_stations_df):
        assert (
            resolve_station_to_source_id("ASOSAWOS_72483023225", legacy_stations_df)
            == "KTES"
        )

    def test_unmapped_hdp_station_id(self, legacy_stations_df):
        assert resolve_station_to_source_id("OtherISD_99999999999", legacy_stations_df) is None

    def test_malformed_identifier(self, legacy_stations_df):
        assert resolve_station_to_source_id("not_a_valid_id", legacy_stations_df) is None


class TestRequestedYears:
    """Test class for _requested_years."""

    def test_time_slice_tuple(self):
        query = {"processes": {"time_slice": (2000, 2003)}}
        assert _requested_years(query) == {2000, 2001, 2002, 2003}

    @patch(f"{MODULE}.get_year_at_gwl")
    def test_warming_level_already_crossed_historically(self, mock_get_year):
        mock_get_year.return_value = pd.DataFrame(
            {"Mean": [2005]}, index=["Historical"]
        )
        query = {"processes": {"warming_level": {"warming_levels": [1.0]}}}
        assert _requested_years(query) == {2005}

    @patch(f"{MODULE}.get_year_at_gwl")
    def test_warming_level_future_crossing(self, mock_get_year):
        mock_get_year.return_value = pd.DataFrame(
            {"Mean": [pd.NA, 2075]}, index=["Historical", "SSP 3-7.0"]
        )
        query = {"processes": {"warming_level": {"warming_levels": [2.0]}}}
        assert _requested_years(query) == {2075}

    def test_neither_processor_present(self):
        assert _requested_years({"processes": {}}) == set()


class TestValidateStationQaForRequestedYears:
    """Test class for validate_station_qa_for_requested_years."""

    def _query(self, variable_id, time_slice, stations=("KTES",)):
        return {
            "variable_id": variable_id,
            "processes": {
                "time_slice": time_slice,
                "bias_adjust_model_to_station": {"stations": list(stations)},
            },
        }

    @patch(f"{MODULE}.read_csv_file")
    @patch(f"{MODULE}.DataCatalog")
    def test_clean_combo_passes(self, mock_catalog, mock_read_csv, legacy_stations_df):
        mock_catalog.return_value.__getitem__.return_value = legacy_stations_df
        mock_read_csv.return_value = _synthetic_qa_df()
        query = self._query("t2", (2000, 2000))
        assert validate_station_qa_for_requested_years(query) is True

    @patch(f"{MODULE}.read_csv_file")
    @patch(f"{MODULE}.DataCatalog")
    def test_error_tier_blocks_with_message(
        self, mock_catalog, mock_read_csv, legacy_stations_df
    ):
        mock_catalog.return_value.__getitem__.return_value = legacy_stations_df
        mock_read_csv.return_value = _synthetic_qa_df()
        query = self._query("t2", (2002, 2002))
        with pytest.warns(UserWarning, match="too much missing data overall"):
            result = validate_station_qa_for_requested_years(query)
        assert result is False

    @patch(f"{MODULE}.read_csv_file")
    @patch(f"{MODULE}.DataCatalog")
    def test_error_tier_with_no_nearby_year(
        self, mock_catalog, mock_read_csv, legacy_stations_df
    ):
        mock_catalog.return_value.__getitem__.return_value = legacy_stations_df
        mock_read_csv.return_value = _all_error_qa_df()
        query = self._query("dew_point", (2001, 2001), stations=("KBAD",))
        with pytest.warns(UserWarning, match="No usable year was found nearby"):
            result = validate_station_qa_for_requested_years(query)
        assert result is False

    @patch(f"{MODULE}.read_csv_file")
    @patch(f"{MODULE}.DataCatalog")
    def test_warning_tier_passes_with_message(
        self, mock_catalog, mock_read_csv, legacy_stations_df
    ):
        mock_catalog.return_value.__getitem__.return_value = legacy_stations_df
        mock_read_csv.return_value = _synthetic_qa_df()
        query = self._query("t2", (2001, 2001))
        with pytest.warns(UserWarning, match="data may be biased"):
            result = validate_station_qa_for_requested_years(query)
        assert result is True

    @patch(f"{MODULE}.read_csv_file")
    @patch(f"{MODULE}.DataCatalog")
    def test_mixed_range_blocks_overall(
        self, mock_catalog, mock_read_csv, legacy_stations_df
    ):
        mock_catalog.return_value.__getitem__.return_value = legacy_stations_df
        mock_read_csv.return_value = _synthetic_qa_df()
        query = self._query("t2", (2000, 2002))
        result = validate_station_qa_for_requested_years(query)
        assert result is False

    @patch(f"{MODULE}.read_csv_file")
    @patch(f"{MODULE}.DataCatalog")
    def test_unmapped_station_skipped_others_checked(
        self, mock_catalog, mock_read_csv, legacy_stations_df
    ):
        mock_catalog.return_value.__getitem__.return_value = legacy_stations_df
        mock_read_csv.return_value = _synthetic_qa_df()
        query = self._query(
            "t2", (2002, 2002), stations=("OtherISD_12345678901", "KTES")
        )
        result = validate_station_qa_for_requested_years(query)
        assert result is False

    @patch(f"{MODULE}.read_csv_file")
    @patch(f"{MODULE}.DataCatalog")
    def test_no_bias_adjust_processor_is_noop(self, mock_catalog, mock_read_csv):
        result = validate_station_qa_for_requested_years(
            {"processes": {"time_slice": (2002, 2002)}}
        )
        assert result is True
        mock_read_csv.assert_not_called()

    @patch(f"{MODULE}.read_csv_file")
    @patch(f"{MODULE}.DataCatalog")
    def test_no_time_slice_or_warming_level_is_noop(
        self, mock_catalog, mock_read_csv, legacy_stations_df
    ):
        mock_catalog.return_value.__getitem__.return_value = legacy_stations_df
        mock_read_csv.return_value = _synthetic_qa_df()
        query = {
            "variable_id": "t2",
            "processes": {"bias_adjust_model_to_station": {"stations": ["KTES"]}},
        }
        assert validate_station_qa_for_requested_years(query) is True
