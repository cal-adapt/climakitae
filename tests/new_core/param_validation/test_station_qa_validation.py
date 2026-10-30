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
    _historical_slice_center_year,
    _load_qa_table,
    _nearest_clean_years,
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


class TestHistoricalSliceCenterYear:
    """Test class for _historical_slice_center_year."""

    def test_even_span(self):
        assert _historical_slice_center_year((1980, 2010)) == 1995

    def test_default_historical_slice(self):
        assert _historical_slice_center_year((1980, 2014)) == 1997

    def test_single_year_span(self):
        assert _historical_slice_center_year((2000, 2000)) == 2000


class TestValidateStationQaForRequestedYears:
    """Test class for validate_station_qa_for_requested_years."""

    def _query(self, variable_id, historical_slice=None, stations=("KTES",)):
        bias_params = {"stations": list(stations)}
        if historical_slice is not None:
            bias_params["historical_slice"] = historical_slice
        return {
            "variable_id": variable_id,
            "processes": {"bias_adjust_model_to_station": bias_params},
        }

    @patch(f"{MODULE}.read_csv_file")
    @patch(f"{MODULE}.DataCatalog")
    def test_clean_combo_passes(self, mock_catalog, mock_read_csv, legacy_stations_df):
        mock_catalog.return_value.__getitem__.return_value = legacy_stations_df
        mock_read_csv.return_value = _synthetic_qa_df()
        # center year = (1999 + 2001) // 2 = 2000, which is clean
        query = self._query("t2", historical_slice=(1999, 2001))
        assert validate_station_qa_for_requested_years(query) is True

    @patch(f"{MODULE}.read_csv_file")
    @patch(f"{MODULE}.DataCatalog")
    def test_error_tier_blocks_with_message(
        self, mock_catalog, mock_read_csv, legacy_stations_df
    ):
        mock_catalog.return_value.__getitem__.return_value = legacy_stations_df
        mock_read_csv.return_value = _synthetic_qa_df()
        # center year = (2001 + 2003) // 2 = 2002, which fails the error tier
        query = self._query("t2", historical_slice=(2001, 2003))
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
        # center year = (2000 + 2002) // 2 = 2001
        query = self._query(
            "dew_point", historical_slice=(2000, 2002), stations=("KBAD",)
        )
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
        # center year = (2000 + 2002) // 2 = 2001, warning-tier only
        query = self._query("t2", historical_slice=(2000, 2002))
        with pytest.warns(UserWarning, match="data may be biased"):
            result = validate_station_qa_for_requested_years(query)
        assert result is True

    @patch(f"{MODULE}.read_csv_file")
    @patch(f"{MODULE}.DataCatalog")
    def test_missing_historical_slice_uses_default(
        self, mock_catalog, mock_read_csv, legacy_stations_df
    ):
        """No historical_slice given -> falls back to (1980, 2014) ->
        center year 1997, which this sheet doesn't cover (passes as
        'unknown')."""
        mock_catalog.return_value.__getitem__.return_value = legacy_stations_df
        mock_read_csv.return_value = _synthetic_qa_df()
        query = self._query("t2")
        assert validate_station_qa_for_requested_years(query) is True

    @patch(f"{MODULE}.read_csv_file")
    @patch(f"{MODULE}.DataCatalog")
    def test_unmapped_station_skipped_others_checked(
        self, mock_catalog, mock_read_csv, legacy_stations_df
    ):
        mock_catalog.return_value.__getitem__.return_value = legacy_stations_df
        mock_read_csv.return_value = _synthetic_qa_df()
        query = self._query(
            "t2",
            historical_slice=(2001, 2003),
            stations=("OtherISD_12345678901", "KTES"),
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
