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
    _historical_slice_offsets,
    _load_qa_table,
    _nearest_passing_years,
    _year_qa_status,
    _year_to_historical_slice,
    resolve_station_to_source_id,
    validate_station_qa_for_requested_years,
)

MODULE = "climakitae.new_core.param_validation.station_qa_validation"


def _synthetic_qa_df() -> pd.DataFrame:
    """A small synthetic QA sheet, independent of the real shipped sheet.

    - 2000: everything clean.
    - 2001: a non-blocking row fails (month_01_nans), total-time-window clear.
    - 2002: total-time-window (the sole blocking check) fails, with no other
      row flagged at that year.
    """
    return pd.DataFrame(
        [
            {"source-id": "KTES", "variable": "tas", "test": "total-time-window", 2000: 0, 2001: 0, 2002: 1},
            {"source-id": "KTES", "variable": "tas", "test": "window_missing_nans", 2000: 0, 2001: 0, 2002: 0},
            {"source-id": "KTES", "variable": "tas", "test": "month_01_nans", 2000: 0, 2001: 1, 2002: 0},
            {"source-id": "KTES", "variable": "tas", "test": "diurnal_day_nans", 2000: 0, 2001: 0, 2002: 0},
        ]
    )


def _error_with_reasons_qa_df() -> pd.DataFrame:
    """total-time-window fails at 2002, alongside window_missing_nans and
    month_01_nans -- exercises the "reasons" list in the error message."""
    return pd.DataFrame(
        [
            {"source-id": "KTES", "variable": "tas", "test": "total-time-window", 2000: 0, 2001: 0, 2002: 1},
            {"source-id": "KTES", "variable": "tas", "test": "window_missing_nans", 2000: 0, 2001: 0, 2002: 1},
            {"source-id": "KTES", "variable": "tas", "test": "month_01_nans", 2000: 0, 2001: 0, 2002: 1},
            {"source-id": "KTES", "variable": "tas", "test": "diurnal_day_nans", 2000: 0, 2001: 0, 2002: 0},
        ]
    )


def _all_error_qa_df() -> pd.DataFrame:
    """A sheet where total-time-window fails every year -- no passing year
    at all for the sole blocking check."""
    return pd.DataFrame(
        [
            {"source-id": "KBAD", "variable": "tdps", "test": "total-time-window", 2000: 1, 2001: 1, 2002: 1},
            {"source-id": "KBAD", "variable": "tdps", "test": "window_missing_nans", 2000: 1, 2001: 1, 2002: 1},
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
    def test_other_row_flagged_total_time_window_clear(self, mock_read_csv):
        mock_read_csv.return_value = _synthetic_qa_df()
        assert _year_qa_status("KTES", "tas", 2001) == (False, ["month_01_nans"])

    @patch(f"{MODULE}.read_csv_file")
    def test_total_time_window_fails_alone(self, mock_read_csv):
        mock_read_csv.return_value = _synthetic_qa_df()
        has_error, other_tests = _year_qa_status("KTES", "tas", 2002)
        assert has_error is True
        assert other_tests == []

    @patch(f"{MODULE}.read_csv_file")
    def test_total_time_window_fails_with_reasons(self, mock_read_csv):
        mock_read_csv.return_value = _error_with_reasons_qa_df()
        has_error, other_tests = _year_qa_status("KTES", "tas", 2002)
        assert has_error is True
        assert other_tests == ["month_01_nans", "window_missing_nans"]

    @patch(f"{MODULE}.read_csv_file")
    def test_unknown_station_variable_year_passes(self, mock_read_csv):
        mock_read_csv.return_value = _synthetic_qa_df()
        assert _year_qa_status("KUNKNOWN", "tas", 2000) == (False, [])
        assert _year_qa_status("KTES", "tas", 1999) == (False, [])
        assert _year_qa_status("KTES", "tdps", 2000) == (False, [])


class TestNearestPassingYears:
    """Test class for _nearest_passing_years."""

    @patch(f"{MODULE}.read_csv_file")
    def test_finds_nearest_before_and_after(self, mock_read_csv):
        df = pd.DataFrame(
            [
                {"source-id": "KTES", "variable": "tas", "test": "total-time-window", 1998: 0, 1999: 1, 2000: 1, 2001: 0},
                {"source-id": "KTES", "variable": "tas", "test": "window_missing_nans", 1998: 0, 1999: 0, 2000: 0, 2001: 0},
            ]
        )
        mock_read_csv.return_value = df
        assert _nearest_passing_years("KTES", "tas", "total-time-window", 1999) == [1998, 2001]

    @patch(f"{MODULE}.read_csv_file")
    def test_different_test_not_conflated(self, mock_read_csv):
        """window_missing_nans passes every year here, even though
        total-time-window fails 1999-2000 -- the two rows are independent."""
        df = pd.DataFrame(
            [
                {"source-id": "KTES", "variable": "tas", "test": "total-time-window", 1998: 0, 1999: 1, 2000: 1, 2001: 0},
                {"source-id": "KTES", "variable": "tas", "test": "window_missing_nans", 1998: 0, 1999: 0, 2000: 0, 2001: 0},
            ]
        )
        mock_read_csv.return_value = df
        assert _nearest_passing_years("KTES", "tas", "window_missing_nans", 1999) == [1998, 2000]

    @patch(f"{MODULE}.read_csv_file")
    def test_no_passing_year_returns_empty(self, mock_read_csv):
        mock_read_csv.return_value = _all_error_qa_df()
        assert _nearest_passing_years("KBAD", "tdps", "window_missing_nans", 2001) == []


class TestDescribeTest:
    """Test class for _describe_test."""

    def test_month_test(self):
        assert _describe_test("month_07_nans") == "July"

    def test_diurnal_test(self):
        assert _describe_test("diurnal_day_nans") == "daytime coverage"
        assert _describe_test("diurnal_night_nans") == "nighttime coverage"

    def test_error_tier_test(self):
        assert _describe_test("total-time-window") == "overall time coverage"
        assert _describe_test("window_missing_nans") == "overall missing-data window"

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


class TestHistoricalSliceOffsets:
    """Test class for _historical_slice_offsets and _year_to_historical_slice."""

    def test_symmetric_offsets(self):
        assert _historical_slice_offsets((1985, 2015)) == (15, 15)

    def test_odd_span_offsets(self):
        # center = (1980 + 2014) // 2 = 1997 -> 17 years back, 17 forward
        assert _historical_slice_offsets((1980, 2014)) == (17, 17)

    def test_year_to_historical_slice_round_trips(self):
        offsets = _historical_slice_offsets((1985, 2015))
        assert _year_to_historical_slice(1999, offsets) == (1984, 2014)
        assert _year_to_historical_slice(2001, offsets) == (1986, 2016)
        assert _year_to_historical_slice(2000, offsets) == (1985, 2015)


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
    def test_error_blocks_with_no_other_reasons(
        self, mock_catalog, mock_read_csv, legacy_stations_df
    ):
        mock_catalog.return_value.__getitem__.return_value = legacy_stations_df
        mock_read_csv.return_value = _synthetic_qa_df()
        # historical_slice=(2001, 2003) centers on 2002: total-time-window
        # fails alone, nothing else flagged; nearest clear year for it is
        # 2001, re-expressed as a 2001-centered range of the same width.
        query = self._query("t2", historical_slice=(2001, 2003))
        with pytest.warns(
            UserWarning,
            match=r"Too much missing data for station 'KTES', variable 'tas', "
            r"from 2001-2003\. Related issues: no other checks are flagged "
            r"for this time range\. Closest usable time ranges: 2000-2002",
        ):
            result = validate_station_qa_for_requested_years(query)
        assert result is False

    @patch(f"{MODULE}.read_csv_file")
    @patch(f"{MODULE}.DataCatalog")
    def test_error_blocks_with_reasons_from_other_rows(
        self, mock_catalog, mock_read_csv, legacy_stations_df
    ):
        mock_catalog.return_value.__getitem__.return_value = legacy_stations_df
        mock_read_csv.return_value = _error_with_reasons_qa_df()
        # center year = 2002: total-time-window fails alongside
        # window_missing_nans and month_01_nans, which should be named as
        # the reasons; nearest clear year for total-time-window is 2001,
        # re-expressed as the 2000-2002 range.
        query = self._query("t2", historical_slice=(2001, 2003))
        with pytest.warns(
            UserWarning,
            match=r"from 2001-2003\. Related issues: January, overall "
            r"missing-data window\. Closest usable time ranges: 2000-2002",
        ):
            result = validate_station_qa_for_requested_years(query)
        assert result is False

    @patch(f"{MODULE}.read_csv_file")
    @patch(f"{MODULE}.DataCatalog")
    def test_error_with_no_nearby_year(
        self, mock_catalog, mock_read_csv, legacy_stations_df
    ):
        mock_catalog.return_value.__getitem__.return_value = legacy_stations_df
        mock_read_csv.return_value = _all_error_qa_df()
        # center year = (2000 + 2002) // 2 = 2001; total-time-window fails
        # every year in this sheet, so no nearby year clears it.
        query = self._query(
            "dew_point", historical_slice=(2000, 2002), stations=("KBAD",)
        )
        with pytest.warns(
            UserWarning,
            match="No usable time range was found nearby in the QA sheet",
        ):
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
        # (month_01_nans); nearest clear years for that row are 2000 and 2002.
        query = self._query("t2", historical_slice=(2000, 2002))
        with pytest.warns(
            UserWarning,
            match=r"January \(nearest clear year\(s\): 2000 or 2002\).*data may be biased",
        ):
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
