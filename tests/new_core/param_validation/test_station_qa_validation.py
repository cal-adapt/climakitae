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
    _check_source_variable,
    _describe_failing_tests,
    _describe_test,
    _group_missing_value_tests,
    _historical_slice_center_year,
    _load_qa_table,
    _nearest_passing_years,
    _recenter_historical_slice,
    _year_qa_status,
    resolve_station_to_source_id,
    validate_station_request_qa,
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
            {
                "source-id": "KTES",
                "variable": "tas",
                "test": "total-time-window",
                2000: 0,
                2001: 0,
                2002: 1,
            },
            {
                "source-id": "KTES",
                "variable": "tas",
                "test": "window_missing_nans",
                2000: 0,
                2001: 0,
                2002: 0,
            },
            {
                "source-id": "KTES",
                "variable": "tas",
                "test": "month_01_nans",
                2000: 0,
                2001: 1,
                2002: 0,
            },
            {
                "source-id": "KTES",
                "variable": "tas",
                "test": "diurnal_day_nans",
                2000: 0,
                2001: 0,
                2002: 0,
            },
        ]
    )


def _error_with_reasons_qa_df() -> pd.DataFrame:
    """total-time-window fails at 2002, alongside window_missing_nans and
    month_01_nans -- exercises the "reasons" list in the error message."""
    return pd.DataFrame(
        [
            {
                "source-id": "KTES",
                "variable": "tas",
                "test": "total-time-window",
                2000: 0,
                2001: 0,
                2002: 1,
            },
            {
                "source-id": "KTES",
                "variable": "tas",
                "test": "window_missing_nans",
                2000: 0,
                2001: 0,
                2002: 1,
            },
            {
                "source-id": "KTES",
                "variable": "tas",
                "test": "month_01_nans",
                2000: 0,
                2001: 0,
                2002: 1,
            },
            {
                "source-id": "KTES",
                "variable": "tas",
                "test": "diurnal_day_nans",
                2000: 0,
                2001: 0,
                2002: 0,
            },
        ]
    )


def _all_error_qa_df() -> pd.DataFrame:
    """A sheet where total-time-window fails every year -- no passing year
    at all for the sole blocking check."""
    return pd.DataFrame(
        [
            {
                "source-id": "KBAD",
                "variable": "tdps",
                "test": "total-time-window",
                2000: 1,
                2001: 1,
                2002: 1,
            },
            {
                "source-id": "KBAD",
                "variable": "tdps",
                "test": "window_missing_nans",
                2000: 1,
                2001: 1,
                2002: 1,
            },
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
        """Test _year_qa_status for a year with every row clear."""
        mock_read_csv.return_value = _synthetic_qa_df()
        assert _year_qa_status("KTES", "tas", 2000) == (False, [])

    @patch(f"{MODULE}.read_csv_file")
    def test_other_row_flagged_total_time_window_clear(self, mock_read_csv):
        """Test _year_qa_status when a non-blocking row fails but
        total-time-window itself passes."""
        mock_read_csv.return_value = _synthetic_qa_df()
        assert _year_qa_status("KTES", "tas", 2001) == (False, ["month_01_nans"])

    @patch(f"{MODULE}.read_csv_file")
    def test_total_time_window_fails_alone(self, mock_read_csv):
        """Test _year_qa_status when total-time-window fails with no other
        row flagged at that year."""
        mock_read_csv.return_value = _synthetic_qa_df()
        has_error, other_tests = _year_qa_status("KTES", "tas", 2002)
        assert has_error is True
        assert other_tests == []

    @patch(f"{MODULE}.read_csv_file")
    def test_total_time_window_fails_with_reasons(self, mock_read_csv):
        """Test _year_qa_status when total-time-window fails alongside
        other flagged rows, which should come back as "other_tests"."""
        mock_read_csv.return_value = _error_with_reasons_qa_df()
        has_error, other_tests = _year_qa_status("KTES", "tas", 2002)
        assert has_error is True
        assert other_tests == ["month_01_nans", "window_missing_nans"]

    @patch(f"{MODULE}.read_csv_file")
    def test_unknown_station_variable_year_passes(self, mock_read_csv):
        """Test _year_qa_status treats an unmapped station, an uncovered
        year, and an uncovered variable all as "unknown" (passing)."""
        mock_read_csv.return_value = _synthetic_qa_df()
        assert _year_qa_status("KUNKNOWN", "tas", 2000) == (False, [])
        assert _year_qa_status("KTES", "tas", 1999) == (False, [])
        assert _year_qa_status("KTES", "tdps", 2000) == (False, [])


class TestNearestPassingYears:
    """Test class for _nearest_passing_years."""

    @patch(f"{MODULE}.read_csv_file")
    def test_finds_nearest_before_and_after(self, mock_read_csv):
        """Test _nearest_passing_years returns the nearest passing year on
        each side of the requested year."""
        df = pd.DataFrame(
            [
                {
                    "source-id": "KTES",
                    "variable": "tas",
                    "test": "total-time-window",
                    1998: 0,
                    1999: 1,
                    2000: 1,
                    2001: 0,
                },
                {
                    "source-id": "KTES",
                    "variable": "tas",
                    "test": "window_missing_nans",
                    1998: 0,
                    1999: 0,
                    2000: 0,
                    2001: 0,
                },
            ]
        )
        mock_read_csv.return_value = df
        assert _nearest_passing_years("KTES", "tas", "total-time-window", 1999) == [
            1998,
            2001,
        ]

    @patch(f"{MODULE}.read_csv_file")
    def test_different_test_not_conflated(self, mock_read_csv):
        """Test that _nearest_passing_years checks only the named test's
        own row: window_missing_nans passes every year here, even though
        total-time-window fails 1999-2000 -- the two rows are independent."""
        df = pd.DataFrame(
            [
                {
                    "source-id": "KTES",
                    "variable": "tas",
                    "test": "total-time-window",
                    1998: 0,
                    1999: 1,
                    2000: 1,
                    2001: 0,
                },
                {
                    "source-id": "KTES",
                    "variable": "tas",
                    "test": "window_missing_nans",
                    1998: 0,
                    1999: 0,
                    2000: 0,
                    2001: 0,
                },
            ]
        )
        mock_read_csv.return_value = df
        assert _nearest_passing_years("KTES", "tas", "window_missing_nans", 1999) == [
            1998,
            2000,
        ]

    @patch(f"{MODULE}.read_csv_file")
    def test_no_passing_year_returns_empty(self, mock_read_csv):
        """Test _nearest_passing_years returns an empty list when the named
        test fails for every year the sheet covers."""
        mock_read_csv.return_value = _all_error_qa_df()
        assert _nearest_passing_years("KBAD", "tdps", "window_missing_nans", 2001) == []


class TestDescribeTest:
    """Test class for _describe_test."""

    def test_month_test(self):
        """Test _describe_test converts a month test name to a month name."""
        assert _describe_test("month_07_nans") == "July"

    def test_diurnal_test(self):
        """Test _describe_test converts the diurnal test names, including
        the winter-specific variant."""
        assert _describe_test("diurnal_day_nans") == "daytime coverage"
        assert _describe_test("diurnal_night_nans") == "nighttime coverage"
        assert (
            _describe_test("diurnal_night_winter_nans") == "winter nighttime coverage"
        )

    def test_error_tier_test(self):
        """Test _describe_test converts the two blocking/overall-coverage
        test names."""
        assert _describe_test("total-time-window") == "overall time coverage"
        assert _describe_test("window_missing_nans") == "overall missing-data window"

    def test_unknown_test_passthrough(self):
        """Test _describe_test passes an unrecognized test name through
        unchanged."""
        assert _describe_test("some_other_test") == "some_other_test"


class TestResolveStationToSourceId:
    """Test class for resolve_station_to_source_id."""

    def test_airport_code(self, legacy_stations_df):
        """Test resolve_station_to_source_id with a direct airport code."""
        assert resolve_station_to_source_id("KTES", legacy_stations_df) == "KTES"

    def test_hdp_station_id(self, legacy_stations_df):
        """Test resolve_station_to_source_id maps an HDP station_id to its
        airport-code equivalent."""
        assert (
            resolve_station_to_source_id("ASOSAWOS_72483023225", legacy_stations_df)
            == "KTES"
        )

    def test_unmapped_hdp_station_id(self, legacy_stations_df):
        """Test resolve_station_to_source_id returns None for an HDP
        station_id with no legacy airport-code equivalent."""
        assert (
            resolve_station_to_source_id("OtherISD_99999999999", legacy_stations_df)
            is None
        )

    def test_malformed_identifier(self, legacy_stations_df):
        """Test resolve_station_to_source_id returns None for an
        unparseable identifier."""
        assert (
            resolve_station_to_source_id("not_a_valid_id", legacy_stations_df) is None
        )


class TestHistoricalSliceCenterYear:
    """Test class for _historical_slice_center_year."""

    def test_even_span(self):
        """Test _historical_slice_center_year with an even total span."""
        assert _historical_slice_center_year((1980, 2010)) == 1995

    def test_default_historical_slice(self):
        """Test _historical_slice_center_year with the processor's own
        default historical_slice."""
        assert _historical_slice_center_year((1980, 2014)) == 1997

    def test_single_year_span(self):
        """Test _historical_slice_center_year with a single-year span."""
        assert _historical_slice_center_year((2000, 2000)) == 2000


class TestRecenterHistoricalSlice:
    """Test class for _recenter_historical_slice."""

    def test_recenters_on_a_different_year(self):
        """Test _recenter_historical_slice re-centers on a nearby year
        while preserving the original slice's width."""
        assert _recenter_historical_slice(1999, (1985, 2015)) == (1984, 2014)
        assert _recenter_historical_slice(2001, (1985, 2015)) == (1986, 2016)

    def test_recentering_on_the_same_center_is_a_no_op(self):
        """Test _recenter_historical_slice returns the original slice when
        re-centered on its own existing center year."""
        assert _recenter_historical_slice(2000, (1985, 2015)) == (1985, 2015)

    def test_odd_span_preserves_width(self):
        """Test _recenter_historical_slice preserves an odd total span's
        exact width rather than assuming a symmetric window."""
        # center = (1980 + 2014) // 2 = 1997 -> 17 years back, 17 forward
        assert _recenter_historical_slice(2000, (1980, 2014)) == (1983, 2017)


class TestGroupMissingValueTests:
    """Test class for _group_missing_value_tests."""

    def test_splits_nans_from_other_tests(self):
        """Test _group_missing_value_tests separates "_nans"-suffixed rows
        from a hypothetical other kind of test."""
        missing, other = _group_missing_value_tests(
            ["month_01_nans", "some_future_check", "window_missing_nans"]
        )
        assert missing == ["month_01_nans", "window_missing_nans"]
        assert other == ["some_future_check"]

    def test_all_nans(self):
        """Test _group_missing_value_tests with only "_nans" rows."""
        missing, other = _group_missing_value_tests(["month_01_nans"])
        assert missing == ["month_01_nans"]
        assert other == []

    def test_empty_input(self):
        """Test _group_missing_value_tests with no failing rows."""
        assert _group_missing_value_tests([]) == ([], [])


class TestDescribeFailingTests:
    """Test class for _describe_failing_tests."""

    def test_groups_missing_value_tests_under_one_phrase(self):
        """Test _describe_failing_tests groups "_nans" rows under one "too
        many missing station values" phrase."""
        result = _describe_failing_tests(
            ["month_02_nans", "month_05_nans"], _describe_test
        )
        assert result == "Too many missing station values for February, May"

    def test_other_test_type_described_separately(self):
        """Test _describe_failing_tests describes a non-"_nans" row on its
        own, not folded into the missing-values phrase."""
        result = _describe_failing_tests(
            ["month_02_nans", "some_future_check"], _describe_test
        )
        assert result == (
            "Too many missing station values for February; some_future_check"
        )

    def test_only_other_test_type(self):
        """Test _describe_failing_tests with no missing-value rows at all."""
        result = _describe_failing_tests(["some_future_check"], _describe_test)
        assert result == "some_future_check"

    def test_uses_the_given_describe_callable(self):
        """Test _describe_failing_tests renders each test through the
        caller-supplied `describe` function, not just _describe_test."""
        result = _describe_failing_tests(["month_02_nans"], lambda t: f"[{t}]")
        assert result == "Too many missing station values for [month_02_nans]"


class TestCheckSourceVariable:
    """Test class for _check_source_variable."""

    @patch(f"{MODULE}._year_qa_status")
    def test_clean_combo_returns_false_silently(self, mock_status):
        """Test _check_source_variable returns False and logs nothing when
        nothing is flagged."""
        mock_status.return_value = (False, [])
        assert _check_source_variable("KTES", "tas", 2000, (1985, 2015)) is False

    @patch(f"{MODULE}._nearest_passing_years")
    @patch(f"{MODULE}._year_qa_status")
    def test_warning_only_returns_false_and_warns(self, mock_status, mock_nearest):
        """Test _check_source_variable returns False but logs a warning
        naming the flagged row and its own nearest passing year(s), when
        total-time-window itself passes."""
        mock_status.return_value = (False, ["month_01_nans"])
        mock_nearest.return_value = [1999]
        with pytest.warns(
            UserWarning,
            match=r"January \(nearest clear year\(s\): 1999\).*resulting localized data may be biased",
        ):
            result = _check_source_variable("KTES", "tas", 2000, (1985, 2015))
        assert result is False

    @patch(f"{MODULE}._nearest_passing_years")
    @patch(f"{MODULE}._year_qa_status")
    def test_error_returns_true_and_errors_with_reasons(
        self, mock_status, mock_nearest
    ):
        """Test _check_source_variable returns True (blocks) and logs an
        error naming the other flagged rows as reasons, plus the
        total-time-window-specific suggested time range(s), when
        total-time-window fails."""
        mock_status.return_value = (True, ["window_missing_nans"])
        mock_nearest.return_value = [1999, 2001]
        with pytest.warns(
            UserWarning,
            match=r"Related issues: Too many missing station values for "
            r"overall missing-data window\. "
            r"Closest usable time ranges: 1984-2014 or 1986-2016",
        ):
            result = _check_source_variable("KTES", "tas", 2000, (1985, 2015))
        assert result is True

    @patch(f"{MODULE}._nearest_passing_years")
    @patch(f"{MODULE}._year_qa_status")
    def test_error_with_no_other_reasons(self, mock_status, mock_nearest):
        """Test _check_source_variable's error message falls back to "no
        other checks are flagged" when total-time-window fails alone (but
        a nearby year does exist)."""
        mock_status.return_value = (True, [])
        mock_nearest.return_value = [1999]
        with pytest.warns(
            UserWarning,
            match="no other checks are flagged for this time range",
        ):
            result = _check_source_variable("KTES", "tas", 2000, (1985, 2015))
        assert result is True

    @patch(f"{MODULE}._nearest_passing_years")
    @patch(f"{MODULE}._year_qa_status")
    def test_error_with_no_nearby_year(self, mock_status, mock_nearest):
        """Test _check_source_variable's error message falls back to "no
        usable time range" when total-time-window has no nearby passing
        year at all (even though other rows are flagged as reasons)."""
        mock_status.return_value = (True, ["window_missing_nans"])
        mock_nearest.return_value = []
        with pytest.warns(
            UserWarning,
            match="No usable time range was found nearby in the QA sheet",
        ):
            result = _check_source_variable("KTES", "tas", 2000, (1985, 2015))
        assert result is True


class TestValidateStationRequestQa:
    """Test class for validate_station_request_qa."""

    def _query(self, variable_id, historical_slice=None, stations=("KTES",)):
        """Build a minimal query dict for bias_adjust_model_to_station."""
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
        """Test validate_station_request_qa passes when the center year is
        fully clean."""
        mock_catalog.return_value.__getitem__.return_value = legacy_stations_df
        mock_read_csv.return_value = _synthetic_qa_df()
        # center year = (1999 + 2001) // 2 = 2000, which is clean
        query = self._query("t2", historical_slice=(1999, 2001))
        assert validate_station_request_qa(query) is True

    @patch(f"{MODULE}.read_csv_file")
    @patch(f"{MODULE}.DataCatalog")
    def test_error_blocks_with_no_other_reasons(
        self, mock_catalog, mock_read_csv, legacy_stations_df
    ):
        """Test validate_station_request_qa blocks end-to-end when
        total-time-window fails alone, suggesting the nearest clear
        same-width time range."""
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
            result = validate_station_request_qa(query)
        assert result is False

    @patch(f"{MODULE}.read_csv_file")
    @patch(f"{MODULE}.DataCatalog")
    def test_error_blocks_with_reasons_from_other_rows(
        self, mock_catalog, mock_read_csv, legacy_stations_df
    ):
        """Test validate_station_request_qa blocks end-to-end and names
        other flagged rows as the likely reason, when total-time-window
        fails alongside other rows."""
        mock_catalog.return_value.__getitem__.return_value = legacy_stations_df
        mock_read_csv.return_value = _error_with_reasons_qa_df()
        # center year = 2002: total-time-window fails alongside
        # window_missing_nans and month_01_nans, which should be named as
        # the reasons; nearest clear year for total-time-window is 2001,
        # re-expressed as the 2000-2002 range.
        query = self._query("t2", historical_slice=(2001, 2003))
        with pytest.warns(
            UserWarning,
            match=r"from 2001-2003\. Related issues: Too many missing "
            r"station values for January, overall missing-data window\. "
            r"Closest usable time ranges: 2000-2002",
        ):
            result = validate_station_request_qa(query)
        assert result is False

    @patch(f"{MODULE}.read_csv_file")
    @patch(f"{MODULE}.DataCatalog")
    def test_error_with_no_nearby_year(
        self, mock_catalog, mock_read_csv, legacy_stations_df
    ):
        """Test validate_station_request_qa blocks end-to-end with the "no
        usable time range" fallback when total-time-window fails every
        year the sheet covers."""
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
            result = validate_station_request_qa(query)
        assert result is False

    @patch(f"{MODULE}.read_csv_file")
    @patch(f"{MODULE}.DataCatalog")
    def test_warning_tier_passes_with_message(
        self, mock_catalog, mock_read_csv, legacy_stations_df
    ):
        """Test validate_station_request_qa still returns True but logs a
        warning naming the flagged row's own nearest clear year(s), when
        only a non-blocking row fails."""
        mock_catalog.return_value.__getitem__.return_value = legacy_stations_df
        mock_read_csv.return_value = _synthetic_qa_df()
        # center year = (2000 + 2002) // 2 = 2001, warning-tier only
        # (month_01_nans); nearest clear years for that row are 2000 and 2002.
        query = self._query("t2", historical_slice=(2000, 2002))
        with pytest.warns(
            UserWarning,
            match=r"January \(nearest clear year\(s\): 2000 or 2002\).*resulting localized data may be biased",
        ):
            result = validate_station_request_qa(query)
        assert result is True

    @patch(f"{MODULE}.read_csv_file")
    @patch(f"{MODULE}.DataCatalog")
    def test_missing_historical_slice_uses_default(
        self, mock_catalog, mock_read_csv, legacy_stations_df
    ):
        """Test validate_station_request_qa falls back to the processor's
        default historical_slice when none is given, and passes when that
        default center year isn't covered by the sheet."""
        mock_catalog.return_value.__getitem__.return_value = legacy_stations_df
        mock_read_csv.return_value = _synthetic_qa_df()
        query = self._query("t2")
        assert validate_station_request_qa(query) is True

    @patch(f"{MODULE}.read_csv_file")
    @patch(f"{MODULE}.DataCatalog")
    def test_unmapped_station_skipped_others_checked(
        self, mock_catalog, mock_read_csv, legacy_stations_df
    ):
        """Test validate_station_request_qa skips a station with no
        airport-code equivalent but still checks the remaining stations."""
        mock_catalog.return_value.__getitem__.return_value = legacy_stations_df
        mock_read_csv.return_value = _synthetic_qa_df()
        query = self._query(
            "t2",
            historical_slice=(2001, 2003),
            stations=("OtherISD_12345678901", "KTES"),
        )
        result = validate_station_request_qa(query)
        assert result is False

    @patch(f"{MODULE}.read_csv_file")
    @patch(f"{MODULE}.DataCatalog")
    def test_no_bias_adjust_processor_is_noop(self, mock_catalog, mock_read_csv):
        """Test validate_station_request_qa is a no-op, and never touches
        the QA sheet, when bias_adjust_model_to_station isn't present."""
        result = validate_station_request_qa(
            {"processes": {"time_slice": (2002, 2002)}}
        )
        assert result is True
        mock_read_csv.assert_not_called()
