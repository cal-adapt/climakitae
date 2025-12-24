"""
Unit tests for climakitae/new_core/param_validation/bias_adjust_model_to_station_param_validator.py

This module contains comprehensive unit tests for the StationBiasCorrection processor
parameter validation functionality.
"""

import logging
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from climakitae.core.constants import UNSET
from climakitae.new_core.param_validation.bias_adjust_model_to_station_param_validator import (
    _get_station_metadata, _validate_catalog_requirement,
    _validate_downscaling_method_requirement, _validate_group,
    _validate_historical_slice, _validate_institution_id_requirement,
    _validate_kind, _validate_nquantiles, _validate_resolution_requirement,
    _validate_scenario_resolution_compatibility, _validate_stations,
    _validate_timescale_requirement, _validate_variable_compatibility,
    _validate_window, validate_bias_correction_station_data_param)


@pytest.fixture
def mock_station_metadata():
    """Fixture providing mock station metadata DataFrame."""
    return pd.DataFrame(
        {
            "station": [
                "Sacramento Executive Airport (KSAC)",
                "San Francisco International Airport (KSFO)",
                "Oakland International Airport (KOAK)",
                "Los Angeles International Airport (KLAX)",
            ],
            "station id": ["KSAC", "KSFO", "KOAK", "KLAX"],
            "latitude": [38.5, 37.6, 37.7, 33.9],
            "longitude": [-121.5, -122.4, -122.2, -118.4],
            "elevation": [10, 5, 3, 38],
        }
    )


@pytest.fixture
def valid_query():
    """Fixture providing a valid query dictionary for cross-validation."""
    return {
        "catalog": "cadcat",
        "variable_id": "tas",
        "table_id": "1hr",
        "activity_id": "WRF",
        "grid_label": "d02",
        "experiment_id": "ssp370",
        "institution_id": "UCLA",
    }


class TestValidateBiasCorrectStationDataParam:
    """Test class for validate_bias_correction_station_data_param function."""

    def setup_method(self):
        """Set up test fixtures for each test method."""
        self.mock_station_metadata = pd.DataFrame(
            {
                "station": [
                    "Sacramento Executive Airport (KSAC)",
                    "San Francisco (KSFO)",
                ],
                "station id": ["KSAC", "KSFO"],
                "latitude": [38.5, 37.6],
                "longitude": [-121.5, -122.4],
                "elevation": [10, 5],
            }
        )

    @patch(
        "climakitae.new_core.param_validation.bias_adjust_model_to_station_param_validator._get_station_metadata"
    )
    @patch(
        "climakitae.new_core.param_validation.bias_adjust_model_to_station_param_validator.find_station_match"
    )
    def test_valid_minimal_parameters(
        self, mock_find_station, mock_get_metadata, valid_query
    ):
        """Test validation with minimal valid parameters."""
        mock_get_metadata.return_value = self.mock_station_metadata
        mock_find_station.return_value = "Sacramento Executive Airport (KSAC)"

        value = {"stations": ["KSAC"]}
        result = validate_bias_correction_station_data_param(value, query=valid_query)

        assert result is True

    @patch(
        "climakitae.new_core.param_validation.bias_adjust_model_to_station_param_validator._get_station_metadata"
    )
    @patch(
        "climakitae.new_core.param_validation.bias_adjust_model_to_station_param_validator.find_station_match"
    )
    def test_valid_full_parameters(
        self, mock_find_station, mock_get_metadata, valid_query
    ):
        """Test validation with all optional parameters."""
        mock_get_metadata.return_value = self.mock_station_metadata
        mock_find_station.return_value = "Sacramento Executive Airport (KSAC)"

        value = {
            "stations": ["KSAC"],
            "historical_slice": (1990, 2010),
            "window": 90,
            "nquantiles": 20,
            "group": "time.dayofyear",
            "kind": "+",
        }
        result = validate_bias_correction_station_data_param(value, query=valid_query)

        assert result is True

    @pytest.mark.parametrize(
        "value",
        [
            None,
            UNSET,
        ],
        ids=["none_value", "unset_value"],
    )
    def test_none_or_unset_value(self, value, valid_query):
        """Test validation with None or UNSET values."""
        with pytest.warns(
            UserWarning, match="Station bias correction parameters cannot be None"
        ):
            result = validate_bias_correction_station_data_param(
                value, query=valid_query
            )
        assert result is False

    @pytest.mark.parametrize(
        "value",
        [
            ["KSAC"],
            "KSAC",
            123,
            ("KSAC",),
        ],
        ids=["list", "string", "integer", "tuple"],
    )
    def test_invalid_type_not_dict(self, value, valid_query):
        """Test validation with non-dictionary input types."""
        with pytest.warns(UserWarning, match="must be a dictionary"):
            result = validate_bias_correction_station_data_param(
                value, query=valid_query
            )
        assert result is False

    def test_missing_required_stations_key(self, valid_query):
        """Test validation with missing 'stations' key."""
        value = {"window": 90}
        with pytest.warns(UserWarning, match="Missing required parameter"):
            result = validate_bias_correction_station_data_param(
                value, query=valid_query
            )
        assert result is False

    @patch(
        "climakitae.new_core.param_validation.bias_adjust_model_to_station_param_validator._validate_stations"
    )
    def test_invalid_station_validation_fails(
        self, mock_validate_stations, valid_query
    ):
        """Test that station validation failure propagates."""
        mock_validate_stations.return_value = False

        value = {"stations": ["InvalidStation"]}
        result = validate_bias_correction_station_data_param(value, query=valid_query)

        assert result is False

    def test_query_none_returns_false(self):
        """Test that missing query returns False."""
        value = {"stations": ["KSAC"]}
        result = validate_bias_correction_station_data_param(value, query=None)
        assert result is False


class TestValidateStations:
    """Test class for _validate_stations function."""

    def setup_method(self):
        """Set up test fixtures."""
        self.mock_station_metadata = pd.DataFrame(
            {
                "station": [
                    "Sacramento Executive Airport (KSAC)",
                    "San Francisco International Airport (KSFO)",
                    "Oakland International Airport (KOAK)",
                ],
                "station id": ["KSAC", "KSFO", "KOAK"],
                "latitude": [38.5, 37.6, 37.7],
                "longitude": [-121.5, -122.4, -122.2],
                "elevation": [10, 5, 3],
            }
        )

    @patch(
        "climakitae.new_core.param_validation.bias_adjust_model_to_station_param_validator._get_station_metadata"
    )
    @patch(
        "climakitae.new_core.param_validation.bias_adjust_model_to_station_param_validator.find_station_match"
    )
    @pytest.mark.parametrize(
        "stations",
        [
            ["KSAC"],
            ["Sacramento Executive Airport (KSAC)"],
            ["KSAC", "KSFO"],
            ["KSAC", "KSFO", "KOAK"],
        ],
        ids=["single_code", "single_full_name", "multiple_codes", "three_stations"],
    )
    def test_valid_stations(self, mock_find_station, mock_get_metadata, stations):
        """Test validation with valid station names and codes."""
        mock_get_metadata.return_value = self.mock_station_metadata
        mock_find_station.return_value = "Sacramento Executive Airport (KSAC)"

        result = _validate_stations(stations)
        assert result is True

    @pytest.mark.parametrize(
        "stations,error_match",
        [
            ("KSAC", "'stations' must be a list"),
            ({"KSAC"}, "'stations' must be a list"),
            (123, "'stations' must be a list"),
        ],
        ids=["string", "set", "integer"],
    )
    def test_invalid_type_not_list(self, stations, error_match):
        """Test validation with non-list input types."""
        with pytest.warns(UserWarning, match=error_match):
            result = _validate_stations(stations)
        assert result is False

    def test_invalid_empty_list(self):
        """Test validation with empty station list."""
        with pytest.warns(UserWarning, match="Station list cannot be empty"):
            result = _validate_stations([])
        assert result is False

    @pytest.mark.parametrize(
        "stations",
        [
            ["KSAC", 123],
            [123, "KSFO"],
            ["KSAC", None, "KSFO"],
        ],
        ids=["string_and_int", "int_first", "with_none"],
    )
    def test_invalid_non_string_elements(self, stations):
        """Test validation with non-string elements in list."""
        with pytest.warns(UserWarning, match="All station names must be strings"):
            result = _validate_stations(stations)
        assert result is False

    @patch(
        "climakitae.new_core.param_validation.bias_adjust_model_to_station_param_validator._get_station_metadata"
    )
    @patch(
        "climakitae.new_core.param_validation.bias_adjust_model_to_station_param_validator.find_station_match"
    )
    def test_invalid_station_not_found(self, mock_find_station, mock_get_metadata):
        """Test validation with invalid station name."""
        mock_get_metadata.return_value = self.mock_station_metadata
        mock_find_station.return_value = None

        with pytest.warns(UserWarning, match="Invalid station"):
            result = _validate_stations(["INVALIDSTATION"])
        assert result is False


class TestValidateHistoricalSlice:
    """Test class for _validate_historical_slice function."""

    @pytest.mark.parametrize(
        "historical_slice",
        [
            (1980, 2014),
            (1990, 2010),
            (1980, 2000),
            (2000, 2014),
        ],
        ids=["full_period", "partial_period", "early_period", "late_period"],
    )
    def test_valid_historical_slices(self, historical_slice):
        """Test validation with valid historical periods."""
        result = _validate_historical_slice(historical_slice)
        assert result is True

    @pytest.mark.parametrize(
        "historical_slice,error_match",
        [
            ([1980, 2014], "must be a tuple"),
            ((1980,), "must contain exactly 2 elements"),
            ((1980, 2000, 2014), "must contain exactly 2 elements"),
            (("1980", 2014), "must be integers"),
            ((1980, "2014"), "must be integers"),
            ((2000, 2000), "must be less than"),
            ((2014, 1980), "must be less than"),
            ((1970, 2014), "before HadISD observational period starts"),
            ((1980, 2020), "after HadISD observational period ends"),
        ],
        ids=[
            "list_not_tuple",
            "too_short",
            "too_long",
            "start_string",
            "end_string",
            "equal_years",
            "reversed_years",
            "before_1980",
            "after_2014",
        ],
    )
    def test_invalid_historical_slices(self, historical_slice, error_match):
        """Test validation with invalid historical periods."""
        with pytest.warns(UserWarning, match=error_match):
            result = _validate_historical_slice(historical_slice)
        assert result is False


class TestValidateWindow:
    """Test class for _validate_window function."""

    @pytest.mark.parametrize(
        "window",
        [1, 30, 90, 180, 365],
        ids=["minimum", "typical_30", "typical_90", "half_year", "maximum"],
    )
    def test_valid_windows(self, window):
        """Test validation with valid window values."""
        result = _validate_window(window)
        assert result is True

    @pytest.mark.parametrize(
        "window,error_match",
        [
            (90.5, "must be an integer"),
            ("90", "must be an integer"),
            (0, "must be positive"),
            (-30, "must be positive"),
            (400, "exceeds one year"),
        ],
        ids=["float", "string", "zero", "negative", "too_large"],
    )
    def test_invalid_windows(self, window, error_match):
        """Test validation with invalid window values."""
        with pytest.warns(UserWarning, match=error_match):
            result = _validate_window(window)
        assert result is False


class TestValidateNquantiles:
    """Test class for _validate_nquantiles function."""

    @pytest.mark.parametrize(
        "nquantiles",
        [2, 10, 20, 30, 50],
        ids=["minimum", "low_typical", "default", "high_typical", "moderate_high"],
    )
    def test_valid_nquantiles(self, nquantiles):
        """Test validation with valid nquantiles values."""
        result = _validate_nquantiles(nquantiles)
        assert result is True

    def test_valid_high_nquantiles_with_warning(self, caplog):
        """Test validation with high but valid nquantiles value."""
        with caplog.at_level(logging.WARNING):
            result = _validate_nquantiles(150)
        assert result is True
        assert "very high" in caplog.text

    @pytest.mark.parametrize(
        "nquantiles,error_match",
        [
            (20.5, "must be an integer"),
            ("20", "must be an integer"),
            (0, "must be at least 2"),
            (1, "must be at least 2"),
            (-10, "must be at least 2"),
        ],
        ids=["float", "string", "zero", "one", "negative"],
    )
    def test_invalid_nquantiles(self, nquantiles, error_match):
        """Test validation with invalid nquantiles values."""
        with pytest.warns(UserWarning, match=error_match):
            result = _validate_nquantiles(nquantiles)
        assert result is False


class TestValidateGroup:
    """Test class for _validate_group function."""

    @pytest.mark.parametrize(
        "group",
        [
            "time.dayofyear",
            "time.month",
            "time.season",
        ],
        ids=["dayofyear", "month", "season"],
    )
    def test_valid_groups(self, group):
        """Test validation with valid group values."""
        result = _validate_group(group)
        assert result is True

    @pytest.mark.parametrize(
        "group,error_match",
        [
            (123, "must be a string"),
            ("time.week", "must be one of"),
            ("time.day_of_year", "must be one of"),
            ("", "must be one of"),
        ],
        ids=["integer", "unknown_group", "typo", "empty_string"],
    )
    def test_invalid_groups(self, group, error_match):
        """Test validation with invalid group values."""
        with pytest.warns(UserWarning, match=error_match):
            result = _validate_group(group)
        assert result is False


class TestValidateKind:
    """Test class for _validate_kind function."""

    @pytest.mark.parametrize("kind", ["+", "*"], ids=["additive", "multiplicative"])
    def test_valid_kinds(self, kind):
        """Test validation with valid kind values."""
        result = _validate_kind(kind)
        assert result is True

    @pytest.mark.parametrize(
        "kind,error_match",
        [
            (1, "must be a string"),
            ("-", "must be '\\+' .* or '\\*'"),
            ("/", "must be '\\+' .* or '\\*'"),
            ("", "must be '\\+' .* or '\\*'"),
        ],
        ids=["integer", "subtract", "divide", "empty_string"],
    )
    def test_invalid_kinds(self, kind, error_match):
        """Test validation with invalid kind values."""
        with pytest.warns(UserWarning, match=error_match):
            result = _validate_kind(kind)
        assert result is False


class TestValidateVariableCompatibility:
    """Test class for _validate_variable_compatibility function."""

    @pytest.mark.parametrize(
        "variable_id",
        [
            "tas",
            "tasmax",
            "tasmin",
            "t2",
            ["tas", "tasmax"],
            ["tas", "tasmax", "tasmin"],
        ],
        ids=["tas", "tasmax", "tasmin", "t2", "multiple_tas_tasmax", "all_temp"],
    )
    def test_valid_temperature_variables(self, variable_id):
        """Test validation with valid temperature variables."""
        query = {"variable_id": variable_id}
        result = _validate_variable_compatibility(query)
        assert result is True

    @pytest.mark.parametrize(
        "query",
        [
            {},
            {"variable_id": None},
        ],
        ids=["no_variable_id", "variable_id_none"],
    )
    def test_no_variable_id_returns_true(self, query):
        """Test that missing variable_id returns True."""
        result = _validate_variable_compatibility(query)
        assert result is True

    @pytest.mark.parametrize(
        "variable_id",
        [
            "pr",
            "huss",
            ["tas", "pr"],
            ["tasmax", "huss"],
        ],
        ids=["precipitation", "humidity", "mixed_tas_pr", "mixed_tasmax_huss"],
    )
    def test_invalid_unsupported_variables(self, variable_id):
        """Test validation with unsupported variables."""
        query = {"variable_id": variable_id}
        with pytest.warns(UserWarning, match="only supports temperature variables"):
            result = _validate_variable_compatibility(query)
        assert result is False


class TestValidateTimescaleRequirement:
    """Test class for _validate_timescale_requirement function."""

    @pytest.mark.parametrize("table_id", ["1hr", "hr"], ids=["1hr", "hr"])
    def test_valid_hourly_table_ids(self, table_id):
        """Test validation with valid hourly table IDs."""
        query = {"table_id": table_id}
        result = _validate_timescale_requirement(query)
        assert result is True

    @pytest.mark.parametrize(
        "query",
        [
            {},
            {"table_id": None},
        ],
        ids=["no_table_id", "table_id_none"],
    )
    def test_no_table_id_returns_true(self, query):
        """Test that missing table_id returns True."""
        result = _validate_timescale_requirement(query)
        assert result is True

    @pytest.mark.parametrize(
        "table_id", ["day", "mon", "3hr"], ids=["daily", "monthly", "3hourly"]
    )
    def test_invalid_non_hourly_table_ids(self, table_id):
        """Test validation with non-hourly table IDs."""
        query = {"table_id": table_id}
        with pytest.warns(UserWarning, match="requires hourly data"):
            result = _validate_timescale_requirement(query)
        assert result is False


class TestValidateDownscalingMethodRequirement:
    """Test class for _validate_downscaling_method_requirement function."""

    def test_valid_wrf_activity_id(self):
        """Test validation with valid WRF activity ID."""
        query = {"activity_id": "WRF"}
        result = _validate_downscaling_method_requirement(query)
        assert result is True

    @pytest.mark.parametrize(
        "query",
        [
            {},
            {"activity_id": None},
        ],
        ids=["no_activity_id", "activity_id_none"],
    )
    def test_no_activity_id_returns_true(self, query):
        """Test that missing activity_id returns True."""
        result = _validate_downscaling_method_requirement(query)
        assert result is True

    @pytest.mark.parametrize("activity_id", ["LOCA2", "CMIP6"], ids=["loca2", "cmip6"])
    def test_invalid_non_wrf_activity_ids(self, activity_id):
        """Test validation with non-WRF activity IDs."""
        query = {"activity_id": activity_id}
        with pytest.warns(UserWarning, match="only supports WRF"):
            result = _validate_downscaling_method_requirement(query)
        assert result is False


class TestValidateResolutionRequirement:
    """Test class for _validate_resolution_requirement function."""

    @pytest.mark.parametrize("grid_label", ["d01", "d02"], ids=["45km", "9km"])
    def test_valid_grid_labels(self, grid_label):
        """Test validation with valid grid labels."""
        query = {"grid_label": grid_label}
        result = _validate_resolution_requirement(query)
        assert result is True

    @pytest.mark.parametrize(
        "query",
        [
            {},
            {"grid_label": None},
        ],
        ids=["no_grid_label", "grid_label_none"],
    )
    def test_no_grid_label_returns_true(self, query):
        """Test that missing grid_label returns True."""
        result = _validate_resolution_requirement(query)
        assert result is True

    def test_invalid_3km_grid_label(self):
        """Test validation with 3km grid label."""
        query = {"grid_label": "d03"}
        with pytest.warns(UserWarning, match="does not support 3km resolution"):
            result = _validate_resolution_requirement(query)
        assert result is False


class TestValidateScenarioResolutionCompatibility:
    """Test class for _validate_scenario_resolution_compatibility function."""

    @pytest.mark.parametrize(
        "experiment_id",
        [
            "ssp245",
            "ssp585",
            "ssp370",
            ["ssp245", "ssp585"],
        ],
        ids=["ssp245_9km", "ssp585_9km", "ssp370_9km", "multiple_9km"],
    )
    def test_valid_9km_with_any_scenario(self, experiment_id):
        """Test validation with 9km and various scenarios."""
        query = {"grid_label": "d02", "experiment_id": experiment_id}
        result = _validate_scenario_resolution_compatibility(query)
        assert result is True

    @pytest.mark.parametrize(
        "query",
        [
            {"grid_label": "d03"},
            {"experiment_id": "ssp245"},
            {},
        ],
        ids=["no_experiment_id", "no_grid_label", "both_missing"],
    )
    def test_incomplete_query_returns_true(self, query):
        """Test that incomplete query returns True."""
        result = _validate_scenario_resolution_compatibility(query)
        assert result is True

    @pytest.mark.parametrize(
        "experiment_id",
        [
            "ssp245",
            "ssp585",
            ["ssp245", "ssp370"],
        ],
        ids=["ssp245_3km", "ssp585_3km", "mixed_with_ssp245"],
    )
    def test_invalid_3km_with_restricted_scenarios(self, experiment_id):
        """Test validation with 3km and restricted scenarios."""
        query = {"grid_label": "d03", "experiment_id": experiment_id}
        with pytest.warns(UserWarning, match="not compatible with"):
            result = _validate_scenario_resolution_compatibility(query)
        assert result is False


class TestValidateInstitutionIdRequirement:
    """Test class for _validate_institution_id_requirement function."""

    def test_valid_ucla_institution_id(self):
        """Test validation with valid UCLA institution ID."""
        query = {"institution_id": "UCLA"}
        result = _validate_institution_id_requirement(query)
        assert result is True

    @pytest.mark.parametrize(
        "institution_id",
        [
            "CNRM",
            "DWD",
            None,
        ],
        ids=["cnrm", "dwd", "none"],
    )
    def test_invalid_non_ucla_institution_ids(self, institution_id):
        """Test validation with non-UCLA institution IDs."""
        query = {"institution_id": institution_id}
        with pytest.warns(
            UserWarning, match="requires 'institution_id' to be set to 'UCLA'"
        ):
            result = _validate_institution_id_requirement(query)
        assert result is False

    def test_missing_institution_id_key(self):
        """Test validation with missing institution_id key."""
        query = {}
        with pytest.warns(
            UserWarning, match="requires 'institution_id' to be set to 'UCLA'"
        ):
            result = _validate_institution_id_requirement(query)
        assert result is False


class TestValidateCatalogRequirement:
    """Test class for _validate_catalog_requirement function."""

    @pytest.mark.parametrize(
        "catalog",
        [
            "cadcat",
            ["cadcat"],
        ],
        ids=["string", "list"],
    )
    def test_valid_cadcat_catalog(self, catalog):
        """Test validation with valid cadcat catalog."""
        query = {"catalog": catalog}
        result = _validate_catalog_requirement(query)
        assert result is True

    @pytest.mark.parametrize(
        "catalog,error_match",
        [
            ("climate", "requires 'catalog' == 'cadcat'"),
            ("renewables", "requires 'catalog' == 'cadcat'"),
            (["cadcat", "climate"], "requires 'catalog' == 'cadcat'"),
            (None, "requires 'catalog' to be set"),
            (123, "'catalog' must be a string or list"),
        ],
        ids=["climate", "renewables", "mixed_list", "none", "integer"],
    )
    def test_invalid_catalogs(self, catalog, error_match):
        """Test validation with invalid catalog values."""
        query = {"catalog": catalog}
        with pytest.warns(UserWarning, match=error_match):
            result = _validate_catalog_requirement(query)
        assert result is False

    def test_missing_catalog_key(self):
        """Test validation with missing catalog key."""
        query = {}
        with pytest.warns(UserWarning, match="requires 'catalog' to be set"):
            result = _validate_catalog_requirement(query)
        assert result is False


class TestGetStationMetadata:
    """Test class for _get_station_metadata function."""

    @patch(
        "climakitae.new_core.param_validation.bias_adjust_model_to_station_param_validator.DataCatalog"
    )
    def test_returns_dataframe(self, mock_catalog_class):
        """Test that function returns DataFrame from DataCatalog."""
        mock_catalog_instance = MagicMock()
        mock_stations_df = pd.DataFrame(
            {
                "station": ["Sacramento (KSAC)"],
                "station id": ["KSAC"],
                "latitude": [38.5],
                "longitude": [-121.5],
                "elevation": [10],
            }
        )
        mock_catalog_instance.__getitem__.return_value = mock_stations_df
        mock_catalog_class.return_value = mock_catalog_instance

        result = _get_station_metadata()

        assert isinstance(result, pd.DataFrame)
        assert "station" in result.columns
        mock_catalog_class.assert_called_once()

    @patch(
        "climakitae.new_core.param_validation.bias_adjust_model_to_station_param_validator.DataCatalog"
    )
    def test_contains_required_columns(self, mock_catalog_class):
        """Test that returned DataFrame has required columns."""
        mock_catalog_instance = MagicMock()
        mock_stations_df = pd.DataFrame(
            {
                "station": ["Sacramento (KSAC)", "San Francisco (KSFO)"],
                "station id": ["KSAC", "KSFO"],
                "latitude": [38.5, 37.6],
                "longitude": [-121.5, -122.4],
                "elevation": [10, 5],
            }
        )
        mock_catalog_instance.__getitem__.return_value = mock_stations_df
        mock_catalog_class.return_value = mock_catalog_instance

        result = _get_station_metadata()

        required_columns = [
            "station",
            "station id",
            "latitude",
            "longitude",
            "elevation",
        ]
        for col in required_columns:
            assert col in result.columns
