"""
Unit tests for climakitae/explore/standard_year_profile.py

This module contains comprehensive unit tests for the Standard Year and climate
profile computation functions that provide climate profile analysis.

"""

from unittest.mock import MagicMock, call, patch

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from climakitae.explore.standard_year_profile import (
    _get_clean_standardyr_filename,
    _check_cached_area,
    _check_lat_lon,
    _check_stations,
    export_profile_to_csv,
)


class TestExportProfile:
    """Test class for file export functions.

    This set of tests mainly checks that file names are correctly
    generated and that separate files are saved for each warming level.

    Attributes
    ----------
    multi_df : pd.DataFrame
        DataFrame with MultiIndex columns.
    multi_df_gwl : pd.DataFrame
        DataFrame with MultiIndex columns.
    multi_df_invalid : pd.DataFrame
        DataFrame with MultiIndex columns.
    """

    def setup_method(self):
        """Set up test fixtures."""
        # Create DataFrame with MultiIndex columns
        hours = list(range(1, 25))
        simulations = ["sim1", "sim2"]
        multi_cols = pd.MultiIndex.from_product(
            [hours, simulations], names=["Hour", "Simulation"]
        )
        self.multi_df = pd.DataFrame(
            np.random.rand(365, len(multi_cols)),
            index=range(1, 366),
            columns=multi_cols,
        )

        # Create DataFrame with MultiIndex columns and warming levels
        hours = list(range(1, 25))
        simulations = ["sim1", "sim2"]
        global_warming_levels = ["WL_1.5", "WL_2.0"]
        multi_cols = pd.MultiIndex.from_product(
            [hours, global_warming_levels, simulations],
            names=["Hour", "Warming_Level", "Simulation"],
        )
        self.multi_df_gwl = pd.DataFrame(
            np.random.rand(365, len(multi_cols)),
            index=range(1, 366),
            columns=multi_cols,
        )

        # Create DataFrame with invalid MultiIndex
        hours = list(range(1, 25))
        simulations = ["sim1", "sim2"]
        global_warming_levels = ["WL_1.5", "WL_2.0"]
        extra_dim = ["val1", "val2"]
        multi_cols = pd.MultiIndex.from_product(
            [hours, global_warming_levels, simulations, extra_dim],
            names=["Hour", "Warming_Level", "Simulation", "Extra_dim"],
        )
        self.multi_df_invalid = pd.DataFrame(
            np.random.rand(365, len(multi_cols)),
            index=range(1, 366),
            columns=multi_cols,
        )

    def test_export_profile_to_csv(self):
        with patch("pandas.DataFrame.to_csv") as to_csv_mock:
            variable = "Air Temperature at 2m"
            q = 0.5
            gwl = [1.5]
            cached_area = "Sacramento County"
            no_delta = False
            profile_selections = {
                "variable": variable,
                "q": q,
                "warming_level": gwl,
                "cached_area": cached_area,
                "no_delta": no_delta,
            }
            export_profile_to_csv(
                self.multi_df,
                **profile_selections,
            )
            expected_filename = "stdyr_t2_50ptile_sacramento_county_near-future_delta_from_historical.csv"
            to_csv_mock.assert_called_with(expected_filename)

        with patch("pandas.DataFrame.to_csv") as to_csv_mock:
            gwls = [1.5, 2.0]
            profile_selections = {
                "variable": variable,
                "q": q,
                "warming_level": gwls,
                "cached_area": cached_area,
                "no_delta": no_delta,
            }
            export_profile_to_csv(self.multi_df_gwl, **profile_selections)
            expected_filenames = [
                call(
                    "stdyr_t2_50ptile_sacramento_county_near-future_delta_from_historical.csv"
                ),
                call(
                    "stdyr_t2_50ptile_sacramento_county_mid-century_delta_from_historical.csv"
                ),
            ]
            to_csv_mock.assert_has_calls(expected_filenames)

    def test_export_profile_to_csv_invalid_profile(self):
        """Test that error is raised by profile with invalid index format."""
        with patch("pandas.DataFrame.to_csv") as to_csv_mock, pytest.raises(ValueError):
            variable = "Air Temperature at 2m"
            q = 0.5
            gwl = [1.5]
            cached_area = "Sacramento County"
            no_delta = False
            profile_selections = {
                "variable": variable,
                "q": q,
                "warming_level": gwl,
                "cached_area": cached_area,
                "no_delta": no_delta,
            }
            export_profile_to_csv(self.multi_df_invalid, **profile_selections)

    @pytest.mark.parametrize(
        "input_value,expected",
        [
            (
                {
                    "var_id": "t2",
                    "q": 0.5,
                    "gwl": 1.2,
                    "location": "sacramento county",
                    "no_delta": False,
                },
                "stdyr_t2_50ptile_sacramento_county_present-day_delta_from_historical.csv",
            ),
            (
                {
                    "var_id": "t2",
                    "q": 0.5,
                    "gwl": 1.2,
                    "location": "sacramento county",
                    "no_delta": True,
                },
                "stdyr_t2_50ptile_sacramento_county_present-day.csv",
            ),
            (
                {
                    "var_id": "t2",
                    "q": 0.5,
                    "gwl": 1.2,
                    "location": "35-5N_122-5W",
                    "no_delta": True,
                },
                "stdyr_t2_50ptile_35-5N_122-5W_present-day.csv",
            ),
            (
                {
                    "var_id": "t2",
                    "q": 0.5,
                    "gwl": 3.0,
                    "location": "san diego lindbergh field ksan",
                    "no_delta": True,
                },
                "stdyr_t2_50ptile_san_diego_lindbergh_field_ksan_late-century.csv",
            ),
            (
                {
                    "var_id": "prec",
                    "q": 0.5,
                    "gwl": 2.5,
                    "location": "35-5N_122-5W",
                    "no_delta": True,
                },
                "stdyr_prec_50ptile_35-5N_122-5W_mid-late-century.csv",
            ),
            (
                {
                    "var_id": "prec",
                    "q": 0.75,
                    "gwl": 3.0,
                    "location": "35-5N_122-5W",
                    "no_delta": False,
                },
                "stdyr_prec_75ptile_35-5N_122-5W_late-century_delta_from_historical.csv",
            ),
        ],
    )
    def test_get_clean_standardyr_filename(self, input_value, expected):
        """Test that file name is correctly formatted based on given inputs."""
        assert _get_clean_standardyr_filename(**input_value) == expected


class TestChainedExportHelpers:
    """Integration test covering the behavior of export helper function that are used in succession to
    construct a string of climate profile location information"""

    @pytest.mark.parametrize(
        "input_value,expected",
        [
            (
                {
                    "stations": ["Sacramento Executive Airport (KSAC)"],
                    "latitude": 34.4041,
                    "longitude": -121.516,
                },
                "34-4041N_121-516W",
            ),
            (
                {
                    "stations": ["Sacramento Executive Airport (KSAC)"],
                    "latitude": 34.4041,
                },
                "sacramento executive airport (ksac)",
            ),
            (
                {
                    "stations": ["Sacramento Executive Airport (KSAC)"],
                    "latitude": 34.4041,
                    "longitude": -121.516,
                    "cached_area": "Los Angeles County",
                },
                "los angeles county",
            ),
            (
                {
                    "stations": ["Sacramento Executive Airport (KSAC)"],
                    "cached_area": "Los Angeles County",
                },
                "los angeles county",
            ),
            (
                {
                    "stations": ["Sacramento Executive Airport (KSAC)"],
                },
                "sacramento executive airport (ksac)",
            ),
            (
                {
                    "stations": ["Custom Station Name"],
                    "latitude": 34.4041,
                    "longitude": -121.516,
                },
                "custom station name_34-4041N_121-516W",
            ),
            (
                {
                    "stations": [
                        "Sacramento Executive Airport (KSAC)",
                        "Santa Barbara Municipal Airport (KSBA)",
                    ],
                },
                "sacramento executive airport (ksac)_santa barbara municipal airport (ksba)",
            ),
            (
                {
                    "latitude": 34.4041,
                    "longitude": -121.516,
                    "cached_area": "Los Angeles County",
                },
                "los angeles county",
            ),
            (
                {
                    "latitude": 34.4041,
                    "cached_area": "Los Angeles County",
                },
                "los angeles county",
            ),
            (
                {
                    "latitude": 34.4041,
                    "longitude": -121.516,
                },
                "34-4041N_121-516W",
            ),
            (
                {
                    "cached_area": "Los Angeles County",
                },
                "los angeles county",
            ),
        ],
    )
    def test_location_string_construction(self, input_value, expected):
        """Test behavior"""
        func_list = [_check_cached_area, _check_lat_lon, _check_stations]
        location_str = ""

        for func in func_list:
            location_str = func(location_str, **input_value)

        assert location_str == expected


class TestCheckCachedArea:
    """Test location string construction"""

    @pytest.mark.parametrize(
        "input_value,expected",
        [
            (
                {
                    "stations": ["Sacramento Executive Airport (KSAC)"],
                    "latitude": 34.4041,
                    "longitude": -121.516,
                },
                "",
            ),
            (
                {
                    "stations": ["Sacramento Executive Airport (KSAC)"],
                    "latitude": 34.4041,
                },
                "",
            ),
            (
                {
                    "stations": ["Sacramento Executive Airport (KSAC)"],
                    "latitude": 34.4041,
                    "longitude": -121.516,
                    "cached_area": "Los Angeles County",
                },
                "los angeles county",
            ),
            (
                {
                    "stations": ["Sacramento Executive Airport (KSAC)"],
                    "cached_area": "Los Angeles County",
                },
                "los angeles county",
            ),
            (
                {
                    "stations": ["Sacramento Executive Airport (KSAC)"],
                },
                "",
            ),
            (
                {
                    "stations": ["Custom Station Name"],
                    "latitude": 34.4041,
                    "longitude": -121.516,
                },
                "",
            ),
            (
                {
                    "stations": ["Custom Station Name"],
                },
                "",
            ),
            (
                {
                    "stations": [
                        "Sacramento Executive Airport (KSAC)",
                        "Santa Barbara Municipal Airport (KSBA)",
                    ],
                },
                "",
            ),
            (
                {
                    "stations": [
                        "Custom Name 1",
                        "Custom Name 2",
                    ],
                },
                "",
            ),
            (
                {
                    "stations": [
                        "Custom Station Name",
                        "Santa Barbara Municipal Airport (KSBA)",
                    ],
                },
                "",
            ),
            (
                {
                    "latitude": 34.4041,
                    "longitude": -121.516,
                    "cached_area": "Los Angeles County",
                },
                "los angeles county",
            ),
            (
                {
                    "latitude": 34.4041,
                    "cached_area": "Los Angeles County",
                },
                "los angeles county",
            ),
            (
                {
                    "latitude": 34.4041,
                    "longitude": -121.516,
                },
                "",
            ),
            (
                {
                    "latitude": 34.4041,
                },
                "",
            ),
            (
                {
                    "cached_area": "Los Angeles County",
                },
                "los angeles county",
            ),
        ],
    )
    def test_check_cached_area(self, input_value, expected):
        """Test that location string is correctly formatted based on given inputs."""
        location_str = ""
        assert _check_cached_area(location_str, **input_value) == expected


class TestCheckLatLon:
    """Test location string construction"""

    @pytest.mark.parametrize(
        "input_string,input_value,expected",
        [
            (
                "",
                {
                    "stations": ["Sacramento Executive Airport (KSAC)"],
                    "latitude": 34.4041,
                    "longitude": -121.516,
                },
                "34-4041N_121-516W",
            ),
            (
                "",
                {
                    "stations": ["Sacramento Executive Airport (KSAC)"],
                    "latitude": 34.4041,
                },
                "",
            ),
            (
                "los angeles county",
                {
                    "stations": ["Sacramento Executive Airport (KSAC)"],
                    "latitude": 34.4041,
                    "longitude": -121.516,
                    "cached_area": "Los Angeles County",
                },
                "los angeles county",
            ),
            (
                "los angeles county",
                {
                    "stations": ["Sacramento Executive Airport (KSAC)"],
                    "cached_area": "Los Angeles County",
                },
                "los angeles county",
            ),
            (
                "",
                {
                    "stations": ["Sacramento Executive Airport (KSAC)"],
                },
                "",
            ),
            (
                "",
                {
                    "stations": ["Custom Station Name"],
                    "latitude": 34.4041,
                    "longitude": -121.516,
                },
                "34-4041N_121-516W",
            ),
            (
                "",
                {
                    "stations": ["Custom Station Name"],
                },
                "",
            ),
            (
                "",
                {
                    "stations": [
                        "Sacramento Executive Airport (KSAC)",
                        "Santa Barbara Municipal Airport (KSBA)",
                    ],
                },
                "",
            ),
            (
                "",
                {
                    "stations": [
                        "Custom Name 1",
                        "Custom Name 2",
                    ],
                },
                "",
            ),
            (
                "",
                {
                    "stations": [
                        "Custom Station Name",
                        "Santa Barbara Municipal Airport (KSBA)",
                    ],
                },
                "",
            ),
            (
                "los angeles county",
                {
                    "latitude": 34.4041,
                    "longitude": -121.516,
                    "cached_area": "Los Angeles County",
                },
                "los angeles county",
            ),
            (
                "los angeles county",
                {
                    "latitude": 34.4041,
                    "cached_area": "Los Angeles County",
                },
                "los angeles county",
            ),
            (
                "",
                {
                    "latitude": 34.4041,
                    "longitude": -121.516,
                },
                "34-4041N_121-516W",
            ),
            (
                "",
                {
                    "latitude": 34.4041,
                },
                "",
            ),
            (
                "los angeles county",
                {
                    "cached_area": "Los Angeles County",
                },
                "los angeles county",
            ),
        ],
    )
    def test_check_lat_lon(self, value, expected):
        """Test that location string is correctly formatted based on given inputs."""
        assert _check_lat_lon(**value) == expected


class TestCheckStations:
    """Test location string construction"""

    @pytest.mark.parametrize(
        "input_string,input_value,expected",
        [
            (
                "34-4041N_121-516W",
                {
                    "stations": ["Sacramento Executive Airport (KSAC)"],
                    "latitude": 34.4041,
                    "longitude": -121.516,
                },
                "34-4041N_121-516W",
            ),
            (
                "",
                {
                    "stations": ["Sacramento Executive Airport (KSAC)"],
                    "latitude": 34.4041,
                },
                "sacramento executive airport (ksac)",
            ),
            (
                "los angeles county",
                {
                    "stations": ["Sacramento Executive Airport (KSAC)"],
                    "latitude": 34.4041,
                    "longitude": -121.516,
                    "cached_area": "Los Angeles County",
                },
                "los angeles county",
            ),
            (
                "los angeles county",
                {
                    "stations": ["Sacramento Executive Airport (KSAC)"],
                    "cached_area": "Los Angeles County",
                },
                "los angeles county",
            ),
            (
                "",
                {
                    "stations": ["Sacramento Executive Airport (KSAC)"],
                },
                "sacramento executive airport (ksac)",
            ),
            (
                "34-4041N_121-516W",
                {
                    "stations": ["Custom Station Name"],
                    "latitude": 34.4041,
                    "longitude": -121.516,
                },
                "custom station name_34-4041N_121-516W",
            ),
            (
                "",
                {
                    "stations": [
                        "Sacramento Executive Airport (KSAC)",
                        "Santa Barbara Municipal Airport (KSBA)",
                    ],
                },
                "sacramento executive airport (ksac)_santa barbara municipal airport (ksba)",
            ),
            (
                "los angeles county",
                {
                    "latitude": 34.4041,
                    "longitude": -121.516,
                    "cached_area": "Los Angeles County",
                },
                "los angeles county",
            ),
            (
                "los angeles county",
                {
                    "latitude": 34.4041,
                    "cached_area": "Los Angeles County",
                },
                "los angeles county",
            ),
            (
                "34-4041N_121-516W",
                {
                    "latitude": 34.4041,
                    "longitude": -121.516,
                },
                "34-4041N_121-516W",
            ),
            (
                "los angeles county",
                {
                    "cached_area": "Los Angeles County",
                },
                "los angeles county",
            ),
        ],
    )
    def test_check_stations(self, value, expected):
        """Test that location string is correctly formatted based on given inputs."""
        assert _check_stations(**value) == expected

    def test_check_stations_raises_error_for_invalid_input(self):
        """Test that _check_stations raises TypeError for incomplete profile parameters."""

        invalid_profile_selections =                 {
                    "latitude": 34.4041,
                },
        with pytest.raises(
            TypeError,
            match="Location must be provided as either `station_name` or `cached_area` or `latitude` plus `longitude`",
        ):
            _check_stations("",invalid_profile_selections)

    def test_check_stations_raises_error_for_custom_list(self):
        """Test that _check_stations raises ValueError for list of custom station names."""

        invalid_profile_selections = (
            {
                "stations": [
                    "Custom Name 1",
                    "Custom Name 2",
                ],
            },
        )
        with pytest.raises(
            ValueError,
            match="If multiple stations are given, all must be HadISD stations.",
        ):
            _check_stations("", invalid_profile_selections)

    def test_check_stations_raises_error_for_mixed_list(self):
        """Test that _check_stations raises ValueError for list of custom and HadISD station names."""

        invalid_profile_selections = (
            {
                "stations": [
                    "Custom Station Name",
                    "Santa Barbara Municipal Airport (KSBA)",
                ],
            },
        )
        with pytest.raises(
            ValueError,
            match="If multiple stations are given, all must be HadISD stations.",
        ):
            _check_stations("", invalid_profile_selections)

    def test_check_stations_raises_error_for_custom_station_without_coordinates(self):
        """Test that _check_stations raises ValueError for a custom station provided without its associated latitude and longitude."""

        invalid_profile_selections = (
            {
                "stations": [
                    "Custom Station Name",
                ],
            },
        )
        with pytest.raises(
            ValueError,
            match="If a custom station name if given, its latitude and longitude must also be provided.",
        ):
            _check_stations("", invalid_profile_selections)
