"""
Explicitly test the DataParameters class and its methods.
This module contains unit tests for the DataParameters class, which is part of the climakitae.core.data_interface module.
These tests cover the initialization, default values, update methods, and retrieval of data parameters.
"""

from unittest.mock import MagicMock, Mock, patch

import geopandas as gpd
import pandas as pd
import pytest
from shapely.geometry import box

from climakitae.core.data_interface import DataParameters

VAR_DESC = pd.read_csv("climakitae/data/variable_descriptions.csv")


def mock_scenario_to_experiment_id(scenario, reverse=False):
    """Mock implementation of scenario_to_experiment_id with additional scenarios"""
    scenario_dict = {
        "Historical Reconstruction": "reanalysis",
        "Historical Climate": "historical",
        "SSP 2-4.5": "ssp245",
        "SSP 5-8.5": "ssp585",
        "SSP 3-7.0": "ssp370",
    }

    if reverse:
        scenario_dict = {v: k for k, v in scenario_dict.items()}

    # Handle special cases used in tests
    if scenario == "n/a" and not reverse:
        return "n/a"
    if scenario == "n/a" and reverse:
        return "n/a"

    return scenario_dict.get(scenario, scenario)


class TestDataParameters:
    """
    Tests for the DataParameters class and its methods.
    """

    @pytest.fixture
    def mock_data_interface(self):
        """Fixture to create a mocked DataInterface to avoid external dependencies."""
        with patch("climakitae.core.data_interface.DataInterface") as mock_di:
            # Create mock instance with required properties
            mock_instance = Mock()

            # Mock variable descriptions
            mock_instance.variable_descriptions = VAR_DESC

            # Mock data catalog
            mock_catalog = Mock()
            mock_catalog.search.return_value = mock_catalog
            mock_subset = mock_catalog

            # Use MagicMock instead of Mock for df to support magic methods like __getitem__
            mock_subset.df = MagicMock()

            # Configure mock dataframe columns with test values
            experiment_id_mock = MagicMock()
            experiment_id_mock.unique.return_value = ["historical", "ssp585"]

            source_id_mock = MagicMock()
            source_id_mock.unique.return_value = ["CESM2", "CNRM-CM6-1"]

            variable_id_mock = MagicMock()
            variable_id_mock.unique.return_value = ["tas", "pr"]

            # Set up the __getitem__ method to return appropriate column mocks
            mock_dict = {
                "experiment_id": experiment_id_mock,
                "source_id": source_id_mock,
                "variable_id": variable_id_mock,
            }
            mock_subset.df.__getitem__.side_effect = mock_dict.__getitem__
            mock_instance.data_catalog = mock_catalog

            # Mock warming level times
            mock_instance.warming_level_times = pd.DataFrame()

            # Mock stations
            stations_gdf = gpd.GeoDataFrame(
                {
                    "station": ["Station1", "Station2"],
                    "geometry": [box(0, 0, 1, 1), box(1, 1, 2, 2)],
                }
            )
            mock_instance.stations_gdf = stations_gdf

            # Mock geographies
            mock_geographies = Mock()
            mock_geographies.boundary_dict.return_value = {
                "none": {"entire domain": 0},
                "lat/lon": {"coordinate selection": 0},
                "states": {"CA": 0, "NV": 1},
                "CA counties": {"Alameda": 0},
            }
            mock_instance.geographies = mock_geographies

            # Return the mocked instance
            mock_di.return_value = mock_instance
            yield mock_instance

    def test_initialization_and_defaults(self, mock_data_interface):
        """
        Test that DataParameters initializes correctly with default values
        and properly connects to DataInterface.
        """
        # Create a DataParameters instance
        with patch(
            "climakitae.core.data_interface._get_user_options"
        ) as mock_get_options:
            mock_get_options.return_value = (
                ["historical"],  # scenario_options
                ["CESM2", "CNRM-CM6-1"],  # simulation_options
                ["tas", "pr"],  # unique_variable_ids
            )

            with patch(
                "climakitae.core.data_interface._get_variable_options_df"
            ) as mock_get_vars:
                # Add the 'variable_id' column to the DataFrame
                var_df = VAR_DESC[
                    (VAR_DESC["display_name"] == "Air Temperature at 2m")
                    & (VAR_DESC["timescale"] == "hourly")
                ].copy()
                print(var_df.columns)
                mock_get_vars.return_value = var_df

                with patch(
                    "climakitae.core.data_interface._get_var_ids"
                ) as mock_get_var_ids:
                    mock_get_var_ids.return_value = ["tas"]

                    # Mock the scenario_to_experiment_id function
                    with patch(
                        "climakitae.core.data_interface.scenario_to_experiment_id",
                        side_effect=mock_scenario_to_experiment_id,
                    ):
                        # Create the instance
                        params = DataParameters()

                        # Check initialization - fix this line
                        assert params.data_interface == mock_data_interface
                        assert params._data_catalog == mock_data_interface.data_catalog
                        assert params.variable == var_df["display_name"].values[0]
                        assert params.units == var_df["unit"].values[0]
                        assert params.colormap == var_df["colormap"].values[0]

                        # Check default parameter values
                        assert (
                            params.downscaling_method
                            == var_df["downscaling_method"].values[0]
                        )
                        assert params.data_type == "Gridded"
                        assert params.approach == "Time"
                        assert params.area_subset == "none"
                        assert params.scenario_ssp == []
                        assert params.scenario_historical == ["Historical Climate"]
                        assert params._data_warning == ""

    def test_update_methods(self):
        """
        Test that update methods correctly modify parameters based on dependencies.
        Focus on testing multiple update methods in a single test to verify interactions.
        """
        # Setup DataParameters instance with mocked internals
        with patch(
            "climakitae.core.data_interface._get_user_options"
        ) as mock_get_options:
            mock_get_options.return_value = (
                ["historical"],  # scenario_options
                ["CESM2", "CNRM-CM6-1"],  # simulation_options
                ["tas", "pr"],  # unique_variable_ids
            )

            with patch(
                "climakitae.core.data_interface._get_variable_options_df"
            ) as mock_get_vars:
                var_df = VAR_DESC[
                    VAR_DESC["display_name"].isin(
                        [
                            "Air Temperature at 2m",
                            "Precipitation",
                        ]
                    )
                    & (VAR_DESC["timescale"] == "hourly")
                ].copy()
                mock_get_vars.return_value = var_df

                with patch(
                    "climakitae.core.data_interface._get_var_ids"
                ) as mock_get_var_ids:
                    mock_get_var_ids.return_value = ["tas"]

                    # Add the missing patch for scenario_to_experiment_id
                    with patch(
                        "climakitae.core.data_interface.scenario_to_experiment_id",
                        side_effect=mock_scenario_to_experiment_id,
                    ):
                        # Create the instance
                        params = DataParameters()

                        # Test approach update method
                        params.approach = "Warming Level"
                        assert params.warming_level == [2.0]
                        assert params.scenario_ssp == ["n/a"]
                        assert params.scenario_historical == ["n/a"]

                        # Change back to Time approach
                        params.approach = "Time"
                        assert params.warming_level == ["n/a"]
                        assert params.scenario_historical == ["Historical Climate"]
                        assert "SSP" not in str(
                            params.scenario_ssp
                        )  # Should not contain SSP

                        # Test data type update
                        params.data_type = "Stations"
                        assert params.area_average == "n/a"
                        assert (
                            params.downscaling_method == "Dynamical"
                        )  # Enforced by _update_data_type_option_for_some_selections
                        assert (
                            params.timescale == "hourly"
                        )  # Enforced by _update_user_options

                        # Test resolution update with area subset
                        params = DataParameters()  # Reset
                        params.area_subset = "states"
                        params.resolution = "3 km"
                        assert "CA" in params.param["cached_area"].objects

                        # Test changing subset to lat/lon
                        params.latitude = (35, 40)
                        params.longitude = (-120, -115)
                        assert params.area_subset == "lat/lon"

    def test_unit_and_scenario_updates(self):
        """
        Test unit updates and scenario selection behavior.
        """
        # Setup DataParameters instance with mocked internals
        with patch(
            "climakitae.core.data_interface._get_user_options"
        ) as mock_get_options:
            mock_get_options.return_value = (
                ["historical", "reanalysis"],  # scenario_options
                ["CESM2", "CNRM-CM6-1"],  # simulation_options
                ["tas", "pr"],  # unique_variable_ids
            )

            with patch(
                "climakitae.core.data_interface._get_variable_options_df"
            ) as mock_get_vars:
                var_df = VAR_DESC[
                    VAR_DESC["display_name"].isin(
                        [
                            "Air Temperature at 2m",
                        ]
                    )
                    & (VAR_DESC["timescale"] == "hourly")
                ].copy()

                mock_get_vars.return_value = var_df

                with patch(
                    "climakitae.core.data_interface._get_var_ids"
                ) as mock_get_var_ids:
                    mock_get_var_ids.return_value = ["tas"]

                    # Add the missing patch for scenario_to_experiment_id
                    with patch(
                        "climakitae.core.data_interface.scenario_to_experiment_id",
                        side_effect=mock_scenario_to_experiment_id,
                    ):
                        # Create the instance
                        params = DataParameters()

                        # Test unit conversion options update
                        with patch.dict(
                            params.unit_options_dict, {"K": ["degC", "degF", "K"]}
                        ):
                            params._update_unit_options()
                            assert params.param["units"].objects == [
                                "degC",
                                "degF",
                                "K",
                            ]
                            assert params.units == "K"

                        # Test scenario update based on downscaling method
                        params.downscaling_method = "Statistical"
                        mock_get_options.return_value = (
                            [
                                "historical",
                                "ssp585",
                            ],  # Different scenarios for Statistical
                            ["CESM2"],  # Fewer models
                            ["tas"],  # Fewer variables
                        )

                        # Force an update to scenarios
                        params._update_scenarios()

                        # Verify time slice update based on scenario
                        params.scenario_historical = ["Historical Climate"]
                        params.scenario_ssp = ["SSP 5-8.5"]  # Using GUI-friendly name
                        params._update_time_slice_range()
                        assert (
                            params.time_slice[0]
                            == params.historical_climate_range_loca[0]
                        )
                        assert params.time_slice[1] == params.ssp_range[1]

    def test_retrieve_and_warnings(self):
        """
        Test the retrieve method and data warning behavior.
        """
        with patch(
            "climakitae.core.data_interface._get_user_options"
        ) as mock_get_options:
            mock_get_options.return_value = (
                [
                    "historical",
                ],  # scenario_options
                ["CESM2", "CNRM-CM6-1"],  # simulation_options
                ["tas", "pr"],  # unique_variable_ids
            )

            with patch(
                "climakitae.core.data_interface._get_variable_options_df"
            ) as mock_get_vars:
                var_df = VAR_DESC[
                    VAR_DESC["display_name"].isin(
                        [
                            "Air Temperature at 2m",
                        ]
                    )
                    & (VAR_DESC["timescale"] == "hourly")
                ].copy()
                mock_get_vars.return_value = var_df

                with patch(
                    "climakitae.core.data_interface._get_var_ids"
                ) as mock_get_var_ids:
                    mock_get_var_ids.return_value = ["tas"]

                    # Add the missing patch for scenario_to_experiment_id
                    with patch(
                        "climakitae.core.data_interface.scenario_to_experiment_id",
                        side_effect=mock_scenario_to_experiment_id,
                    ):
                        # Create the instance
                        params = DataParameters()

                        # Test data warnings
                        params.scenario_historical = ["Historical Climate"]
                        params.scenario_ssp = ["SSP 3-7.0"]
                        params.time_slice = (1970, 2030)  # Outside historical range
                        params._update_data_warning()
                        assert "time slice" in params._data_warning.lower()

                        # Test conflicting scenarios warning
                        params.scenario_historical = ["Historical Reconstruction"]
                        params._update_data_warning()
                        assert (
                            "Historical Reconstruction data is not available with SSP data"
                            in params._data_warning
                        )

                        # Test retrieve method
                        with patch(
                            "climakitae.core.data_interface.read_catalog_from_select"
                        ) as mock_read:
                            mock_data = Mock()
                            mock_data.nbytes = 6 * 10**9  # 6GB - should trigger warning
                            mock_read.return_value = mock_data

                            with patch("builtins.print") as mock_print:
                                _ = params.retrieve()
                                # Check warning for large data was printed
                                mock_print.assert_called()
                                any_size_warning = any(
                                    "large" in call[0][0].lower()
                                    for call in mock_print.call_args_list
                                )
                                assert any_size_warning

    def test_station_list_updates(self):
        """
        Test station list updates based on area selection.
        """
        with patch(
            "climakitae.core.data_interface._get_user_options"
        ) as mock_get_options:
            mock_get_options.return_value = (
                ["historical"],  # scenario_options
                ["CESM2"],  # simulation_options
                ["tas"],  # unique_variable_ids
            )

            with patch(
                "climakitae.core.data_interface._get_variable_options_df"
            ) as mock_get_vars:
                var_df = VAR_DESC[
                    VAR_DESC["display_name"].isin(
                        [
                            "Air Temperature at 2m",
                        ]
                    )
                    & (VAR_DESC["timescale"] == "hourly")
                ].copy()
                mock_get_vars.return_value = var_df

                with patch(
                    "climakitae.core.data_interface._get_var_ids"
                ) as mock_get_var_ids:
                    mock_get_var_ids.return_value = ["tas"]

                    # Create the instance
                    params = DataParameters()

                    # Test station updates
                    with patch(
                        "climakitae.core.data_interface._get_overlapping_station_names"
                    ) as mock_get_stations:
                        # Test with stations available
                        mock_get_stations.return_value = [
                            "Station1",
                            "Station2",
                        ]
                        params.data_type = "Stations"
                        params._update_station_list()
                        assert params.param["stations"].objects == [
                            "Station1",
                            "Station2",
                        ]
                        assert params.stations == ["Station1", "Station2"]

                        # Test with no stations available
                        mock_get_stations.return_value = []
                        params._update_station_list()
                        assert params.param["stations"].objects == [
                            "No stations available at this location"
                        ]
                        assert params.stations == [
                            "No stations available at this location"
                        ]

                        # Test with gridded data
                        params.data_type = "Gridded"
                        params._update_station_list()
                        assert params.param["stations"].objects == [
                            "Set data type to 'Station' to see options"
                        ]
                        assert params.stations == [
                            "Set data type to 'Station' to see options"
                        ]
