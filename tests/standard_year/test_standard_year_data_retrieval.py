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
    retrieve_profile_data,
    _handle_approach_params,
    _filter_by_ssp,
)


class TestRetrieveProfileData:
    """Test class for retrieve_profile_data function.

    Tests the function's ability to retrieve climate profile data based on
    various parameter configurations and location specifications.

    Attributes
    ----------
    mock_get_data : MagicMock
        Mock for get_data function.
    """

    def setup_method(self):
        """Set up test fixtures."""
        self.mock_get_data_patcher = patch(
            "climakitae.explore.standard_year_profile.get_data"
        )
        self.mock_get_data = self.mock_get_data_patcher.start()

    def teardown_method(self):
        """Clean up test fixtures."""
        self.mock_get_data_patcher.stop()

    def test_retrieve_profile_data_returns_tuple(self):
        """Test that retrieve_profile_data returns a tuple of two datasets.
        This is not a type test for what that tuple contains."""
        # Setup mock return values
        mock_historic = MagicMock(spec=xr.Dataset)
        mock_future = MagicMock(spec=xr.Dataset)
        self.mock_get_data.side_effect = [mock_historic, mock_future]

        # Execute function
        result = retrieve_profile_data(warming_level=[2.0])

        # Verify outcome: returns tuple of two datasets
        assert isinstance(result, tuple), "Should return a tuple"
        assert len(result) == 2, "Tuple should contain exactly 2 elements"
        historic_data, future_data = result
        assert historic_data == mock_historic, "First element should be historic data"
        assert future_data == mock_future, "Second element should be future data"

    def test_retrieve_profile_data_with_invalid_parameters_raises_error(self):
        """Test that retrieve_profile_data raises error for invalid parameter keys."""
        # Execute and verify outcome: should raise ValueError for invalid keys
        with pytest.raises(ValueError, match="Invalid input"):
            retrieve_profile_data(invalid_param="test", another_invalid=123)

    def test_retrieve_profile_data_with_float_warming_level_window_raises_error(self):
        """Test that retrieve_profile_data raises error for a warming_level_window input outside of range 5-25"""

        with pytest.raises(ValueError):
            retrieve_profile_data(
                variable="Air Temperature at 2m",
                resolution="3 km",
                warming_level=[1.5],
                cached_area="bay area",
                units="degC",
                warming_level_window=2,
            )

    def test_retrieve_profile_data_with_no_delta_returns_none_historic(self):
        """Test that retrieve_profile_data returns None for historic when no_delta=True."""
        # Setup mock return value
        mock_future = MagicMock(spec=xr.Dataset)
        self.mock_get_data.return_value = mock_future

        # Execute function with no_delta=True
        result = retrieve_profile_data(no_delta=True, warming_level=[2.0])

        # Verify outcome: historic data should be None when no_delta=True
        assert isinstance(result, tuple), "Should return a tuple"
        historic_data, future_data = result
        assert historic_data is None, "Historic data should be None when no_delta=True"
        assert future_data == mock_future, "Future data should be returned"

    def test_retrieve_profile_data_accepts_valid_parameters(self):
        """Test that retrieve_profile_data accepts valid parameter combinations."""
        # Setup mock return values
        mock_historic = MagicMock(spec=xr.Dataset)
        mock_future = MagicMock(spec=xr.Dataset)
        self.mock_get_data.side_effect = [mock_historic, mock_future]

        # Execute function with basic valid parameters
        result = retrieve_profile_data(warming_level=[1.5, 2.0])

        # Verify outcome: function should complete successfully with valid parameters
        assert isinstance(result, tuple), "Should return a tuple with valid parameters"
        assert len(result) == 2, "Should return tuple of two elements"

        # Verify get_data was called twice (once for historic, once for future)
        assert self.mock_get_data.call_count == 2, "Should call get_data twice"

    def test_retrieve_profile_data_with_complex_parameters(self):
        """Test that retrieve_profile_data handles complex parameter combinations."""
        # Setup mock return values
        mock_historic = MagicMock(spec=xr.Dataset)
        mock_future = MagicMock(spec=xr.Dataset)
        self.mock_get_data.side_effect = [mock_historic, mock_future]

        # Execute function with complex valid parameters that previously caused the bug
        result = retrieve_profile_data(
            variable="Air Temperature at 2m",
            resolution="45 km",
            warming_level=[1.5, 2.0],
            cached_area="bay area",
            units="degC",
        )

        # Verify outcome: function should complete successfully
        assert isinstance(result, tuple), "Should handle complex parameters"
        assert len(result) == 2, "Should return two datasets"
        historic_data, future_data = result
        assert historic_data == mock_historic, "Historic data should be returned"
        assert future_data == mock_future, "Future data should be returned"

    def test_retrieve_profile_data_time_based_approach(self):
        """Test that retrieve_profile_data accepts valid time-based approach parameters."""

        # Execute function with time-based approach and no scenario specified
        result = retrieve_profile_data(
            variable="Air Temperature at 2m",
            resolution="3 km",
            cached_area="bay area",
            units="degC",
            approach="Time",
            centered_year=2016,
        )

        # Verify outcome: function should complete successfully
        assert isinstance(result, tuple), "Should handle time-based approach"
        assert len(result) == 2, "Should return two datasets"

        # Execute function with time-based approach and scenario specified
        result = retrieve_profile_data(
            variable="Air Temperature at 2m",
            resolution="9 km",
            cached_area="bay area",
            units="degC",
            approach="Time",
            centered_year=2016,
            time_profile_scenario="SSP 2-4.5",
        )

        # Verify outcome: function should complete successfully
        assert isinstance(result, tuple), "Should handle time-based approach"
        assert len(result) == 2, "Should return two datasets"


class TestRetrieveProfileDataWithStations:
    """Test class for retrieve_profile_data with stations parameter.

    Tests that the stations parameter is correctly intercepted and converted
    to lat/lon coordinates with buffer before calling get_data.

    Attributes
    ----------
    mock_get_data : MagicMock
        Mock for get_data function.
    mock_stations_gdf : pd.DataFrame
        Mock GeoDataFrame with station data.
    """

    def setup_method(self):
        """Set up test fixtures."""
        # Create mock stations GeoDataFrame
        self.mock_stations_gdf = pd.DataFrame(
            {
                "station": [
                    "San Diego Lindbergh Field (KSAN)",
                    "Los Angeles International Airport (KLAX)",
                ],
                "LAT_Y": [32.7336, 33.9416],
                "LON_X": [-117.1831, -118.4085],
            }
        )

    def test_retrieve_profile_data_converts_stations_to_lat_lon(self):
        """Test that stations parameter is converted to lat/lon before calling get_data."""
        # Setup mocks
        with (
            patch("climakitae.explore.standard_year_profile.get_data") as mock_get_data,
            patch(
                "climakitae.explore.standard_year_profile.DataInterface"
            ) as mock_data_interface_class,
            patch("builtins.print"),
        ):
            # Setup DataInterface mock
            mock_instance = MagicMock()
            mock_instance.stations_gdf = self.mock_stations_gdf
            mock_data_interface_class.return_value = mock_instance

            # Setup get_data mock
            mock_dataset = MagicMock(spec=xr.Dataset)
            mock_get_data.return_value = mock_dataset

            # Execute function with stations parameter
            retrieve_profile_data(
                stations=["San Diego Lindbergh Field (KSAN)"], warming_level=[2.0]
            )

            # Verify outcome: get_data should be called with lat/lon, not stations
            assert mock_get_data.call_count >= 1, "Should call get_data"

            # Check that lat/lon were passed in at least one call
            found_lat_lon = False
            for call in mock_get_data.call_args_list:
                kwargs = call.kwargs if hasattr(call, "kwargs") else call[1]
                if "latitude" in kwargs and "longitude" in kwargs:
                    found_lat_lon = True
                    # Verify stations was NOT passed
                    assert (
                        "stations" not in kwargs
                    ), "stations parameter should not be passed to get_data"
                    # Verify lat/lon are tuples with min/max
                    assert isinstance(
                        kwargs["latitude"], tuple
                    ), "latitude should be a tuple"
                    assert isinstance(
                        kwargs["longitude"], tuple
                    ), "longitude should be a tuple"
                    assert (
                        len(kwargs["latitude"]) == 2
                    ), "latitude should have min and max"
                    assert (
                        len(kwargs["longitude"]) == 2
                    ), "longitude should have min and max"

            assert found_lat_lon, "At least one call should include lat/lon parameters"

    def test_retrieve_profile_data_applies_correct_buffer_to_stations(self):
        """Test that stations are converted with correct 0.02 degree buffer."""
        # Setup mocks
        with (
            patch("climakitae.explore.standard_year_profile.get_data") as mock_get_data,
            patch(
                "climakitae.explore.standard_year_profile.DataInterface"
            ) as mock_data_interface_class,
            patch("builtins.print"),
        ):
            # Setup DataInterface mock
            mock_instance = MagicMock()
            mock_instance.stations_gdf = self.mock_stations_gdf
            mock_data_interface_class.return_value = mock_instance

            # Setup get_data mock
            mock_dataset = MagicMock(spec=xr.Dataset)
            mock_get_data.return_value = mock_dataset

            # Execute function with stations parameter
            retrieve_profile_data(
                stations=["San Diego Lindbergh Field (KSAN)"], warming_level=[2.0]
            )

            # Verify outcome: lat/lon should have 0.02 buffer applied
            # San Diego: lat=32.7336, lon=-117.1831
            expected_lat_min = 32.7336 - 0.02
            expected_lat_max = 32.7336 + 0.02
            expected_lon_min = -117.1831 - 0.02
            expected_lon_max = -117.1831 + 0.02

            # Find the call with lat/lon
            for call in mock_get_data.call_args_list:
                kwargs = call.kwargs if hasattr(call, "kwargs") else call[1]
                if "latitude" in kwargs:
                    lat_bounds = kwargs["latitude"]
                    lon_bounds = kwargs["longitude"]

                    assert (
                        abs(lat_bounds[0] - expected_lat_min) < 1e-6
                    ), "Should apply 0.02 buffer to min latitude"
                    assert (
                        abs(lat_bounds[1] - expected_lat_max) < 1e-6
                    ), "Should apply 0.02 buffer to max latitude"
                    assert (
                        abs(lon_bounds[0] - expected_lon_min) < 1e-6
                    ), "Should apply 0.02 buffer to min longitude"
                    assert (
                        abs(lon_bounds[1] - expected_lon_max) < 1e-6
                    ), "Should apply 0.02 buffer to max longitude"

    def test_retrieve_profile_data_cached_area_takes_priority_over_stations(self):
        """Test that cached_area parameter takes priority over stations."""
        # Setup mocks
        with (
            patch("climakitae.explore.standard_year_profile.get_data") as mock_get_data,
            patch(
                "climakitae.explore.standard_year_profile.DataInterface"
            ) as mock_data_interface_class,
            patch("builtins.print"),
        ):
            # Setup DataInterface mock
            mock_instance = MagicMock()
            mock_instance.stations_gdf = self.mock_stations_gdf
            mock_data_interface_class.return_value = mock_instance

            # Setup get_data mock
            mock_dataset = MagicMock(spec=xr.Dataset)
            mock_get_data.return_value = mock_dataset

            # Execute function with both cached_area and stations
            retrieve_profile_data(
                cached_area="Los Angeles",
                stations=["San Diego Lindbergh Field (KSAN)"],
                warming_level=[2.0],
            )

            # Verify outcome: cached_area should be used, stations should be ignored
            for call in mock_get_data.call_args_list:
                kwargs = call.kwargs if hasattr(call, "kwargs") else call[1]
                assert (
                    kwargs.get("cached_area") == "Los Angeles"
                ), "Should use cached_area"
                # Stations should not be converted to lat/lon when cached_area is present
                # (cached_area takes priority)

    def test_retrieve_profile_data_explicit_lat_lon_takes_priority_over_stations(self):
        """Test that explicit lat/lon parameters take priority over stations."""
        # Setup mocks
        with (
            patch("climakitae.explore.standard_year_profile.get_data") as mock_get_data,
            patch(
                "climakitae.explore.standard_year_profile.DataInterface"
            ) as mock_data_interface_class,
            patch("builtins.print"),
        ):
            # Setup DataInterface mock
            mock_instance = MagicMock()
            mock_instance.stations_gdf = self.mock_stations_gdf
            mock_data_interface_class.return_value = mock_instance

            # Setup get_data mock
            mock_dataset = MagicMock(spec=xr.Dataset)
            mock_get_data.return_value = mock_dataset

            # Execute function with both explicit lat/lon and stations
            explicit_lat = (32.0, 34.0)
            explicit_lon = (-118.0, -116.0)
            retrieve_profile_data(
                latitude=explicit_lat,
                longitude=explicit_lon,
                stations=["San Diego Lindbergh Field (KSAN)"],
                warming_level=[2.0],
            )

            # Verify outcome: explicit lat/lon should be used, stations should be ignored
            for call in mock_get_data.call_args_list:
                kwargs = call.kwargs if hasattr(call, "kwargs") else call[1]
                if "latitude" in kwargs:
                    assert (
                        kwargs["latitude"] == explicit_lat
                    ), "Should use explicit latitude"
                    assert (
                        kwargs["longitude"] == explicit_lon
                    ), "Should use explicit longitude"


class TestHandleApproachParams:
    """Test class for _handle_approach_params helper function.

    Tests the function's ability to update parameters for valid inputs

    """

    @pytest.mark.parametrize(
        "input_value,expected",
        [
            (
                {
                    "var_id": "t2",
                    "q": 0.5,
                    "gwl": 1.5,
                    "location": "sacramento county",
                    "no_delta": False,
                },
                {
                    "var_id": "t2",
                    "q": 0.5,
                    "gwl": 1.5,
                    "location": "sacramento county",
                    "no_delta": False,
                },
            ),
            (
                {
                    "var_id": "t2",
                    "q": 0.5,
                    "location": "sacramento county",
                    "no_delta": False,
                },
                {
                    "var_id": "t2",
                    "q": 0.5,
                    "location": "sacramento county",
                    "no_delta": False,
                },
            ),
            (
                {
                    "var_id": "t2",
                    "q": 0.5,
                    "warming_level_window": 5,
                    "location": "sacramento county",
                    "no_delta": False,
                },
                {
                    "var_id": "t2",
                    "q": 0.5,
                    "warming_level_window": 5,
                    "location": "sacramento county",
                    "no_delta": False,
                },
            ),
            (
                {
                    "var_id": "t2",
                    "q": 0.5,
                    "location": "sacramento county",
                    "no_delta": False,
                    "approach": "Time",
                    "centered_year": 2016,
                },
                {
                    "var_id": "t2",
                    "q": 0.5,
                    "location": "sacramento county",
                    "no_delta": False,
                    "approach": "Warming Level",
                    "centered_year": 2016,
                    "warming_level": [1.12],
                    "time_profile_scenario": "SSP 3-7.0",
                },
            ),
            (
                {
                    "var_id": "t2",
                    "q": 0.5,
                    "warming_level_window": 5,
                    "location": "sacramento county",
                    "no_delta": False,
                    "approach": "Time",
                    "centered_year": 2016,
                },
                {
                    "var_id": "t2",
                    "q": 0.5,
                    "warming_level_window": 5,
                    "location": "sacramento county",
                    "no_delta": False,
                    "approach": "Warming Level",
                    "centered_year": 2016,
                    "warming_level": [1.12],
                    "time_profile_scenario": "SSP 3-7.0",
                },
            ),
            (
                {
                    "var_id": "t2",
                    "q": 0.5,
                    "gwl": 1.5,
                    "warming_level_window": 5,
                    "location": "sacramento county",
                    "no_delta": False,
                    "approach": "Warming Level",
                },
                {
                    "var_id": "t2",
                    "q": 0.5,
                    "gwl": 1.5,
                    "warming_level_window": 5,
                    "location": "sacramento county",
                    "no_delta": False,
                    "approach": "Warming Level",
                },
            ),
            (
                {
                    "var_id": "t2",
                    "q": 0.5,
                    "gwl": 1.5,
                    "location": "sacramento county",
                    "no_delta": False,
                    "approach": "Warming Level",
                },
                {
                    "var_id": "t2",
                    "q": 0.5,
                    "gwl": 1.5,
                    "location": "sacramento county",
                    "no_delta": False,
                    "approach": "Warming Level",
                },
            ),
            (
                {
                    "var_id": "t2",
                    "q": 0.5,
                    "warming_level_window": 5,
                    "location": "sacramento county",
                    "no_delta": False,
                    "approach": "Warming Level",
                },
                {
                    "var_id": "t2",
                    "q": 0.5,
                    "warming_level_window": 5,
                    "location": "sacramento county",
                    "no_delta": False,
                    "approach": "Warming Level",
                },
            ),
            (
                {
                    "var_id": "t2",
                    "q": 0.5,
                    "warming_level_window": 5,
                    "location": "sacramento county",
                    "no_delta": False,
                    "approach": "Time",
                    "centered_year": 2030,
                    "time_profile_scenario": "SSP 2-4.5",
                    "resolution": "9 km",
                },
                {
                    "var_id": "t2",
                    "q": 0.5,
                    "warming_level_window": 5,
                    "location": "sacramento county",
                    "no_delta": False,
                    "approach": "Warming Level",
                    "centered_year": 2030,
                    "time_profile_scenario": "SSP 2-4.5",
                    "resolution": "9 km",
                    "warming_level": [1.48],
                },
            ),
            (
                {
                    "var_id": "t2",
                    "q": 0.5,
                    "warming_level_window": 5,
                    "location": "sacramento county",
                    "no_delta": False,
                    "approach": "Time",
                    "centered_year": 1980,
                    "time_profile_scenario": "SSP 2-4.5",
                    "resolution": "9 km",
                },
                {
                    "var_id": "t2",
                    "q": 0.5,
                    "warming_level_window": 5,
                    "location": "sacramento county",
                    "no_delta": False,
                    "approach": "Warming Level",
                    "centered_year": 1980,
                    "time_profile_scenario": "SSP 2-4.5",
                    "resolution": "9 km",
                    "warming_level": [0.35],
                },
            ),
            (
                {
                    "var_id": "t2",
                    "q": 0.5,
                    "warming_level_window": 5,
                    "location": "sacramento county",
                    "no_delta": False,
                    "approach": "Time",
                    "centered_year": 1980,
                    "resolution": "45 km",
                },
                {
                    "var_id": "t2",
                    "q": 0.5,
                    "warming_level_window": 5,
                    "location": "sacramento county",
                    "no_delta": False,
                    "approach": "Warming Level",
                    "centered_year": 1980,
                    "time_profile_scenario": "SSP 3-7.0",
                    "resolution": "45 km",
                    "warming_level": [0.35],
                },
            ),
        ],
    )
    def test_handle_approach_params(self, input_value, expected):
        """Test that parameters are correctly modified based on given inputs."""
        assert _handle_approach_params(**input_value) == expected


class TestHandleApproachParamsInvalidInputs:
    """Test class for _handle_approach_params function.

    Tests the function's ability to field invalid approach parameter combinations.

    """

    def test_handle_approach_params_with_invalid_centered_year_raises_error(self):
        """Test that _handle_approach_params raises error for 'centered_year' outside of 2015-2099"""
        with pytest.raises(ValueError):
            retrieve_profile_data(
                variable="Air Temperature at 2m",
                resolution="3 km",
                cached_area="bay area",
                units="degF",
                approach="Time",
                centered_year=2014,
            )

    def test_handle_approach_params_with_invalid_centered_year_raises_error(self):
        """Test that _handle_approach_params raises error for 'warming_level' provided in addition to time-based approach inputs"""
        with pytest.raises(ValueError):
            retrieve_profile_data(
                variable="Air Temperature at 2m",
                resolution="3 km",
                cached_area="bay area",
                units="degF",
                warming_level=[1.5],
                approach="Time",
                centered_year=2016,
            )

    def test_handle_approach_params_with_missing_centered_year_raises_error(self):
        """Test that _handle_approach_params raises error if 'centered_year' not provided with 'approach'='Time'"""
        with pytest.raises(ValueError):
            retrieve_profile_data(
                variable="Air Temperature at 2m",
                resolution="3 km",
                cached_area="bay area",
                units="degF",
                approach="Time",
            )

    def test_handle_approach_params_with_centered_year_and_warming_level_approach_raises_error(
        self,
    ):
        """Test that _handle_approach_params raises error for 'centered_year' with 'approach'='Warming Level'"""
        with pytest.raises(ValueError):
            retrieve_profile_data(
                variable="Air Temperature at 2m",
                resolution="3 km",
                cached_area="bay area",
                units="degF",
                approach="Warming Level",
                centered_year=2016,
            )

    def test_handle_approach_params_with_centered_year_and_no_approach_raises_error(
        self,
    ):
        """Test that _handle_approach_params raises error for 'centered_year' with no 'approach'"""
        with pytest.raises(ValueError):
            retrieve_profile_data(
                variable="Air Temperature at 2m",
                resolution="3 km",
                cached_area="bay area",
                units="degF",
                centered_year=2016,
            )

    def test_handle_approach_params_with_invalid_approach_raises_error(self):
        """Test that _handle_approach_params raises error for invalid 'approach' input"""
        with pytest.raises(ValueError):
            retrieve_profile_data(
                variable="Air Temperature at 2m",
                resolution="3 km",
                cached_area="bay area",
                units="degF",
                approach="other",
                centered_year=2016,
            )

    def test_handle_approach_params_with_invalid_scenario_raises_error(self):
        """Test that _handle_approach_params raises error for invalid 'scenario' and "resolution" combination"""
        with pytest.raises(ValueError):
            retrieve_profile_data(
                variable="Air Temperature at 2m",
                resolution="3 km",
                cached_area="bay area",
                units="degF",
                approach="Time",
                centered_year=2016,
                scenario="SSP 5-8.5",
            )
        with pytest.raises(ValueError):
            retrieve_profile_data(
                variable="Air Temperature at 2m",
                resolution="3 km",
                cached_area="bay area",
                units="degF",
                approach="Time",
                centered_year=2016,
                scenario="SSP 2-4.5",
            )


class TestFilterBySSP:
    """Test class for _filter_by_ssp(), which filters retrieved data by SSP"""

    def setup_method(self):
        """Set up test fixtures."""
        # Create smaller sample for faster testing (just enough for the algorithm)
        time_delta = pd.date_range(
            "2020-01-01", periods=8760, freq="h"
        )  # 1 year of hourly data
        warming_levels = [1.5]
        simulations = [
            "WRF_CESM2_r11i1p1f1_historical+ssp245",
            "WRF_CESM2_r11i1p1f1_historical+ssp370",
            "WRF_CESM2_r11i1p1f1_historical+ssp585",
        ]

        # Create test data with proper dimensions
        data = np.random.rand(len(warming_levels), len(time_delta), len(simulations))

        self.sample_data = xr.DataArray(
            data,
            dims=["warming_level", "time_delta", "simulation"],
            coords={
                "warming_level": warming_levels,
                "time_delta": time_delta,
                "simulation": simulations,
            },
            attrs={"units": "degF", "variable_id": "t2"},
        )

    def test_filter_by_ssp_returns_correct_simulation(self):
        """Test that _filter_by_spp returns data from desired simulation."""
        # Execute function
        result = _filter_by_ssp(self.sample_data, scenario="SSP 2-4.5")

        # Verify the result contains only the target simulation
        simulations = np.array(["WRF_CESM2_r11i1p1f1_historical+ssp245"])
        assert np.array_equal(
            result.simulation.values, simulations
        ), "Incorrect simulation(s) returned"
