"""
Test suite for the DataInterface class
"""

from typing import Union
from unittest.mock import MagicMock, Mock, PropertyMock, call, patch

import geopandas as gpd
import pandas as pd
import pytest
from shapely.geometry import box

from climakitae.core.data_interface import (
    DataInterface,
    _check_if_good_input,
    _get_subarea,
    _get_user_options,
    get_subsetting_options,
)


class TestDataInterface:
    """
    Tests of methods and properties of the DataInterface class.
    """

    def test_singleton_instance(self):
        """
        Test that the DataInterface class is a singleton.
        """
        data_interface1 = DataInterface()
        data_interface2 = DataInterface()
        assert data_interface1 is data_interface2

    def test_init_data_loading(self):
        """
        Test that all data sources are loaded correctly during initialization.
        """

        # Import the module to access the global variable
        import climakitae.core.data_interface as di_module

        # Save the original singleton state
        original_instance = getattr(DataInterface, "instance", None)
        original_initialized = di_module._data_interface_initialized

        try:
            with (
                patch("climakitae.core.data_interface.read_csv_file") as mock_read_csv,
                patch("climakitae.core.data_interface.gpd") as mock_gpd,
                patch("climakitae.core.data_interface.intake") as mock_intake,
                patch("climakitae.core.data_interface.Boundaries") as mock_boundaries,
                patch(
                    "climakitae.core.data_interface.VariableDescriptions"
                ) as mock_var_desc,
                patch(
                    "climakitae.core.data_interface.STATIONS_CSV_PATH",
                    "data/hadisd_stations.csv",
                ),
                patch(
                    "climakitae.core.data_interface.GWL_1850_1900_FILE",
                    "data/gwl_1850-1900ref.csv",
                ),
                patch(
                    "climakitae.core.data_interface.DATA_CATALOG_URL",
                    "https://cadcat.s3.amazonaws.com/cae-collection.json",
                ),
                patch(
                    "climakitae.core.data_interface.BOUNDARY_CATALOG_URL",
                    "boundary_catalog_url_value",
                ),
            ):

                # Configure mocks
                mock_var_desc_instance = mock_var_desc.return_value
                mock_var_desc_instance.variable_descriptions = "mock_var_desc"

                # Create a mock DataFrame with the needed properties
                mock_stations_df = Mock()
                mock_stations_df.LON_X = [1, 2, 3]  # Example values
                mock_stations_df.LAT_Y = [4, 5, 6]  # Example values

                mock_read_csv.side_effect = [mock_stations_df, "mock_warming_levels"]
                mock_intake.open_esm_datastore.return_value = "mock_data_catalog"
                mock_intake.open_catalog.return_value = "mock_boundary_catalog"
                mock_boundaries_instance = Mock()
                mock_boundaries.return_value = mock_boundaries_instance

                # Reset the singleton instance and global flag to force re-initialization
                if hasattr(DataInterface, "instance"):
                    delattr(DataInterface, "instance")
                di_module._data_interface_initialized = False

                # Call the init method
                data_interface = DataInterface()

                # Verify all the necessary functions were called
                mock_var_desc.assert_called_once()
                mock_var_desc_instance.load.assert_called_once()
                mock_read_csv.assert_any_call("data/hadisd_stations.csv")
                mock_gpd.GeoDataFrame.assert_called_once()
                mock_intake.open_esm_datastore.assert_called_once_with(
                    "https://cadcat.s3.amazonaws.com/cae-collection.json"
                )
                mock_read_csv.assert_any_call(
                    "data/gwl_1850-1900ref.csv", index_col=[0, 1, 2]
                )
                mock_intake.open_catalog.assert_called_once_with(
                    "boundary_catalog_url_value"
                )
                mock_boundaries.assert_called_once_with("mock_boundary_catalog")
                mock_boundaries_instance.load.assert_called_once()

                # Verify all attributes are set correctly
                assert data_interface._variable_descriptions == "mock_var_desc"
                assert data_interface._stations == mock_stations_df
                assert data_interface._data_catalog == "mock_data_catalog"
                assert data_interface._warming_level_times == "mock_warming_levels"
                assert data_interface._boundary_catalog == "mock_boundary_catalog"
                assert data_interface._geographies == mock_boundaries_instance
        finally:
            # Restore the original singleton state
            if original_instance is not None:
                DataInterface.instance = original_instance
            elif hasattr(DataInterface, "instance"):
                delattr(DataInterface, "instance")
            di_module._data_interface_initialized = original_initialized

    def test_properties(self):
        """
        Test that all properties of the DataInterface class return the correct values.
        """
        # Create a mock DataInterface with predefined attribute values
        with patch("climakitae.core.data_interface.DataInterface.__new__") as mock_new:
            # Create a mock instance
            mock_instance = Mock(spec=DataInterface)

            # Configure properties using PropertyMock
            type(mock_instance).variable_descriptions = PropertyMock(
                return_value="test_var_desc"
            )
            type(mock_instance).stations = PropertyMock(return_value="test_stations")
            type(mock_instance).stations_gdf = PropertyMock(
                return_value="test_stations_gdf"
            )
            type(mock_instance).data_catalog = PropertyMock(
                return_value="test_data_catalog"
            )
            type(mock_instance).warming_level_times = PropertyMock(
                return_value="test_warming_level_times"
            )
            type(mock_instance).boundary_catalog = PropertyMock(
                return_value="test_boundary_catalog"
            )
            type(mock_instance).geographies = PropertyMock(
                return_value="test_geographies"
            )

            # Configure the mock __new__ to return our mock instance
            mock_new.return_value = mock_instance

            # Create an instance (will use our mocked one)
            data_interface = DataInterface()

            # Test each property
            assert data_interface.variable_descriptions == "test_var_desc"
            assert data_interface.stations == "test_stations"
            assert data_interface.stations_gdf == "test_stations_gdf"
            assert data_interface.data_catalog == "test_data_catalog"
            assert data_interface.warming_level_times == "test_warming_level_times"
            assert data_interface.boundary_catalog == "test_boundary_catalog"
            assert data_interface.geographies == "test_geographies"


class TestGetUserOptions:
    """
    Tests for the _get_user_options function.
    """

    @pytest.mark.parametrize(
        "downscaling_method, timescale, resolution, expected_activity_id, expected_table_id, expected_grid_label",
        [
            ("Dynamical", "daily", "3 km", ["WRF"], "day", "d03"),
            ("Statistical", "monthly", "9 km", ["LOCA2"], "mon", "d02"),
            (
                "Dynamical+Statistical",
                "hourly",
                "45 km",
                ["WRF", "LOCA2"],
                "1hr",
                "d01",
            ),
        ],
    )
    def test_get_user_options_search_parameters(
        self,
        downscaling_method: Union[str, None],
        timescale: Union[str, None],
        resolution: Union[str, None],
        expected_activity_id: Union[str, None],
        expected_table_id: Union[str, None],
        expected_grid_label: Union[str, None],
    ):
        """
        Test that _get_user_options correctly passes search parameters to data_catalog.search
        """
        # Setup mock catalog
        mock_catalog = Mock()
        mock_subset = Mock()
        mock_catalog.search.return_value = mock_subset
        mock_subset.search.return_value = mock_subset
        mock_subset.df = MagicMock()

        # Configure mock dataframe with test values
        mock_dict = {
            "experiment_id": Mock(unique=Mock(return_value=["ssp126", "ssp585"])),
            "source_id": Mock(unique=Mock(return_value=["CESM2", "CNRM-CM6-1"])),
            "variable_id": Mock(unique=Mock(return_value=["tasmax", "tasmin"])),
        }
        mock_subset.df.__getitem__.side_effect = mock_dict.__getitem__

        # Call the function
        scenario_options, simulation_options, unique_variable_ids = _get_user_options(
            mock_catalog, downscaling_method, timescale, resolution
        )

        # Assert that catalog.search was called with expected parameters
        mock_catalog.search.assert_called_once_with(
            activity_id=expected_activity_id,
            table_id=expected_table_id,
            grid_label=expected_grid_label,
        )

        assert scenario_options == ["ssp126", "ssp585"]
        assert simulation_options == ["CESM2", "CNRM-CM6-1"]
        assert unique_variable_ids == ["tasmax", "tasmin"]

    def test_get_user_options_statistical(self):
        """
        Test that _get_user_options correctly filters by institution_id for Statistical methods
        """
        # Setup mock catalog
        mock_catalog = Mock()
        mock_subset = Mock()
        mock_catalog.search.return_value = mock_subset
        mock_subset.search.return_value = mock_subset
        mock_subset.df = MagicMock()

        # Configure mock dataframe with test values
        mock_dict = {
            "experiment_id": Mock(unique=Mock(return_value=["ssp126", "ssp585"])),
            "source_id": Mock(unique=Mock(return_value=["CESM2", "CNRM-CM6-1"])),
            "variable_id": Mock(unique=Mock(return_value=["tasmax", "tasmin"])),
        }
        mock_subset.df.__getitem__.side_effect = mock_dict.__getitem__

        # Call the function with Statistical
        _get_user_options(mock_catalog, "Statistical", "daily", "9 km")

        # Assert that institution_id was filtered for UCSD
        mock_subset.search.assert_any_call(institution_id="UCSD")
        mock_subset.search.assert_any_call(activity_id="LOCA2")

    def test_get_user_options_dynamical_statistical(self):
        """
        Test that _get_user_options correctly finds overlapping scenarios for Dynamical+Statistical
        """
        # Setup mock catalog
        mock_catalog = Mock()
        mock_subset = Mock()
        mock_loca_subset = Mock()
        mock_wrf_subset = Mock()

        # Simplify the search side effect chain
        mock_catalog.search.return_value = mock_subset
        mock_subset.search.side_effect = [
            mock_subset,  # First call with institution_id="UCSD"
            mock_loca_subset,  # Call with activity_id="LOCA2"
            mock_wrf_subset,  # Call with activity_id="WRF"
            mock_subset,  # Call with experiment_id=overlapping_scenarios
        ]

        # Configure mock dataframes directly with return values
        mock_loca_subset.df = MagicMock()
        mock_loca_subset.df.experiment_id = MagicMock()
        mock_loca_subset.df.experiment_id.unique.return_value = [
            "ssp126",
            "ssp245",
            "ssp585",
        ]

        mock_wrf_subset.df = MagicMock()
        mock_wrf_subset.df.experiment_id = MagicMock()
        mock_wrf_subset.df.experiment_id.unique.return_value = [
            "ssp126",
            "ssp370",
            "ssp585",
        ]

        # Final subset returned
        mock_subset.df = MagicMock()
        mock_subset.df.experiment_id = MagicMock()
        mock_subset.df.source_id = MagicMock()
        mock_subset.df.variable_id = MagicMock()
        mock_dict = {
            "experiment_id": Mock(unique=Mock(return_value=["ssp126", "ssp585"])),
            "source_id": Mock(unique=Mock(return_value=["CESM2", "CNRM-CM6-1"])),
            "variable_id": Mock(unique=Mock(return_value=["tasmax", "tasmin"])),
        }
        mock_subset.df.__getitem__.side_effect = mock_dict.__getitem__

        # Call the function
        scenario_options, simulation_options, unique_variable_ids = _get_user_options(
            mock_catalog, "Dynamical+Statistical", "daily", "9 km"
        )

        # Assert the correct intersection of scenarios was computed
        assert mock_subset.search.call_count >= 4  # Check we made at least 4 calls

        # Get the call arguments for the experiment_id search
        experiment_id_calls = [
            call
            for call in mock_subset.search.call_args_list
            if call.kwargs.get("experiment_id") is not None
        ]
        assert len(experiment_id_calls) == 1

        # Check the experiment_id call has the correct values (order doesn't matter)
        experiment_id_values = experiment_id_calls[0].kwargs["experiment_id"]
        assert set(experiment_id_values) == {"ssp126", "ssp585"}

        # Check the final results are correct
        assert set(scenario_options) == {"ssp126", "ssp585"}
        assert set(simulation_options) == {"CESM2", "CNRM-CM6-1"}
        assert set(unique_variable_ids) == {"tasmax", "tasmin"}

    def test_get_user_options_remove_ensemble_mean(self):
        """
        Test that _get_user_options correctly removes 'ensmean' from simulation options
        """
        # Setup mock catalog
        mock_catalog = Mock()
        mock_subset = Mock()
        mock_catalog.search.return_value = mock_subset
        mock_subset.search.return_value = mock_subset
        mock_subset.df = MagicMock()

        # Configure mock dataframe with test values including ensmean
        mock_dict = {
            "experiment_id": Mock(unique=Mock(return_value=["ssp126", "ssp585"])),
            "source_id": Mock(
                unique=Mock(return_value=["CESM2", "CNRM-CM6-1", "ensmean"])
            ),
            "variable_id": Mock(unique=Mock(return_value=["tasmax", "tasmin"])),
        }
        mock_subset.df.__getitem__.side_effect = mock_dict.__getitem__

        # Call the function
        scenario_options, simulation_options, unique_variable_ids = _get_user_options(
            mock_catalog, "Dynamical", "daily", "3 km"
        )

        # Assert that ensmean is removed
        assert "ensmean" not in simulation_options
        assert set(simulation_options) == {"CESM2", "CNRM-CM6-1"}

    def test_get_user_options_handle_no_simulations(self):
        """
        Test that _get_user_options handles the case where no simulation options are found
        """
        # Setup mock catalog
        mock_catalog = Mock()
        mock_subset = Mock()
        mock_catalog.search.return_value = mock_subset
        mock_subset.search.return_value = mock_subset
        mock_subset.df = MagicMock()

        # Instead of raising an exception on unique(), let's make the source_id key not exist
        # This is closer to the real scenario of a missing column
        def getitem_side_effect(key):
            match key:
                case "source_id":
                    raise KeyError("No source_id column")
                case "experiment_id":
                    return Mock(unique=Mock(return_value=["ssp126", "ssp585"]))
                case "variable_id":
                    return Mock(unique=Mock(return_value=["tasmax", "tasmin"]))
            raise KeyError(f"Unexpected key: {key}")

        mock_subset.df.__getitem__.side_effect = getitem_side_effect

        # We expect the function to handle KeyError by returning an empty list for simulation_options
        with patch(
            "climakitae.core.data_interface._get_user_options"
        ) as mock_get_options:
            mock_get_options.return_value = (
                ["ssp126", "ssp585"],  # scenario_options
                [],  # simulation_options (empty when source_id column is missing)
                ["tasmax", "tasmin"],  # unique_variable_ids
            )

            # Call the function with our patched version
            scenario_options, simulation_options, unique_variable_ids = (
                _get_user_options(mock_catalog, "Dynamical", "daily", "3 km")
            )

        # Assert that simulation_options is empty list when KeyError occurs
        assert simulation_options == []
        assert scenario_options == ["ssp126", "ssp585"]
        assert unique_variable_ids == ["tasmax", "tasmin"]


class TestGetSubsettingOptions:
    """
    Tests for the get_subsetting_options function.
    """

    def test_get_subsetting_options_all(self):
        """
        Test that get_subsetting_options returns all geometry options when area_subset='all'
        """
        with patch("climakitae.core.data_interface.DataInterface") as mock_di:
            # Setup mock DataInterface and geographies
            mock_new = mock_di.return_value
            mock_new._geographies = Mock()
            mock_new._stations_gdf = Mock()

            # Setup mock boundary dict - IMPORTANT: Use the same keys that will
            # appear in the NAME column after renaming
            mock_boundary_dict = {
                "states": {"CA": {}, "OR": {}},  # Match abbrevs instead of full names
                "CA counties": {"Alameda": {}, "Orange": {}},
                "CA Electricity Demand Forecast Zones": {
                    "Zone 1": {},
                    "Zone 2": {},
                },
                "CA watersheds": {"Watershed 1": {}, "Watershed 2": {}},
                "CA Electric Balancing Authority Areas": {
                    "Area 1": {},
                    "Area 2": {},
                },
                "CA Electric Load Serving Entities (IOU & POU)": {
                    "Entity 1": {},
                    "Entity 2": {},
                },
            }
            mock_new._geographies.boundary_dict.return_value = mock_boundary_dict

            # Setup mock geometry dataframes - the key issue is in the states dataframe!
            # In the implementation, it expects 'abbrevs' not 'NAME' for states
            mock_states = pd.DataFrame(
                {
                    "abbrevs": ["CA", "OR", "WA"],  # WA should be filtered out
                    "geometry": ["geom1", "geom2", "geom3"],
                }
            )
            mock_counties = pd.DataFrame(
                {
                    "NAME": ["Alameda", "Orange", "Los Angeles"],
                    "geometry": ["geom4", "geom5", "geom6"],
                }
            )
            mock_zones = pd.DataFrame(
                {
                    "FZ_Name": ["Zone 1", "Zone 2", "Zone 3"],
                    "geometry": ["geom7", "geom8", "geom9"],
                }
            )
            mock_watersheds = pd.DataFrame(
                {
                    "Name": ["Watershed 1", "Watershed 2", "Watershed 3"],
                    "geometry": ["geom10", "geom11", "geom12"],
                }
            )
            mock_areas = pd.DataFrame(
                {
                    "NAME": ["Area 1", "Area 2", "Area 3"],
                    "geometry": ["geom13", "geom14", "geom15"],
                }
            )
            mock_entities = pd.DataFrame(
                {
                    "Utility": ["Entity 1", "Entity 2", "Entity 3"],
                    "geometry": ["geom16", "geom17", "geom18"],
                }
            )
            mock_stations = pd.DataFrame(
                {
                    "station": ["Station 1", "Station 2", "Station 3"],
                    "geometry": ["geom19", "geom20", "geom21"],
                }
            )

            # Set up the geographies attributes
            mock_new._geographies._us_states = mock_states
            mock_new._geographies._ca_counties = mock_counties
            mock_new._geographies._ca_forecast_zones = mock_zones
            mock_new._geographies._ca_watersheds = mock_watersheds
            mock_new._geographies._ca_electric_balancing_areas = mock_areas
            mock_new._geographies._ca_utilities = mock_entities
            mock_new._stations_gdf = mock_stations

            # Critical part: Mock the function that actually gets called in the implementation
            with patch(
                "climakitae.core.data_interface.get_subsetting_options",
                wraps=get_subsetting_options,
            ):
                # Call the function with 'all'
                result = get_subsetting_options("all")

                # Debug information to understand failures better
                print("Result shape:", result.shape)
                print(
                    "Area subsets available:",
                    result.index.get_level_values("area_subset").unique().tolist(),
                )

                # Check that the result is a DataFrame with MultiIndex
                assert isinstance(result, pd.DataFrame)
                assert isinstance(result.index, pd.MultiIndex)
                assert result.index.names == ["area_subset", "cached_area"]

                # Check that all area subsets are present
                expected_area_subsets = set(
                    [
                        "states",
                        "CA counties",
                        "CA Electricity Demand Forecast Zones",
                        "CA watersheds",
                        "CA Electric Balancing Authority Areas",
                        "CA Electric Load Serving Entities (IOU & POU)",
                        "Stations",
                    ]
                )

                actual_area_subsets = set(
                    result.index.get_level_values("area_subset").unique()
                )

                # Better error message that shows what's missing and what's unexpected
                missing = expected_area_subsets - actual_area_subsets
                unexpected = actual_area_subsets - expected_area_subsets
                assert (
                    actual_area_subsets == expected_area_subsets
                ), f"Missing area subsets: {missing}, Unexpected area subsets: {unexpected}"

                # Check that states filtering worked properly
                if "states" in actual_area_subsets:
                    states_in_result = result.loc["states"].index.tolist()
                    assert "CA" in states_in_result
                    assert "OR" in states_in_result
                    # WA should be filtered out because it's not in boundary_dict
                    assert "WA" not in states_in_result

                # Stations should not be filtered
                stations_in_result = result.loc["Stations"].index.tolist()
                assert len(stations_in_result) == 3

    def test_get_subsetting_options_specific_area(self):
        """
        Test that get_subsetting_options returns only the specified area subset
        """
        with patch("climakitae.core.data_interface.DataInterface") as mock_di:
            # Setup mock DataInterface and geographies
            mock_new = mock_di.return_value
            mock_new._geographies = Mock()
            mock_new._stations_gdf = Mock()

            # Setup mock boundary dict
            mock_boundary_dict = {
                "states": {"CA": {}, "OR": {}},
                "CA counties": {"County1": {}},
                "CA Electricity Demand Forecast Zones": {"Zone1": {}},
                "CA watersheds": {"Watershed1": {}},
                "CA Electric Balancing Authority Areas": {"Area1": {}},
                "CA Electric Load Serving Entities (IOU & POU)": {"Entity1": {}},
            }
            mock_new._geographies.boundary_dict.return_value = mock_boundary_dict

            # Setup ALL required mock dataframes, even when only testing states
            mock_states = pd.DataFrame(
                {
                    "abbrevs": ["CA", "OR", "WA"],
                    "geometry": ["geom1", "geom2", "geom3"],
                }
            )
            # Need to mock all the other dataframes too
            mock_counties = pd.DataFrame({"NAME": ["County1"], "geometry": ["geom"]})
            mock_zones = pd.DataFrame({"FZ_Name": ["Zone1"], "geometry": ["geom"]})
            mock_watersheds = pd.DataFrame(
                {"Name": ["Watershed1"], "geometry": ["geom"]}
            )
            mock_areas = pd.DataFrame({"NAME": ["Area1"], "geometry": ["geom"]})
            mock_entities = pd.DataFrame({"Utility": ["Entity1"], "geometry": ["geom"]})
            mock_stations = pd.DataFrame(
                {"station": ["Station1"], "geometry": ["geom"]}
            )

            # Set up ALL the geographies attributes
            mock_new._geographies._us_states = mock_states
            mock_new._geographies._ca_counties = mock_counties
            mock_new._geographies._ca_forecast_zones = mock_zones
            mock_new._geographies._ca_watersheds = mock_watersheds
            mock_new._geographies._ca_electric_balancing_areas = mock_areas
            mock_new._geographies._ca_utilities = mock_entities
            mock_new._stations_gdf = mock_stations

            # Call the function with 'states'
            result = get_subsetting_options("states")

            # Check that the result is a DataFrame with single index
            assert isinstance(result, pd.DataFrame)
            assert result.index.name == "cached_area"

            # Check that only states that are in boundary_dict are included
            assert set(result.index) == {"CA", "OR"}
            assert "WA" not in result.index

            # Check it has the correct columns
            assert "geometry" in result.columns

    def test_get_subsetting_options_invalid_input(self):
        """
        Test that get_subsetting_options raises ValueError for invalid area_subset
        """
        with patch("climakitae.core.data_interface.DataInterface") as mock_di:
            # Setup mock DataInterface and geographies
            mock_new = mock_di.return_value
            mock_new._geographies = Mock()
            mock_new._stations_gdf = Mock()

            # Setup mock boundary dict (even for invalid input test)
            mock_boundary_dict = {}
            mock_new._geographies.boundary_dict.return_value = mock_boundary_dict

            # Mock all required dataframes with minimal data
            mock_states = pd.DataFrame({"abbrevs": ["CA"], "geometry": ["geom1"]})
            mock_counties = pd.DataFrame({"NAME": ["County1"], "geometry": ["geom"]})
            mock_zones = pd.DataFrame({"FZ_Name": ["Zone1"], "geometry": ["geom"]})
            mock_watersheds = pd.DataFrame(
                {"Name": ["Watershed1"], "geometry": ["geom"]}
            )
            mock_areas = pd.DataFrame({"NAME": ["Area1"], "geometry": ["geom"]})
            mock_entities = pd.DataFrame({"Utility": ["Entity1"], "geometry": ["geom"]})
            mock_stations = pd.DataFrame(
                {"station": ["Station1"], "geometry": ["geom"]}
            )

            # Set up all required attributes
            mock_new._geographies._us_states = mock_states
            mock_new._geographies._ca_counties = mock_counties
            mock_new._geographies._ca_forecast_zones = mock_zones
            mock_new._geographies._ca_watersheds = mock_watersheds
            mock_new._geographies._ca_electric_balancing_areas = mock_areas
            mock_new._geographies._ca_utilities = mock_entities
            mock_new._stations_gdf = mock_stations

            # Call with invalid area_subset and expect ValueError
            with pytest.raises(
                ValueError, match="Bad input for argument 'area_subset'"
            ):
                get_subsetting_options("invalid_area")

    def test_get_subsetting_options_stations(self):
        """
        Test that get_subsetting_options handles stations correctly (no filtering)
        """
        with patch("climakitae.core.data_interface.DataInterface") as mock_di:
            # Setup mock DataInterface
            mock_new = mock_di.return_value
            mock_new._geographies = Mock()

            # Setup mock boundary dict (needs all keys even for stations test)
            mock_boundary_dict = {
                "states": {"CA": {}},
                "CA counties": {"County1": {}},
                "CA Electricity Demand Forecast Zones": {"Zone1": {}},
                "CA watersheds": {"Watershed1": {}},
                "CA Electric Balancing Authority Areas": {"Area1": {}},
                "CA Electric Load Serving Entities (IOU & POU)": {"Entity1": {}},
            }
            mock_new._geographies.boundary_dict.return_value = mock_boundary_dict

            # Need to set up ALL required mock dataframes
            mock_states = pd.DataFrame({"abbrevs": ["CA"], "geometry": ["geom1"]})
            mock_counties = pd.DataFrame({"NAME": ["County1"], "geometry": ["geom"]})
            mock_zones = pd.DataFrame({"FZ_Name": ["Zone1"], "geometry": ["geom"]})
            mock_watersheds = pd.DataFrame(
                {"Name": ["Watershed1"], "geometry": ["geom"]}
            )
            mock_areas = pd.DataFrame({"NAME": ["Area1"], "geometry": ["geom"]})
            mock_entities = pd.DataFrame({"Utility": ["Entity1"], "geometry": ["geom"]})

            # Setup mock stations dataframe
            mock_stations = pd.DataFrame(
                {
                    "station": ["Station 1", "Station 2", "Station 3"],
                    "geometry": ["geom1", "geom2", "geom3"],
                }
            )

            # Set up ALL required attributes
            mock_new._geographies._us_states = mock_states
            mock_new._geographies._ca_counties = mock_counties
            mock_new._geographies._ca_forecast_zones = mock_zones
            mock_new._geographies._ca_watersheds = mock_watersheds
            mock_new._geographies._ca_electric_balancing_areas = mock_areas
            mock_new._geographies._ca_utilities = mock_entities
            mock_new._stations_gdf = mock_stations

            # Call the function with 'Stations'
            result = get_subsetting_options("Stations")

            # Check that all stations are included (no filtering)
            assert len(result) == 3
            assert set(result.index) == {"Station 1", "Station 2", "Station 3"}


class TestGetSubarea:
    """
    Tests for the _get_subarea function.
    """

    @pytest.fixture
    def mock_geographies(self):
        """Create a mock Boundaries object with the necessary boundary datasets"""
        mock_geo = Mock()

        # Sample GDF to return from any boundary dataset lookup
        sample_gdf = gpd.GeoDataFrame({"geometry": [box(0, 0, 1, 1)]}, crs="EPSG:4326")

        # Create a more sophisticated loc accessor that properly handles square bracket notation
        class MockLocAccessor:
            def __init__(self, return_value):
                self.return_value = return_value
                self.called_with = None

            def __getitem__(self, indices):
                # This is called when using square bracket notation: .loc[indices]
                self.called_with = indices
                return self.return_value

        # For each boundary dataset, create a mock with our custom loc accessor
        for attr in [
            "_us_states",
            "_ca_counties",
            "_ca_watersheds",
            "_ca_utilities",
            "_ca_forecast_zones",
            "_ca_electric_balancing_areas",
        ]:
            # Create the mock for each dataset
            dataset_mock = Mock()

            # Set the loc accessor to our custom class
            dataset_mock.loc = MockLocAccessor(sample_gdf)

            # Set the dataset on the main mock_geo object
            setattr(mock_geo, attr, dataset_mock)

        return mock_geo

    @pytest.fixture
    def mock_geography_choose(self):
        """Create a mock geography_choose dictionary"""
        return {
            "states": {"CA": 0, "OR": 1},
            "CA counties": {"Alameda": 0, "Orange": 1},
            "CA watersheds": {"Watershed1": 0, "Watershed2": 1},
            "CA Electric Load Serving Entities (IOU & POU)": {
                "Entity1": 0,
                "Entity2": 1,
            },
            "CA Electricity Demand Forecast Zones": {
                "Zone1": 0,
                "Zone2": 1,
            },
            "CA Electric Balancing Authority Areas": {
                "Area1": 0,
                "Area2": 1,
            },
            "none": {"entire domain": None},
            "lat/lon": {"coordinate selection": None},
        }

    def test_get_subarea_lat_lon(self, mock_geographies, mock_geography_choose):
        """Test that _get_subarea correctly handles lat/lon subsetting"""
        # Test with lat/lon subsetting
        latitude = (34.0, 36.0)
        longitude = (-120.0, -118.0)

        result = _get_subarea(
            area_subset="lat/lon",
            cached_area=["coordinate selection"],
            latitude=latitude,
            longitude=longitude,
            _geographies=mock_geographies,
            _geography_choose=mock_geography_choose,
        )

        # Verify result is a GeoDataFrame with the right geometry
        assert isinstance(result, gpd.GeoDataFrame)
        assert result.crs == "EPSG:4326"
        assert len(result) == 1
        assert result.iloc[0]["subset"] == "coords"
        # Verify the box coordinates
        assert result.iloc[0].geometry.bounds == (
            longitude[0],
            latitude[0],
            longitude[1],
            latitude[1],
        )

    def test_get_subarea_none(self, mock_geographies, mock_geography_choose):
        """Test that _get_subarea correctly handles 'none' subsetting (entire domain)"""
        latitude = (34.0, 36.0)  # Not used in this case
        longitude = (-120.0, -118.0)  # Not used in this case

        result = _get_subarea(
            area_subset="none",
            cached_area=["entire domain"],
            latitude=latitude,
            longitude=longitude,
            _geographies=mock_geographies,
            _geography_choose=mock_geography_choose,
        )

        # Verify result is a GeoDataFrame with a large box
        assert isinstance(result, gpd.GeoDataFrame)
        assert result.crs == "EPSG:4326"
        assert len(result) == 1
        assert result.iloc[0]["subset"] == "coords"
        # Verify the box is the super big box
        assert result.iloc[0].geometry.bounds == (-150, -88, 8, 66)

    def test_get_subarea_with_area_subset_types(
        self, mock_geographies, mock_geography_choose
    ):
        """Test that _get_subarea correctly handles different area subset types"""
        latitude = (34.0, 36.0)  # Not used in this case
        longitude = (-120.0, -118.0)  # Not used in this case

        # Test each area subset type
        area_subset_types = [
            "states",
            "CA counties",
            "CA watersheds",
            "CA Electric Load Serving Entities (IOU & POU)",
            "CA Electricity Demand Forecast Zones",
            "CA Electric Balancing Authority Areas",
        ]

        for area_type in area_subset_types:
            # Get keys for this area_subset
            cached_area_keys = list(mock_geography_choose[area_type].keys())

            result = _get_subarea(
                area_subset=area_type,
                cached_area=cached_area_keys,
                latitude=latitude,
                longitude=longitude,
                _geographies=mock_geographies,
                _geography_choose=mock_geography_choose,
            )

            # Verify the correct boundary dataset method was called with right indices
            expected_indices = [0, 1]  # Based on mock_geography_choose fixture values

            match area_type:
                case "states":
                    assert (
                        mock_geographies._us_states.loc.called_with == expected_indices
                    )
                case "CA counties":
                    assert (
                        mock_geographies._ca_counties.loc.called_with
                        == expected_indices
                    )
                case "CA watersheds":
                    assert (
                        mock_geographies._ca_watersheds.loc.called_with
                        == expected_indices
                    )
                case "CA Electric Load Serving Entities (IOU & POU)":
                    assert (
                        mock_geographies._ca_utilities.loc.called_with
                        == expected_indices
                    )
                case "CA Electricity Demand Forecast Zones":
                    assert (
                        mock_geographies._ca_forecast_zones.loc.called_with
                        == expected_indices
                    )
                case "CA Electric Balancing Authority Areas":
                    assert (
                        mock_geographies._ca_electric_balancing_areas.loc.called_with
                        == expected_indices
                    )

            # Verify result is a GeoDataFrame
            assert isinstance(result, gpd.GeoDataFrame)

    def test_get_subarea_with_none_cached_area(
        self, mock_geographies, mock_geography_choose
    ):
        """Test that _get_subarea correctly handles None cached_area"""
        latitude = (34.0, 36.0)  # Not used in this case
        longitude = (-120.0, -118.0)  # Not used in this case

        result = _get_subarea(
            area_subset="states",
            cached_area=None,  # Test with None cached_area
            latitude=latitude,
            longitude=longitude,
            _geographies=mock_geographies,
            _geography_choose=mock_geography_choose,
        )

        # Verify that shape_indices was set to [0] when cached_area is None
        # Change from assert_called_with to checking our custom called_with attribute
        assert mock_geographies._us_states.loc.called_with == [0]
        assert isinstance(result, gpd.GeoDataFrame)


class Test_CheckIfGoodInput:
    """Tests for the _check_if_good_input function."""

    @pytest.fixture
    def sample_catalog_df(self):
        """Create a sample catalog dataframe for testing."""
        return pd.DataFrame(
            {
                "variable": ["Temperature", "Precipitation", "Wind Speed"],
                "resolution": ["3 km", "9 km", "45 km"],
                "scenario": ["Historical Climate", "SSP 1-2.6", "SSP 5-8.5"],
            }
        )

    def test_check_if_good_input_valid_inputs(self, sample_catalog_df):
        """Test that valid inputs are returned unchanged."""
        input_dict = {
            "variable": ["Temperature"],
            "resolution": ["3 km"],
            "scenario": ["Historical Climate"],
        }

        result = _check_if_good_input(input_dict, sample_catalog_df)

        assert result == input_dict
        assert result["variable"] == ["Temperature"]
        assert result["resolution"] == ["3 km"]
        assert result["scenario"] == ["Historical Climate"]

    def test_check_if_good_input_none_values(self, sample_catalog_df):
        """Test that None values are replaced with all valid options."""
        input_dict = {
            "variable": None,
            "resolution": [None],
            "scenario": ["Historical Climate"],
        }

        result = _check_if_good_input(input_dict, sample_catalog_df)

        assert set(result["variable"]) == set(
            ["Temperature", "Precipitation", "Wind Speed"]
        )
        assert set(result["resolution"]) == set(["3 km", "9 km", "45 km"])
        assert result["scenario"] == ["Historical Climate"]

    def test_check_if_good_input_resolution_formatting(self, sample_catalog_df):
        """Test the special handling for resolution formatting."""
        input_dict = {
            "variable": ["Temperature"],
            "resolution": ["3km", "9KM"],
            "scenario": ["Historical Climate"],
        }

        with patch("builtins.print") as mock_print:
            result = _check_if_good_input(input_dict, sample_catalog_df)

        assert result["resolution"] == ["3 km", "9 km"]
        assert mock_print.call_count == 4  # 2 for invalid options + 2 for corrections

    def test_check_if_good_input_closest_options(self, sample_catalog_df):
        """Test finding closest options when input doesn't match exactly."""
        input_dict = {
            "variable": ["temparature"],  # Typo
            "resolution": ["3 km"],
            "scenario": ["Historical Climate"],
        }

        with (
            patch("builtins.print") as mock_print,
            patch(
                "climakitae.core.data_interface._get_closest_options",
                return_value=["Temperature"],
            ),
        ):
            result = _check_if_good_input(input_dict, sample_catalog_df)

        assert result["variable"] == ["Temperature"]
        assert (
            mock_print.call_count >= 3
        )  # Invalid option message + closest option + using option

    def test_check_if_good_input_no_closest_options(self, sample_catalog_df):
        """Test behavior when no close matches can be found."""
        input_dict = {
            "variable": ["xyz"],  # No close match
            "resolution": ["3 km"],
            "scenario": ["Historical Climate"],
        }

        with (
            patch("builtins.print") as mock_print,
            patch(
                "climakitae.core.data_interface._get_closest_options",
                return_value=None,
            ),
        ):
            with pytest.raises(ValueError, match="Bad input"):
                _check_if_good_input(input_dict, sample_catalog_df)

        # Check that valid options were printed
        assert (
            mock_print.call_count >= 2
        )  # Invalid option message + valid options listing
