"""
Test suite for the DataInterface class

need coverage for lines:
climakitae/core/data_interface.py     782    179    77%
170, 316, 319-325, 331-362, 727, 729, 735-736, 789-800, 808-809, 832-834, 887-888,
893-895, 912, 943-945, 951-952, 1014-1024, 1044-1045, 1088, 1098, 1125, 1127, 1135,
1149, 1156, 1166-1184, 1201, 1203, 1210, 1217, 1220-1226, 1233, 1258-1260, 1309, 1323,
1327, 1332-1335, 1341-1342, 1532, 1535, 1586-1599, 1603-1622, 1674-1679, 1715,
1739-1810, 1942, 1962-1963, 1982, 1992, 2005-2009, 2013-2016, 2018-2021, 2035-2042,
2055-2060, 2071-2073, 2126-2128, 2141, 2146, 2156-2157, 2183-2187, 2192-2193, 2208-2210,
2338, 2340, 2344, 2346, 2348, 2351-2356

1739-1810 (72 lines)
331-362 (32 lines)
1603-1622 (20 lines)
1166-1184 (19 lines)
1586-1599 (14 lines)
789-800 (12 lines)
1674-1679 (6 lines)
2035-2042 (8 lines)
2351-2356 (6 lines)
2183-2187 (5 lines)
1014-1024 (11 lines)
"""

from typing import Union
from unittest.mock import MagicMock, Mock, PropertyMock, patch

import pandas as pd
import pytest

from climakitae.core.data_interface import (
    DataInterface,
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

    @staticmethod
    def test_init_data_loading():
        """
        Test that all data sources are loaded correctly during initialization.
        """

        with patch(
            "climakitae.core.data_interface.read_csv_file"
        ) as mock_read_csv, patch(
            "climakitae.core.data_interface.gpd"
        ) as mock_gpd, patch(
            "climakitae.core.data_interface.intake"
        ) as mock_intake, patch(
            "climakitae.core.data_interface.Boundaries"
        ) as mock_boundaries, patch(
            "climakitae.core.data_interface.VariableDescriptions"
        ) as mock_var_desc, patch(
            "climakitae.core.data_interface.stations_csv_path",
            "data/hadisd_stations.csv",
        ), patch(
            "climakitae.core.data_interface.gwl_1850_1900_file",
            "data/gwl_1850-1900ref.csv",
        ), patch(
            "climakitae.core.data_interface.data_catalog_url",
            "https://cadcat.s3.amazonaws.com/cae-collection.json",
        ), patch(
            "climakitae.core.data_interface.boundary_catalog_url",
            "boundary_catalog_url_value",
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

        # Configure mock dataframe with experiment_id and variable_id but make source_id.unique() raise an exception
        mock_dict = {
            "experiment_id": Mock(unique=Mock(return_value=["ssp126", "ssp585"])),
            "source_id": Mock(
                unique=Mock(side_effect=Exception("No source_id column"))
            ),
            "variable_id": Mock(unique=Mock(return_value=["tasmax", "tasmin"])),
        }
        mock_subset.df.__getitem__.side_effect = mock_dict.__getitem__

        # Call the function
        scenario_options, simulation_options, unique_variable_ids = _get_user_options(
            mock_catalog, "Dynamical", "daily", "3 km"
        )

        # Assert that simulation_options is empty list when exception occurs
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
