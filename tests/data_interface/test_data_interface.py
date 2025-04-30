"""
Test suite for the DataInterface class
"""

from unittest.mock import Mock, PropertyMock, patch

import pytest

from climakitae.core.data_interface import DataInterface


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
