"""
Unit tests for climakitae/new_core/processors/add_catalog_coords.py.

This module contains comprehensive unit tests for the AddCatalogCoords processor
that adds catalog metadata (network_id) as coordinates to HDP datasets.
"""

from unittest.mock import MagicMock, Mock
import pandas as pd
import pytest
import xarray as xr

from climakitae.core.constants import UNSET
from climakitae.new_core.processors.add_catalog_coords import AddCatalogCoords


class TestAddCatalogCoordsInitialization:
    """Test class for AddCatalogCoords initialization."""

    def test_init_default(self):
        """Test initialization with default value."""
        processor = AddCatalogCoords()
        assert processor.value is UNSET
        assert processor.name == "add_catalog_coords"
        assert processor.catalog is None
        assert processor.needs_catalog is True

    def test_init_with_value(self):
        """Test initialization with custom value."""
        processor = AddCatalogCoords("test_value")
        assert processor.value == "test_value"
        assert processor.name == "add_catalog_coords"


class TestAddCatalogCoordsDataAccessor:
    """Test class for data accessor functionality."""

    def test_set_data_accessor(self):
        """Test setting data accessor."""
        processor = AddCatalogCoords()
        mock_catalog = MagicMock()
        processor.set_data_accessor(mock_catalog)
        assert processor.catalog is mock_catalog


class TestAddCatalogCoordsExecute:
    """Test class for execute method functionality."""

    def setup_method(self):
        """Set up test fixtures."""
        self.processor = AddCatalogCoords()

        # Create mock catalog
        self.mock_catalog = MagicMock()
        self.mock_hdp_catalog = MagicMock()
        self.mock_catalog.hdp = self.mock_hdp_catalog

        self.processor.set_data_accessor(self.mock_catalog)

    def test_execute_with_single_network_broadcasts_to_all_stations(self):
        """Test execute with single network_id broadcasts to all stations."""
        # Create dataset with station_id dimension
        dataset = xr.Dataset(
            {"temp": (["station_id", "time"], [[1, 2], [3, 4], [5, 6]])},
            coords={
                "station_id": ["ASOSAWOS_1", "ASOSAWOS_2", "ASOSAWOS_3"],
                "time": [0, 1],
            },
        )

        # Mock catalog search result
        mock_subset = MagicMock()
        mock_subset.df = pd.DataFrame(
            {
                "network_id": ["ASOSAWOS", "ASOSAWOS", "ASOSAWOS"],
                "station_id": ["ASOSAWOS_1", "ASOSAWOS_2", "ASOSAWOS_3"],
            }
        )
        self.mock_hdp_catalog.search.return_value = mock_subset

        context = {"station_id": ["ASOSAWOS_1", "ASOSAWOS_2", "ASOSAWOS_3"]}
        result = self.processor.execute(dataset, context)

        # Check network_id was added as coordinate
        assert "network_id" in result.coords
        assert result.coords["network_id"].dims == ("station_id",)
        # All stations from same network
        assert list(result.coords["network_id"].values) == [
            "ASOSAWOS",
            "ASOSAWOS",
            "ASOSAWOS",
        ]

        # Check attributes
        assert "long_name" in result["network_id"].attrs
        assert "description" in result["network_id"].attrs

    def test_execute_with_multiple_networks(self):
        """Test execute with multiple network_ids."""
        # Create dataset with station_id dimension
        dataset = xr.Dataset(
            {"temp": (["station_id", "time"], [[1, 2], [3, 4]])},
            coords={
                "station_id": ["ASOSAWOS_1", "CIMIS_1"],
                "time": [0, 1],
            },
        )

        # Mock catalog search result with different networks
        mock_subset = MagicMock()
        mock_subset.df = pd.DataFrame(
            {
                "network_id": ["ASOSAWOS", "CIMIS"],
                "station_id": ["ASOSAWOS_1", "CIMIS_1"],
            }
        )
        self.mock_hdp_catalog.search.return_value = mock_subset

        context = {"station_id": ["ASOSAWOS_1", "CIMIS_1"]}
        result = self.processor.execute(dataset, context)

        # Check network_id was added with correct values
        assert "network_id" in result.coords
        assert result.coords["network_id"].dims == ("station_id",)
        assert list(result.coords["network_id"].values) == ["ASOSAWOS", "CIMIS"]

    def test_execute_no_station_id_in_context(self):
        """Test execute when station_id is not in context."""
        dataset = xr.Dataset({"temp": (["time"], [1, 2, 3])})
        context = {}

        result = self.processor.execute(dataset, context)

        # Should return dataset unchanged
        assert result is dataset
        assert "network_id" not in result.coords

    def test_execute_with_dataarray(self):
        """Test execute with xarray DataArray."""
        dataarray = xr.DataArray(
            [[1, 2], [3, 4]],
            dims=["station_id", "time"],
            coords={
                "station_id": ["ASOSAWOS_1", "ASOSAWOS_2"],
                "time": [0, 1],
            },
            name="temp",
        )

        # Mock catalog search result
        mock_subset = MagicMock()
        mock_subset.df = pd.DataFrame(
            {
                "network_id": ["ASOSAWOS", "ASOSAWOS"],
                "station_id": ["ASOSAWOS_1", "ASOSAWOS_2"],
            }
        )
        self.mock_hdp_catalog.search.return_value = mock_subset

        context = {"station_id": ["ASOSAWOS_1", "ASOSAWOS_2"]}
        result = self.processor.execute(dataarray, context)

        # Should be a DataArray with network_id coordinate
        assert isinstance(result, xr.DataArray)
        assert "network_id" in result.coords

    def test_execute_with_dict_of_datasets(self):
        """Test execute with dictionary of datasets."""
        ds1 = xr.Dataset(
            {"temp": (["station_id"], [1, 2])},
            coords={"station_id": ["ASOSAWOS_1", "ASOSAWOS_2"]},
        )
        ds2 = xr.Dataset(
            {"temp": (["station_id"], [3, 4])},
            coords={"station_id": ["ASOSAWOS_1", "ASOSAWOS_2"]},
        )

        # Mock catalog search result
        mock_subset = MagicMock()
        mock_subset.df = pd.DataFrame(
            {
                "network_id": ["ASOSAWOS", "ASOSAWOS"],
                "station_id": ["ASOSAWOS_1", "ASOSAWOS_2"],
            }
        )
        self.mock_hdp_catalog.search.return_value = mock_subset

        context = {"station_id": ["ASOSAWOS_1", "ASOSAWOS_2"]}
        result = self.processor.execute({"key1": ds1, "key2": ds2}, context)

        # Both datasets should have network_id coordinate
        assert "network_id" in result["key1"].coords
        assert "network_id" in result["key2"].coords

    def test_execute_with_list_of_datasets(self):
        """Test execute with list of datasets."""
        ds1 = xr.Dataset(
            {"temp": (["station_id"], [1, 2])},
            coords={"station_id": ["ASOSAWOS_1", "ASOSAWOS_2"]},
        )
        ds2 = xr.Dataset(
            {"temp": (["station_id"], [3, 4])},
            coords={"station_id": ["ASOSAWOS_1", "ASOSAWOS_2"]},
        )

        # Mock catalog search result
        mock_subset = MagicMock()
        mock_subset.df = pd.DataFrame(
            {
                "network_id": ["ASOSAWOS", "ASOSAWOS"],
                "station_id": ["ASOSAWOS_1", "ASOSAWOS_2"],
            }
        )
        self.mock_hdp_catalog.search.return_value = mock_subset

        context = {"station_id": ["ASOSAWOS_1", "ASOSAWOS_2"]}
        result = self.processor.execute([ds1, ds2], context)

        # Both datasets should have network_id coordinate
        assert "network_id" in result[0].coords
        assert "network_id" in result[1].coords

    def test_execute_without_station_id_dimension(self):
        """Test execute with dataset that doesn't have station_id dimension."""
        dataset = xr.Dataset({"temp": (["time"], [1, 2, 3])})

        # Mock catalog search result
        mock_subset = MagicMock()
        mock_subset.df = pd.DataFrame(
            {
                "network_id": ["ASOSAWOS"],
                "station_id": ["ASOSAWOS_1"],
            }
        )
        self.mock_hdp_catalog.search.return_value = mock_subset

        context = {"station_id": ["ASOSAWOS_1"]}
        result = self.processor.execute(dataset, context)

        # Should add network_id as scalar coordinate
        assert "network_id" in result.coords
        assert result.coords["network_id"].dims == ()


class TestAddCatalogCoordsUpdateContext:
    """Test class for update_context method."""

    def test_update_context_with_network_id(self):
        """Test update_context when network_id is in context."""
        processor = AddCatalogCoords()
        context = {"network_id": "ASOSAWOS"}

        processor.update_context(context)

        from climakitae.core.constants import _NEW_ATTRS_KEY

        assert _NEW_ATTRS_KEY in context
        assert "add_catalog_coords" in context[_NEW_ATTRS_KEY]
        assert "ASOSAWOS" in context[_NEW_ATTRS_KEY]["add_catalog_coords"]

    def test_update_context_without_network_id(self):
        """Test update_context when network_id is not in context."""
        processor = AddCatalogCoords()
        context = {}

        processor.update_context(context)

        from climakitae.core.constants import _NEW_ATTRS_KEY

        assert _NEW_ATTRS_KEY in context
        assert "add_catalog_coords" not in context[_NEW_ATTRS_KEY]
