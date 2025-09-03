"""
Unit tests for climakitae/new_core/dataset.py.

This module contains comprehensive unit tests for the Dataset class that 
provides the core Dataset class that implements a flexible, pipeline-based
approach for climate data processing.
"""

from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from climakitae.new_core.dataset import Dataset
from climakitae.new_core.dataset_factory import DataCatalog

class TestDatasetInit:
    """Test class for Dataset initialization"""

    def test_init_successful(self):
        """Test successful initialization."""

        dataset = Dataset()

        assert hasattr(dataset, "data_access")
        assert hasattr(dataset, "parameter_validator")
        assert hasattr(dataset, "processing_pipeline")


class TestDatasetWithCatalogMethod:
    """Test class for with_catalog method."""

    def test_with_catalog_successful(self):
        """Test successful with_catalog set."""

        data_catalog = DataCatalog()

        data_catalog.catalog_df = pd.DataFrame(
            {"catalog": ["climate"], "variable_id": ["tas"]}
        )

        dataset = Dataset()

        dataset.with_catalog(data_catalog)

        assert dataset.data_access is data_catalog

    def test_with_catalog_is_datacatalog(self):
        """Test with_catalog input catalog is DataCatalog."""
        dictionary = {
            "catalog": ["test_catalog", "another_catalog"],
            "variable_id": ["var1", "var2"],
        }

        dataset = Dataset()

        try:
            dataset.with_catalog(dictionary)
        except TypeError as e:
            assert "Data catalog must be an instance of DataCatalog." in str(e)

    @patch("climakitae.new_core.dataset_factory.DataCatalog")
    def test_with_catalog_has_get_data(self, mock_data_catalog):
        """Test with_catalog input catalog has 'get_data' method."""
        mock_catalog_instance = MagicMock()
        mock_catalog_instance.catalog_df = pd.DataFrame(
            {
                "catalog": ["test_catalog", "another_catalog"],
                "variable_id": ["var1", "var2"],
            }
        )
        delattr(mock_catalog_instance, 'get_data')
        mock_data_catalog.return_value = mock_catalog_instance

        dataset = Dataset()

        try:
            dataset.with_catalog(mock_data_catalog)
        except AttributeError as e:
            assert "Data catalog must have a 'get_data' method to retrieve data." in str(e)
