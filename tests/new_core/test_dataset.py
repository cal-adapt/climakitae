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
from climakitae.new_core.data_access.data_access import DataCatalog

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

    def test_with_catalog_has_get_data_error(self):
        """Test with_catalog has 'get_data' method error message."""
        delattr(DataCatalog, 'get_data')
        data_catalog = DataCatalog()

        data_catalog.catalog_df = pd.DataFrame(
            {"catalog": ["climate"], "variable_id": ["tas"]}
        )

        dataset = Dataset()

        try:
            dataset.with_catalog(data_catalog)
        except AttributeError as e:
            assert "Data catalog must have a 'get_data' method to retrieve data." in str(e)

    def test_with_catalog_get_data_callable_error(self):
        """Test with_catalog callable 'get_data' method error message."""
        DataCatalog.get_data = None
        data_catalog = DataCatalog()

        data_catalog.catalog_df = pd.DataFrame(
            {"catalog": ["climate"], "variable_id": ["tas"]}
        )

        dataset = Dataset()

        try:
            dataset.with_catalog(data_catalog)
        except TypeError as e:
            assert "'get_data' method in data catalog must be callable." in str(e)
