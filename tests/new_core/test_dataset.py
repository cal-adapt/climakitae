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

    @patch("climakitae.new_core.dataset_factory.DataCatalog")
    def test_with_catalog_successful(self, mock_data_catalog):
        """Test successful with_catalog set."""
        mock_catalog_instance = MagicMock()
        mock_catalog_instance.catalog_df = pd.DataFrame(
            {"catalog": ["climate"], "variable_id": ["tas"]}
        )
        mock_data_catalog.return_value = mock_catalog_instance

        dataset = Dataset()

        dataset.with_catalog(mock_catalog_instance)
        print(dataset.data_access)

        assert dataset is mock_catalog_instance
