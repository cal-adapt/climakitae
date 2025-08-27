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
        print(dataset.data_access)

        assert dataset.data_access is data_catalog
