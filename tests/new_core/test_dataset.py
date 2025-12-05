"""
Unit tests for climakitae/new_core/dataset.py.

This module contains comprehensive unit tests for the Dataset class
that implements a pipeline-based approach for climate data processing.
"""

from unittest.mock import MagicMock

import pytest
import xarray as xr

from climakitae.core.constants import UNSET
from climakitae.new_core.data_access.data_access import DataCatalog
from climakitae.new_core.dataset import Dataset


class TestDatasetInit:
    """Test class for Dataset initialization."""

    def test_init_default_values(self):
        """Test initialization sets all attributes to UNSET."""
        dataset = Dataset()

        assert dataset.data_access is UNSET
        assert dataset.parameter_validator is UNSET
        assert dataset.processing_pipeline is UNSET


class TestDatasetWithCatalog:
    """Test class for with_catalog method."""

    def test_with_catalog_valid(self):
        """Test with_catalog with valid DataCatalog instance."""
        dataset = Dataset()
        mock_catalog = MagicMock(spec=DataCatalog)
        mock_catalog.get_data = MagicMock()

        dataset.with_catalog(mock_catalog)

        assert dataset.data_access is mock_catalog
