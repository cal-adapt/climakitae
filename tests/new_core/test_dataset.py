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
from climakitae.new_core.param_validation.abc_param_validation import ParameterValidator


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

    def test_with_catalog_returns_self(self):
        """Test with_catalog returns Dataset instance for chaining."""
        dataset = Dataset()
        mock_catalog = MagicMock(spec=DataCatalog)
        mock_catalog.get_data = MagicMock()

        result = dataset.with_catalog(mock_catalog)

        assert result is dataset

    def test_with_catalog_invalid_type(self):
        """Test with_catalog raises TypeError for non-DataCatalog."""
        dataset = Dataset()
        invalid_catalog = {"get_data": lambda x: x}  # dict, not DataCatalog

        with pytest.raises(TypeError, match="must be an instance of DataCatalog"):
            dataset.with_catalog(invalid_catalog)

    def test_with_catalog_missing_get_data(self):
        """Test with_catalog raises AttributeError if catalog lacks get_data method."""
        dataset = Dataset()
        # Create a mock that passes isinstance check but lacks get_data
        mock_catalog = MagicMock(spec=DataCatalog)
        del mock_catalog.get_data  # Remove the get_data attribute

        with pytest.raises(AttributeError, match="must have a 'get_data' method"):
            dataset.with_catalog(mock_catalog)

    def test_with_catalog_non_callable_get_data(self):
        """Test with_catalog raises TypeError if get_data is not callable."""
        dataset = Dataset()
        mock_catalog = MagicMock(spec=DataCatalog)
        mock_catalog.get_data = "not_a_function"  # Non-callable

        with pytest.raises(TypeError, match="'get_data' method.*must be callable"):
            dataset.with_catalog(mock_catalog)


class TestDatasetWithParamValidator:
    """Test class for with_param_validator method."""

    def test_with_param_validator_valid(self):
        """Test with_param_validator with valid ParameterValidator instance."""
        dataset = Dataset()
        mock_validator = MagicMock(spec=ParameterValidator)

        dataset.with_param_validator(mock_validator)

        assert dataset.parameter_validator is mock_validator
