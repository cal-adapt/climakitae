"""
Unit tests for climakitae/new_core/dataset.py.

This module contains comprehensive unit tests for the Dataset class that
provides the core Dataset class that implements a flexible, pipeline-based
approach for climate data processing.
"""

from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from climakitae.core.constants import UNSET
from climakitae.new_core.dataset import Dataset
from climakitae.new_core.data_access.data_access import DataCatalog
from climakitae.new_core.param_validation.abc_param_validation import ParameterValidator


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
        delattr(DataCatalog, "get_data")
        data_catalog = DataCatalog()

        data_catalog.catalog_df = pd.DataFrame(
            {"catalog": ["climate"], "variable_id": ["tas"]}
        )

        dataset = Dataset()

        try:
            dataset.with_catalog(data_catalog)
        except AttributeError as e:
            assert (
                "Data catalog must have a 'get_data' method to retrieve data." in str(e)
            )

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


class ConcreteValidator(ParameterValidator):
    """Concrete implementation for testing abstract class."""

    def is_valid_query(self, query):
        """Implement abstract method for testing."""
        return self._is_valid_query(query)


class TestDatasetWithParamValidatorMethod:
    """Test class for with_param_validator method."""

    def test_with_param_validator_successful(self):
        """Test successful with_param_validator set."""

        data_catalog = DataCatalog()

        data_catalog.catalog_df = pd.DataFrame(
            {
                "variable": ["tas", "pr", "tasmax"],
                "experiment_id": ["ssp245", "ssp245", "historical"],
                "source_id": ["model1", "model2", "model1"],
            }
        )
        data_validator = ConcreteValidator()
        data_validator.all_catalog_keys = {
            "variable": UNSET,
            "experiment_id": UNSET,
            "source_id": UNSET,
        }

        data_validator.catalog = data_catalog
        query = {"variable": "tas", "experiment_id": "ssp245", "extra_key": "ignored"}
        data_validator.populate_catalog_keys(query)

        dataset = Dataset()

        dataset.with_param_validator(data_validator)

        assert dataset.validator.all_catalog_keys == {
            "variable": "tas",
            "experiment_id": "ssp245",
        }
