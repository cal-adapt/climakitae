"""
Unit tests for climakitae/new_core/param_validation/abc_param_validation.py

This module contains comprehensive unit tests for the parameter validation
framework including decorator registration, abstract base class functionality,
and validation logic.
"""

import warnings
from unittest.mock import patch

import pandas as pd
import pytest

from climakitae.core.constants import PROC_KEY, UNSET
from climakitae.new_core.param_validation.abc_param_validation import (
    _CATALOG_VALIDATOR_REGISTRY,
    _PROCESSOR_VALIDATOR_REGISTRY,
    ParameterValidator,
    register_catalog_validator,
)


class TestRegisterCatalogValidator:
    """Test class for the register_catalog_validator decorator."""

    def setup_method(self):
        """Clear the registry before each test."""
        _CATALOG_VALIDATOR_REGISTRY.clear()

    def test_register_catalog_validator_successful(self):
        """Test successful registration of a catalog validator."""

        @register_catalog_validator("test_catalog")
        class TestValidator:
            pass

        assert "test_catalog" in _CATALOG_VALIDATOR_REGISTRY
        assert _CATALOG_VALIDATOR_REGISTRY["test_catalog"] is TestValidator

    def test_register_catalog_validator_returns_class(self):
        """Test that decorator returns the class unchanged."""

        @register_catalog_validator("test_catalog")
        class TestValidator:
            def test_method(self):
                return "test"

        # Class should be returned unchanged
        instance = TestValidator()
        assert instance.test_method() == "test"

    def test_register_multiple_validators(self):
        """Test registering multiple catalog validators."""

        @register_catalog_validator("catalog1")
        class Validator1:
            pass

        @register_catalog_validator("catalog2")
        class Validator2:
            pass

        assert "catalog1" in _CATALOG_VALIDATOR_REGISTRY
        assert "catalog2" in _CATALOG_VALIDATOR_REGISTRY
        assert _CATALOG_VALIDATOR_REGISTRY["catalog1"] is Validator1
        assert _CATALOG_VALIDATOR_REGISTRY["catalog2"] is Validator2

    def test_register_overwrite_existing(self):
        """Test that registering with same name overwrites existing."""

        @register_catalog_validator("test_catalog")
        class Validator1:
            pass

        @register_catalog_validator("test_catalog")
        class Validator2:
            pass

        assert _CATALOG_VALIDATOR_REGISTRY["test_catalog"] is Validator2


class ConcreteValidator(ParameterValidator):
    """Concrete implementation for testing abstract class."""

    def is_valid_query(self, query):
        """Implementation of abstract method."""
        return self._is_valid_query(query)


class TestParameterValidatorInit:
    """Test class for ParameterValidator initialization."""

    @patch("climakitae.new_core.param_validation.abc_param_validation.DataCatalog")
    def test_init_successful(self, mock_data_catalog):
        """Test successful initialization of ParameterValidator."""
        mock_catalog_df = pd.DataFrame({"test": [1, 2, 3]})
        mock_data_catalog.return_value.catalog_df = mock_catalog_df

        validator = ConcreteValidator()

        assert validator.catalog_path == "climakitae/data/catalogs.csv"
        assert validator.catalog is UNSET
        assert validator.all_catalog_keys is UNSET
        assert validator.catalog_df.equals(mock_catalog_df)
        mock_data_catalog.assert_called_once()

    @patch("climakitae.new_core.param_validation.abc_param_validation.DataCatalog")
    def test_init_with_datacatalog_error(self, mock_data_catalog):
        """Test initialization when DataCatalog raises an exception."""
        mock_data_catalog.side_effect = Exception("Catalog error")

        with pytest.raises(Exception, match="Catalog error"):
            ConcreteValidator()


class TestParameterValidatorMethods:
    """Test class for ParameterValidator methods."""

    def setup_method(self):
        """Set up test fixtures."""
        with patch(
            "climakitae.new_core.param_validation.abc_param_validation.DataCatalog"
        ):
            self.validator = ConcreteValidator()
            self.validator.catalog_df = pd.DataFrame(
                {
                    "variable": ["tas", "pr", "tasmax"],
                    "experiment_id": ["ssp245", "ssp245", "historical"],
                    "source_id": ["model1", "model2", "model1"],
                }
            )

    def test_populate_catalog_keys_with_values(self):
        """Test populate_catalog_keys with set values."""
        self.validator.all_catalog_keys = {
            "variable": UNSET,
            "experiment_id": UNSET,
            "source_id": UNSET,
        }

        query = {"variable": "tas", "experiment_id": "ssp245", "extra_key": "ignored"}

        self.validator.populate_catalog_keys(query)

        assert self.validator.all_catalog_keys == {
            "variable": "tas",
            "experiment_id": "ssp245",
        }

    def test_populate_catalog_keys_all_unset(self):
        """Test populate_catalog_keys when all values are unset."""
        self.validator.all_catalog_keys = {"variable": UNSET, "experiment_id": UNSET}

        query = {}

        self.validator.populate_catalog_keys(query)

        assert self.validator.all_catalog_keys == {}

    def test_load_catalog_df(self):
        """Test load_catalog_df method."""
        mock_df = pd.DataFrame({"col1": [1, 2], "col2": [3, 4]})

        with patch(
            "climakitae.new_core.param_validation.abc_param_validation.DataCatalog"
        ) as mock_dc:
            mock_dc.return_value.catalog_df = mock_df

            self.validator.load_catalog_df()

            assert self.validator.catalog_df.equals(mock_df)
            mock_dc.assert_called_once()


class TestHasValidProcesses:
    """Test class for _has_valid_processes method."""

    def setup_method(self):
        """Set up test fixtures."""
        _PROCESSOR_VALIDATOR_REGISTRY.clear()
        with patch(
            "climakitae.new_core.param_validation.abc_param_validation.DataCatalog"
        ):
            self.validator = ConcreteValidator()

    def test_has_valid_processes_no_processes(self):
        """Test _has_valid_processes with no processes in query."""
        query = {"variable": "tas"}

        result = self.validator._has_valid_processes(query)

        assert result is True

    def test_has_valid_processes_all_valid(self):
        """Test _has_valid_processes with all valid processes."""
        # Register validators
        _PROCESSOR_VALIDATOR_REGISTRY["proc1"] = lambda v, query=None: True
        _PROCESSOR_VALIDATOR_REGISTRY["proc2"] = lambda v, query=None: True

        query = {PROC_KEY: {"proc1": "value1", "proc2": "value2"}}

        result = self.validator._has_valid_processes(query)

        assert result is True

    def test_has_valid_processes_invalid_processor(self):
        """Test _has_valid_processes with invalid processor."""
        _PROCESSOR_VALIDATOR_REGISTRY["proc1"] = lambda v, query=None: False

        query = {PROC_KEY: {"proc1": "invalid_value"}}

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            result = self.validator._has_valid_processes(query)

            assert result is False
            assert len(w) == 1
            assert "proc1 with value invalid_value is not valid" in str(w[0].message)

    def test_has_valid_processes_unregistered_processor(self):
        """Test _has_valid_processes with unregistered processor."""
        query = {PROC_KEY: {"unregistered_proc": "value"}}

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            result = self.validator._has_valid_processes(query)

            assert result is True  # Unregistered processors don't fail validation
            assert len(w) == 1
            assert "unregistered_proc is not registered" in str(w[0].message)

    def test_has_valid_processes_modifies_query(self):
        """Test that processor validators can modify query in place."""

        def modifying_validator(value, query=None):
            if query:
                query["modified"] = True
            return True

        _PROCESSOR_VALIDATOR_REGISTRY["modifier"] = modifying_validator

        query = {PROC_KEY: {"modifier": "value"}}

        result = self.validator._has_valid_processes(query)

        assert result is True
        assert query.get("modified") is True
