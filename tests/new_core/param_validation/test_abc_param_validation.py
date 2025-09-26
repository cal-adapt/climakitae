"""
Unit tests for climakitae/new_core/param_validation/abc_param_validation.py

This module contains comprehensive unit tests for the parameter validation
framework including decorator registration, abstract base class functionality,
and validation logic.
"""

import warnings
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from climakitae.core.constants import PROC_KEY, UNSET
from climakitae.new_core.param_validation.abc_param_validation import (
    _CATALOG_VALIDATOR_REGISTRY,
    _PROCESSOR_VALIDATOR_REGISTRY,
    ParameterValidator,
    register_catalog_validator,
    register_processor_validator,
)

# Suppress known external warnings that are not relevant to our tests
warnings.filterwarnings(
    "ignore",
    message="The 'shapely.geos' module is deprecated",
    category=DeprecationWarning,
)
warnings.filterwarnings(
    "ignore", message="pkg_resources is deprecated", category=DeprecationWarning
)


class TestRegisterCatalogValidator:
    """Test class for the register_catalog_validator decorator."""

    def setup_method(self):
        """Save original registry state and clear for testing."""
        self._original_catalog_registry = _CATALOG_VALIDATOR_REGISTRY.copy()
        _CATALOG_VALIDATOR_REGISTRY.clear()

    def teardown_method(self):
        """Restore original registry state."""
        _CATALOG_VALIDATOR_REGISTRY.clear()
        _CATALOG_VALIDATOR_REGISTRY.update(self._original_catalog_registry)

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


class TestRegisterProcessorValidator:
    """Test class for the register_processor_validator decorator."""

    def setup_method(self):
        """Save original registry state and clear for testing."""
        self._original_processor_registry = _PROCESSOR_VALIDATOR_REGISTRY.copy()
        _PROCESSOR_VALIDATOR_REGISTRY.clear()

    def teardown_method(self):
        """Restore original registry state."""
        _PROCESSOR_VALIDATOR_REGISTRY.clear()
        _PROCESSOR_VALIDATOR_REGISTRY.update(self._original_processor_registry)

    def test_register_processor_validator_successful(self):
        """Test successful registration of a processor validator."""

        @register_processor_validator("spatial_subset")
        def validate_spatial(value, query=None):
            return True

        assert "spatial_subset" in _PROCESSOR_VALIDATOR_REGISTRY
        assert _PROCESSOR_VALIDATOR_REGISTRY["spatial_subset"] is validate_spatial

    def test_register_processor_validator_returns_function(self):
        """Test that decorator returns the function unchanged."""

        @register_processor_validator("test_processor")
        def validate_test(value, query=None):
            return value == "valid"

        # Function should work normally
        assert validate_test("valid") is True
        assert validate_test("invalid") is False

    def test_register_multiple_processors(self):
        """Test registering multiple processor validators."""

        @register_processor_validator("proc1")
        def validate1(value, query=None):
            return True

        @register_processor_validator("proc2")
        def validate2(value, query=None):
            return False

        assert "proc1" in _PROCESSOR_VALIDATOR_REGISTRY
        assert "proc2" in _PROCESSOR_VALIDATOR_REGISTRY
        assert _PROCESSOR_VALIDATOR_REGISTRY["proc1"]("test") is True
        assert _PROCESSOR_VALIDATOR_REGISTRY["proc2"]("test") is False


class ConcreteValidator(ParameterValidator):
    """Concrete implementation for testing abstract class."""

    def is_valid_query(self, query):
        """Implement abstract method for testing."""
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
        self._original_processor_registry = _PROCESSOR_VALIDATOR_REGISTRY.copy()
        _PROCESSOR_VALIDATOR_REGISTRY.clear()
        with patch(
            "climakitae.new_core.param_validation.abc_param_validation.DataCatalog"
        ):
            self.validator = ConcreteValidator()

    def teardown_method(self):
        """Restore original registry state."""
        _PROCESSOR_VALIDATOR_REGISTRY.clear()
        _PROCESSOR_VALIDATOR_REGISTRY.update(self._original_processor_registry)

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


class TestIsValidQuery:
    """Test class for _is_valid_query method."""

    def setup_method(self):
        """Set up test fixtures."""
        self._original_processor_registry = _PROCESSOR_VALIDATOR_REGISTRY.copy()
        _PROCESSOR_VALIDATOR_REGISTRY.clear()
        with patch(
            "climakitae.new_core.param_validation.abc_param_validation.DataCatalog"
        ):
            self.validator = ConcreteValidator()
            self.validator.catalog_df = pd.DataFrame(
                {
                    "variable": ["tas", "pr", "tasmax", "tasmin"],
                    "experiment_id": ["ssp245", "ssp245", "historical", "ssp585"],
                    "source_id": ["model1", "model2", "model1", "model3"],
                    "grid_label": ["gr1", "gr1", "gr2", "gr1"],
                }
            )

            # Mock catalog
            self.validator.catalog = MagicMock()
            self.validator.catalog.df = self.validator.catalog_df

            # Initialize all_catalog_keys
            self.validator.all_catalog_keys = {
                "variable": UNSET,
                "experiment_id": UNSET,
                "source_id": UNSET,
                "grid_label": UNSET,
            }

    def teardown_method(self):
        """Restore original registry state."""
        _PROCESSOR_VALIDATOR_REGISTRY.clear()
        _PROCESSOR_VALIDATOR_REGISTRY.update(self._original_processor_registry)

    def test_is_valid_query_successful_match(self):
        """Test _is_valid_query with successful dataset match."""
        query = {"variable": "tas", "experiment_id": "ssp245", "source_id": "model1"}

        # Mock successful search
        self.validator.catalog.search.return_value = MagicMock(__len__=lambda self: 5)

        with patch("builtins.print") as mock_print:
            result = self.validator._is_valid_query(query)

        assert result == {
            "variable": "tas",
            "experiment_id": "ssp245",
            "source_id": "model1",
        }
        mock_print.assert_any_call("Found 5 datasets matching your query.")

    def test_is_valid_query_no_matches(self):
        """Test _is_valid_query when no datasets match."""
        query = {"variable": "nonexistent"}

        # Mock catalog search returning empty result (0 length)
        self.validator.catalog.search.return_value = MagicMock(__len__=lambda self: 0)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=UserWarning)
            with patch("builtins.print"):
                result = self.validator._is_valid_query(query)

        assert result is None
        # Note: No warning about "Query did not match any datasets" since we're
        # not raising ValueError but the method should still return None when
        # no datasets are found

    @patch(
        "climakitae.new_core.param_validation.abc_param_validation._get_closest_options"
    )
    def test_is_valid_query_with_typo(self, mock_get_closest):
        """Test _is_valid_query suggests corrections for typos."""
        query = {"variable": "tass"}  # Typo for 'tas'

        self.validator.catalog.search.return_value = MagicMock(__len__=lambda self: 0)
        mock_get_closest.return_value = ["tas", "tasmax"]

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            with patch("builtins.print") as mock_print:
                result = self.validator._is_valid_query(query)

        assert result is None
        mock_print.assert_any_call("Could not find any datasets with variable = tass.")
        assert any(
            "Did you mean one of these options" in str(warning.message) for warning in w
        )

    @patch(
        "climakitae.new_core.param_validation.abc_param_validation._validate_experimental_id_param"
    )
    def test_is_valid_query_experiment_id_list(self, mock_validate_exp):
        """Test _is_valid_query with experiment_id as list."""
        query = {"experiment_id": ["ssp245", "ssp585"]}

        self.validator.catalog.search.return_value = MagicMock(__len__=lambda self: 0)
        mock_validate_exp.return_value = True

        with patch("builtins.print"):
            self.validator._is_valid_query(query)

        mock_validate_exp.assert_called_once_with(
            ["ssp245", "ssp585"], ["ssp245", "historical", "ssp585"]
        )

    def test_is_valid_query_invalid_key(self):
        """Test _is_valid_query with key not in catalog."""
        query = {"invalid_key": "value"}

        self.validator.catalog.search.return_value = MagicMock(__len__=lambda self: 0)

        with patch("builtins.print"):
            result = self.validator._is_valid_query(query)

        # The method returns empty dict when no valid catalog keys are provided
        # rather than None (this is the current implementation behavior)
        assert result == {}

    def test_is_valid_query_conflicting_parameters(self):
        """Test _is_valid_query with conflicting parameters."""
        query = {
            "variable": "tas",
            "grid_label": "gr2",  # Conflicts with tas which has gr1
        }

        self.validator.catalog.search.return_value = MagicMock(__len__=lambda self: 0)

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            with patch("builtins.print"):
                result = self.validator._is_valid_query(query)

        assert result is None
        assert any(
            "conflict between grid_label and variable" in str(warning.message)
            for warning in w
        )

    def test_is_valid_query_with_invalid_processes(self):
        """Test _is_valid_query fails when processes are invalid."""
        query = {"variable": "tas", PROC_KEY: {"invalid_proc": "value"}}

        # Mock successful catalog search
        self.validator.catalog.search.return_value = MagicMock(__len__=lambda self: 5)

        # Register a failing processor validator
        _PROCESSOR_VALIDATOR_REGISTRY["invalid_proc"] = lambda v, query=None: False

        with warnings.catch_warnings(record=True):
            warnings.simplefilter("always")
            with patch("builtins.print"):
                result = self.validator._is_valid_query(query)

        assert result is None


class TestParameterValidatorAbstract:
    """Test class for abstract method enforcement."""

    def test_cannot_instantiate_abstract_class(self):
        """Test that ParameterValidator cannot be instantiated directly."""
        with pytest.raises(TypeError, match="Can't instantiate abstract class"):
            ParameterValidator()

    def test_must_implement_is_valid_query(self):
        """Test that subclasses must implement is_valid_query."""

        class IncompleteValidator(ParameterValidator):
            pass

        with pytest.raises(TypeError, match="Can't instantiate abstract class"):
            IncompleteValidator()


class TestParameterValidatorIntegration:
    """Integration tests for complete validation workflows."""

    def setup_method(self):
        """Set up test fixtures."""
        self._original_catalog_registry = _CATALOG_VALIDATOR_REGISTRY.copy()
        self._original_processor_registry = _PROCESSOR_VALIDATOR_REGISTRY.copy()
        _CATALOG_VALIDATOR_REGISTRY.clear()
        _PROCESSOR_VALIDATOR_REGISTRY.clear()

    def teardown_method(self):
        """Restore original registry states."""
        _CATALOG_VALIDATOR_REGISTRY.clear()
        _CATALOG_VALIDATOR_REGISTRY.update(self._original_catalog_registry)
        _PROCESSOR_VALIDATOR_REGISTRY.clear()
        _PROCESSOR_VALIDATOR_REGISTRY.update(self._original_processor_registry)

    def test_complete_validation_workflow(self):
        """Test complete validation workflow with catalog and processor validators."""

        # Register a catalog validator
        @register_catalog_validator("test_catalog")
        class TestCatalogValidator(ParameterValidator):
            def is_valid_query(self, query):
                return self._is_valid_query(query)

        # Register processor validators
        @register_processor_validator("spatial_subset")
        def validate_spatial(value, query=None):
            return isinstance(value, dict) and "bounds" in value

        @register_processor_validator("temporal_average")
        def validate_temporal(value, query=None):
            return value in ["daily", "monthly", "yearly"]

        # Create validator instance
        with patch(
            "climakitae.new_core.param_validation.abc_param_validation.DataCatalog"
        ) as mock_dc:
            mock_dc.return_value.catalog_df = pd.DataFrame(
                {"variable": ["tas", "pr"], "experiment_id": ["ssp245", "ssp245"]}
            )

            validator = _CATALOG_VALIDATOR_REGISTRY["test_catalog"]()
            validator.catalog = MagicMock()
            validator.catalog.df = validator.catalog_df
            validator.catalog.search.return_value = MagicMock(__len__=lambda self: 2)
            validator.all_catalog_keys = {"variable": UNSET, "experiment_id": UNSET}

            # Valid query
            query = {
                "variable": "tas",
                "experiment_id": "ssp245",
                PROC_KEY: {
                    "spatial_subset": {"bounds": [1, 2, 3, 4]},
                    "temporal_average": "monthly",
                },
            }

            with patch("builtins.print"):
                result = validator.is_valid_query(query)

            assert result == {"variable": "tas", "experiment_id": "ssp245"}
