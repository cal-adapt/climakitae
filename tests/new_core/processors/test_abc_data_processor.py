"""
Unit tests for the abc_data_processor module.

This module contains comprehensive tests for the DataProcessor abstract base class,
the processor registry system, and the example processor implementations.
Tests cover initialization, registration, method contracts, and error handling.
"""

from abc import ABC
from unittest.mock import MagicMock

import numpy as np
import pytest
import xarray as xr

from climakitae.core.constants import UNSET
from climakitae.new_core.data_access.data_access import DataCatalog
from climakitae.new_core.processors.abc_data_processor import (
    _PROCESSOR_REGISTRY,
    ApplyBiasCorrection,
    DataProcessor,
    FilterData,
    RenameVariables,
    register_processor,
)


class TestRegistrySystem:
    """Test the processor registry system and decorator functionality."""

    def setup_method(self):
        """Clear the registry before each test."""
        global _PROCESSOR_REGISTRY
        self._original_registry = _PROCESSOR_REGISTRY.copy()
        _PROCESSOR_REGISTRY.clear()

    def teardown_method(self):
        """Restore the original registry after each test."""
        global _PROCESSOR_REGISTRY
        _PROCESSOR_REGISTRY.clear()
        _PROCESSOR_REGISTRY.update(self._original_registry)

    def test_register_processor_with_explicit_key(self):
        """Test registering a processor with an explicit key."""

        @register_processor(key="test_processor", priority=5)
        class TestProcessor(DataProcessor):
            def execute(self, result, context):
                return result

            def update_context(self, context):
                pass

            def set_data_accessor(self, catalog):
                pass

        assert "test_processor" in _PROCESSOR_REGISTRY
        registered_class, priority = _PROCESSOR_REGISTRY["test_processor"]
        assert registered_class is TestProcessor
        assert priority == 5

    def test_register_processor_with_auto_generated_key(self):
        """Test registering a processor with auto-generated key from class name."""

        @register_processor(priority=10)
        class MyCustomProcessor(DataProcessor):
            def execute(self, result, context):
                return result

            def update_context(self, context):
                pass

            def set_data_accessor(self, catalog):
                pass

        # Should generate "my_custom_processor" from "MyCustomProcessor"
        assert "my_custom_processor" in _PROCESSOR_REGISTRY
        registered_class, priority = _PROCESSOR_REGISTRY["my_custom_processor"]
        assert registered_class is MyCustomProcessor
        assert priority == 10

    def test_register_processor_without_parameters(self):
        """Test registering a processor without key or priority."""

        @register_processor()
        class SimpleProcessor(DataProcessor):
            def execute(self, result, context):
                return result

            def update_context(self, context):
                pass

            def set_data_accessor(self, catalog):
                pass

        assert "simple_processor" in _PROCESSOR_REGISTRY
        registered_class, priority = _PROCESSOR_REGISTRY["simple_processor"]
        assert registered_class is SimpleProcessor
        assert priority is UNSET

    def test_register_processor_with_unset_key(self):
        """Test registering a processor with UNSET key falls back to auto-generation."""

        @register_processor(key=UNSET, priority=1)
        class AnotherTestProcessor(DataProcessor):
            def execute(self, result, context):
                return result

            def update_context(self, context):
                pass

            def set_data_accessor(self, catalog):
                pass

        assert "another_test_processor" in _PROCESSOR_REGISTRY
        registered_class, priority = _PROCESSOR_REGISTRY["another_test_processor"]
        assert registered_class is AnotherTestProcessor
        assert priority == 1

    def test_key_generation_from_class_name(self):
        """Test various class name patterns for key generation."""

        @register_processor()
        class HTMLParser(DataProcessor):
            def execute(self, result, context):
                return result

            def update_context(self, context):
                pass

            def set_data_accessor(self, catalog):
                pass

        @register_processor()
        class XMLHTTPRequest(DataProcessor):
            def execute(self, result, context):
                return result

            def update_context(self, context):
                pass

            def set_data_accessor(self, catalog):
                pass

        assert "h_t_m_l_parser" in _PROCESSOR_REGISTRY
        assert "x_m_l_h_t_t_p_request" in _PROCESSOR_REGISTRY

    def test_registry_allows_overwriting(self):
        """Test that registering with the same key overwrites the previous entry."""

        @register_processor(key="overwrite_test")
        class FirstProcessor(DataProcessor):
            def execute(self, result, context):
                return result

            def update_context(self, context):
                pass

            def set_data_accessor(self, catalog):
                pass

        @register_processor(key="overwrite_test")
        class SecondProcessor(DataProcessor):
            def execute(self, result, context):
                return result

            def update_context(self, context):
                pass

            def set_data_accessor(self, catalog):
                pass

        # Should have the second processor
        registered_class, _ = _PROCESSOR_REGISTRY["overwrite_test"]
        assert registered_class is SecondProcessor


class TestDataProcessorABC:
    """Test the DataProcessor abstract base class."""

    def test_cannot_instantiate_abstract_class(self):
        """Test that DataProcessor cannot be instantiated directly."""
        with pytest.raises(TypeError):
            DataProcessor()

    def test_subclass_must_implement_all_abstract_methods(self):
        """Test that subclasses must implement all abstract methods."""

        class IncompleteProcessor(DataProcessor):
            # Missing all abstract methods
            pass

        with pytest.raises(TypeError):
            IncompleteProcessor()

    def test_subclass_with_missing_execute_method(self):
        """Test that subclasses must implement execute method."""

        class NoExecuteProcessor(DataProcessor):
            def update_context(self, context):
                pass

            def set_data_accessor(self, catalog):
                pass

        with pytest.raises(TypeError):
            NoExecuteProcessor()

    def test_subclass_with_missing_update_context_method(self):
        """Test that subclasses must implement update_context method."""

        class NoUpdateContextProcessor(DataProcessor):
            def execute(self, result, context):
                return result

            def set_data_accessor(self, catalog):
                pass

        with pytest.raises(TypeError):
            NoUpdateContextProcessor()

    def test_subclass_with_missing_set_data_accessor_method(self):
        """Test that subclasses must implement set_data_accessor method."""

        class NoSetDataAccessorProcessor(DataProcessor):
            def execute(self, result, context):
                return result

            def update_context(self, context):
                pass

        with pytest.raises(TypeError):
            NoSetDataAccessorProcessor()

    def test_complete_subclass_can_be_instantiated(self):
        """Test that a complete subclass can be instantiated."""

        class CompleteProcessor(DataProcessor):
            def execute(self, result, context):
                return result

            def update_context(self, context):
                pass

            def set_data_accessor(self, catalog):
                pass

        # Should not raise any exception
        processor = CompleteProcessor()
        assert isinstance(processor, DataProcessor)
        assert isinstance(processor, ABC)

    def test_abstract_method_signatures(self):
        """Test that abstract methods have the expected signatures."""

        class TestProcessor(DataProcessor):
            def execute(self, result, context):
                # Verify signature matches expected types
                assert isinstance(context, dict)
                return result

            def update_context(self, context):
                assert isinstance(context, dict)

            def set_data_accessor(self, catalog):
                assert isinstance(catalog, DataCatalog)

        processor = TestProcessor()

        # Test with mock data
        mock_dataset = xr.Dataset()
        context = {"test": "value"}
        mock_catalog = MagicMock(spec=DataCatalog)

        result = processor.execute(mock_dataset, context)
        processor.update_context(context)
        processor.set_data_accessor(mock_catalog)

        assert result is mock_dataset


class TestConcreteProcessors:
    """Test the concrete processor implementations."""

    @pytest.fixture
    def sample_dataset(self):
        """Create a sample xarray Dataset for testing."""
        data = np.random.rand(10, 5)
        coords = {"time": range(10), "location": range(5)}
        return xr.Dataset({"temperature": (["time", "location"], data)}, coords=coords)

    @pytest.fixture
    def sample_dataarray(self):
        """Create a sample xarray DataArray for testing."""
        data = np.random.rand(8, 3)
        coords = {"x": range(8), "y": range(3)}
        return xr.DataArray(data, coords=coords, dims=["x", "y"])

    @pytest.fixture
    def sample_context(self):
        """Create a sample context dictionary."""
        return {
            "variable_id": "temperature",
            "time_range": "2020-2030",
            "location": "California",
        }

    @pytest.fixture
    def mock_datacatalog(self):
        """Create a mock DataCatalog."""
        return MagicMock(spec=DataCatalog)

    def test_rename_variables_processor(
        self, sample_dataset, sample_context, mock_datacatalog
    ):
        """Test RenameVariables processor."""
        processor = RenameVariables()

        # Test execute method (placeholder implementation)
        result = processor.execute(sample_dataset, sample_context)
        assert result is sample_dataset  # Should return unchanged for now

        # Test update_context method
        original_context = sample_context.copy()
        processor.update_context(sample_context)
        assert sample_context == original_context  # Should be unchanged for now

        # Test set_data_accessor method
        processor.set_data_accessor(mock_datacatalog)  # Should not raise exception

    def test_apply_bias_correction_processor(
        self, sample_dataarray, sample_context, mock_datacatalog
    ):
        """Test ApplyBiasCorrection processor."""
        processor = ApplyBiasCorrection()

        # Test execute method with DataArray
        result = processor.execute(sample_dataarray, sample_context)
        assert result is sample_dataarray

        # Test with iterable of datasets
        dataset_list = [sample_dataarray, sample_dataarray]
        result = processor.execute(dataset_list, sample_context)
        assert result is dataset_list

        # Test other methods
        processor.update_context(sample_context)
        processor.set_data_accessor(mock_datacatalog)

    def test_filter_data_processor(
        self, sample_dataset, sample_context, mock_datacatalog
    ):
        """Test FilterData processor."""
        processor = FilterData()

        # Test execute method
        result = processor.execute(sample_dataset, sample_context)
        assert result is sample_dataset

        # Test other methods
        processor.update_context(sample_context)
        processor.set_data_accessor(mock_datacatalog)

    def test_processors_inherit_from_dataprocessor(self):
        """Test that all concrete processors inherit from DataProcessor."""
        processors = [RenameVariables(), ApplyBiasCorrection(), FilterData()]

        for processor in processors:
            assert isinstance(processor, DataProcessor)
            assert hasattr(processor, "execute")
            assert hasattr(processor, "update_context")
            assert hasattr(processor, "set_data_accessor")

    def test_processors_handle_different_input_types(
        self, sample_dataset, sample_dataarray, sample_context
    ):
        """Test that processors can handle different input types."""
        processors = [RenameVariables(), ApplyBiasCorrection(), FilterData()]

        for processor in processors:
            # Test with Dataset
            result1 = processor.execute(sample_dataset, sample_context)
            assert result1 is not None

            # Test with DataArray
            result2 = processor.execute(sample_dataarray, sample_context)
            assert result2 is not None

            # Test with list of datasets
            dataset_list = [sample_dataset, sample_dataarray]
            result3 = processor.execute(dataset_list, sample_context)
            assert result3 is not None


class TestProcessorIntegration:
    """Test processor integration scenarios."""

    def setup_method(self):
        """Clear registry before each test."""
        global _PROCESSOR_REGISTRY
        self._original_registry = _PROCESSOR_REGISTRY.copy()
        _PROCESSOR_REGISTRY.clear()

    def teardown_method(self):
        """Restore registry after each test."""
        global _PROCESSOR_REGISTRY
        _PROCESSOR_REGISTRY.clear()
        _PROCESSOR_REGISTRY.update(self._original_registry)

    def test_register_and_retrieve_processor(self):
        """Test registering a processor and retrieving it from registry."""

        @register_processor(key="integration_test", priority=1)
        class IntegrationProcessor(DataProcessor):
            def execute(self, result, context):
                return result

            def update_context(self, context):
                context["processed"] = True

            def set_data_accessor(self, catalog):
                self.catalog = catalog

        # Verify registration
        assert "integration_test" in _PROCESSOR_REGISTRY

        # Retrieve and instantiate
        processor_class, _ = _PROCESSOR_REGISTRY["integration_test"]
        processor = processor_class()

        # Test functionality
        context = {}
        processor.update_context(context)
        assert context["processed"] is True

        mock_catalog = MagicMock(spec=DataCatalog)
        processor.set_data_accessor(mock_catalog)
        assert processor.catalog is mock_catalog

    def test_multiple_processors_with_priorities(self):
        """Test registering multiple processors with different priorities."""

        @register_processor(key="high_priority", priority=1)
        class HighPriorityProcessor(DataProcessor):
            def execute(self, result, context):
                return result

            def update_context(self, context):
                pass

            def set_data_accessor(self, catalog):
                pass

        @register_processor(key="low_priority", priority=10)
        class LowPriorityProcessor(DataProcessor):
            def execute(self, result, context):
                return result

            def update_context(self, context):
                pass

            def set_data_accessor(self, catalog):
                pass

        # Check both are registered with correct priorities
        _, high_priority = _PROCESSOR_REGISTRY["high_priority"]
        _, low_priority = _PROCESSOR_REGISTRY["low_priority"]

        assert high_priority < low_priority

    def test_processor_registry_persistence(self):
        """Test that registry maintains state across multiple registrations."""

        @register_processor("first")
        class FirstProcessor(DataProcessor):
            def execute(self, result, context):
                return result

            def update_context(self, context):
                pass

            def set_data_accessor(self, catalog):
                pass

        assert len(_PROCESSOR_REGISTRY) == 1

        @register_processor("second")
        class SecondProcessor(DataProcessor):
            def execute(self, result, context):
                return result

            def update_context(self, context):
                pass

            def set_data_accessor(self, catalog):
                pass

        assert len(_PROCESSOR_REGISTRY) == 2
        assert "first" in _PROCESSOR_REGISTRY
        assert "second" in _PROCESSOR_REGISTRY


class TestEdgeCases:
    """Test edge cases and error conditions."""

    def test_register_processor_with_empty_string_key(self):
        """Test registering with empty string key."""
        global _PROCESSOR_REGISTRY
        _PROCESSOR_REGISTRY.clear()

        @register_processor(key="", priority=1)
        class EmptyKeyProcessor(DataProcessor):
            def execute(self, result, context):
                return result

            def update_context(self, context):
                pass

            def set_data_accessor(self, catalog):
                pass

        assert "" in _PROCESSOR_REGISTRY

    def test_register_processor_with_none_priority(self):
        """Test registering with None priority."""
        global _PROCESSOR_REGISTRY
        _PROCESSOR_REGISTRY.clear()

        @register_processor(key="none_priority", priority=None)
        class NonePriorityProcessor(DataProcessor):
            def execute(self, result, context):
                return result

            def update_context(self, context):
                pass

            def set_data_accessor(self, catalog):
                pass

        _, priority = _PROCESSOR_REGISTRY["none_priority"]
        assert priority is None

    def test_processors_with_complex_data_types(self):
        """Test processors with complex xarray data structures."""
        processor = RenameVariables()

        # Create complex dataset with multiple variables and dimensions
        data = xr.Dataset(
            {
                "temperature": (["x", "y", "time"], np.random.rand(5, 5, 10)),
                "humidity": (["x", "y", "time"], np.random.rand(5, 5, 10)),
            }
        )

        context = {"complex": True}
        result = processor.execute(data, context)

        # Should handle complex data without issues
        assert result is data
        assert isinstance(result, xr.Dataset)
        assert "temperature" in result.data_vars
        assert "humidity" in result.data_vars

    def test_registry_state_isolation(self):
        """Test that registry modifications don't affect other tests."""
        # This test verifies our setup/teardown works correctly
        global _PROCESSOR_REGISTRY

        # Start with clean registry
        original_keys = set(_PROCESSOR_REGISTRY.keys())

        @register_processor("isolation_test")
        class IsolationProcessor(DataProcessor):
            def execute(self, result, context):
                return result

            def update_context(self, context):
                pass

            def set_data_accessor(self, catalog):
                pass

        # Should have one more key
        assert len(_PROCESSOR_REGISTRY) == len(original_keys) + 1
        assert "isolation_test" in _PROCESSOR_REGISTRY
