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

    def test_with_param_validator_returns_self(self):
        """Test with_param_validator returns Dataset instance for chaining."""
        dataset = Dataset()
        mock_validator = MagicMock(spec=ParameterValidator)

        result = dataset.with_param_validator(mock_validator)

        assert result is dataset

    def test_with_param_validator_invalid_type(self):
        """Test with_param_validator raises TypeError for non-ParameterValidator."""
        dataset = Dataset()
        invalid_validator = {"is_valid_query": lambda x: x}  # dict, not validator

        with pytest.raises(
            TypeError, match="must be an instance of ParameterValidator"
        ):
            dataset.with_param_validator(invalid_validator)

    def test_with_param_validator_none(self):
        """Test with_param_validator raises TypeError for None value."""
        dataset = Dataset()

        with pytest.raises(
            TypeError, match="must be an instance of ParameterValidator"
        ):
            dataset.with_param_validator(None)


def _create_mock_processor(return_value=None, needs_catalog=False):
    """Create a mock processor with required methods for testing.

    Parameters
    ----------
    return_value : xr.Dataset or None
        Value to return from execute method
    needs_catalog : bool
        Whether the processor needs catalog access

    Returns
    -------
    MagicMock
        Mock processor with execute, update_context, and set_data_accessor methods
    """
    mock_processor = MagicMock()
    mock_processor.execute = MagicMock(return_value=return_value)
    mock_processor.update_context = MagicMock()
    mock_processor.set_data_accessor = MagicMock()
    mock_processor.needs_catalog = needs_catalog
    mock_processor.name = "MockProcessor"
    return mock_processor


class TestDatasetWithProcessingStep:
    """Test class for with_processing_step method."""

    def test_with_processing_step_valid(self):
        """Test with_processing_step with valid processor."""
        dataset = Dataset()
        mock_processor = _create_mock_processor()

        dataset.with_processing_step(mock_processor)

        assert dataset.processing_pipeline == [mock_processor]

    def test_with_processing_step_returns_self(self):
        """Test with_processing_step returns Dataset instance for chaining."""
        dataset = Dataset()
        mock_processor = _create_mock_processor()

        result = dataset.with_processing_step(mock_processor)

        assert result is dataset

    def test_with_processing_step_initializes_pipeline(self):
        """Test with_processing_step converts UNSET to list on first call."""
        dataset = Dataset()
        assert dataset.processing_pipeline is UNSET

        mock_processor = _create_mock_processor()
        dataset.with_processing_step(mock_processor)

        assert isinstance(dataset.processing_pipeline, list)
        assert len(dataset.processing_pipeline) == 1

    def test_with_processing_step_multiple_steps(self):
        """Test with_processing_step appends multiple steps in order."""
        dataset = Dataset()
        processor1 = _create_mock_processor()
        processor2 = _create_mock_processor()
        processor3 = _create_mock_processor()

        dataset.with_processing_step(processor1)
        dataset.with_processing_step(processor2)
        dataset.with_processing_step(processor3)

        assert len(dataset.processing_pipeline) == 3
        assert dataset.processing_pipeline[0] is processor1
        assert dataset.processing_pipeline[1] is processor2
        assert dataset.processing_pipeline[2] is processor3

    def test_with_processing_step_missing_execute(self):
        """Test with_processing_step raises TypeError if step lacks execute method."""
        dataset = Dataset()
        mock_processor = MagicMock()
        del mock_processor.execute  # Remove execute
        mock_processor.update_context = MagicMock()
        mock_processor.set_data_accessor = MagicMock()

        with pytest.raises(TypeError, match="must have an 'execute' method"):
            dataset.with_processing_step(mock_processor)

    def test_with_processing_step_missing_update_context(self):
        """Test with_processing_step raises AttributeError if step lacks update_context."""
        dataset = Dataset()
        mock_processor = MagicMock()
        mock_processor.execute = MagicMock()
        del mock_processor.update_context  # Remove update_context
        mock_processor.set_data_accessor = MagicMock()

        with pytest.raises(
            AttributeError, match="must have an 'update_context' method"
        ):
            dataset.with_processing_step(mock_processor)

    def test_with_processing_step_missing_set_data_accessor(self):
        """Test with_processing_step raises TypeError if step lacks set_data_accessor."""
        dataset = Dataset()
        mock_processor = MagicMock()
        mock_processor.execute = MagicMock()
        mock_processor.update_context = MagicMock()
        del mock_processor.set_data_accessor  # Remove set_data_accessor

        with pytest.raises(TypeError, match="must have a 'set_data_accessor' method"):
            dataset.with_processing_step(mock_processor)

    def test_with_processing_step_non_callable_execute(self):
        """Test with_processing_step raises TypeError if execute is not callable."""
        dataset = Dataset()
        mock_processor = MagicMock()
        mock_processor.execute = "not_a_function"  # Non-callable
        mock_processor.update_context = MagicMock()
        mock_processor.set_data_accessor = MagicMock()

        with pytest.raises(TypeError, match="must have an 'execute' method"):
            dataset.with_processing_step(mock_processor)


class TestDatasetMethodChaining:
    """Test class for fluent interface / method chaining behavior."""

    def test_method_chaining_all_methods(self):
        """Test chaining all with_* methods together in fluent interface."""
        mock_catalog = MagicMock(spec=DataCatalog)
        mock_catalog.get_data = MagicMock()
        mock_validator = MagicMock(spec=ParameterValidator)
        mock_processor = _create_mock_processor()

        dataset = (
            Dataset()
            .with_catalog(mock_catalog)
            .with_param_validator(mock_validator)
            .with_processing_step(mock_processor)
        )

        assert dataset.data_access is mock_catalog
        assert dataset.parameter_validator is mock_validator
        assert dataset.processing_pipeline == [mock_processor]

    def test_method_chaining_returns_same_instance(self):
        """Test each chained call returns the same Dataset instance."""
        mock_catalog = MagicMock(spec=DataCatalog)
        mock_catalog.get_data = MagicMock()
        mock_validator = MagicMock(spec=ParameterValidator)
        mock_processor = _create_mock_processor()

        dataset = Dataset()
        result1 = dataset.with_catalog(mock_catalog)
        result2 = result1.with_param_validator(mock_validator)
        result3 = result2.with_processing_step(mock_processor)

        assert result1 is dataset
        assert result2 is dataset
        assert result3 is dataset


class TestDatasetExecute:
    """Test class for execute method - basic execution paths."""

    def setup_method(self):
        """Set up test fixtures."""
        self.sample_dataset = xr.Dataset(
            {"temp": (["x", "y"], [[1.0, 2.0], [3.0, 4.0]])},
            coords={"x": [0, 1], "y": [0, 1]},
        )

    def test_execute_minimal_no_pipeline(self):
        """Test execute with just catalog, no validator or pipeline - returns raw data."""
        mock_catalog = MagicMock(spec=DataCatalog)
        mock_catalog.get_data = MagicMock(return_value=self.sample_dataset)

        dataset = Dataset().with_catalog(mock_catalog)

        result = dataset.execute({"variable": "temp", "_catalog_key": "cadcat"})

        assert isinstance(result, xr.Dataset)
        mock_catalog.get_data.assert_called_once()

    def test_execute_with_parameters(self):
        """Test execute with parameters - context should be initialized."""
        mock_catalog = MagicMock(spec=DataCatalog)
        mock_catalog.get_data = MagicMock(return_value=self.sample_dataset)

        dataset = Dataset().with_catalog(mock_catalog)
        parameters = {"variable": "temp", "grid_label": "d03", "_catalog_key": "cadcat"}

        result = dataset.execute(parameters)

        assert isinstance(result, xr.Dataset)
        # Verify parameters were passed to get_data via validation (UNSET without validator)
        mock_catalog.get_data.assert_called_once()

    def test_execute_without_parameters(self):
        """Test execute with minimal parameters - catalog_key is still required."""
        mock_catalog = MagicMock(spec=DataCatalog)
        mock_catalog.get_data = MagicMock(return_value=self.sample_dataset)

        dataset = Dataset().with_catalog(mock_catalog)

        # _catalog_key is now required for thread-safe operation
        result = dataset.execute({"_catalog_key": "cadcat"})

        assert isinstance(result, xr.Dataset)
        # Should call get_data with UNSET (no validator, so valid_query stays UNSET)
        mock_catalog.get_data.assert_called_once()

    def test_execute_returns_xarray_dataset(self):
        """Test execute returns xr.Dataset."""
        mock_catalog = MagicMock(spec=DataCatalog)
        mock_catalog.get_data = MagicMock(return_value=self.sample_dataset)

        dataset = Dataset().with_catalog(mock_catalog)

        result = dataset.execute({"variable": "temp", "_catalog_key": "cadcat"})

        assert isinstance(result, xr.Dataset)
        assert "temp" in result.data_vars


class TestDatasetExecuteValidation:
    """Test class for execute method - validation paths."""

    def setup_method(self):
        """Set up test fixtures."""
        self.sample_dataset = xr.Dataset(
            {"temp": (["x", "y"], [[1.0, 2.0], [3.0, 4.0]])},
            coords={"x": [0, 1], "y": [0, 1]},
        )

    def test_execute_validation_success(self):
        """Test execute with validator that returns valid query."""
        mock_catalog = MagicMock(spec=DataCatalog)
        mock_catalog.get_data = MagicMock(return_value=self.sample_dataset)

        mock_validator = MagicMock(spec=ParameterValidator)
        validated_query = {
            "variable": "temp",
            "validated": True,
            "_catalog_key": "cadcat",
        }
        mock_validator.is_valid_query = MagicMock(return_value=validated_query)

        dataset = (
            Dataset().with_catalog(mock_catalog).with_param_validator(mock_validator)
        )

        result = dataset.execute({"variable": "temp", "_catalog_key": "cadcat"})

        assert isinstance(result, xr.Dataset)
        mock_validator.is_valid_query.assert_called_once()
        mock_catalog.get_data.assert_called_once()

    def test_execute_validation_failure_returns_empty_dataset(self):
        """Test execute returns None when validation fails."""
        mock_catalog = MagicMock(spec=DataCatalog)
        mock_catalog.get_data = MagicMock(return_value=self.sample_dataset)

        mock_validator = MagicMock(spec=ParameterValidator)
        mock_validator.is_valid_query = MagicMock(return_value=None)  # Validation fails

        dataset = (
            Dataset().with_catalog(mock_catalog).with_param_validator(mock_validator)
        )

        result = dataset.execute({"invalid": "params"})

        assert result is None  # Returns None when validation fails
        mock_catalog.get_data.assert_not_called()  # Should not call get_data

    def test_execute_without_validator(self):
        """Test execute skips validation when no validator is set."""
        mock_catalog = MagicMock(spec=DataCatalog)
        mock_catalog.get_data = MagicMock(return_value=self.sample_dataset)

        dataset = Dataset().with_catalog(mock_catalog)
        # No validator set

        result = dataset.execute({"variable": "temp", "_catalog_key": "cadcat"})

        assert isinstance(result, xr.Dataset)
        # get_data should be called with UNSET since no validator processed the query
        mock_catalog.get_data.assert_called_once()


class TestDatasetExecuteProcessing:
    """Test class for execute method - processing step execution paths."""

    def setup_method(self):
        """Set up test fixtures."""
        self.sample_dataset = xr.Dataset(
            {"temp": (["x", "y"], [[1.0, 2.0], [3.0, 4.0]])},
            coords={"x": [0, 1], "y": [0, 1]},
        )
        self.processed_dataset = xr.Dataset(
            {"temp_processed": (["x", "y"], [[10.0, 20.0], [30.0, 40.0]])},
            coords={"x": [0, 1], "y": [0, 1]},
        )

    def test_execute_single_processing_step(self):
        """Test execute with single processor transforms data."""
        mock_catalog = MagicMock(spec=DataCatalog)
        mock_catalog.get_data = MagicMock(return_value=self.sample_dataset)

        mock_processor = _create_mock_processor(return_value=self.processed_dataset)

        dataset = (
            Dataset().with_catalog(mock_catalog).with_processing_step(mock_processor)
        )

        result = dataset.execute({"variable": "temp", "_catalog_key": "cadcat"})

        assert result is self.processed_dataset
        mock_processor.execute.assert_called_once()

    def test_execute_multiple_processing_steps_sequential(self):
        """Test execute with multiple processors executes in order."""
        mock_catalog = MagicMock(spec=DataCatalog)
        mock_catalog.get_data = MagicMock(return_value=self.sample_dataset)

        intermediate_dataset = xr.Dataset(
            {"temp_intermediate": (["x", "y"], [[5.0, 10.0], [15.0, 20.0]])},
            coords={"x": [0, 1], "y": [0, 1]},
        )

        processor1 = _create_mock_processor(return_value=intermediate_dataset)
        processor2 = _create_mock_processor(return_value=self.processed_dataset)

        dataset = (
            Dataset()
            .with_catalog(mock_catalog)
            .with_processing_step(processor1)
            .with_processing_step(processor2)
        )

        result = dataset.execute({"variable": "temp", "_catalog_key": "cadcat"})

        assert result is self.processed_dataset
        processor1.execute.assert_called_once()
        processor2.execute.assert_called_once()
        # Verify processor2 received output from processor1
        call_args = processor2.execute.call_args
        assert call_args[0][0] is intermediate_dataset

    def test_execute_processing_step_receives_context(self):
        """Test processing step receives context dictionary."""
        mock_catalog = MagicMock(spec=DataCatalog)
        mock_catalog.get_data = MagicMock(return_value=self.sample_dataset)

        mock_processor = _create_mock_processor(return_value=self.processed_dataset)

        dataset = (
            Dataset().with_catalog(mock_catalog).with_processing_step(mock_processor)
        )

        parameters = {"variable": "temp", "grid_label": "d03", "_catalog_key": "cadcat"}
        dataset.execute(parameters)

        # Verify execute was called with context
        call_args = mock_processor.execute.call_args
        context = call_args[0][1]  # Second positional argument
        assert isinstance(context, dict)

    def test_execute_processing_step_needs_catalog(self):
        """Test processing step with needs_catalog=True receives catalog."""
        mock_catalog = MagicMock(spec=DataCatalog)
        mock_catalog.get_data = MagicMock(return_value=self.sample_dataset)

        mock_processor = _create_mock_processor(
            return_value=self.processed_dataset, needs_catalog=True
        )

        dataset = (
            Dataset().with_catalog(mock_catalog).with_processing_step(mock_processor)
        )

        dataset.execute({"variable": "temp", "_catalog_key": "cadcat"})

        # Verify set_data_accessor was called with the catalog
        mock_processor.set_data_accessor.assert_called_once_with(mock_catalog)

    def test_execute_processing_step_returns_none_continues(self):
        """Test execute continues when processor returns None (with warning)."""
        mock_catalog = MagicMock(spec=DataCatalog)
        mock_catalog.get_data = MagicMock(return_value=self.sample_dataset)

        # Processor that returns None
        mock_processor = _create_mock_processor(return_value=None)

        dataset = (
            Dataset().with_catalog(mock_catalog).with_processing_step(mock_processor)
        )

        # Should not raise, but result will be None
        result = dataset.execute({"variable": "temp", "_catalog_key": "cadcat"})

        assert result is None
        mock_processor.execute.assert_called_once()

    def test_execute_empty_pipeline_returns_raw_data(self):
        """Test execute with empty pipeline list returns raw data."""
        mock_catalog = MagicMock(spec=DataCatalog)
        mock_catalog.get_data = MagicMock(return_value=self.sample_dataset)

        dataset = Dataset().with_catalog(mock_catalog)
        dataset.processing_pipeline = []  # Explicitly set to empty list

        result = dataset.execute({"variable": "temp", "_catalog_key": "cadcat"})

        assert result is self.sample_dataset


class TestDatasetExecuteErrorHandling:
    """Test class for execute method - error scenarios."""

    def setup_method(self):
        """Set up test fixtures."""
        self.sample_dataset = xr.Dataset(
            {"temp": (["x", "y"], [[1.0, 2.0], [3.0, 4.0]])},
            coords={"x": [0, 1], "y": [0, 1]},
        )

    def test_execute_missing_data_accessor_raises(self):
        """Test execute raises ValueError when data_access is UNSET."""
        dataset = Dataset()
        # data_access is UNSET by default

        with pytest.raises(ValueError, match="Data accessor is not configured"):
            dataset.execute({"variable": "temp", "_catalog_key": "cadcat"})

    def test_execute_processing_step_exception_raises_runtime_error(self):
        """Test execute wraps processor exceptions in RuntimeError."""
        mock_catalog = MagicMock(spec=DataCatalog)
        mock_catalog.get_data = MagicMock(return_value=self.sample_dataset)

        mock_processor = _create_mock_processor()
        mock_processor.execute.side_effect = ValueError("Processing failed")

        dataset = (
            Dataset().with_catalog(mock_catalog).with_processing_step(mock_processor)
        )

        with pytest.raises(RuntimeError, match="Error in processing pipeline"):
            dataset.execute({"variable": "temp", "_catalog_key": "cadcat"})

    def test_execute_processing_step_exception_includes_original_message(self):
        """Test RuntimeError message includes original exception info."""
        mock_catalog = MagicMock(spec=DataCatalog)
        mock_catalog.get_data = MagicMock(return_value=self.sample_dataset)

        mock_processor = _create_mock_processor()
        original_error_msg = "Specific error from processor"
        mock_processor.execute.side_effect = ValueError(original_error_msg)

        dataset = (
            Dataset().with_catalog(mock_catalog).with_processing_step(mock_processor)
        )

        with pytest.raises(RuntimeError) as exc_info:
            dataset.execute({"variable": "temp", "_catalog_key": "cadcat"})

        assert original_error_msg in str(exc_info.value)
        # Verify the original exception is chained
        assert exc_info.value.__cause__ is not None
        assert isinstance(exc_info.value.__cause__, ValueError)
