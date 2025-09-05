"""
Units tests for `filter_unadjusted_models` processor.

This module contains comprehensive unit tests for the `FilterUnadjustedModels` processor class
that filters out unadjusted climate models from datasets. The tests cover various scenarios,
including different input types (xarray Dataset, DataArray, lists, tuples, dictionaries),
and edge cases such as all models being unadjusted or none being unadjusted.
"""

from typing import Iterable, Union

import pandas as pd
import pytest
import xarray as xr

from climakitae.core.constants import (
    _NEW_ATTRS_KEY,
    NON_WRF_BA_MODELS,
    UNSET,
    WRF_BA_MODELS,
)
from climakitae.new_core.processors.filter_unadjusted_models import (
    FilterUnAdjustedModels,
)


class TestFilterUnAdjustedModelsInitialization:
    """Test class for FilterUnAdjustedModels processor."""

    def setup_method(self):
        """Set up text fixtures."""
        pass

    def test_init_default_params(self):
        """Setup method to initialize common variables."""
        processor = FilterUnAdjustedModels()
        assert processor.value == "yes"
        assert processor.name == "filter_unadjusted_models"
        assert processor.valid_values == ["yes", "no"]

    @pytest.mark.parametrize("value", ["yes", "no", "YES", "No", "YeS"])
    def test_init_custom_value(self, value):
        """Test initialization with custom valid values to test capitalization validation."""
        processor = FilterUnAdjustedModels(value=value)
        assert processor.value == value.lower()

    @pytest.mark.parametrize("invalid_value", ["y", "n", "maybe", "", "123"])
    def test_init_invalid_value(self, invalid_value):
        """Test initialization and dummy execution with invalid values."""
        with pytest.raises(ValueError) as excinfo:
            obj = FilterUnAdjustedModels(value=invalid_value)
            result = xr.Dataset()
            obj.execute(result, context={})
        assert "Invalid value" in str(excinfo.value)
        assert "Valid values are:" in str(excinfo.value)
        assert any(valid in str(excinfo.value) for valid in ["yes", "no"])


class TestFilterUnAdjustedModelsContainsUnAdjustedModels:
    """Test class for _contains_unadjusted_models method."""

    def setup_method(self):
        """Setup method to initialize common variables."""
        self.processor_yes = FilterUnAdjustedModels(value="yes")
        self.processor_no = FilterUnAdjustedModels(value="no")

        # Create sample xarray Datasets and DataArrays for testing
        self.ds_empty = xr.Dataset()
        self.ds_adjusted = xr.Dataset(
            {"temp": (["time"], [1, 2, 3])},
            coords={
                "time": pd.date_range("2000-01-01", periods=3),
                "simulation": ["model_a"],
            },
            attrs={
                "intake_esm_attrs:activity_id": "WRF",
                "intake_esm_attrs:source_id": "EC-Earth3",
                "intake_esm_attrs:member_id": "r1i1p1f1",
            },
        )
        self.ds_not_adjusted = xr.Dataset(
            {"temp": (["time"], [4, 5, 6])},
            coords={
                "time": pd.date_range("2000-01-01", periods=3),
                "simulation": ["model_b"],
            },
            attrs={
                "intake_esm_attrs:activity_id": "WRF",
                "intake_esm_attrs:source_id": "FGOALS-g3",
                "intake_esm_attrs:member_id": "r1i1p1f1",
            },
        )

    def test_contains_unadjusted_models_single_dataset_adjusted(self):
        """Test _contains_unadjusted_models with a single adjusted Dataset."""
        assert not self.processor_yes._contains_unadjusted_models(self.ds_adjusted)
        assert not self.processor_no._contains_unadjusted_models(self.ds_adjusted)

    def test_contains_unadjusted_models_single_dataset_not_adjusted(self):
        """Test _contains_unadjusted_models with a single unadjusted Dataset."""
        assert self.processor_yes._contains_unadjusted_models(self.ds_not_adjusted)
        assert self.processor_no._contains_unadjusted_models(self.ds_not_adjusted)

    def test_contains_unadjusted_models_empty_dataset(self):
        """Test _contains_unadjusted_models with an empty Dataset."""
        assert not self.processor_yes._contains_unadjusted_models(self.ds_empty)
        assert not self.processor_no._contains_unadjusted_models(self.ds_empty)

    def test_contains_unadjusted_models_list_mixed(self):
        """Test _contains_unadjusted_models with a list of mixed Datasets."""
        assert self.processor_yes._contains_unadjusted_models(
            [self.ds_adjusted, self.ds_not_adjusted]
        )
        assert self.processor_no._contains_unadjusted_models(
            [self.ds_adjusted, self.ds_not_adjusted]
        )

    def test_contains_unadjusted_models_tuple_mixed(self):
        """Test _contains_unadjusted_models with a tuple of mixed Datasets."""
        assert self.processor_yes._contains_unadjusted_models(
            (self.ds_not_adjusted, self.ds_adjusted)
        )
        assert self.processor_no._contains_unadjusted_models(
            (self.ds_not_adjusted, self.ds_adjusted)
        )

    def test_contains_unadjusted_models_list_all_not_adjusted(self):
        """Test _contains_unadjusted_models with a list of unadjusted Datasets."""
        assert self.processor_yes._contains_unadjusted_models(
            [self.ds_not_adjusted, self.ds_not_adjusted]
        )
        assert self.processor_no._contains_unadjusted_models(
            [self.ds_not_adjusted, self.ds_not_adjusted]
        )

    def test_contains_unadjusted_models_tuple_all_not_adjusted(self):
        """Test _contains_unadjusted_models with a tuple of unadjusted Datasets."""
        assert self.processor_yes._contains_unadjusted_models(
            (self.ds_not_adjusted, self.ds_not_adjusted)
        )
        assert self.processor_no._contains_unadjusted_models(
            (self.ds_not_adjusted, self.ds_not_adjusted)
        )

    def test_contains_unadjusted_models_empty_list(self):
        """Test _contains_unadjusted_models with an empty list."""
        assert not self.processor_yes._contains_unadjusted_models([])
        assert not self.processor_no._contains_unadjusted_models([])

    def test_contains_unadjusted_models_empty_tuple(self):
        """Test _contains_unadjusted_models with an empty tuple."""
        assert not self.processor_yes._contains_unadjusted_models(())
        assert not self.processor_no._contains_unadjusted_models(())

    def test_contains_unadjusted_models_list_all_adjusted(self):
        """Test _contains_unadjusted_models with a list of adjusted Datasets."""
        assert not self.processor_yes._contains_unadjusted_models(
            [self.ds_adjusted, self.ds_adjusted]
        )
        assert not self.processor_no._contains_unadjusted_models(
            [self.ds_adjusted, self.ds_adjusted]
        )

    def test_contains_unadjusted_models_tuple_all_adjusted(self):
        """Test _contains_unadjusted_models with a tuple of adjusted Datasets."""
        assert not self.processor_yes._contains_unadjusted_models(
            (self.ds_adjusted, self.ds_adjusted)
        )
        assert not self.processor_no._contains_unadjusted_models(
            (self.ds_adjusted, self.ds_adjusted)
        )

    def test_contains_unadjusted_models_dict_mixed(self):
        """Test _contains_unadjusted_models with a dictionary of mixed Datasets."""
        assert self.processor_yes._contains_unadjusted_models(
            {"data1": self.ds_adjusted, "data2": self.ds_not_adjusted}
        )
        assert self.processor_no._contains_unadjusted_models(
            {"data1": self.ds_adjusted, "data2": self.ds_not_adjusted}
        )

    def test_contains_unadjusted_models_dict_all_not_adjusted(self):
        """Test _contains_unadjusted_models with a dictionary of unadjusted Datasets."""
        assert self.processor_yes._contains_unadjusted_models(
            {"data1": self.ds_not_adjusted, "data2": self.ds_not_adjusted}
        )
        assert self.processor_no._contains_unadjusted_models(
            {"data1": self.ds_not_adjusted, "data2": self.ds_not_adjusted}
        )

    def test_contains_unadjusted_models_empty_dict(self):
        """Test _contains_unadjusted_models with an empty dictionary."""
        assert not self.processor_yes._contains_unadjusted_models({})
        assert not self.processor_no._contains_unadjusted_models({})

    def test_contains_unadjusted_models_dict_mixed_multiple(self):
        """Test _contains_unadjusted_models with a dictionary with mixed Datasets."""
        assert self.processor_yes._contains_unadjusted_models(
            {
                "data1": self.ds_adjusted,
                "data2": self.ds_adjusted,
                "data3": self.ds_not_adjusted,
            }
        )
        assert self.processor_no._contains_unadjusted_models(
            {
                "data1": self.ds_adjusted,
                "data2": self.ds_adjusted,
                "data3": self.ds_not_adjusted,
            }
        )

    def test_contains_unadjusted_models_dict_all_adjusted(self):
        """Test _contains_unadjusted_models with a dictionary of adjusted Datasets."""
        assert not self.processor_yes._contains_unadjusted_models(
            {
                "data1": self.ds_adjusted,
                "data2": self.ds_adjusted,
                "data3": self.ds_adjusted,
            }
        )
        assert not self.processor_no._contains_unadjusted_models(
            {
                "data1": self.ds_adjusted,
                "data2": self.ds_adjusted,
                "data3": self.ds_adjusted,
            }
        )

    def test_contains_unadjusted_models_invalid_type(self):
        """Test _contains_unadjusted_models with an invalid type."""
        with pytest.raises(TypeError) as excinfo:
            self.processor_yes._contains_unadjusted_models(123)
        assert "Unsupported type for result" in str(excinfo.value)

        with pytest.raises(TypeError) as excinfo:
            self.processor_no._contains_unadjusted_models(123)
        assert "Unsupported type for result" in str(excinfo.value)


class TestFilterUnAdjustedModelsRemoveUnAdjustedModels:
    """Test class for _remove_unadjusted_models method."""

    def setup_method(self):
        """Setup method to initialize common variables."""
        self.processor_yes = FilterUnAdjustedModels(value="yes")
        self.processor_no = FilterUnAdjustedModels(value="no")

        # Create sample xarray Datasets and DataArrays for testing
        self.ds_empty = xr.Dataset()
        self.ds_adjusted = xr.Dataset(
            {"temp": (["time"], [1, 2, 3])},
            coords={
                "time": pd.date_range("2000-01-01", periods=3),
                "simulation": ["model_a"],
            },
            attrs={
                "intake_esm_attrs:activity_id": "WRF",
                "intake_esm_attrs:source_id": "EC-Earth3",
                "intake_esm_attrs:member_id": "r1i1p1f1",
            },
        )
        self.ds_not_adjusted = xr.Dataset(
            {"temp": (["time"], [4, 5, 6])},
            coords={
                "time": pd.date_range("2000-01-01", periods=3),
                "simulation": ["model_b"],
            },
            attrs={
                "intake_esm_attrs:activity_id": "WRF",
                "intake_esm_attrs:source_id": "FGOALS-g3",
                "intake_esm_attrs:member_id": "r1i1p1f1",
            },
        )

    def test_remove_unadjusted_models_single_dataset_adjusted(self):
        """Test _remove_unadjusted_models with a single adjusted Dataset."""
        result_yes = self.processor_yes._remove_unadjusted_models(self.ds_adjusted)
        assert result_yes is self.ds_adjusted

    def test_remove_unadjusted_models_single_dataset_not_adjusted(self):
        """Test _remove_unadjusted_models with a single unadjusted Dataset."""
        result_yes = self.processor_yes._remove_unadjusted_models(self.ds_not_adjusted)
        assert result_yes is None

    def test_remove_unadjusted_models_empty_dataset(self):
        """Test _remove_unadjusted_models with an empty Dataset."""
        result = self.processor_yes._remove_unadjusted_models(self.ds_empty)
        assert result is self.ds_empty

    def test_remove_unadjusted_models_list_mixed(self):
        """Test _remove_unadjusted_models with a list of mixed Datasets."""
        result_yes = self.processor_yes._remove_unadjusted_models(
            [self.ds_adjusted, self.ds_not_adjusted]
        )
        assert result_yes == [self.ds_adjusted]

    def test_remove_unadjusted_models_tuple_mixed(self):
        """Test _remove_unadjusted_models with a tuple of mixed Datasets."""
        result_yes = self.processor_yes._remove_unadjusted_models(
            (self.ds_not_adjusted, self.ds_adjusted)
        )
        assert result_yes == (self.ds_adjusted)

    def test_remove_unadjusted_models_list_all_not_adjusted(self):
        """Test _remove_unadjusted_models with a list of unadjusted Datasets."""
        result_yes = self.processor_yes._remove_unadjusted_models(
            [self.ds_not_adjusted, self.ds_not_adjusted]
        )
        assert result_yes == []

    def test_remove_unadjusted_models_tuple_all_not_adjusted(self):
        """Test _remove_unadjusted_models with a tuple of unadjusted Datasets."""
        result_yes = self.processor_yes._remove_unadjusted_models(
            (self.ds_not_adjusted, self.ds_not_adjusted)
        )
        assert result_yes is None


class TestFilterUnAdjustedModelsUpdateContext:
    """Test class for update_context method."""

    def setup_method(self):
        """Setup method to initialize common variables."""
        self.processor = FilterUnAdjustedModels()

    def test_update_context_yes(self):
        """Test update_context when value is 'yes'."""
        context = {}
        self.processor.update_context(context)
        assert f"Process '{self.processor.name}' applied to the data" in context[
            _NEW_ATTRS_KEY
        ].get("filter_unadjusted_models")

    def test_update_context_with_attrs_key_existing(self):
        """Test update_context when _NEW_ATTRS_KEY already exists in context."""
        context = {_NEW_ATTRS_KEY: {"existing_key": "existing_value"}}
        self.processor.update_context(context)
        assert "existing_key" in context[_NEW_ATTRS_KEY]
        assert f"Process '{self.processor.name}' applied to the data" in context[
            _NEW_ATTRS_KEY
        ].get("filter_unadjusted_models")


class TestFilterUnAdjustedModelsExecute:
    """Test class for execute method."""

    def setup_method(self):
        """Setup method to initialize common variables."""
        self.processor_yes = FilterUnAdjustedModels(value="yes")
        self.processor_no = FilterUnAdjustedModels(value="no")

        # Create sample xarray Datasets and DataArrays for testing
        self.ds_empty = xr.Dataset()
        self.ds_adjusted = xr.Dataset(
            {"temp": (["time"], [1, 2, 3])},
            coords={
                "time": pd.date_range("2000-01-01", periods=3),
                "simulation": ["model_a"],
            },
            attrs={
                "intake_esm_attrs:activity_id": "WRF",
                "intake_esm_attrs:source_id": "EC-Earth3",
                "intake_esm_attrs:member_id": "r1i1p1f1",
            },
        )
        self.ds_not_adjusted = xr.Dataset(
            {"temp": (["time"], [4, 5, 6])},
            coords={
                "time": pd.date_range("2000-01-01", periods=3),
                "simulation": ["model_b"],
            },
            attrs={
                "intake_esm_attrs:activity_id": "WRF",
                "intake_esm_attrs:source_id": "FGOALS-g3",
                "intake_esm_attrs:member_id": "r1i1p1f1",
            },
        )

    def test_execute_single_dataset_adjusted_yes(self):
        """Test execute with a single adjusted Dataset and value 'yes'."""
        result = self.processor_yes.execute(self.ds_adjusted, context={})
        assert result is self.ds_adjusted

    def test_execute_single_dataset_not_adjusted_yes(self):
        """Test execute with a single unadjusted Dataset and value 'yes'."""
        result = self.processor_yes.execute(self.ds_not_adjusted, context={})
        assert result is None

    def test_execute_single_dataset_adjusted_no(self):
        """Test execute with a single adjusted Dataset and value 'no'."""
        result = self.processor_no.execute(self.ds_adjusted, context={})
        assert result is self.ds_adjusted

    def test_execute_single_dataset_not_adjusted_no(self):
        """Test execute with a single unadjusted Dataset and value 'no'."""
        result = self.processor_no.execute(self.ds_not_adjusted, context={})
        assert result is self.ds_not_adjusted

    def test_execute_list_mixed_yes(self):
        """Test execute with a list of mixed Datasets and value 'yes'."""
        result = self.processor_yes.execute(
            [self.ds_adjusted, self.ds_not_adjusted], context={}
        )
        assert result == [self.ds_adjusted]

    def test_execute_list_mixed_no(self):
        """Test execute with a list of mixed Datasets and value 'no'."""
        result = self.processor_no.execute(
            [self.ds_adjusted, self.ds_not_adjusted], context={}
        )
        assert result == [self.ds_adjusted, self.ds_not_adjusted]

    def test_execute_tuple_mixed_yes(self):
        """Test execute with a tuple of mixed Datasets and value 'yes'."""
        result = self.processor_yes.execute(
            (self.ds_not_adjusted, self.ds_adjusted), context={}
        )
        assert result == (self.ds_adjusted,)

    def test_execute_tuple_mixed_no(self):
        """Test execute with a tuple of mixed Datasets and value 'no'."""
        result = self.processor_no.execute(
            (self.ds_not_adjusted, self.ds_adjusted), context={}
        )
        assert result == (self.ds_not_adjusted, self.ds_adjusted)

    def test_execute_empty_list_yes(self):
        """Test execute with an empty list and value 'yes'."""
        result = self.processor_yes.execute([], context={})
        assert result == []

    def test_execute_empty_list_no(self):
        """Test execute with an empty list and value 'no'."""
        result = self.processor_no.execute([], context={})
        assert result == []

    def test_execute_throws_warning_case_yes(self):
        """Test execute raises warning when all models are unadjusted and value 'yes'."""
        with pytest.warns(UserWarning) as record:
            result = self.processor_yes.execute(
                [self.ds_not_adjusted, self.ds_not_adjusted], context={}
            )
            assert result == []
        assert any(
            "Your query selected models that do not have a-priori bias adjustment."
            in str(warning.message)
            for warning in record
        )
