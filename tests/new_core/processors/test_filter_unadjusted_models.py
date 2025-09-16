"""
Unit tests for the `filter_unadjusted_models` processor.

This module contains comprehensive unit tests for the `FilterUnadjustedModels` processor class
that filters out unadjusted climate models from datasets. The tests cover various scenarios,
including different input types (xarray Dataset, DataArray, lists, tuples, dictionaries),
and edge cases such as empty datasets, datasets with only adjusted or unadjusted models, and
datasets with a mixed set of adjusted and unadjusted models.
"""

from typing import Union

import pandas as pd
import pytest
import xarray as xr

from climakitae.core.constants import _NEW_ATTRS_KEY
from climakitae.new_core.processors.filter_unadjusted_models import (
    FilterUnAdjustedModels,
)


@pytest.fixture
def da_empty() -> xr.DataArray:
    """Fixture for an empty xarray DataArray."""
    return xr.DataArray([], dims=["time"], coords={"time": []})


@pytest.fixture
def ds_empty() -> xr.Dataset:
    """Fixture for an empty xarray Dataset."""
    return xr.Dataset()


@pytest.fixture
def da_adjusted() -> xr.DataArray:
    """Fixture for an adjusted xarray DataArray."""
    return xr.DataArray(
        [1, 2, 3],
        dims=["time"],
        coords={
            "time": pd.date_range("2000-01-01", periods=3),
        },
        attrs={
            "intake_esm_attrs:activity_id": "WRF",
            "intake_esm_attrs:source_id": "EC-Earth3",
            "intake_esm_attrs:member_id": "r1i1p1f1",
        },
    )


@pytest.fixture
def ds_adjusted() -> xr.Dataset:
    """Fixture for an adjusted xarray Dataset."""
    return xr.Dataset(
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


@pytest.fixture
def da_not_adjusted() -> xr.DataArray:
    """Fixture for an unadjusted xarray DataArray."""
    return xr.DataArray(
        [4, 5, 6],
        dims=["time"],
        coords={
            "time": pd.date_range("2000-01-01", periods=3),
        },
        attrs={
            "intake_esm_attrs:activity_id": "WRF",
            "intake_esm_attrs:source_id": "FGOALS-g3",
            "intake_esm_attrs:member_id": "r1i1p1f1",
        },
    )


@pytest.fixture
def ds_not_adjusted() -> xr.Dataset:
    """Fixture for an unadjusted xarray Dataset."""
    return xr.Dataset(
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


@pytest.fixture
def processor_yes() -> FilterUnAdjustedModels:
    """Fixture for FilterUnAdjustedModels processor with value 'yes'."""
    return FilterUnAdjustedModels(value="yes")


@pytest.fixture
def processor_no() -> FilterUnAdjustedModels:
    """Fixture for FilterUnAdjustedModels processor with value 'no'."""
    return FilterUnAdjustedModels(value="no")


@pytest.fixture
def empty_processor() -> FilterUnAdjustedModels:
    """Fixture for FilterUnAdjustedModels processor with default value."""
    return FilterUnAdjustedModels()


class TestFilterUnAdjustedModelsInitialization:
    """Tests for FilterUnAdjustedModels initialization."""

    def test_init_default_params(self) -> None:
        """Test default initialization."""
        processor = FilterUnAdjustedModels()
        assert processor.value == "yes"
        assert processor.name == "filter_unadjusted_models"
        assert processor.valid_values == ["yes", "no"]

    @pytest.mark.parametrize("value", ["yes", "no", "YES", "No", "YeS"])
    def test_init_custom_value(self, value: str) -> None:
        """Test initialization with custom valid values."""
        processor = FilterUnAdjustedModels(value=value)
        assert processor.value == value.lower()

    @pytest.mark.parametrize("invalid_value", ["y", "n", "maybe", "", "123"])
    def test_init_invalid_value(self, invalid_value: str) -> None:
        """Test initialization with invalid values."""
        with pytest.raises(ValueError) as excinfo:
            obj = FilterUnAdjustedModels(value=invalid_value)
            result = xr.Dataset()
            obj.execute(result, context={})
        assert "Invalid value" in str(excinfo.value)
        assert "Valid values are:" in str(excinfo.value)
        assert any(valid in str(excinfo.value) for valid in ["yes", "no"])

    def test_set_data_accessor(self) -> None:
        """Test set_data_accessor method."""
        processor = FilterUnAdjustedModels()
        processor.set_data_accessor(catalog=None)
        assert processor  # Just to ensure no exceptions are raised


class TestFilterUnAdjustedModelsContainsUnAdjustedModels:
    """Test class for _contains_unadjusted_models method."""

    @pytest.mark.parametrize("empty_iterable", [[], (), {}])
    def test_contains_unadjusted_models_empty_iterable(
        self,
        empty_iterable: Union[list, tuple, dict],
        processor_yes: FilterUnAdjustedModels,
        processor_no: FilterUnAdjustedModels,
    ) -> None:
        """
        Test _contains_unadjusted_models with an empty iterable.
        """
        assert not processor_yes._contains_unadjusted_models(empty_iterable)
        assert not processor_no._contains_unadjusted_models(empty_iterable)

    @pytest.mark.parametrize("container_type", [list, tuple, dict])
    def test_contains_unadjusted_models_all_adjusted(
        self,
        container_type: type,
        processor_yes: FilterUnAdjustedModels,
        processor_no: FilterUnAdjustedModels,
        ds_adjusted: xr.Dataset,
    ) -> None:
        """
        Test _contains_unadjusted_models with only adjusted datasets.
        """
        if container_type is dict:
            input_data = {1: ds_adjusted, 2: ds_adjusted}
        else:
            input_data = container_type([ds_adjusted, ds_adjusted])
        assert not processor_yes._contains_unadjusted_models(input_data)
        assert not processor_no._contains_unadjusted_models(input_data)

    @pytest.mark.parametrize("container_type", [list, tuple, dict])
    def test_contains_unadjusted_models_all_unadjusted(
        self,
        container_type: type,
        processor_yes: FilterUnAdjustedModels,
        processor_no: FilterUnAdjustedModels,
        ds_not_adjusted: xr.Dataset,
    ) -> None:
        """
        Test _contains_unadjusted_models with only unadjusted datasets.
        """
        if container_type is dict:
            input_data = {1: ds_not_adjusted, 2: ds_not_adjusted}
        else:
            input_data = container_type([ds_not_adjusted, ds_not_adjusted])
        assert processor_yes._contains_unadjusted_models(input_data)
        assert processor_no._contains_unadjusted_models(input_data)

    @pytest.mark.parametrize("container_type", [list, tuple, dict])
    def test_contains_unadjusted_models_mixed(
        self,
        container_type: type,
        processor_yes: FilterUnAdjustedModels,
        processor_no: FilterUnAdjustedModels,
        ds_adjusted: xr.Dataset,
        ds_not_adjusted: xr.Dataset,
    ) -> None:
        """
        Test _contains_unadjusted_models with a mix of adjusted and unadjusted datasets.
        """
        if container_type is dict:
            input_data = {1: ds_not_adjusted, 2: ds_adjusted}
        else:
            input_data = container_type([ds_not_adjusted, ds_adjusted])
        assert processor_yes._contains_unadjusted_models(input_data)
        assert processor_no._contains_unadjusted_models(input_data)

    @pytest.mark.parametrize("xr_obj_name", ["da_empty", "ds_empty"])
    def test_contains_unadjusted_models_empty_xr(
        self,
        xr_obj_name: str,
        request: pytest.FixtureRequest,
        processor_yes: FilterUnAdjustedModels,
        processor_no: FilterUnAdjustedModels,
    ) -> None:
        """
        Test _contains_unadjusted_models with an empty xarray object.
        """
        xr_obj = request.getfixturevalue(xr_obj_name)
        assert not processor_yes._contains_unadjusted_models(xr_obj)
        assert not processor_no._contains_unadjusted_models(xr_obj)

    @pytest.mark.parametrize("xr_obj_name", ["da_adjusted", "ds_adjusted"])
    def test_contains_unadjusted_models_all_adjusted_xr(
        self,
        xr_obj_name: str,
        request: pytest.FixtureRequest,
        processor_yes: FilterUnAdjustedModels,
        processor_no: FilterUnAdjustedModels,
    ) -> None:
        """
        Test _contains_unadjusted_models with an xarray object with an adjusted model.
        """
        xr_obj = request.getfixturevalue(xr_obj_name)
        assert not processor_yes._contains_unadjusted_models(xr_obj)
        assert not processor_no._contains_unadjusted_models(xr_obj)

    @pytest.mark.parametrize("xr_obj_name", ["da_not_adjusted", "ds_not_adjusted"])
    def test_contains_unadjusted_models_all_unadjusted_xr(
        self,
        xr_obj_name: str,
        request: pytest.FixtureRequest,
        processor_yes: FilterUnAdjustedModels,
        processor_no: FilterUnAdjustedModels,
    ) -> None:
        """
        Test _contains_unadjusted_models with an xarray object with an unadjusted model.
        """
        xr_obj = request.getfixturevalue(xr_obj_name)
        assert processor_yes._contains_unadjusted_models(xr_obj)
        assert processor_no._contains_unadjusted_models(xr_obj)

    def test_contains_unadjusted_models_invalid_type(
        self, empty_processor: FilterUnAdjustedModels
    ) -> None:
        """
        Test _contains_unadjusted_models with an invalid type.
        """
        with pytest.raises(TypeError) as excinfo:
            empty_processor._contains_unadjusted_models(123)
        assert "Unsupported type for result" in str(excinfo.value)


class TestFilterUnAdjustedModelsRemoveUnAdjustedModels:
    """Tests for _remove_unadjusted_models method."""

    @pytest.mark.parametrize("empty_iterable", [[], (), {}])
    def test_remove_unadjusted_models_empty_iterable(
        self,
        empty_iterable: Union[list, tuple, dict],
        processor_yes: FilterUnAdjustedModels,
    ) -> None:
        """Test with an empty iterable."""
        result_yes = processor_yes._remove_unadjusted_models(empty_iterable)
        assert result_yes == empty_iterable

    @pytest.mark.parametrize("container_type", [list, tuple, dict])
    def test_remove_unadjusted_models_all_adjusted(
        self,
        container_type: type,
        processor_yes: FilterUnAdjustedModels,
        ds_adjusted: xr.Dataset,
    ) -> None:
        """Test with only adjusted models datasets."""
        if container_type is dict:
            input_data = {1: ds_adjusted, 2: ds_adjusted}
        else:
            input_data = container_type([ds_adjusted, ds_adjusted])
        result_yes = processor_yes._remove_unadjusted_models(input_data)
        if isinstance(input_data, dict):
            assert result_yes == {1: ds_adjusted, 2: ds_adjusted}
        else:
            assert result_yes == container_type([ds_adjusted, ds_adjusted])

    @pytest.mark.parametrize("container_type", [list, tuple, dict])
    def test_remove_unadjusted_models_all_unadjusted(
        self,
        container_type: type,
        processor_yes: FilterUnAdjustedModels,
        ds_not_adjusted: xr.Dataset,
    ) -> None:
        """Test with only unadjusted models datasets."""
        if container_type is dict:
            input_data = {1: ds_not_adjusted, 2: ds_not_adjusted}
        else:
            input_data = container_type([ds_not_adjusted, ds_not_adjusted])
        result_yes = processor_yes._remove_unadjusted_models(input_data)
        if isinstance(input_data, dict):
            assert result_yes == {}
        else:
            assert result_yes == container_type()

    @pytest.mark.parametrize("container_type", [list, tuple, dict])
    def test_remove_unadjusted_models_mixed(
        self,
        container_type: type,
        processor_yes: FilterUnAdjustedModels,
        ds_adjusted: xr.Dataset,
        ds_not_adjusted: xr.Dataset,
    ) -> None:
        """Test with datasets with a mix of adjusted and unadjusted models."""
        if container_type is dict:
            input_data = {1: ds_not_adjusted, 2: ds_adjusted}
        else:
            input_data = container_type([ds_not_adjusted, ds_adjusted])
        result_yes = processor_yes._remove_unadjusted_models(input_data)
        if isinstance(input_data, dict):
            assert result_yes == {2: ds_adjusted}
        else:
            assert result_yes == container_type([ds_adjusted])

    @pytest.mark.parametrize("xr_obj_name", ["da_empty", "ds_empty"])
    def test_remove_unadjusted_models_empty_xr(
        self,
        xr_obj_name: str,
        request: pytest.FixtureRequest,
        processor_yes: FilterUnAdjustedModels,
    ) -> None:
        """Test with an empty xarray object."""
        xr_obj = request.getfixturevalue(xr_obj_name)
        result_yes = processor_yes._remove_unadjusted_models(xr_obj)
        assert result_yes.identical(xr_obj)

    @pytest.mark.parametrize("xr_obj_name", ["da_adjusted", "ds_adjusted"])
    def test_remove_unadjusted_models_all_adjusted_xr(
        self,
        xr_obj_name: str,
        request: pytest.FixtureRequest,
        processor_yes: FilterUnAdjustedModels,
    ) -> None:
        """Test with an xarray object with an adjusted model."""
        xr_obj = request.getfixturevalue(xr_obj_name)
        result_yes = processor_yes._remove_unadjusted_models(xr_obj)
        assert result_yes.identical(xr_obj)

    @pytest.mark.parametrize("xr_obj_name", ["da_not_adjusted", "ds_not_adjusted"])
    def test_remove_unadjusted_models_all_unadjusted_xr(
        self,
        xr_obj_name: str,
        request: pytest.FixtureRequest,
        processor_yes: FilterUnAdjustedModels,
    ) -> None:
        """Test with an xarray object with an unadjusted model."""
        xr_obj = request.getfixturevalue(xr_obj_name)
        result_yes = processor_yes._remove_unadjusted_models(xr_obj)
        assert result_yes is None

    def test_remove_unadjusted_models_with_invalid_dtype(
        self, processor_yes: FilterUnAdjustedModels
    ) -> None:
        """Test with an invalid type."""
        with pytest.raises(TypeError) as excinfo:
            processor_yes._remove_unadjusted_models(123)
        assert "Unsupported type for result" in str(excinfo.value)


class TestFilterUnAdjustedModelsUpdateContext:
    """Tests for the update_context method of FilterUnAdjustedModels."""

    def test_update_context(self, empty_processor: FilterUnAdjustedModels) -> None:
        """Test update_context with an empty context."""
        context: dict = {}
        empty_processor.update_context(context)
        assert f"Process '{empty_processor.name}' applied to the data" in context[
            _NEW_ATTRS_KEY
        ].get("filter_unadjusted_models")

    def test_update_context_with_attrs_key_existing(
        self, empty_processor: FilterUnAdjustedModels
    ) -> None:
        """Test update_context when _NEW_ATTRS_KEY exists in context."""
        context: dict = {_NEW_ATTRS_KEY: {"existing_key": "existing_value"}}
        empty_processor.update_context(context)
        assert "existing_key" in context[_NEW_ATTRS_KEY]
        assert f"Process '{empty_processor.name}' applied to the data" in context[
            _NEW_ATTRS_KEY
        ].get("filter_unadjusted_models")


class TestFilterUnAdjustedModelsExecute:
    """Tests for the execute method."""

    def test_execute_empty_list_yes(
        self, processor_yes: FilterUnAdjustedModels
    ) -> None:
        """Test execute with an empty list and processor value 'yes'."""
        result: list = processor_yes.execute([], context={})
        assert result == []

    def test_execute_empty_list_no(self, processor_no: FilterUnAdjustedModels) -> None:
        """Test execute with an empty list and processor value 'no'."""
        result: list = processor_no.execute([], context={})
        assert result == []

    def test_execute_yes_with_only_adjusted_models(
        self,
        recwarn: pytest.WarningsRecorder,
        processor_yes: FilterUnAdjustedModels,
        ds_adjusted: xr.Dataset,
    ) -> None:
        """Test execute with only adjusted models and processor value 'yes'."""
        result: list = processor_yes.execute([ds_adjusted, ds_adjusted], context={})
        assert result == [ds_adjusted, ds_adjusted]
        assert len(recwarn) == 0

    def test_execute_yes_with_only_unadjusted_models(
        self,
        recwarn: pytest.WarningsRecorder,
        processor_yes: FilterUnAdjustedModels,
        ds_not_adjusted: xr.Dataset,
    ) -> None:
        """Test execute with only unadjusted models and processor value 'yes'."""
        result: list = processor_yes.execute(
            [ds_not_adjusted, ds_not_adjusted], context={}
        )
        assert result == []
        assert len(recwarn) > 0
        assert any(
            "These models have been removed from the returned query." in str(w.message)
            for w in recwarn.list
        )

    def test_execute_yes_with_mixed_models(
        self,
        recwarn: pytest.WarningsRecorder,
        processor_yes: FilterUnAdjustedModels,
        ds_adjusted: xr.Dataset,
        ds_not_adjusted: xr.Dataset,
    ) -> None:
        """Test execute with mixed models and processor value 'yes'."""
        result: list = processor_yes.execute([ds_adjusted, ds_not_adjusted], context={})
        assert result == [ds_adjusted]
        assert len(recwarn) > 0
        assert any(
            "These models have been removed from the returned query." in str(w.message)
            for w in recwarn.list
        )

    def test_execute_no_with_only_adjusted_models(
        self,
        recwarn: pytest.WarningsRecorder,
        processor_no: FilterUnAdjustedModels,
        ds_adjusted: xr.Dataset,
    ) -> None:
        """Test execute with only adjusted models and processor value 'no'."""
        result: list = processor_no.execute([ds_adjusted, ds_adjusted], context={})
        assert result == [ds_adjusted, ds_adjusted]
        assert len(recwarn) == 0

    def test_execute_no_with_only_unadjusted_models(
        self,
        recwarn: pytest.WarningsRecorder,
        processor_no: FilterUnAdjustedModels,
        ds_not_adjusted: xr.Dataset,
    ) -> None:
        """Test execute with only unadjusted models and processor value 'no'."""
        result: list = processor_no.execute(
            [ds_not_adjusted, ds_not_adjusted], context={}
        )
        assert result == [ds_not_adjusted, ds_not_adjusted]
        assert len(recwarn) > 0
        assert any(
            "These models HAVE NOT been removed from the returned query."
            in str(w.message)
            for w in recwarn.list
        )

    def test_execute_no_with_mixed_models(
        self,
        recwarn: pytest.WarningsRecorder,
        processor_no: FilterUnAdjustedModels,
        ds_adjusted: xr.Dataset,
        ds_not_adjusted: xr.Dataset,
    ) -> None:
        """Test execute with a mixed set models and processor value 'no'."""
        result: list = processor_no.execute([ds_adjusted, ds_not_adjusted], context={})
        assert result == [ds_adjusted, ds_not_adjusted]
        assert len(recwarn) > 0
        assert any(
            "These models HAVE NOT been removed from the returned query."
            in str(w.message)
            for w in recwarn.list
        )

    def test_execute_not_yes_or_no(self, ds_adjusted: xr.Dataset) -> None:
        """Test execute raises ValueError for invalid value."""
        processor_invalid: FilterUnAdjustedModels = FilterUnAdjustedModels(
            value="no idea"
        )
        with pytest.raises(ValueError) as excinfo:
            processor_invalid.execute(ds_adjusted, context={})
        assert "Invalid value" in str(excinfo.value)
        assert "Valid values are:" in str(excinfo.value)
