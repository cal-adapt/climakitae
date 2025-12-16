"""
Unit tests for climakitae/new_core/processors/update_attributes.py.
"""

import numpy as np
import pandas as pd
import pytest
import xarray as xr
from unittest.mock import MagicMock

from climakitae.core.constants import _NEW_ATTRS_KEY, UNSET
from climakitae.new_core.processors.update_attributes import (
    UpdateAttributes,
    common_attrs,
)


class TestUpdateAttributesInit:
    """Test class for UpdateAttributes processor initialization."""

    def test_init_defaults(self):
        """Test initialization with default values."""
        processor = UpdateAttributes()
        assert processor.value is UNSET
        assert processor.name == "update_attributes"

    def test_init_with_value(self):
        """Test initialization with custom value."""
        processor = UpdateAttributes(value="custom_value")
        assert processor.value == "custom_value"
        assert processor.name == "update_attributes"


class TestUpdateAttributesUpdateContext:
    """Test class for UpdateAttributes update_context method."""

    def test_update_context_creates_new_attrs_key(self):
        """Test that update_context creates _NEW_ATTRS_KEY if not present."""
        processor = UpdateAttributes()
        context = {}

        processor.update_context(context)

        assert _NEW_ATTRS_KEY in context
        assert isinstance(context[_NEW_ATTRS_KEY], dict)

    def test_update_context_adds_processor_entry(self):
        """Test that update_context adds processor description entry."""
        processor = UpdateAttributes()
        context = {}

        processor.update_context(context)

        assert processor.name in context[_NEW_ATTRS_KEY]
        assert "update_attributes" in context[_NEW_ATTRS_KEY][processor.name]
        assert "applied to the data" in context[_NEW_ATTRS_KEY][processor.name]

    def test_update_context_preserves_existing_attrs(self):
        """Test that update_context preserves existing entries in _NEW_ATTRS_KEY."""
        processor = UpdateAttributes()
        context = {_NEW_ATTRS_KEY: {"existing_key": "existing_value"}}

        processor.update_context(context)

        assert "existing_key" in context[_NEW_ATTRS_KEY]
        assert context[_NEW_ATTRS_KEY]["existing_key"] == "existing_value"
        assert processor.name in context[_NEW_ATTRS_KEY]


class TestUpdateAttributesExecuteDataset:
    """Test class for UpdateAttributes execute method with xr.Dataset."""

    def setup_method(self):
        """Set up test fixtures."""
        self.processor = UpdateAttributes()
        self.sample_dataset = xr.Dataset(
            {"temp": (["time", "lat", "lon"], np.random.rand(2, 3, 4))},
            coords={
                "time": pd.date_range("2020-01-01", periods=2),
                "lat": [34.0, 35.0, 36.0],
                "lon": [-118.0, -117.0, -116.0, -115.0],
            },
        )

    def test_execute_dataset_adds_new_attrs(self):
        """Test that execute adds new attributes from context to Dataset."""
        context = {_NEW_ATTRS_KEY: {"test_attr": "test_value"}}

        result = self.processor.execute(self.sample_dataset, context)

        assert "test_attr" in result.attrs
        assert result.attrs["test_attr"] == "test_value"

    def test_execute_dataset_preserves_existing_attrs(self):
        """Test that execute preserves existing Dataset attributes."""
        self.sample_dataset.attrs["original_attr"] = "original_value"
        context = {_NEW_ATTRS_KEY: {"new_attr": "new_value"}}

        result = self.processor.execute(self.sample_dataset, context)

        assert "original_attr" in result.attrs
        assert result.attrs["original_attr"] == "original_value"
        assert "new_attr" in result.attrs

    def test_execute_dataset_updates_dim_attrs(self):
        """Test that execute updates dimension attributes with common_attrs."""
        context = {_NEW_ATTRS_KEY: {"test_attr": "test_value"}}

        result = self.processor.execute(self.sample_dataset, context)

        # Check lat dimension attrs
        assert result["lat"].attrs["standard_name"] == "latitude"
        assert result["lat"].attrs["units"] == "degrees_north"
        # Check lon dimension attrs
        assert result["lon"].attrs["standard_name"] == "longitude"
        assert result["lon"].attrs["units"] == "degrees_east"
        # Check time dimension attrs
        assert result["time"].attrs["standard_name"] == "time"
        assert result["time"].attrs["axis"] == "T"
