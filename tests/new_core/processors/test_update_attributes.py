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
