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
