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
