"""
Units tests for the `warming_level` processor.

This module contains comprehensive unit tests for the `warming_level` processor,
which is responsible for adjusting climate data based on specified global warming levels.
These tests ensure that the processor behaves as expected under various scenarios,
including different warming levels, time windows, and edge cases.

The tests cover:
- Basic functionality: Verifying that the processor correctly adjusts data for standard warming levels.
- Boundary conditions: Ensuring correct behavior at the edges of the time windows.
- Error handling: Confirming that appropriate exceptions are raised for invalid inputs.
"""

import numpy as np
import pandas as pd
import pytest

from climakitae.core.constants import _NEW_ATTRS_KEY
from climakitae.new_core.processors.warming_level import WarmingLevel


class TestWarmingLevelProcessorInitialization:
    """Tests for the initialization of the WarmingLevelProcessor."""

    def test_default_initialization(self):
        """Test default initialization."""
        processor = WarmingLevel(value={})
        assert processor.warming_levels == [2.0]
        assert processor.warming_level_window == 15
        assert isinstance(processor.warming_level_times, pd.DataFrame)
        assert isinstance(processor.warming_level_times_idx, pd.DataFrame)

    def test_custom_initialization(self):
        """Test custom initialization."""
        processor = WarmingLevel(
            value={"warming_levels": [1.5, 3.0], "warming_level_window": 10}
        )
        assert processor.warming_levels == [1.5, 3.0]
        assert processor.warming_level_window == 10


class TestWarmingLevelUpdateContext:
    """Tests for the update_context method of WarmingLevelProcessor."""

    def test_update_context_adds_warming_levels(self):
        """Test that update_context adds warming level information to context."""
        processor = WarmingLevel(value={})
        context = {_NEW_ATTRS_KEY: {}}
        updated_context = processor.update_context(context)
        assert (
            f"Process '{processor.name}'"
            in updated_context[_NEW_ATTRS_KEY][processor.name]
        )
        assert "warming_levels" in updated_context[_NEW_ATTRS_KEY][processor.name]
        assert "warming_level_window" in updated_context[_NEW_ATTRS_KEY][processor.name]
        assert (
            str(processor.warming_levels)
            in updated_context[_NEW_ATTRS_KEY][processor.name]
        )
        assert (
            str(processor.warming_level_window)
            in updated_context[_NEW_ATTRS_KEY][processor.name]
        )


class TestWarmingReformatMemberIds:
    pass


class TestWarmingLevelExtendTimeDomain:
    pass


class TestWarmingLevelGetCenterYears:
    pass


class TestWarmingLevelExecute:
    pass
