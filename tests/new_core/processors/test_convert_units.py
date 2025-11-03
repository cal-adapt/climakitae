"""
Unit tests for climakitae/new_core/processors/convert_units.py

This module tests the ConvertUnits processor to ensure it correctly converts
data units as specified. It verifies that the processor modifies the data
array in the expected manner and handles edge cases appropriately.
"""

import numpy as np
import pytest

from climakitae.core.constants import UNSET
from climakitae.new_core.processors.convert_units import ConvertUnits


class TestConvertUnitsProcessor:
    """Test class for ConvertUnits processor."""

    @pytest.fixture
    def sample_data(self):
        """Fixture to provide sample data for testing."""
        return np.array([0.0, 10.0, 20.0, 30.0])  # Sample data in degC

    def test_convert_c_to_k(self, sample_data):
        """Test conversion from Celsius to Kelvin."""

        processor = ConvertUnits(target_unit="K")
        converted_data = processor.process(sample_data, current_unit="degC")

        expected_data = sample_data + 273.15  # Convert degC to K
        np.testing.assert_array_almost_equal(converted_data, expected_data)

    def test_convert_k_to_c(self, sample_data):
        """Test conversion from Kelvin to Celsius."""

        kelvin_data = sample_data + 273.15  # Sample data in K
        processor = ConvertUnits(target_unit="degC")
        converted_data = processor.process(kelvin_data, current_unit="K")

        expected_data = kelvin_data - 273.15  # Convert K to degC
        np.testing.assert_array_almost_equal(converted_data, expected_data)

    def test_invalid_unit_conversion(self, sample_data):
        """Test handling of invalid unit conversion."""

        processor = ConvertUnits(target_unit="invalid_unit")
        with pytest.raises(ValueError, match="Unsupported unit conversion"):
            processor.process(sample_data, current_unit="degC")

    def test_no_conversion_needed(self, sample_data):
        """Test case where no conversion is needed."""

        processor = ConvertUnits(target_unit="degC")
        converted_data = processor.process(sample_data, current_unit="degC")

        np.testing.assert_array_almost_equal(converted_data, sample_data)

    def test_convert_mm_to_inches(self):
        """Test conversion from millimeters to inches."""

        mm_data = np.array([0.0, 25.4, 50.8, 76.2])  # Sample data in mm
        processor = ConvertUnits(target_unit="inches")
        converted_data = processor.process(mm_data, current_unit="mm")

        expected_data = mm_data / 25.4  # Convert mm to inches
        np.testing.assert_array_almost_equal(converted_data, expected_data)
