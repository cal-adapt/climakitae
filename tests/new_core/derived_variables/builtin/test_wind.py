"""
Unit tests for wind-related derived variables.

This module tests the wind calculation functions, validating both:
1. **Functional correctness**: Ensuring functions execute without error and return expected
   types and value ranges
2. **Computational accuracy**: Verifying that wind calculations use correct vector mathematics
   including wind speed magnitude and meteorological wind direction conventions

The tests validate two core wind derivations:
- **Wind speed**: Calculated as vector magnitude from u and v components
- **Wind direction**: Derived using meteorological convention (direction FROM which wind blows)
"""

import numpy as np

from climakitae.new_core.derived_variables.builtin.wind import (
    calc_wind_speed_10m,
    calc_wind_direction_10m,
)


class TestWindCalculations:
    """Test class for wind-related derived variable calculations."""

    def test_wind_speed_and_direction(self, wind_dataset):
        """Test wind speed and direction calculation functional and mathematical correctness.

        Validates that wind speed is correctly computed as vector magnitude from u and v
        components, and that wind direction follows meteorological convention (270 - atan2
        transformation). Tests both functional execution and mathematical accuracy of
        vector calculations.
        """
        ds = wind_dataset.copy()

        out_speed = calc_wind_speed_10m(ds.copy())
        expected_speed = np.sqrt(ds.u10.values**2 + ds.v10.values**2)
        assert np.allclose(out_speed.wind_speed_10m.values, expected_speed)

        out_dir = calc_wind_direction_10m(ds.copy())
        expected_dir = (
            270 - np.arctan2(ds.v10.values, ds.u10.values) * 180 / np.pi
        ) % 360
        assert np.allclose(out_dir.wind_direction_10m.values, expected_dir)
