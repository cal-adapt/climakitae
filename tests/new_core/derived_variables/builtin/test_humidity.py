"""
Unit tests for humidity-related derived variables.

This module tests the humidity calculation functions, validating both:
1. **Functional correctness**: Ensuring functions execute without error and return expected
   types, units, and value ranges (e.g., relative humidity clipped to 0-100%)
2. **Computational accuracy**: Verifying that humidity calculations use meteorologically correct
   formulas including saturation vapor pressure (Magnus formula), psychrometric equations,
   and proper handling of thermodynamic relationships between temperature, humidity, and pressure

The tests validate three core humidity derivations:
- **Relative humidity**: Calculated from specific humidity and saturation vapor pressure using
  the Magnus approximation
- **Dew point temperature**: Derived from relative humidity using the Magnus formula inversion
- **Specific humidity**: Computed from relative humidity and saturation vapor pressure with
  pressure-dependent corrections
"""

import numpy as np

from climakitae.new_core.derived_variables.builtin.humidity import (
    calc_relative_humidity_2m,
    calc_dew_point_2m,
    calc_specific_humidity_2m,
)


class TestHumidityCalculations:
    """Test class for humidity-related derived variable calculations."""

    def test_relative_humidity(self, humidity_dataset):
        """Test relative humidity calculation functional correctness and meteorological accuracy.

        Validates that relative humidity is correctly computed from specific humidity and
        saturation vapor pressure using the Magnus approximation formula. Tests both that
        the function executes properly and that the mathematical implementation matches
        the expected psychrometric relationship, with proper clipping to 0-100% bounds.
        """
        ds = humidity_dataset.copy()
        out = calc_relative_humidity_2m(ds.copy())

        t_celsius = ds.t2 - 273.15
        es = 611.2 * np.exp(17.67 * t_celsius / (t_celsius + 243.5))
        e = ds.q2 * ds.psfc / (0.622 + 0.378 * ds.q2)
        expected = 100.0 * e / es
        expected = expected.clip(0, 100)

        assert np.allclose(out.relative_humidity_2m.values, expected.values)
        assert out.relative_humidity_2m.attrs.get("units") == "%"

    def test_dew_point(self, humidity_dataset):
        """Test dew point temperature calculation functional correctness and thermodynamic accuracy.

        Validates that dew point is correctly derived from relative humidity using the
        Magnus formula inversion with Magnus coefficients (a=17.27, b=237.7°C). Tests both
        that the function executes properly and that the temperature derivation follows
        correct psychrometric principles with stable RH input for predictable validation.
        """
        ds = humidity_dataset.copy()
        ds2 = ds.copy()
        # set a stable RH for predictable result
        ds2["rh"] = 50.0
        out = calc_dew_point_2m(ds2.copy())

        a = 17.27
        b = 237.7
        t_celsius = ds2.t2 - 273.15
        gamma = (a * t_celsius / (b + t_celsius)) + np.log(ds2.rh / 100.0)
        expected = b * gamma / (a - gamma) + 273.15

        assert np.allclose(out.dew_point_2m.values, expected.values)

    def test_specific_humidity(self, humidity_dataset):
        """Test specific humidity calculation functional correctness and physical accuracy.

        Validates that specific humidity is correctly computed from relative humidity,
        saturation vapor pressure, and surface pressure using the proper psychrometric
        mixing ratio conversion. Tests both that the function executes properly and that
        the mass fraction calculation accurately reflects the thermodynamic relationship
        between water vapor and dry air with pressure-dependent corrections.
        """
        ds = humidity_dataset.copy()
        out = calc_specific_humidity_2m(ds.copy())

        t_celsius = ds.t2 - 273.15
        es = 611.2 * np.exp(17.67 * t_celsius / (t_celsius + 243.5))
        e = (ds.rh / 100.0) * es
        q = 0.622 * e / (ds.psfc - 0.378 * e)

        assert np.allclose(out.specific_humidity_2m.values, q.values)
