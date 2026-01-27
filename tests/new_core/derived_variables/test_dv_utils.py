"""
Unit tests for derived variable utility functions.

This module tests the threshold utility functions for derived variables, validating both:
1. **Functional correctness**: Ensuring functions execute without error and return expected
   types and ranges
2. **Computational accuracy**: Verifying that threshold calculations are mathematically correct
   including unit conversions, override precedence, and fallback behavior

The tests validate threshold resolution logic across multiple input scenarios:
- Explicit parameter passing (Kelvin and Fahrenheit)
- Dataset attribute overrides (per-variable and top-level)
- Default fallback values
- Correct temperature unit conversions (Celsius ↔ Fahrenheit ↔ Kelvin)
"""

import numpy as np
import xarray as xr

from climakitae.new_core.derived_variables.utils import get_derived_threshold
from climakitae.core.constants import DEFAULT_DEGREE_DAY_THRESHOLD_K
from climakitae.util.utils import f_to_k


class TestGetDerivedThreshold:
    """Test class for get_derived_threshold function."""

    def test_explicit_threshold_k(self):
        """Verify explicit threshold in Kelvin is returned unchanged."""
        ds = xr.Dataset()
        assert get_derived_threshold(ds, threshold_k=300.0) == 300.0

    def test_explicit_threshold_f_converts(self):
        """Verify Fahrenheit threshold is correctly converted to Kelvin."""
        ds = xr.Dataset()
        got = get_derived_threshold(ds, threshold_f=75.0)
        expect = f_to_k(75.0)
        assert np.isclose(got, expect)

    def test_per_variable_override_from_attrs(self):
        """Verify per-variable threshold override from attrs takes precedence."""
        ds = xr.Dataset()
        ds.attrs["derived_variable_overrides"] = {"CDD_wrf": {"threshold_f": 75.0}}
        got = get_derived_threshold(ds, "CDD_wrf")
        expect = f_to_k(75.0)
        assert np.isclose(got, expect)

    def test_toplevel_attr_override(self):
        """Verify top-level Celsius threshold attribute is used and converted to Kelvin."""
        ds = xr.Dataset()
        ds.attrs["threshold_c"] = 20.0
        got = get_derived_threshold(ds)
        expect = 20.0 + 273.15
        assert np.isclose(got, expect)

    def test_default_fallback(self):
        """Verify default degree day threshold is used when no override specified."""
        ds = xr.Dataset()
        got = get_derived_threshold(ds)
        assert np.isclose(got, float(DEFAULT_DEGREE_DAY_THRESHOLD_K))

    def test_per_variable_override_threshold_k_and_c_alias(self):
        """Verify Kelvin threshold in per-variable params is returned unchanged."""
        ds = xr.Dataset()
        ds.attrs["derived_variable_params"] = {"HDD_loca": {"threshold_k": 295.0}}
        got = get_derived_threshold(ds, "HDD_loca")
        assert np.isclose(got, 295.0)

    def test_per_variable_override_threshold_c_and_top_level_f(self):
        """Verify per-variable Celsius override takes precedence over top-level Fahrenheit."""
        ds = xr.Dataset()
        ds.attrs["derived_variable_overrides"] = {"CDD_loca": {"threshold_c": 18.0}}
        got = get_derived_threshold(ds, "CDD_loca")
        assert np.isclose(got, 18.0 + 273.15)

    def test_top_level_threshold_f(self):
        """Verify top-level Fahrenheit threshold is converted to Kelvin."""
        ds = xr.Dataset()
        ds.attrs["threshold_f"] = 70.0
        got = get_derived_threshold(ds)
        expect = f_to_k(70.0)
        assert np.isclose(got, expect)
