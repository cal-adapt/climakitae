import numpy as np
import xarray as xr

from climakitae.new_core.derived_variables.utils import get_derived_threshold
from climakitae.core.constants import DEFAULT_DEGREE_DAY_THRESHOLD_K
from climakitae.util.utils import f_to_k


def test_explicit_threshold_k():
    ds = xr.Dataset()
    assert get_derived_threshold(ds, threshold_k=300.0) == 300.0


def test_explicit_threshold_f_converts():
    ds = xr.Dataset()
    got = get_derived_threshold(ds, threshold_f=75.0)
    expect = f_to_k(75.0)
    assert np.isclose(got, expect)


def test_per_variable_override_from_attrs():
    ds = xr.Dataset()
    ds.attrs["derived_variable_overrides"] = {"CDD_wrf": {"threshold_f": 75.0}}
    got = get_derived_threshold(ds, "CDD_wrf")
    expect = f_to_k(75.0)
    assert np.isclose(got, expect)


def test_toplevel_attr_override():
    ds = xr.Dataset()
    ds.attrs["threshold_c"] = 20.0
    got = get_derived_threshold(ds)
    expect = 20.0 + 273.15
    assert np.isclose(got, expect)


def test_default_fallback():
    ds = xr.Dataset()
    got = get_derived_threshold(ds)
    assert np.isclose(got, float(DEFAULT_DEGREE_DAY_THRESHOLD_K))


def test_per_variable_override_threshold_k_and_c_alias():
    ds = xr.Dataset()
    ds.attrs["derived_variable_params"] = {"HDD_loca": {"threshold_k": 295.0}}
    got = get_derived_threshold(ds, "HDD_loca")
    assert np.isclose(got, 295.0)


def test_per_variable_override_threshold_c_and_top_level_f():
    ds = xr.Dataset()
    ds.attrs["derived_variable_overrides"] = {"CDD_loca": {"threshold_c": 18.0}}
    got = get_derived_threshold(ds, "CDD_loca")
    assert np.isclose(got, 18.0 + 273.15)


def test_top_level_threshold_f():
    ds = xr.Dataset()
    ds.attrs["threshold_f"] = 70.0
    got = get_derived_threshold(ds)
    expect = f_to_k(70.0)
    assert np.isclose(got, expect)
