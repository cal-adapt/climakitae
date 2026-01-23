import numpy as np

from climakitae.new_core.derived_variables.builtin.humidity import (
    calc_relative_humidity_2m,
    calc_dew_point_2m,
    calc_specific_humidity_2m,
)


def test_relative_humidity(humidity_dataset):
    ds = humidity_dataset.copy()
    out = calc_relative_humidity_2m(ds.copy())

    t_celsius = ds.t2 - 273.15
    es = 611.2 * np.exp(17.67 * t_celsius / (t_celsius + 243.5))
    e = ds.q2 * ds.psfc / (0.622 + 0.378 * ds.q2)
    expected = 100.0 * e / es
    expected = expected.clip(0, 100)

    assert np.allclose(out.relative_humidity_2m.values, expected.values)
    assert out.relative_humidity_2m.attrs.get("units") == "%"


def test_dew_point(humidity_dataset):
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


def test_specific_humidity(humidity_dataset):
    ds = humidity_dataset.copy()
    out = calc_specific_humidity_2m(ds.copy())

    t_celsius = ds.t2 - 273.15
    es = 611.2 * np.exp(17.67 * t_celsius / (t_celsius + 243.5))
    e = (ds.rh / 100.0) * es
    q = 0.622 * e / (ds.psfc - 0.378 * e)

    assert np.allclose(out.specific_humidity_2m.values, q.values)
