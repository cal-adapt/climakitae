import numpy as np

from climakitae.new_core.derived_variables.builtin.wind import (
    calc_wind_speed_10m,
    calc_wind_direction_10m,
)


def test_wind_speed_and_direction(wind_dataset):
    ds = wind_dataset.copy()

    out_speed = calc_wind_speed_10m(ds.copy())
    expected_speed = np.sqrt(ds.u10.values ** 2 + ds.v10.values ** 2)
    assert np.allclose(out_speed.wind_speed_10m.values, expected_speed)

    out_dir = calc_wind_direction_10m(ds.copy())
    expected_dir = (270 - np.arctan2(ds.v10.values, ds.u10.values) * 180 / np.pi) % 360
    assert np.allclose(out_dir.wind_direction_10m.values, expected_dir)
