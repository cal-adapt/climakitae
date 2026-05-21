"""Unit tests for fire weather index derived variables."""

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from climakitae.new_core.derived_variables.builtin.indices import (
    _equilibrium_moisture_constant,
    _moisture_dampening_coeff,
    calc_fosberg_fire_weather_index,
)


@pytest.fixture
def ffwi_dataset():
    """Synthetic dataset with all inputs required for FFWI."""
    time = pd.date_range("2020-01-01", periods=3)
    shape = (3, 2, 2)

    ds = xr.Dataset(
        {
            "t2": (("time", "y", "x"), np.full(shape, 300.0)),  # K
            "q2": (("time", "y", "x"), np.full(shape, 0.008)),  # kg/kg
            "psfc": (("time", "y", "x"), np.full(shape, 100000.0)),  # Pa
            "u10": (("time", "y", "x"), np.full(shape, 3.0)),  # m/s
            "v10": (("time", "y", "x"), np.full(shape, 4.0)),  # m/s
        },
        coords={"time": time, "y": np.arange(2), "x": np.arange(2)},
    )
    return ds


class TestFosbergFireWeatherIndex:

    def test_output_variable_present(self, ffwi_dataset):
        """FFWI variable is added to the dataset."""
        out = calc_fosberg_fire_weather_index(ffwi_dataset.copy())
        assert "fosberg_fire_weather_index" in out.data_vars

    def test_output_range(self, ffwi_dataset):
        """FFWI values are clipped to [0, 100]."""
        out = calc_fosberg_fire_weather_index(ffwi_dataset.copy())
        ffwi = out["fosberg_fire_weather_index"]
        assert float(ffwi.min()) >= 0.0
        assert float(ffwi.max()) <= 100.0

    def test_intermediate_vars_dropped(self, ffwi_dataset):
        """relative_humidity_2m and wind_speed_10m are not in the output."""
        out = calc_fosberg_fire_weather_index(ffwi_dataset.copy())
        assert "relative_humidity_2m" not in out.data_vars
        assert "wind_speed_10m" not in out.data_vars

    def test_attrs(self, ffwi_dataset):
        """Output has expected units attribute."""
        out = calc_fosberg_fire_weather_index(ffwi_dataset.copy())
        assert out["fosberg_fire_weather_index"].attrs.get("units") == "[0 to 100]"

    def test_output_shape(self, ffwi_dataset):
        """Output shape matches input shape."""
        out = calc_fosberg_fire_weather_index(ffwi_dataset.copy())
        assert out["fosberg_fire_weather_index"].shape == (3, 2, 2)

    def test_value_accuracy(self, ffwi_dataset):
        """FFWI values match manual calculation."""
        out = calc_fosberg_fire_weather_index(ffwi_dataset.copy())

        # Reproduce the calculation manually
        t2_F = (300.0 - 273.15) * 9 / 5 + 32
        wind_ms = np.sqrt(3.0**2 + 4.0**2)
        wind_mph = wind_ms * 2.23694

        # RH from q2, psfc, t2 (same formula as humidity.py)
        t_c = 300.0 - 273.15
        es = 611.2 * np.exp(17.67 * t_c / (t_c + 243.5))
        e = 0.008 * 100000.0 / (0.622 + 0.378 * 0.008)
        rh = np.clip(100.0 * e / es, 0, 100)

        m_low, m_mid, m_high = _equilibrium_moisture_constant(rh, t2_F)
        m = m_low if rh < 10 else m_mid
        m = m_high if rh > 50 else m
        n = _moisture_dampening_coeff(m)
        expected = np.clip(n * ((1 + wind_mph**2) ** 0.5) / 0.3002, 0, 100)

        assert np.allclose(out["fosberg_fire_weather_index"].values, expected)

    @pytest.mark.parametrize(
        "rh_regime,q2",
        [
            ("low", 0.0001),  # RH < 10%
            ("mid", 0.008),  # 10% < RH <= 50%
            ("high", 0.018),  # RH > 50%
        ],
    )
    def test_rh_regimes(self, ffwi_dataset, rh_regime, q2):
        """FFWI runs without error across all three RH regimes."""
        ds = ffwi_dataset.copy()
        ds["q2"] = xr.full_like(ds["q2"], q2)
        out = calc_fosberg_fire_weather_index(ds)
        ffwi = out["fosberg_fire_weather_index"]
        assert float(ffwi.min()) >= 0.0
        assert float(ffwi.max()) <= 100.0
