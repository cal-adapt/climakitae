"""Fire weather and climate indices derived variables.

This module provides derived variables for fire weather and climate index
calculations.

Derived Variables
-----------------
fosberg_fire_weather_index
    Fosberg Fire Weather Index from temperature, humidity, and wind.

"""

import logging

import xarray as xr

from climakitae.new_core.derived_variables.builtin.humidity import (
    calc_relative_humidity_2m,
)
from climakitae.new_core.derived_variables.builtin.wind import calc_wind_speed_10m
from climakitae.new_core.derived_variables.registry import register_derived

logger = logging.getLogger(__name__)


def _equilibrium_moisture_constant(
    h: xr.DataArray, T: xr.DataArray
) -> tuple[xr.DataArray, xr.DataArray, xr.DataArray]:
    """Compute the equilibrium moisture constant.
    Dependent on relative humidity percent.
    Used to compute Fosberg Fire Weather Index.
    Will return three values corresponding to the level of humidity.

    Parameters
    ----------
    h : xr.DataArray
        relative humidity in units of 0-100 (percent)
    T : xr.DataArray
        air temperature in units of Fahrenheit

    Returns
    -------
    m_low : xr.DataArray
        equilibrium moisture constant for low humidity (<10%)
    m_mid : xr.DataArray
        equilibrium moisture constant for 10% < humidity <= 50%
    m_high : xr.DataArray
        equilibrium moisture constant for high humidity (>50%)

    """
    # h < 10: Low humidity
    m_low = 0.03229 + 0.281073 * h - 0.000578 * h * T

    # (10 < h <= 50): Mid humiditiy
    m_mid = 2.22749 + 0.160107 * h - 0.01478 * T

    # h > 50: High humidity
    m_high = 21.0606 + 0.005565 * (h**2) - 0.00035 * h * T - 0.483199 * h

    return (m_low, m_mid, m_high)


def _moisture_dampening_coeff(m: xr.DataArray) -> xr.DataArray:
    """Compute the moisture dampening coefficient.
    Used to compute Fosberg Fire Weather Index.

    Parameters
    ----------
    m : xr.DataArray
        equilibrium moisture constant

    Returns
    -------
    n : xr.DataArray
        moisture dampening coefficient

    """
    n = 1 - 2 * (m / 30) + 1.5 * (m / 30) ** 2 - 0.5 * (m / 30) ** 3
    return n


@register_derived(
    variable="fosberg_fire_weather_index",
    query={"variable_id": ["t2", "q2", "psfc", "u10", "v10"]},
    description="Fosberg Fire Weather Index computed from temperature, humidity, and wind speed",
    units="[0 to 100]",
    source="builtin",
)
def calc_fosberg_fire_weather_index(ds):
    """Calculate the Fosberg Fire Weather Index (FFWI).

    Parameters
    ----------
    ds : xr.Dataset
        Dataset containing:
        - 't2': 2m temperature (K)
        - 'q2': 2m specific humidity (kg/kg)
        - 'psfc': Surface pressure (Pa)
        - 'u10': U-component of wind at 10m (m/s)
        - 'v10': V-component of wind at 10m (m/s)

    Returns
    -------
    xr.Dataset
        Dataset with 'fosberg_fire_weather_index' variable added (0-100 scale).

    References
    ----------
    https://a.atmos.washington.edu/wrfrt/descript/definitions/fosbergindex.html
    https://www.spc.noaa.gov/exper/firecomp/INFO/fosbinfo.html

    """
    logger.debug(
        "Computing fosberg fire weather index (FFWI) from t2, q2, psfc, u10, v10"
    )
    logger.debug(
        "Relative humidity and wind speed will be computed as intermediate steps"
    )

    ds = calc_relative_humidity_2m(ds)
    ds = calc_wind_speed_10m(ds)

    # Convert units
    t2_F = (ds.t2 - 273.15) * 9 / 5 + 32  # K -> Fahrenheit
    rh = ds["relative_humidity_2m"]  # already 0-100
    wind_mph = ds["wind_speed_10m"] * 2.23694  # m/s -> mph

    # Equilibrium moisture content
    m_low, m_mid, m_high = _equilibrium_moisture_constant(h=rh, T=t2_F)
    m = xr.where(rh < 10, m_low, m_mid)
    m = xr.where(rh > 50, m_high, m)

    # Moisture dampening coefficient
    n = _moisture_dampening_coeff(m)

    # FFWI, clipped to [0, 100]
    ffwi = (n * ((1 + wind_mph**2) ** 0.5) / 0.3002).clip(0, 100)

    # Restore coordinate attributes lost in xr.where
    for coord in list(ffwi.coords):
        if coord in ds.coords:
            ffwi[coord].attrs = ds[coord].attrs

    ds["fosberg_fire_weather_index"] = ffwi
    ds["fosberg_fire_weather_index"].attrs = {
        "units": "[0 to 100]",
        "long_name": "Fosberg Fire Weather Index",
        "derived_from": "t2, q2, psfc, u10, v10",
        "derived_by": "climakitae",
    }

    # Drop intermediate derived variables
    ds = ds.drop_vars(["relative_humidity_2m", "wind_speed_10m"])
    return ds
