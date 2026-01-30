"""Humidity-related derived variables.

This module provides derived variables for humidity calculations including
relative humidity, dew point temperature, and specific humidity.

Derived Variables
-----------------
relative_humidity_2m
    Relative humidity at 2m from temperature, specific humidity, and pressure.
dew_point_2m
    Dew point temperature at 2m from temperature and relative humidity.

"""

import logging

import numpy as np

from climakitae.new_core.derived_variables.registry import register_derived

logger = logging.getLogger(__name__)


@register_derived(
    variable="relative_humidity_2m",
    query={"variable_id": ["t2", "q2", "psfc"]},
    description="Relative humidity at 2m computed from temperature, specific humidity, and surface pressure",
    units="%",
    source="builtin",
)
def calc_relative_humidity_2m(ds):
    """Calculate relative humidity at 2m.

    Parameters
    ----------
    ds : xr.Dataset
        Dataset containing:
        - 't2': 2m temperature (K)
        - 'q2': 2m specific humidity (kg/kg)
        - 'psfc': Surface pressure (Pa)

    Returns
    -------
    xr.Dataset
        Dataset with 'relative_humidity_2m' variable added (0-100 scale).

    Notes
    -----
    Uses the approximation:
    - Saturation vapor pressure: es = 611.2 * exp(17.67 * (T-273.15) / (T-29.65))
    - Vapor pressure from specific humidity: e = q * p / (0.622 + 0.378*q)
    - Relative humidity: RH = 100 * e / es

    """
    logger.debug("Computing relative_humidity_2m from t2, q2, psfc")

    # Temperature in Celsius
    t_celsius = ds.t2 - 273.15

    # Saturation vapor pressure (Pa) using Tetens formula
    es = 611.2 * np.exp(17.67 * t_celsius / (t_celsius + 243.5))

    # Vapor pressure from specific humidity (Pa)
    # q = 0.622 * e / (p - 0.378 * e) => e = q * p / (0.622 + 0.378 * q)
    e = ds.q2 * ds.psfc / (0.622 + 0.378 * ds.q2)

    # Relative humidity (%)
    rh = 100.0 * e / es

    # Clip to valid range
    rh = rh.clip(0, 100)

    ds["relative_humidity_2m"] = rh
    ds["relative_humidity_2m"].attrs = {
        "units": "%",
        "long_name": "Relative Humidity at 2m",
        "standard_name": "relative_humidity",
        "valid_min": 0,
        "valid_max": 100,
        "derived_from": "t2, q2, psfc",
        "derived_by": "climakitae",
    }
    return ds


@register_derived(
    variable="dew_point_2m",
    query={"variable_id": ["t2", "rh"]},
    description="Dew point temperature at 2m from temperature and relative humidity",
    units="K",
    source="builtin",
)
def calc_dew_point_2m(ds):
    """Calculate dew point temperature at 2m.

    Parameters
    ----------
    ds : xr.Dataset
        Dataset containing:
        - 't2': 2m temperature (K)
        - 'rh': Relative humidity (0-100 scale)

    Returns
    -------
    xr.Dataset
        Dataset with 'dew_point_2m' variable added (K).

    Notes
    -----
    Uses the Magnus formula approximation for dew point.

    """
    logger.debug("Computing dew_point_2m from t2 and rh")

    # Constants for Magnus formula
    a = 17.27
    b = 237.7  # °C

    # Temperature in Celsius
    t_celsius = ds.t2 - 273.15

    # Gamma function
    gamma = (a * t_celsius / (b + t_celsius)) + np.log(ds.rh / 100.0)

    # Dew point in Celsius
    tdp_celsius = b * gamma / (a - gamma)

    # Convert back to Kelvin
    ds["dew_point_2m"] = tdp_celsius + 273.15
    ds["dew_point_2m"].attrs = {
        "units": "K",
        "long_name": "Dew Point Temperature at 2m",
        "standard_name": "dew_point_temperature",
        "derived_from": "t2, rh",
        "derived_by": "climakitae",
    }
    return ds


@register_derived(
    variable="specific_humidity_2m",
    query={"variable_id": ["t2", "rh", "psfc"]},
    description="Specific humidity at 2m from temperature, relative humidity, and pressure",
    units="kg/kg",
    source="builtin",
)
def calc_specific_humidity_2m(ds):
    """Calculate specific humidity at 2m.

    Parameters
    ----------
    ds : xr.Dataset
        Dataset containing:
        - 't2': 2m temperature (K)
        - 'rh': Relative humidity (0-100 scale)
        - 'psfc': Surface pressure (Pa)

    Returns
    -------
    xr.Dataset
        Dataset with 'specific_humidity_2m' variable added (kg/kg).

    """
    logger.debug("Computing specific_humidity_2m from t2, rh, psfc")

    # Temperature in Celsius
    t_celsius = ds.t2 - 273.15

    # Saturation vapor pressure (Pa)
    es = 611.2 * np.exp(17.67 * t_celsius / (t_celsius + 243.5))

    # Actual vapor pressure (Pa)
    e = (ds.rh / 100.0) * es

    # Specific humidity (kg/kg)
    # q = 0.622 * e / (p - 0.378 * e)
    q = 0.622 * e / (ds.psfc - 0.378 * e)

    ds["specific_humidity_2m"] = q
    ds["specific_humidity_2m"].attrs = {
        "units": "kg/kg",
        "long_name": "Specific Humidity at 2m",
        "standard_name": "specific_humidity",
        "derived_from": "t2, rh, psfc",
        "derived_by": "climakitae",
    }
    return ds
