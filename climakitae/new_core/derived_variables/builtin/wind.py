"""Wind-related derived variables.

This module provides derived variables for wind calculations including
wind speed and wind direction from U and V wind components.

Derived Variables
-----------------
wind_speed_10m
    Wind speed at 10m computed from u10 and v10 components.
wind_direction_10m
    Wind direction at 10m computed from u10 and v10 components.

"""

import logging

import numpy as np

from climakitae.new_core.derived_variables.registry import register_derived

logger = logging.getLogger(__name__)


@register_derived(
    variable="wind_speed_10m",
    query={"variable_id": ["u10", "v10"]},
    description="Wind speed at 10m height computed from U and V components",
    units="m/s",
    source="builtin",
)
def calc_wind_speed_10m(ds):
    """Calculate wind speed at 10m from U and V components.

    Parameters
    ----------
    ds : xr.Dataset
        Dataset containing 'u10' and 'v10' variables.

    Returns
    -------
    xr.Dataset
        Dataset with 'wind_speed_10m' variable added.

    Notes
    -----
    Wind speed is computed as: sqrt(u10² + v10²)

    """
    logger.debug("Computing wind_speed_10m from u10 and v10")
    ds["wind_speed_10m"] = np.sqrt(ds.u10**2 + ds.v10**2)
    ds["wind_speed_10m"].attrs = {
        "units": "m/s",
        "long_name": "Wind Speed at 10m",
        "standard_name": "wind_speed",
        "derived_from": "u10, v10",
        "derived_by": "climakitae",
    }
    return ds


@register_derived(
    variable="wind_direction_10m",
    query={"variable_id": ["u10", "v10"]},
    description="Wind direction at 10m height (meteorological convention: direction wind comes FROM)",
    units="degrees",
    source="builtin",
)
def calc_wind_direction_10m(ds):
    """Calculate wind direction at 10m from U and V components.

    Parameters
    ----------
    ds : xr.Dataset
        Dataset containing 'u10' and 'v10' variables.

    Returns
    -------
    xr.Dataset
        Dataset with 'wind_direction_10m' variable added.

    Notes
    -----
    Wind direction uses meteorological convention: the direction the wind
    is coming FROM, measured clockwise from north.

    Direction = (270 - atan2(v10, u10) * 180/π) mod 360

    """
    logger.debug("Computing wind_direction_10m from u10 and v10")
    # Meteorological wind direction: direction wind comes FROM
    wind_dir = (270 - np.arctan2(ds.v10, ds.u10) * 180 / np.pi) % 360
    ds["wind_direction_10m"] = wind_dir
    ds["wind_direction_10m"].attrs = {
        "units": "degrees",
        "long_name": "Wind Direction at 10m",
        "standard_name": "wind_from_direction",
        "comment": "Meteorological convention: direction wind is coming FROM, clockwise from north",
        "derived_from": "u10, v10",
        "derived_by": "climakitae",
    }
    return ds
