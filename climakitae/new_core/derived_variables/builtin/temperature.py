"""Temperature-related derived variables.

This module provides derived variables for temperature calculations including
heat index, wind chill, apparent temperature, and degree days.

Derived Variables
-----------------
heat_index
    Heat index (feels-like temperature accounting for humidity).
wind_chill
    Wind chill (feels-like temperature accounting for wind).
apparent_temperature
    Apparent temperature combining temperature, humidity, and wind effects.
diurnal_temperature_range
    Daily temperature range (tasmax - tasmin).
HDD_wrf
    Heating degree days from WRF data (base 65°F).
CDD_wrf
    Cooling degree days from WRF data (base 65°F).
HDD_loca
    Heating degree days from LOCA2 data (base 65°F).
CDD_loca
    Cooling degree days from LOCA2 data (base 65°F).

"""

import logging

import numpy as np
import xarray as xr

from climakitae.new_core.derived_variables.registry import register_derived
from climakitae.new_core.derived_variables.utils import get_derived_threshold

logger = logging.getLogger(__name__)


@register_derived(
    variable="heat_index",
    query={"variable_id": ["t2", "rh"]},
    description="Heat index (feels-like temperature accounting for humidity effects)",
    units="K",
    source="builtin",
)
def calc_heat_index(ds):
    """Calculate heat index from temperature and relative humidity.

    Parameters
    ----------
    ds : xr.Dataset
        Dataset containing:
        - 't2': 2m temperature (K)
        - 'rh': Relative humidity (0-100 scale)

    Returns
    -------
    xr.Dataset
        Dataset with 'heat_index' variable added (K).

    Notes
    -----
    Uses the NOAA/NWS heat index formula. The heat index is only meaningful
    when temperature is above ~27°C (80°F) and relative humidity is above ~40%.
    For lower values, the original temperature is returned.

    Reference: https://www.weather.gov/media/ffc/ta_htindx.PDF

    """
    logger.debug("Computing heat_index from t2 and rh")

    # Convert to Fahrenheit for the standard formula
    t_fahrenheit = (ds.t2 - 273.15) * 9 / 5 + 32
    rh = ds.rh

    # Simple approximation for low temperatures
    hi_simple = 0.5 * (t_fahrenheit + 61.0 + (t_fahrenheit - 68.0) * 1.2 + rh * 0.094)

    # Full Rothfusz regression for higher temperatures
    c1 = -42.379
    c2 = 2.04901523
    c3 = 10.14333127
    c4 = -0.22475541
    c5 = -6.83783e-3
    c6 = -5.481717e-2
    c7 = 1.22874e-3
    c8 = 8.5282e-4
    c9 = -1.99e-6

    hi_full = (
        c1
        + c2 * t_fahrenheit
        + c3 * rh
        + c4 * t_fahrenheit * rh
        + c5 * t_fahrenheit**2
        + c6 * rh**2
        + c7 * t_fahrenheit**2 * rh
        + c8 * t_fahrenheit * rh**2
        + c9 * t_fahrenheit**2 * rh**2
    )

    # Adjustments for edge cases
    # Low humidity adjustment
    low_rh_mask = (rh < 13) & (t_fahrenheit >= 80) & (t_fahrenheit <= 112)
    adjustment1 = ((13 - rh) / 4) * np.sqrt((17 - np.abs(t_fahrenheit - 95)) / 17)
    hi_full = xr.where(low_rh_mask, hi_full - adjustment1, hi_full)

    # High humidity adjustment
    high_rh_mask = (rh > 85) & (t_fahrenheit >= 80) & (t_fahrenheit <= 87)
    adjustment2 = ((rh - 85) / 10) * ((87 - t_fahrenheit) / 5)
    hi_full = xr.where(high_rh_mask, hi_full + adjustment2, hi_full)

    # Use simple formula for lower temps, full formula for higher
    heat_index_f = xr.where(hi_simple < 80, hi_simple, hi_full)

    # Use original temp if below heat index threshold
    heat_index_f = xr.where(t_fahrenheit < 80, t_fahrenheit, heat_index_f)

    # Convert back to Kelvin
    heat_index_k = (heat_index_f - 32) * 5 / 9 + 273.15

    ds["heat_index"] = heat_index_k
    ds["heat_index"].attrs = {
        "units": "K",
        "long_name": "Heat Index",
        "standard_name": "apparent_temperature",
        "comment": "Feels-like temperature accounting for humidity effects",
        "derived_from": "t2, rh",
        "derived_by": "climakitae",
    }
    return ds


@register_derived(
    variable="wind_chill",
    query={"variable_id": ["t2", "u10", "v10"]},
    description="Wind chill (feels-like temperature accounting for wind effects)",
    units="K",
    source="builtin",
)
def calc_wind_chill(ds):
    """Calculate wind chill from temperature and wind components.

    Parameters
    ----------
    ds : xr.Dataset
        Dataset containing:
        - 't2': 2m temperature (K)
        - 'u10': 10m U wind component (m/s)
        - 'v10': 10m V wind component (m/s)

    Returns
    -------
    xr.Dataset
        Dataset with 'wind_chill' variable added (K).

    Notes
    -----
    Uses the NWS wind chill formula (2001 revision). Wind chill is only
    calculated when temperature is below 10°C (50°F) and wind speed is
    above 4.8 km/h (3 mph). Otherwise, the original temperature is returned.

    Reference: https://www.weather.gov/media/epz/wxcalc/windChill.pdf

    """
    logger.debug("Computing wind_chill from t2, u10, v10")

    # Wind speed in mph (formula uses mph)
    wind_speed_ms = np.sqrt(ds.u10**2 + ds.v10**2)
    wind_speed_mph = wind_speed_ms * 2.237  # m/s to mph

    # Temperature in Fahrenheit
    t_fahrenheit = (ds.t2 - 273.15) * 9 / 5 + 32

    # NWS Wind Chill formula
    wind_chill_f = (
        35.74
        + 0.6215 * t_fahrenheit
        - 35.75 * (wind_speed_mph**0.16)
        + 0.4275 * t_fahrenheit * (wind_speed_mph**0.16)
    )

    # Only apply when temp < 50°F and wind > 3 mph
    valid_mask = (t_fahrenheit <= 50) & (wind_speed_mph > 3)
    wind_chill_f = xr.where(valid_mask, wind_chill_f, t_fahrenheit)

    # Convert back to Kelvin
    wind_chill_k = (wind_chill_f - 32) * 5 / 9 + 273.15

    ds["wind_chill"] = wind_chill_k
    ds["wind_chill"].attrs = {
        "units": "K",
        "long_name": "Wind Chill Temperature",
        "standard_name": "apparent_temperature",
        "comment": "Feels-like temperature accounting for wind effects",
        "derived_from": "t2, u10, v10",
        "derived_by": "climakitae",
    }
    return ds


@register_derived(
    variable="diurnal_temperature_range_loca",
    query={"variable_id": ["tasmax", "tasmin"]},
    description="Daily temperature range (maximum minus minimum)",
    units="K",
    source="builtin",
)
def calc_diurnal_temperature_range_loca(ds):
    """Calculate diurnal (daily) temperature range.

    Parameters
    ----------
    ds : xr.Dataset
        Dataset containing:
        - 'tasmax': Daily maximum temperature (K)
        - 'tasmin': Daily minimum temperature (K)

    Returns
    -------
    xr.Dataset
        Dataset with 'diurnal_temperature_range_loca' variable added (K).

    """
    logger.debug("Computing diurnal_temperature_range_loca from tasmax and tasmin")

    ds["diurnal_temperature_range_loca"] = ds.tasmax - ds.tasmin
    ds["diurnal_temperature_range_loca"].attrs = {
        "units": "K",
        "long_name": "Diurnal Temperature Range",
        "comment": "Daily maximum minus daily minimum temperature",
        "derived_from": "tasmax, tasmin",
        "derived_by": "climakitae",
    }
    return ds


@register_derived(
    variable="diurnal_temperature_range_wrf",
    query={"variable_id": ["t2max", "t2min"]},
    description="Daily temperature range from WRF data (maximum minus minimum)",
    units="K",
    source="builtin",
)
def calc_diurnal_temperature_range_wrf(ds):
    """Calculate diurnal (daily) temperature range from WRF variables.

    Parameters
    ----------
    ds : xr.Dataset
        Dataset containing:
        - 't2max': Daily maximum 2m temperature (K)
        - 't2min': Daily minimum 2m temperature (K)

    Returns
    -------
    xr.Dataset
        Dataset with 'diurnal_temperature_range_wrf' variable added (K).

    """
    logger.debug("Computing diurnal_temperature_range_wrf from t2max and t2min")

    ds["diurnal_temperature_range_wrf"] = ds.t2max - ds.t2min
    ds["diurnal_temperature_range_wrf"].attrs = {
        "units": "K",
        "long_name": "Diurnal Temperature Range",
        "comment": "Daily maximum minus daily minimum 2m temperature",
        "derived_from": "t2max, t2min",
        "derived_by": "climakitae",
    }
    return ds


@register_derived(
    variable="HDD_wrf",
    query={"variable_id": ["t2"]},
    description="Heating degree days from WRF data (base 65°F)",
    units="K",
    source="builtin",
)
def calc_hdd_wrf(ds, threshold_k=None, threshold_c=None, threshold_f=None):
    """Calculate heating degree days from WRF temperature data.

    Parameters
    ----------
    ds : xr.Dataset
        Dataset containing:
        - 't2max': Daily maximum 2m temperature (K)
        - 't2min': Daily minimum 2m temperature (K)
    threshold_k : float, optional
        Threshold for heating degree days in Kelvin.
    threshold_c : float, optional
        Threshold for heating degree days in Celsius.
    threshold_f : float, optional
        Threshold for heating degree days in Fahrenheit.

    Returns
    -------
    xr.Dataset
        Dataset with 'HDD_wrf' variable added (K).

    Notes
    -----
    Heating degree days (HDD) represent the energy demand for heating.
    Calculated as max(0, threshold - average_temp) where threshold is 65°F (291.48K)
    and average_temp is (t2max + t2min) / 2.

    """
    logger.debug("Computing HDD_wrf from t2max and t2min")

    # Calculate daily average temperature
    t_avg = ds.t2

    # Resolve threshold (explicit args > dataset attrs > global default)
    threshold_k = get_derived_threshold(
        ds,
        "HDD_wrf",
        threshold_k=threshold_k,
        threshold_c=threshold_c,
        threshold_f=threshold_f,
    )

    # HDD = max(0, threshold - avg_temp)
    ds["HDD_wrf"] = np.maximum(0, threshold_k - t_avg)
    ds["HDD_wrf"] = (ds["HDD_wrf"] > 0).astype(int)  # Binary mask data
    ds["HDD_wrf"].attrs = {
        "units": "K",
        "long_name": "Heating Degree Days (WRF)",
        "comment": f"Heating degree days calculated from daily average temperature with base {threshold_k} K",
        "derived_from": "t2",
        "derived_by": "climakitae",
        "threshold": f"{threshold_k} K",
    }
    return ds


@register_derived(
    variable="CDD_wrf",
    query={"variable_id": ["t2"]},
    description="Cooling degree days from WRF data (base 65°F)",
    units="K",
    source="builtin",
)
def calc_cdd_wrf(ds, threshold_k=None, threshold_c=None, threshold_f=None):
    """Calculate cooling degree days from WRF temperature data.

    Parameters
    ----------
    ds : xr.Dataset
        Dataset containing:
        - 't2max': Daily maximum 2m temperature (K)
        - 't2min': Daily minimum 2m temperature (K)
    threshold_k : float, optional
        Threshold for cooling degree days in Kelvin.
    threshold_c : float, optional
        Threshold for cooling degree days in Celsius.
    threshold_f : float, optional
        Threshold for cooling degree days in Fahrenheit.

    Returns
    -------
    xr.Dataset
        Dataset with 'CDD_wrf' variable added (K).

    Notes
    -----
    Cooling degree days (CDD) represent the energy demand for cooling.
    Calculated as max(0, average_temp - threshold) where threshold is 65°F (291.48K)
    and average_temp is (t2max + t2min) / 2.

    """
    logger.debug("Computing CDD_wrf from t2max and t2min")

    # Calculate daily average temperature
    t_avg = ds.t2

    # Resolve threshold (explicit args > dataset attrs > global default)
    threshold_k = get_derived_threshold(
        ds,
        "CDD_wrf",
        threshold_k=threshold_k,
        threshold_c=threshold_c,
        threshold_f=threshold_f,
    )

    # CDD = max(0, avg_temp - threshold)
    ds["CDD_wrf"] = np.maximum(0, t_avg - threshold_k)
    ds["CDD_wrf"] = (ds["CDD_wrf"] > 0).astype(int)  # Binary mask data
    ds["CDD_wrf"].attrs = {
        "units": "K",
        "long_name": "Cooling Degree Days (WRF)",
        "comment": f"Cooling degree days calculated from daily average temperature with base {threshold_k} K",
        "derived_from": "t2",
        "derived_by": "climakitae",
        "threshold": f"{threshold_k} K",
    }
    return ds


@register_derived(
    variable="HDD_loca",
    query={"variable_id": ["tasmax", "tasmin"]},
    description="Heating degree days from LOCA2 data (base 65°F)",
    units="K",
    source="builtin",
)
def calc_hdd_loca(ds, threshold_k=None, threshold_c=None, threshold_f=None):
    """Calculate heating degree days from LOCA2 temperature data.

    Parameters
    ----------
    ds : xr.Dataset
        Dataset containing:
        - 'tasmax': Daily maximum temperature (K)
        - 'tasmin': Daily minimum temperature (K)
    threshold_k : float, optional
        Threshold for heating degree days in Kelvin.
    threshold_c : float, optional
        Threshold for heating degree days in Celsius.
    threshold_f : float, optional
        Threshold for heating degree days in Fahrenheit.

    Returns
    -------
    xr.Dataset
        Dataset with 'HDD_loca' variable added (K).

    Notes
    -----
    Heating degree days (HDD) represent the energy demand for heating.
    Calculated as max(0, threshold - average_temp) where threshold is 65°F (291.48K)
    and average_temp is (tasmax + tasmin) / 2.

    """
    logger.debug("Computing HDD_loca from tasmax and tasmin")

    # Calculate daily average temperature
    t_avg = (ds.tasmax + ds.tasmin) / 2

    # Resolve threshold (explicit args > dataset attrs > global default)
    threshold_k = get_derived_threshold(
        ds,
        "HDD_loca",
        threshold_k=threshold_k,
        threshold_c=threshold_c,
        threshold_f=threshold_f,
    )

    # HDD = max(0, threshold - avg_temp)
    ds["HDD_loca"] = np.maximum(0, threshold_k - t_avg)
    ds["HDD_loca"] = (ds["HDD_loca"] > 0).astype(int)  # Binary mask data
    ds["HDD_loca"].attrs = {
        "units": "K",
        "long_name": "Heating Degree Days (LOCA2)",
        "comment": f"Heating degree days calculated from daily average temperature with base {threshold_k} K",
        "derived_from": "tasmax, tasmin",
        "derived_by": "climakitae",
        "threshold": f"{threshold_k} K",
    }
    return ds


@register_derived(
    variable="CDD_loca",
    query={"variable_id": ["tasmax", "tasmin"]},
    description="Cooling degree days from LOCA2 data (base 65°F)",
    units="K",
    source="builtin",
)
def calc_cdd_loca(ds, threshold_k=None, threshold_c=None, threshold_f=None):
    """Calculate cooling degree days from LOCA2 temperature data.

    Parameters
    ----------
    ds : xr.Dataset
        Dataset containing:
        - 'tasmax': Daily maximum temperature (K)
        - 'tasmin': Daily minimum temperature (K)
    threshold_k : float, optional
        Threshold for cooling degree days in Kelvin.
    threshold_c : float, optional
        Threshold for cooling degree days in Celsius.
    threshold_f : float, optional
        Threshold for cooling degree days in Fahrenheit.

    Returns
    -------
    xr.Dataset
        Dataset with 'CDD_loca' variable added (K).

    Notes
    -----
    Cooling degree days (CDD) represent the energy demand for cooling.
    Calculated as max(0, average_temp - threshold) where threshold is 65°F (291.48K)
    and average_temp is (tasmax + tasmin) / 2.

    """
    logger.debug("Computing CDD_loca from tasmax and tasmin")

    # Calculate daily average temperature
    t_avg = (ds.tasmax + ds.tasmin) / 2

    # Resolve threshold (explicit args > dataset attrs > global default)
    threshold_k = get_derived_threshold(
        ds,
        "CDD_loca",
        threshold_k=threshold_k,
        threshold_c=threshold_c,
        threshold_f=threshold_f,
    )

    # CDD = max(0, avg_temp - threshold)
    ds["CDD_loca"] = np.maximum(0, t_avg - threshold_k)
    ds["CDD_loca"] = (ds["CDD_loca"] > 0).astype(int)  # Binary mask data
    ds["CDD_loca"].attrs = {
        "units": "K",
        "long_name": "Cooling Degree Days (LOCA2)",
        "comment": f"Cooling degree days calculated from daily average temperature with base {threshold_k} K",
        "derived_from": "tasmax, tasmin",
        "derived_by": "climakitae",
        "threshold": f"{threshold_k} K",
    }
    return ds
