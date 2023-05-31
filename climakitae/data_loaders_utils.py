"""Helper functions to clean up the data_loaders module. """

from .unit_conversions import _convert_units
from .data_loaders import _get_data_one_var
from .derive_variables import (
    _compute_relative_humidity,
    _compute_wind_mag,
    _compute_wind_dir,
    _compute_dewpointtemp,
    _compute_specific_humidity,
)
from .fire import fosberg_fire_index


def _get_wind_speed_derived(selections, cat):
    """Get input data and derive wind speed for hourly data"""
    # Load u10 data
    selections.variable_id = ["u10"]
    selections.units = (
        "m s-1"  # Need to set units to required units for _compute_wind_mag
    )
    u10_da = _get_data_one_var(selections, cat)

    # Load v10 data
    selections.variable_id = ["v10"]
    selections.units = "m s-1"
    v10_da = _get_data_one_var(selections, cat)

    # Derive the variable
    da = _compute_wind_mag(u10=u10_da, v10=v10_da)  # m/s
    return da


def _get_wind_dir_derived(selections, cat):
    """Get input data and derive wind direction for hourly data"""
    # Load u10 data
    selections.variable_id = ["u10"]
    selections.units = (
        "m s-1"  # Need to set units to required units for _compute_wind_mag
    )
    u10_da = _get_data_one_var(selections, cat)

    # Load v10 data
    selections.variable_id = ["v10"]
    selections.units = "m s-1"
    v10_da = _get_data_one_var(selections, cat)

    # Derive the variable
    da = _compute_wind_dir(u10=u10_da, v10=v10_da)
    return da


def _get_monthly_daily_dewpoint(selections, cat):
    """Derive dew point temp for monthly/daily data."""
    # Daily/monthly dew point inputs have different units
    # Hourly dew point temp derived differently because you also have to derive relative humidity

    # Load temperature data
    selections.variable_id = ["t2"]
    selections.units = "K"  # Kelvin required for humidity and dew point computation
    t2_da = _get_data_one_var(selections, cat)

    selections.variable_id = ["rh"]
    selections.units = "[0 to 100]"
    rh_da = _get_data_one_var(selections, cat)

    # Derive dew point temperature
    # Returned in units of Kelvin
    da = _compute_dewpointtemp(temperature=t2_da, rel_hum=rh_da)  # Kelvin  # [0-100]
    return da


def _get_hourly_dewpoint(selections, cat):
    """Derive dew point temp for hourly data.
    Requires first deriving relative humidity.
    """
    # Load temperature data
    selections.variable_id = ["t2"]
    selections.units = "K"  # Kelvin required for humidity and dew point computation
    t2_da = _get_data_one_var(selections, cat)

    # Load mixing ratio data
    selections.variable_id = ["q2"]
    selections.units = "kg kg-1"
    q2_da = _get_data_one_var(selections, cat)

    # Load pressure data
    selections.variable_id = ["psfc"]
    selections.units = "Pa"
    pressure_da = _get_data_one_var(selections, cat)

    # Derive relative humidity
    # Returned in units of [0-100]
    rh_da = _compute_relative_humidity(
        pressure=pressure_da,  # Pa
        temperature=t2_da,  # Kelvin
        mixing_ratio=q2_da,  # kg/kg
    )

    # Derive dew point temperature
    # Returned in units of Kelvin
    da = _compute_dewpointtemp(temperature=t2_da, rel_hum=rh_da)  # Kelvin  # [0-100]
    return da


def _get_hourly_rh(selections, cat):
    """Derive hourly relative humidity."""
    # Load temperature data
    selections.variable_id = ["t2"]
    selections.units = "K"  # Kelvin required for humidity and dew point computation
    t2_da = _get_data_one_var(selections, cat)

    # Load mixing ratio data
    selections.variable_id = ["q2"]
    selections.units = "kg kg-1"
    q2_da = _get_data_one_var(selections, cat)

    # Load pressure data
    selections.variable_id = ["psfc"]
    selections.units = "Pa"
    pressure_da = _get_data_one_var(selections, cat)

    # Derive relative humidity
    # Returned in units of [0-100]
    da = _compute_relative_humidity(
        pressure=pressure_da,  # Pa
        temperature=t2_da,  # Kelvin
        mixing_ratio=q2_da,  # kg/kg
    )
    return da


def _get_hourly_specific_humidity(selections, cat):
    """Derive hourly specific humidity.
    Requires first deriving relative humidity, then dew point temp.
    """
    # Load temperature data
    selections.variable_id = ["t2"]
    selections.units = "K"  # Kelvin required for humidity and dew point computation
    t2_da = _get_data_one_var(selections, cat)

    # Load mixing ratio data
    selections.variable_id = ["q2"]
    selections.units = "kg kg-1"
    q2_da = _get_data_one_var(selections, cat)

    # Load pressure data
    selections.variable_id = ["psfc"]
    selections.units = "Pa"
    pressure_da = _get_data_one_var(selections, cat)

    # Derive relative humidity
    # Returned in units of [0-100]
    rh_da = _compute_relative_humidity(
        pressure=pressure_da,  # Pa
        temperature=t2_da,  # Kelvin
        mixing_ratio=q2_da,  # kg/kg
    )

    # Derive dew point temperature
    # Returned in units of Kelvin
    dew_pnt_da = _compute_dewpointtemp(
        temperature=t2_da, rel_hum=rh_da  # Kelvin  # [0-100]
    )

    # Derive specific humidity
    # Returned in units of g/kg
    da = _compute_specific_humidity(
        tdps=dew_pnt_da, pressure=pressure_da  # Kelvin  # Pa
    )
    return da


def _get_fosberg_fire_index(selections, cat):
    """Derive the fosberg fire index."""

    # Load temperature data
    selections.variable_id = ["t2"]
    selections.units = "K"  # Kelvin required for humidity and dew point computation
    t2_da_K = _get_data_one_var(selections, cat)

    # Load mixing ratio data
    selections.variable_id = ["q2"]
    selections.units = "kg kg-1"
    q2_da = _get_data_one_var(selections, cat)

    # Load pressure data
    selections.variable_id = ["psfc"]
    selections.units = "Pa"
    pressure_da = _get_data_one_var(selections, cat)

    # Load u10 data
    selections.variable_id = ["u10"]
    selections.units = (
        "m s-1"  # Need to set units to required units for _compute_wind_mag
    )
    u10_da = _get_data_one_var(selections, cat)

    # Load v10 data
    selections.variable_id = ["v10"]
    selections.units = "m s-1"
    v10_da = _get_data_one_var(selections, cat)

    # Derive relative humidity
    # Returned in units of [0-100]
    rh_da = _compute_relative_humidity(
        pressure=pressure_da,  # Pa
        temperature=t2_da_K,  # Kelvin
        mixing_ratio=q2_da,  # kg/kg
    )

    # Derive windspeed
    # Returned in units of m/s
    windspeed_da_ms = _compute_wind_mag(u10=u10_da, v10=v10_da)  # m/s

    # Convert units to proper units for fosberg index
    t2_da_F = _convert_units(t2_da_K, "degF")
    windspeed_da_mph = _convert_units(windspeed_da_ms, "mph")

    # Compute the index
    da = fosberg_fire_index(
        t2_F=t2_da_F, rh_percent=rh_da, windspeed_mph=windspeed_da_mph
    )
    return da
