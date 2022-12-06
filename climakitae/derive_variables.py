import numpy as np


def _compute_relative_humidity(
    pressure, temperature, mixing_ratio
):
    """Compute relative humidity.
    Variable attributes need to be assigned outside of this function because the metpy function removes them


    Args:
        pressure (xr.DataArray): Pressure in Pascals
        temperature (xr.DataArray): Temperature in Kelvin
        mixing_ratio (xr.DataArray): Dimensionless mass mixing ratio in kg/kg

    Returns:
        rel_hum (xr.DataArray): Relative humidity

    """

    # Calculates saturated vapor pressure, unit is in kPa
    e_s = 0.611 * np.exp((2500000 / 461) * ((1 / 273) - (1 / temperature)))

    # Calculates saturated mixing ratio, unit is kg/kg, pressure has to be divided by 1000 to get to kPa at this step
    r_s = (0.622 * e_s) / ((pressure / 1000) - e_s)

    # Calculates relative humidity, unit is 0 to 100
    rel_hum = 100 * (mixing_ratio / r_s)

    # Assign descriptive name
    rel_hum.name = "rh_derived"
    rel_hum.attrs["units"] = "[0 to 100]"

    return rel_hum


def _compute_wind_mag(u10, v10):
    """Compute wind magnitude at 10 meters

    Args:
        u10 (xr.DataArray): Zonal velocity at 10 meters height in m/s
        v10 (xr.DataArray): Meridonal velocity at 10 meters height in m/s

    Returns:
        wind_mag (xr.DataArray): Wind magnitude

    """
    wind_mag = np.sqrt(np.square(u10) + np.square(v10))
    wind_mag.name = "wind_speed_derived"
    return wind_mag
