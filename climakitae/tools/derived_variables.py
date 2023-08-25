"""Functions for deriving frequently used variables"""

import numpy as np
import xarray as xr


def compute_hdd_cdd(t2, hdd_threshold, cdd_threshold):
    """Compute heating degree days (HDD) and cooling degree days (CDD)

    Parameters
    -----------
    t2: xr.DataArray
        Air temperature at 2m gridded data
    hdd_threshold: int, optional
        Standard temperature in Fahrenheit.
    cdd_threshold: int, optional
        Standard temperature in Fahrenheit.

    Returns
    -------
    tuple of xr.DataArray
        (hdd, cdd)
    """

    # Check that temperature data was passed to function, throw error if not
    if t2.name != "Air Temperature at 2m":
        raise Exception(
            "Invalid input data, please provide Air Temperature at 2m data to CDD/HDD calculation"
        )

    # Subtract t2 from the threshold inputs
    hdd_deg_less_than_standard = hdd_threshold - t2
    cdd_deg_less_than_standard = cdd_threshold - t2

    # Compute HDD: Find positive difference (i.e. days < 65 degF)
    hdd = hdd_deg_less_than_standard.clip(0, None)
    # Replace negative values with 0
    hdd.name = "Heating Degree Days"
    hdd.attrs["hdd_threshold"] = (
        str(hdd_threshold) + " degF"
    )  # add attribute of threshold value

    # Compute CDD: Find negative difference (i.e. days > 65 degF)
    cdd = (-1) * cdd_deg_less_than_standard.clip(None, 0)
    # Replace positive values with 0
    cdd.name = "Cooling Degree Days"
    cdd.attrs["cdd_threshold"] = (
        str(cdd_threshold) + " degF"
    )  # add attribute of threshold value

    return (hdd, cdd)


def compute_hdh_cdh(t2, hdh_threshold, cdh_threshold):
    """Compute heating degree hours (HDH) and cooling degree hours (CDH)

    Parameters
    -----------
    t2: xr.DataArray
        Air temperature at 2m gridded data
    hdh_threshold: int, optional
        Standard temperature in Fahrenheit.
    cdh_threshold: int, optional
        Standard temperature in Fahrenheit.

    Returns
    -------
    tuple of xr.DataArray
        (hdh, cdh)
    """

    # Check that temperature data was passed to function, throw error if not
    if t2.name != "Air Temperature at 2m":
        raise Exception(
            "Invalid input data, please provide Air Temperature at 2m data to CDH/HDH calculation"
        )

    # Calculate heating and cooling hours
    cooling_hours = t2.where(
        t2 > cdh_threshold
    )  # temperatures above threshold, require cooling
    heating_hours = t2.where(
        t2 < hdh_threshold
    )  # temperatures below threshold, require heating

    # Compute CDH: count number of hours and resample to daily (max 24 value)
    cdh = cooling_hours.resample(time="1D").count(dim="time").squeeze()
    cdh.name = "Cooling Degree Hours"
    cdh.attrs["cdh_threshold"] = str(cdh_threshold) + " degF"

    # Compute HDH: count number of hours and resample to daily (max 24 value)
    hdh = heating_hours.resample(time="1D").count(dim="time").squeeze()
    hdh.name = "Heating Degree Hours"
    hdh.attrs["hdh_threshold"] = str(hdh_threshold) + " degF"

    return (hdh, cdh)


def compute_dewpointtemp(temperature, rel_hum):
    """Calculate dew point temperature

    Args:
        temperature (xr.DataArray): Temperature in Kelvin
        rel_hum (xr.DataArray): Relative humidity (0-100 scale)

    Returns
        dew_point (xr.DataArray): Dew point (K)

    """
    es = 0.611 * np.exp(
        5423 * ((1 / 273) - (1 / temperature))
    )  # calculates saturation vapor pressure
    e_vap = (es * rel_hum) / 100.0  # calculates vapor pressure
    tdps = (
        (1 / 273) - 0.0001844 * np.log(e_vap / 0.611)
    ) ** -1  # calculates dew point temperature, units = K

    # Assign descriptive name
    tdps.name = "dew_point_derived"
    tdps.attrs["units"] = "K"
    return tdps


def compute_specific_humidity(tdps, pressure, name="q2_derived"):
    """Compute specific humidity.

    Args:
        tdps (xr.DataArray): Dew-point temperature, in K
        pressure (xr.DataArray): Air pressure, in Pascals
        name (str, optional): Name to assign to output DataArray

    Returns:
        spec_hum (xr.DataArray): Specific humidity

    """

    # Calculate vapor pressure, unit is in kPa
    e = 0.611 * np.exp((2500000 / 461) * ((1 / 273) - (1 / tdps)))

    # Calculate specific humidity, unit is g/g, pressure has to be divided by 1000 to get to kPa at this step
    q = (0.622 * e) / (pressure / 1000)

    # Convert from g/g to g/kg for more understandable value
    q = q * 1000

    # Assign descriptive name
    q.name = name
    q.attrs["units"] = "g/kg"
    return q


def compute_relative_humidity(pressure, temperature, mixing_ratio, name="rh_derived"):
    """Compute relative humidity.
    Variable attributes need to be assigned outside of this function because the metpy function removes them

    Args:
        pressure (xr.DataArray): Pressure in hPa
        temperature (xr.DataArray): Temperature in Celsius
        mixing_ratio (xr.DataArray): Dimensionless mass mixing ratio in g/kg
        name (str, optional): Name to assign to output DataArray

    Returns:
        rel_hum (xr.DataArray): Relative humidity

    Source: https://www.weather.gov/media/epz/wxcalc/mixingRatio.pdf
    """

    # Calculates saturated vapor pressure
    e_s = 6.11 * 10 ** (7.5 * (temperature / (237.7 + temperature)))

    # calculate saturation mixing ratio, unit is g/kg
    w_s = 621.97 * (e_s / (pressure - e_s))

    # Calculates relative humidity, unit is 0 to 100
    rel_hum = 100 * (mixing_ratio / w_s)

    # Reset unrealistically low relative humidity values
    # Lowest recorded relative humidity value in CA is 0.8%
    rel_hum = xr.where(rel_hum > 0.5, rel_hum, 0.5)

    # Reset values above 100 to 100
    rel_hum = xr.where(rel_hum < 100, rel_hum, 100)

    # Reassign coordinate attributes
    # For some reason, these get improperly assigned in the xr.where step
    for coord in list(rel_hum.coords):
        rel_hum[coord].attrs = temperature[coord].attrs

    # Assign descriptive name
    rel_hum.name = name
    rel_hum.attrs["units"] = "[0 to 100]"
    return rel_hum


def _convert_specific_humidity_to_relative_humidity(
    temperature, q, pressure, name="rh_derived"
):
    """Converts specific humidity to relative humidity.

    Args:
        temperature (xr.DataArray): Temperature in Kelvin
        q (xr.DataArray): Specific humidity, in g/kg
        pressure (xr.DataArray): Pressure, in Pascals
        name (str, optional): Name to assign to output DataArray

    Returns:
        rel_hum (xr.DataArray): Relative humidity
    """

    # Calculates saturated vapor pressure, unit is in kPa
    e_s = 0.611 * np.exp((2500000 / 461) * ((1 / 273) - (1 / temperature)))

    # Convert pressure unit to be compatible with e_s, unit to kPa
    pressure = pressure / 1000

    # Convert specific humidity unit to be compatible with epsilon (0.622), unit g/g
    q = q / 1000

    # Calculate relative humidity
    rel_hum = (q * pressure) * (0.622 * e_s)

    # Assign descriptive name
    rel_hum.name = name
    rel_hum.attrs["units"] = "[0 to 100]"
    return rel_hum


def compute_wind_mag(u10, v10, name="wind_speed_derived"):
    """Compute wind magnitude at 10 meters

    Args:
        u10 (xr.DataArray): Zonal velocity at 10 meters height in m/s
        v10 (xr.DataArray): Meridonal velocity at 10 meters height in m/s
        name (str, optional): Name to assign to output DataArray

    Returns:
        wind_mag (xr.DataArray): Wind magnitude

    """
    wind_mag = np.sqrt(np.square(u10) + np.square(v10))
    wind_mag.name = "wind_speed_derived"
    wind_mag.attrs["units"] = "m s-1"
    return wind_mag


def compute_wind_dir(u10, v10, name="wind_direction_derived"):
    """Compute wind direction at 10 meters

    Args:
        u10 (xr.DataArray): Zonal velocity at 10 meters height in m/s
        v10 (xr.DataArray): Meridional velocity at 10 meters height in m/s
        name (str, optional): Name to assign to output DataArray

    Returns:
        wind_dir (xr.DataArray): Wind direction, in [0, 360] degrees,
            with 0/360 defined as north, by meteorological convention

    Notes:
        source:  https://sites.google.com/view/raybellwaves/cheat-sheets/xarray
    """

    wind_dir = np.mod(90 - np.arctan2(-v10, -u10) * (180 / np.pi), 360)
    wind_dir.name = name
    wind_dir.attrs["units"] = "degrees"
    return wind_dir
