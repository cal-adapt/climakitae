import numpy as np

def compute_hdd_cdd(t2, standard_temp=65): 
    """Compute heating degree days (HDD) and cooling degree days (CDD) 
    
    Args: 
        t2 (xr.DataArray): Air temperature at 2m gridded data 
        standard_temp (int, optional): standard temperature in Fahrenheit (default to 65)
        
    Returns: 
        hdd, cdd (xr.DataArray) 
    """

    # Subtract t2 from the standard reference temperature 
    deg_less_than_standard = standard_temp - t2

    # Compute HDD: Find positive difference (i.e. days < 65 degF) 
    hdd = deg_less_than_standard.where(deg_less_than_standard > 0, 0) # Replace negative values with 0
    hdd.name = "Heating Degree Days"

    # Compute CDD: Find negative difference (i.e. days > 65 degF)
    cdd = (-1)*deg_less_than_standard.where(deg_less_than_standard < 0, 0) # Replace positive values with 0
    cdd.name = "Cooling Degree Days"
    
    return (hdd, cdd) 

def _compute_dewpointtemp(temperature, rel_hum):
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


def _compute_relative_humidity(pressure, temperature, mixing_ratio):
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
