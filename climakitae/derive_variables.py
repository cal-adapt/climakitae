import metpy.calc 
import metpy
import numpy as np

def _compute_total_precip(cumulus_precip, gridcell_precip, variable_name="TOT_PRECIP"): 
    """ Compute total precipitation 
    
    Args: 
        cumulus_precip (xr.DataArray): Accumulated total cumulus precipitation (mm)
        gridcell_precip (xr.DataArray): Accumulated total grid scale precipitation (mm) 
        variable_name (string): Name to assign DataArray object (default to "TOT_PRECIP")
        
    Returns: 
        total_precip (xr.DataArray): Total precipitation (mm)
    """
    
    total_precip = cumulus_precip + gridcell_precip 
    
    # Assign descriptive name and attributes 
    total_precip.name = variable_name 
    total_precip.attrs["description"] = "Total precipitation" 
    
    return total_precip


def _compute_relative_humidity(pressure, temperature, mixing_ratio, variable_name="REL_HUMIDITY"): 
    """Compute relative humidity 
    
    Args: 
        pressure (xr.DataArray): Pressure in Pascals 
        temperature (xr.DataArray): Temperature in Kelvin 
        mixing_ratio (xr.DataArray): Dimensionless mass mixing ratio in kg/kg
        variable_name (string): Name to assign DataArray object (default to "REL_HUMIDITY")
        
    Returns: 
        rel_hum (xr.DataArray): Relative humidity
    
    """
    rel_hum = metpy.calc.relative_humidity_from_mixing_ratio(pressure=pressure, temperature=temperature, mixing_ratio=mixing_ratio) 
    rel_hum = rel_hum.metpy.dequantify() # metpy function returns a pint.Quantity object, which can cause issues with dask. This can be undone using the dequantify function. For more info: https://unidata.github.io/MetPy/latest/tutorials/xarray_tutorial.html
    
    # Assign descriptive name and attributes 
    rel_hum.name = variable_name 
    rel_hum.attrs["description"] = "Relative humidity" 
    
    return rel_hum 


def _compute_wind_mag(u10, v10, variable_name="WIND_MAG"): 
    """Compute wind magnitude at 10 meters 
    
    Args: 
        u10 (xr.DataArray): Zonal velocity at 10 meters height in m/s 
        v10 (xr.DataArray): Meridonal velocity at 10 meters height in m/s
        variable_name (string): Name to assign DataArray object (default to "WIND_MAG")
    
    Returns: 
        wind_mag (xr.DataArray): Wind magnitude 
    
    """
    wind_mag = np.sqrt(np.square(u10) + np.square(v10))
    
    # Assign descriptive name and attributes 
    wind_mag.name = variable_name 
    wind_mag.attrs["description"] = "Wind magnitude at 10 meters"
    wind_mag.attrs["units"] = "m2 s-1 2"
    
    return wind_mag


def _compute_wind_direction(u10, v10, variable_name="WIND_DIR"): 
    """Compute wind direction at 10 meters 
    
    Args: 
        u10 (xr.DataArray): Zonal velocity at 10 meters height in m/s 
        v10 (xr.DataArray): Meridonal velocity at 10 meters height in m/s
        variable_name (string): Name to assign DataArray object (default to "WIND_DIR")
    
    Returns: 
        wind_dir (xr.DataArray): Wind direction
    
    """
    
    wind_dir = metpy.calc.wind_direction(u10, v10, convention="from")
    wind_dir = wind_dir.metpy.dequantify() # metpy function returns a pint.Quantity object, which can cause issues with dask. This can be undone using the dequantify function. For more info: https://unidata.github.io/MetPy/latest/tutorials/xarray_tutorial.html
    
    # Assign descriptive name and attributes 
    wind_dir.name = variable_name 
    wind_dir.attrs["description"] = "Direction where the wind is coming from at 10m" 
    #wind_dir.attrs["units"] = "Degrees (0-360)"
    return wind_dir