"""
This script calculates alternative units for variables with multiple commonly used units, following NWS conversions for pressure and wind speed.
Wind speed: https://www.weather.gov/media/epz/wxcalc/windConversion.pdf
Pressure: https://www.weather.gov/media/epz/wxcalc/pressureConversion.pdf

"""

def _get_unit_conversion_options(): 
    """Get dictionary of unit conversion options offered for each unit"""
    options = {
        "K":["K","degC","degF"],
        "hPa":["Pa","hPa","mb","inHg"], 
        "Pa":["Pa","hPa","mb","inHg"], 
        "m/s":["m/s","knots"], 
        "m s-1": ["m s-1","knots"], 
        "[0 to 100]":["[0 to 100]","fraction"], 
        "mm":["mm","inches"],
        "kg/kg":["kg/kg","g/kg"],
        "kg kg-1":["kg kg-1","g kg-1"]
        }
    return options 

def _convert_units(da, selected_units):
    """ Converts units for any variable

    Args:
      da (xr.DataArray): Data
      selected_units (str): Selected units of data, from selections.units

    Returns:
      da (xr.DataArray): Data with converted units and updated units attribute
    """

    # Get native units of data from attributes
    try:
        native_units = da.attrs["units"]
    except:
        raise ValueError("You've encountered a bug in the code. This variable does not have identifiable native units. The data for this variable will need to have a 'units' attribute added in the catalog.")
        
    # Convert hPa to Pa to make conversions easier 
    # Monthly data native unit is hPa, hourly is Pa 
    if native_units == "hPa" and selected_units != "hPa": 
        da = da * 100.
        da.attrs["units"] = "Pa"
        native_units = "Pa" 

    # Pass if chosen units is the same as native units
    if native_units == selected_units:
        pass

    # Precipitation units
    elif native_units == "mm":
        if selected_units == "inches":
            da = da / 25.4
            da.attrs["units"] = selected_units

    # Moisture ratio units
    elif native_units == in ["kg/kg","kg kg-1"]:
        if selected_units == ["g/kg","g kg-1"]:
            da = da * 1000
            da.attrs["units"] = selected_units

    # Pressure units
    elif native_units == "Pa":
        if selected_units == "hPa":
            da = da / 100.
            da.attrs["units"] = selected_units
        elif selected_units == "mb":
            da = da / 100.
            da.attrs["units"] = selected_units
        elif selected_units == "inHg":
            da = da * 0.000295300
            da.attrs["units"] = selected_units

    # Wind units
    elif native_units == ["m/s","m s-1"]:
        if selected_units == "knots":
            da = da * 1.9438445
            da.attrs["units"] = selected_units

    # Temperature units
    elif native_units == "K":
        if selected_units == "degC":
            da = da - 273.15
            da.attrs["units"] = selected_units
        elif selected_units == "degF":
            da = (1.8 * (da - 273.15)) + 32
            da.attrs["units"] = selected_units
            
    # Fraction/percentage units (relative humidity) 
    elif native_units == "[0 to 100]": 
        if selected_units == "fraction":
            da = da / 100. 
            da.attrs["units"] = selected_units

    return da
