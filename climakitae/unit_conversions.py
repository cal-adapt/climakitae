"""
This script calculates alternative units for variables with multiple commonly used units.

"""

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


    # Pass if chosen units is the same as native units
    if native_units == selected_units:
        pass

    # Precipitation units
    elif native_units == "mm":
        if selected_units == "inches":
            da = da / 25.4
            da.attrs["units"] = selected_units

    # Moisture ratio units
    elif native_units == "kg/kg":
        if selected_units == "g/kg":
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
            da = da / 3386.39
            da.attrs["units"] = selected_units

    # Wind units
    elif native_units == "m/s":
        if selected_units == "knots":
            da = da * 1.94
            da.attrs["units"] = selected_units

    # Temperature units
    elif native_units == "K":
        if selected_units == "degC":
            da = da - 273.15
            da.attrs["units"] = selected_units
        elif selected_units == "degF":
            da = (1.8 * (da - 273.15)) + 32
            da.attrs["units"] = selected_units

    return da
