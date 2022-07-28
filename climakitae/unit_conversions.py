"""
This script calculates alternative units for variables with multiple commonly used units.

"""

def _convert_units(da, native_units, selected_units):
    """ Converts units for any variable

    Args:
      da (xr.DataArray): Data
      native_units (str): Native units of data, from selections.native_units
      selected_units (str): Selected units of data, from selections.units

    Returns:
      da (xr.DataArray): Data with converted units and updated units attribute
    """

    # Units that have existing conversions
    units_with_conversions = ["mm","Pa","m/s","K","kg/kg"]

    # Pass if chosen units is the same as native units
    if native_units == selected_units:
        pass

    # Raise error; Unit selected exists as button, but is not completely integrated into the code selection process.
    elif (native_units != selected_units) and (native_units not in units_with_conversions):
        raise ValueError("You've encountered a bug in the code. Selected unit " + selected_units + " is not a valid unit option. Check package data and/or source code for data_loaders and unit_conversions modules.")

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
