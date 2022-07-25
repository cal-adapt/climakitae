"""
This script calculates alternative units for variables with multiple
commonly used units.
"""

def _unit_convert_precip(variable, preferred_units):
    """ Converts units for any precipitation-related variable

    Args:
      variable (xr.DataArray): Precipitation (default unit mm)
      preferred_units (string): Option of either "mm" or "in" for unit

    Returns:
      variable (xr.DataArray): Total precipitation (converted unit in)
    """

  # If choice of units is the default unit, do nothing
    if variable.units == preferred_units:
        return variable

    if variable.units == "mm" and preferred_units == "in":
        variable = total_precip / 25.4
        variable.units = "in"

    return variable


def _unit_convert_pressure(pressure, preferred_units):
    """ Converts units for air pressure

    Args:
      pressure (xr.DataArray): Pressure (default unit Pa)
      preferred_units (string): Option of Pa, hPa, mb, inHg

    Returns:
      pressure (xr.DataArray): Pressue (with unit of choice)
    """

    # If choice of units is the default unit, do nothing
    if pressure.units == preferred_units:
        return pressure

    if pressure.units == "Pa" and preferred_units == "hPa":
        pressure = pressure / 100.
        pressure.units = "hPa"

    elif pressure.units == "Pa" and preferred_units == "mb":
        pressure = pressure / 100.
        pressure.units = "mb"

    elif pressure.units == "Pa" and preferred_units == "inHg":
        pressure = pressure / 100. / 29.92
        pressure.units = "inHg"

    return pressure


def _unit_convert_winds(variable, preferred_units):
    """ Converts units for any wind magnitude-related variable

        Args:
            variable (xr.DataArray): Wind magnitude (default unit m/s)
            preferred_units (string): Option of m/s, knots

        Returns:
            variable (xr.DataArray): Wind magnitude (with unit of choice)
    """

    # If choice of units is the default unit, do nothing
    if variable.units == preferred_units:
        return variable

    if variable.units == "m/s" and preferred_units == "kts":
        variable = variable * 1.94
        variable.units = "kts"

    return variable

def _unit_convert_temp(variable, preferred_units):
    """ Converts units for any temperature-related variable

        Args:
            variable (xr.DataArray): temperature (default unit K)
            preferred_units (string): Option of K, degC, degF, degR

        Returns:
            variable (xr.DataArray): temperature (with unit of choice)
    """

    #If choice of units is the default unit, do nothing
    if variable.units == preferred_units:
        return variable

    if variable.units == "K" and preferred_units == "degC":
        variable = variable - 273.15
        variable.units = "degC"

    elif variable.units == "K" and preferred_units == "degF":
        variable = 1.8 * (variable - 273.15) - 32
        variable.units = "degF"

    elif variable.units == "K" and preferred_units == "degR":
        variable = variable * 1.8
        variable.units = "degR"

    return variable
