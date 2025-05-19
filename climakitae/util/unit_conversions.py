"""
Calculates alternative units for variables with multiple commonly
used units, following NWS conversions for pressure and wind speed.
"""

import xarray as xr


def get_unit_conversion_options() -> dict:
    """Get dictionary of unit conversion options offered for each unit"""
    options = {
        "K": ["K", "degC", "degF"],
        "degF": ["K", "degC", "degF"],
        "degC": ["K", "degC", "degF"],
        "hPa": ["Pa", "hPa", "mb", "inHg"],
        "Pa": ["Pa", "hPa", "mb", "inHg"],
        "m/s": ["m/s", "mph", "knots"],
        "m s-1": ["m s-1", "mph", "knots"],
        "[0 to 100]": ["[0 to 100]", "fraction"],
        "mm": ["mm", "inches"],
        "mm/d": ["mm/d", "inches/d"],
        "mm/h": ["mm/h", "inches/h"],
        "kg/kg": ["kg/kg", "g/kg"],
        "kg kg-1": ["kg kg-1", "g kg-1"],
        "kg m-2 s-1": ["kg m-2 s-1", "mm", "inches"],
        "g/kg": ["g/kg", "kg/kg"],
    }
    return options


def convert_units(da: xr.DataArray, selected_units: str) -> xr.DataArray:
    """Converts units for any variable

    Parameters
    ----------
    da: xr.DataArray
        data
    selected_units: str
        selected units of data, from selections.units

    Returns
    -------
    da: xr.DataArray
        data with converted units and updated units attribute

    References
    ----------
    Wind speed: https://www.weather.gov/media/epz/wxcalc/windConversion.pdf
    Pressure: https://www.weather.gov/media/epz/wxcalc/pressureConversion.pdf
    """

    # Get native units of data from attributes
    try:
        native_units = da.attrs["units"]
    except:
        raise ValueError(
            (
                "You've encountered a bug in the code. This variable "
                "does not have identifiable native units. The data"
                " for this variable will need to have a 'units'"
                " attribute added in the catalog."
            )
        )

    # Convert hPa to Pa to make conversions easier
    # Monthly data native unit is hPa, hourly is Pa
    if native_units == "hPa" and selected_units != "hPa":
        da = da / 100.0
        da.attrs["units"] = "Pa"
        native_units = "Pa"

    # Pass if chosen units is the same as native units
    match native_units:
        case native_units if native_units == selected_units:
            return da

        # Precipitation units
        case "mm" | "mm/d" | "mm/h":
            match selected_units:
                case "inches" | "inches/h" | "inches/d":
                    da = da / 25.4
                case "kg m-2 s-1":
                    da = da / 86400
                    if da.attrs["frequency"] == "monthly":
                        da_name = da.name
                        da = da / da["time"].dt.days_in_month
                        da.name = da_name  # Name is lost during computation above
        case "kg m-2 s-1":
            if da.attrs["frequency"] == "monthly":
                da_name = da.name
                da = da * da["time"].dt.days_in_month
                da.name = da_name  # Name is lost during computation above
            match selected_units:
                case "mm":
                    da = da * 86400
                case "inches":
                    da = (da * 86400) / 25.4

        # Moisture ratio units
        case "kg/kg" | "kg kg-1":
            if selected_units in ["g/kg", "g kg-1"]:
                da = da * 1000

        # Specific humidity
        case "g/kg":
            if selected_units == "kg/kg":
                da = da / 1000
                da.attrs["units"] = selected_units

        # Pressure units
        case "Pa":
            match selected_units:
                case "hPa":
                    da = da / 100.0
                case "mb":
                    da = da / 100.0
                case "inHg":
                    da = da * 0.000295300

        # Wind units
        case "m/s" | "m s-1":
            match selected_units:
                case "knots":
                    da = da * 1.9438445
                case "mph":
                    da = da * 2.236936

        # Temperature units
        case "K":
            match selected_units:
                case "degC":
                    da = da - 273.15
                case "degF":
                    da = (1.8 * (da - 273.15)) + 32
        case "degC":
            match selected_units:
                case "K":
                    da = da + 273.15
                case "degF":
                    da = (1.8 * da) + 32
        case "degF":
            # Convert to C
            match selected_units:
                case "degC":
                    da = (da - 32) / 1.8
                # Then, if K is selected, convert to K
                case "K":
                    da = (da - 32) / 1.8
                    da = da + 273.15

        # Fraction/percentage units (relative humidity)
        case "[0 to 100]":
            if selected_units == "fraction":
                da = da / 100.0

    # Update unit attribute to reflect converted unit
    da.attrs["units"] = selected_units
    return da
