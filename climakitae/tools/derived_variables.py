"""Functions for deriving frequently used variables"""

import logging
from typing import Union

import numpy as np
import xarray as xr
from pyproj import Geod

from climakitae.tools.derived_variables import compute_wind_dir

# Module logger
logger = logging.getLogger(__name__)


def compute_hdd_cdd(
    t2: xr.DataArray, hdd_threshold: int, cdd_threshold: int
) -> tuple[xr.DataArray, xr.DataArray]:
    """Compute heating degree days (HDD) and cooling degree days (CDD)

    Parameters
    ----------
    t2 : xr.DataArray
        Air temperature at 2m gridded data
    hdd_threshold : int, optional
        Standard temperature in Fahrenheit.
    cdd_threshold : int, optional
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


def compute_hdh_cdh(
    t2: xr.DataArray, hdh_threshold: int, cdh_threshold: int
) -> tuple[xr.DataArray, xr.DataArray]:
    """Compute heating degree hours (HDH) and cooling degree hours (CDH)

    Parameters
    ----------
    t2 : xr.DataArray
        Air temperature at 2m gridded data
    hdh_threshold : int, optional
        Standard temperature in Fahrenheit.
    cdh_threshold : int, optional
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


def compute_dewpointtemp(
    temperature: xr.DataArray, rel_hum: xr.DataArray
) -> xr.DataArray:
    """Calculate dew point temperature

    Parameters
    ----------
        temperature : xr.DataArray
            Temperature in Kelvin (K)
        rel_hum : xr.DataArray
            Relative humidity (0-100 scale)

    Returns
    -------
        dew_point : xr.DataArray
            Dew point (K)

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


def compute_specific_humidity(
    tdps: xr.DataArray, pressure: xr.DataArray, name: str = "q2_derived"
) -> xr.DataArray:
    """Compute specific humidity.

    Parameters
    ----------
        tdps : xr.DataArray
            Dew-point temperature, in K
        pressure : xr.DataArray
            Air pressure, in Pascals
        name : str, optional
            Name to assign to output DataArray

    Returns
    -------
        spec_hum : xr.DataArray
            Specific humidity

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


def compute_relative_humidity(
    pressure: xr.DataArray,
    temperature: xr.DataArray,
    mixing_ratio: xr.DataArray,
    name: str = "rh_derived",
) -> xr.DataArray:
    """Compute relative humidity.
    Variable attributes need to be assigned outside of this function because the metpy function removes them

    Parameters
    ----------
        pressure : xr.DataArray
            Pressure in hPa
        temperature : xr.DataArray
            Temperature in Celsius
        mixing_ratio : xr.DataArray
            Dimensionless mass mixing ratio in g/kg
        name : str, optional
            Name to assign to output DataArray

    Returns
    -------
        rel_hum : xr.DataArray
            Relative humidity

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
    temperature: xr.DataArray,
    q: xr.DataArray,
    pressure: xr.DataArray,
    name: str = "rh_derived",
) -> xr.DataArray:
    """Converts specific humidity to relative humidity.

    Parameters
    ----------
        temperature : xr.DataArray
            Temperature in Kelvin
        q : xr.DataArray
            Specific humidity, in g/kg
        pressure : xr.DataArray
            Pressure, in Pascals
        name : str, optional
            Name to assign to output DataArray

    Returns
    -------
        rel_hum : xr.DataArray
            Relative humidity

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


def compute_wind_mag(
    u10: xr.DataArray, v10: xr.DataArray, name: str = "wind_speed_derived"
) -> xr.DataArray:
    """Compute wind magnitude at 10 meters

    Parameters
    ----------
        u10 : xr.DataArray
            Zonal velocity at 10 meters height in m/s
        v10 : xr.DataArray
            Meridonal velocity at 10 meters height in m/s
        name : str, optional
            Name to assign to output DataArray

    Returns
    -------
        wind_mag: xr.DataArray
            Wind magnitude

    """
    wind_mag = np.sqrt(np.square(u10) + np.square(v10))
    wind_mag.name = name
    wind_mag.attrs["units"] = "m s-1"
    return wind_mag


def compute_wind_dir(
    u10: xr.DataArray, v10: xr.DataArray, name: str = "wind_direction_derived"
) -> xr.DataArray:
    """Compute wind direction at 10 meters

    Parameters
    ----------
        u10 : xr.DataArray
            Zonal velocity at 10 meters height in m/s
        v10 : xr.DataArray
            Meridional velocity at 10 meters height in m/s
        name : str, optional
            Name to assign to output DataArray

    Returns
    -------
        wind_dir : xr.DataArray
            Wind direction, in [0, 360] degrees, with 0/360 defined as north, by meteorological convention

    Notes
    -----
        source: https://sites.google.com/view/raybellwaves/cheat-sheets/xarray

    """

    wind_dir = np.mod(90 - np.arctan2(-v10, -u10) * (180 / np.pi), 360)
    wind_dir.name = name
    wind_dir.attrs["units"] = "degrees"
    return wind_dir


def compute_sea_level_pressure(
    psfc: xr.DataArray,
    t2: xr.DataArray,
    q2: xr.DataArray,
    elevation: xr.DataArray,
    lapse_rate: Union[float, xr.DataArray] = 0.0065,
    average_t2: bool = True,
    name: str = "slp_derived",
) -> xr.DataArray:
    """Calculate sea level pressure from hourly surface pressure, temperature, and mixing ratio.

    This function uses the basic method derived from the hydrostatic balance equation
    and the equation of state (Hess 1979). The SLP calculation method used here may not produce
    satisfactory results in locations with high terrain.

    By default this method uses a standard lapse rate of 6.5°K/km when calculating the
    sea level virtual temperature (see Pauley 1998). Users should consider what lapse rate is
    appropriate for their location.

    An option is provided to use a 12-hour average temperature when computing the lapse rate;
    this option is expected to produce more moderate SLP values that are less influenced by
    extreme temperatures.

    Parameters
    ----------
        psfc : xr.DataArray
            Hourly surface pressure in Pascals
        t2 : xr.DataArray
            Hourly surface air temperature in Kelvin
        q2 : xr.DataArray
            Hourly surface mixing ratio
        elevation : xr.DataArray
            Elevation in meters
        lapse_rate : Union[float, xr.DataArray]
            Lapse rate in K/m. Default is 0.0065 K/m
        average_t2 : bool (default True)
            True to use 12-hour mean temperature
        name : str, optional
            Name to assign to output DataArray

    Returns
    -------
    xr.DataArray
        Sea level pressure in Pascals

    Notes
    -----
    Virtual temperature is computed in the following way:
    T_virtual = ((1 + 1.609 q2) / (1 + q2)) * t2
    T_virtual_mean = (2 * T_virtual + lapse_rate * elevation) / 2

    Sea level pressure is calculated as:
    slp = psfc * np.exp(elevation / ((Rd * T_virtual_mean)/g))
       where Rd is the specific gas constant for dry air
       and g is the acceleration due to gravity.

    References
    ----------
    Hess, S. L., 1979: Introduction to Theoretical Meteorology. Robert E. Krieger Publishing Company, 362 pp.
    Pauley, P. M., 1998: An Example of Uncertainty in Sea Level Pressure Reduction. Wea. Forecasting, 13, 833–850, https://doi.org/10.1175/1520-0434(1998)013<0833:AEOUIS>2.0.CO;2.
    """
    # Get mean virtual temperature
    if average_t2:
        logger.info("compute_sea_level_pressure: Using 12-timestep mean temperature.")
        if "time" in t2.dims:
            t2 = t2.rolling(time=12).mean()
        elif "time_delta" in t2.dims:
            t2 = t2.rolling(time_delta=12).mean()
        else:
            raise KeyError(
                "No time or time_delta axis found in t2. Use `average_t2=False` for data without time axis."
            )

    t_virtual_sfc = ((1 + 1.609 * q2) / (1 + q2)) * t2
    t_virtual_mean = (2 * t_virtual_sfc + lapse_rate * elevation) / 2

    # Adjust pressure with hypsometric equation
    Rd = 287.04  # gas constant for dry air, J kg−1 K−1
    g = 9.81  # acceleration due to gravity, m s-2

    h = (Rd * t_virtual_mean) / g
    slp = psfc * np.exp(elevation / h)
    slp.name = name
    slp.attrs["units"] = "Pa"
    return slp


def _wrf_deltas(h: xr.DataArray) -> tuple[xr.DataArray]:
    """Get the actual x and y spacing in meters.

    Find the distance between lat/lon points on a great circle. Assumes a
    spherical geoid. The returned deltas are assigned the coordinates of
    the terminus point of the delta.

    Parameters
    ----------
    h : xr.DataArray
        DataArray with x and y dimensions on WRF grid

    Returns
    -------
    Tuple[xr.DataArray]
        X and Y direction deltas.
    """
    g = Geod(ellps="sphere")
    forward_az, _, dy = g.inv(
        h.lon[0:-1, :], h.lat[0:-1, :], h.lon[1:, :], h.lat[1:, :]
    )
    dy[(forward_az < -90.0) | (forward_az > 90.0)] *= -1

    forward_az, _, dx = g.inv(
        h.lon[:, 0:-1], h.lat[:, 0:-1], h.lon[:, 1:], h.lat[:, 1:]
    )
    dx[(forward_az < -90.0) | (forward_az > 90.0)] *= -1
    # Convert to data array with coordinates of terminus point
    dx = xr.DataArray(
        data=dx,
        dims=["y", "x"],
        coords={
            "y": (["y"], h.y.data),
            "x": (["x"], h.x.data[1:]),
            "lon": (["y", "x"], h.lon.data[:, 1:]),
            "lat": (["y", "x"], h.lat.data[:, 1:]),
        },
    )
    dy = xr.DataArray(
        data=dy,
        dims=["y", "x"],
        coords={
            "y": (["y"], h.y.data[1:]),
            "x": (["x"], h.x.data),
            "lon": (["y", "x"], h.lon.data[1:, :]),
            "lat": (["y", "x"], h.lat.data[1:, :]),
        },
    )
    return dx, dy


def _align_dim(
    da_to_update: xr.DataArray, da_to_copy: xr.DataArray, copy_dim: str
) -> xr.DataArray:
    """Copy `dim` dimension along with `lat` and `lon` from  da_to_copy to da_to_update.

    Parameters
    ----------
    da_to_update : xr.DataArray
        Will be returned with copied dimension.
    da_to_copy : xr.DataArray
        Dimension will be copied from this array.
    copy_dim : str
        Name of the dimension to copy.

    Returns
    -------
    xr.DataArray
    """
    da_to_update[copy_dim] = da_to_copy[copy_dim]
    da_to_update["lat"] = da_to_copy["lat"]
    da_to_update["lon"] = da_to_copy["lon"]
    return da_to_update


def _get_spatial_derivatives(h: xr.DataArray) -> tuple[xr.DataArray]:
    """Get the spatial derivative in the x and y direction on the WRF grid.

    This code borrows heavily from Metpy.calc.tools.first_derivative. It
    uses a method developed to take spatial derivatives on an unevenly spaced
    grid. See the References for more information.

    Parameters
    ----------
    h : xr.DataArray
        Spatial data on WRF grid

    Returns
    -------
    tuple[xr.DataArray]
        Derivative of h with respect to x and y

    References
    ----------
    M.K Bowen, Ronald Smith; Derivative formulae and errors for non-uniformly spaced points. Proc. A 1 July 2005; 461 (2059): 1975–1997. https://doi.org/10.1098/rspa.2004.1430
    Metpy, 2026: first_derivative. Accessed 10 Feb 2026, https://unidata.github.io/MetPy/latest/api/generated/metpy.calc.first_derivative.html#first-derivative.
    """
    delta_x, delta_y = _wrf_deltas(h)
    deltas = {"x": delta_x, "y": delta_y}
    derivatives = []

    for indexer in ["x", "y"]:
        delta = deltas[indexer]
        concat_axis = h.get_axis_num(indexer)

        # Centered derivative
        back_one = h.isel({indexer: slice(0, -2)})
        center = h.isel({indexer: slice(1, -1)})
        forward_one = h.isel({indexer: slice(2, None)})
        # Delta has coordinates of terminus point
        delta_0 = delta.isel({indexer: slice(0, -1)})
        delta_1 = delta.isel({indexer: slice(1, None)})

        # Align the indexer dimension to correctly add shifted slices
        back_one = _align_dim(back_one, center, indexer)
        forward_one = _align_dim(forward_one, center, indexer)
        delta_0 = _align_dim(delta_0, center, indexer)
        delta_1 = _align_dim(delta_1, center, indexer)

        combined_delta = delta_0 + delta_1
        center_derivative = (
            (-delta_1) / (combined_delta * delta_0) * back_one
            + (delta_1 - delta_0) / (delta_0 * delta_1) * center
            + (delta_0) / (combined_delta * delta_1) * forward_one
        )

        # Left edge
        center = h.isel({indexer: slice(0, 1)})
        forward_one = h.isel({indexer: slice(1, 2)})
        forward_two = h.isel({indexer: slice(2, 3)})
        delta_0 = delta.isel({indexer: slice(0, 1)})
        delta_1 = delta.isel({indexer: slice(1, 2)})

        forward_one = _align_dim(forward_one, center, indexer)
        forward_two = _align_dim(forward_two, center, indexer)
        delta_0 = _align_dim(delta_0, center, indexer)
        delta_1 = _align_dim(delta_1, center, indexer)

        combined_delta = delta_0 + delta_1
        left_derivative = (
            -(combined_delta + delta_0) / (combined_delta * delta_0) * center
            + combined_delta / (delta_0 * delta_1) * forward_one
            - delta_0 / (combined_delta * delta_1) * forward_two
        )

        # Right edge
        back_two = h.isel({indexer: slice(-3, -2)})
        back_one = h.isel({indexer: slice(-2, -1)})
        center = h.isel({indexer: slice(-1, None)})
        delta_0 = delta.isel({indexer: slice(-2, -1)})
        delta_1 = delta.isel({indexer: slice(-1, None)})

        back_two = _align_dim(back_two, center, indexer)
        back_one = _align_dim(back_one, center, indexer)
        delta_0 = _align_dim(delta_0, center, indexer)
        delta_1 = _align_dim(delta_1, center, indexer)

        combined_delta = delta_0 + delta_1
        right_derivative = (
            delta_1 / (combined_delta * delta_0) * back_two
            - combined_delta / (delta_0 * delta_1) * back_one
            + (combined_delta + delta_1) / (combined_delta * delta_1) * center
        )

        # Combine into one data array
        derivative = xr.concat(
            [left_derivative, center_derivative, right_derivative], dim=indexer
        )
        derivative = xr.DataArray(
            data=derivative.transpose(*h.dims),
            dims=["sim", "warming_level", "time", "y", "x"],
            coords={
                "sim": (["sim"], h.sim.data),
                "warming_level": (["warming_level"], h.warming_level.data),
                "time": (["time"], h.time.data),
                "y": (["y"], h.y.data),
                "x": (["x"], h.x.data),
                "lon": (["y", "x"], h.lon.data),
                "lat": (["y", "x"], h.lat.data),
            },
            name="derivative",
        )
        derivatives.append(derivative)
    return tuple(derivatives)


def _get_rotated_geostrophic_wind(
    u: xr.DataArray, v: xr.DataArray, gridlabel: str
) -> tuple[xr.DataArray]:
    """Convert WRF-relative winds to Earth-relative winds.

    This is the code from data_load._get_Uearth and
    data_load._get_Vearth but adapted to take u and v as parameters.

    Parameters
    ----------
    u : xr.DataArray
        U component of wind
    v : xr.DataArray
        V component of wind
    gridlabel : str
        Grid label (e.g. "d01")

    Returns
    -------
    tuple[xr.DataArray]
        Earth-relative U and V wind components
    """
    # Read in the appropriate file depending on the data resolution
    # This file contains sinalpha and cosalpha for the WRF grid
    wrf_angles_ds = xr.open_zarr(
        "s3://cadcat/tmp/era/wrf/wrf_angles_{}.zarr/".format(gridlabel),
        storage_options={"anon": True},
    )
    wrf_angles_ds = wrf_angles_ds.sel(x=u.x, y=u.y, method="nearest")
    sinalpha = wrf_angles_ds.SINALPHA
    cosalpha = wrf_angles_ds.COSALPHA

    # Wind components
    Uearth = u * cosalpha - v * sinalpha
    Vearth = v * cosalpha + u * sinalpha

    # Add variable name
    Uearth.name = "u"
    Vearth.name = "v"

    return Uearth, Vearth


def compute_geostrophic_wind(geopotential_height: xr.DataArray) -> tuple[xr.DataArray]:
    """Calculate the geostrophic wind at a single point on a constant pressure surface.

    Currently only implemented for data on the WRF grid. This code follows the
    MetPy code for calculating the geostrophic wind on an unevenly spaced grid.

    Parameters
    ----------
    geopotential_height : xr.DataArray
        Geopotential height in meters on WRF grid. May include multiple pressure levels

    Returns
    -------
    tuple[xr.DataArray]
        Earth-relative U and V components of the geostrophic wind.

    References
    ----------
    Hess, S. L., 1979: Introduction to Theoretical Meteorology. Robert E. Krieger Publishing Company, 362 pp.
    MetPy, 2026: geostrophic_wind. Accessed 10 Feb 2026, https://unidata.github.io/MetPy/latest/api/generated/metpy.calc.geostrophic_wind.html#geostrophic-wind.
    """
    lat_to_radian = geopotential_height.lat.data * np.pi / 180
    omega = 7292115e-11  # rad/s
    g = 9.81  # m/s2
    f = 2 * omega * np.sin(lat_to_radian)
    norm_factor = g / f

    dhdx, dhdy = _get_spatial_derivatives(geopotential_height)

    # These components are u and v on the WRF grid
    geo_u, geo_v = -norm_factor * dhdy, norm_factor * dhdx

    # Rotate these components to an earth-relative E/W orientation
    geo_u_earth, geo_v_earth = _get_rotated_geostrophic_wind(geo_u, geo_v, "d01")

    # Update attributes for results
    geo_u_earth.name = "u"
    geo_u_earth.attrs["long_name"] = "Geostrophic Wind U Component"
    geo_v_earth.name = "v"
    geo_v_earth.attrs["long_name"] = "Geostrophic Wind V Component"

    return geo_u_earth, geo_v_earth
