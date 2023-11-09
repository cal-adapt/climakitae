"""Miscellaneous utility functions."""

import os
import numpy as np
import datetime
import xarray as xr
import pyproj
import rioxarray as rio
import pandas as pd
import matplotlib.colors as mcolors
import matplotlib
import copy
from timezonefinder import TimezoneFinder

from climakitae.core.paths import (
    ae_orange,
    ae_diverging,
    ae_blue,
    ae_diverging_r,
    categorical_cb,
)


def scenario_to_experiment_id(scenario, reverse=False):
    """
    Convert scenario format to experiment_id format matching catalog names.
    Set reverse=True to get scenario format from input experiement_id.
    """
    scenario_dict = {
        "Historical Reconstruction": "reanalysis",
        "Historical Climate": "historical",
        "SSP 2-4.5 -- Middle of the Road": "ssp245",
        "SSP 5-8.5 -- Burn it All": "ssp585",
        "SSP 3-7.0 -- Business as Usual": "ssp370",
    }

    if reverse == True:
        scenario_dict = {v: k for k, v in scenario_dict.items()}
    return scenario_dict[scenario]


def area_average(dset):
    """Weighted area-average

    Parameters
    ----------
    dset: xr.Dataset
        one dataset from the catalog

    Returns
    ----------
    xr.Dataset
        sub-setted output data

    """
    weights = np.cos(np.deg2rad(dset.lat))
    if set(["x", "y"]).issubset(set(dset.dims)):
        # WRF data has x,y
        dset = dset.weighted(weights).mean("x").mean("y")
    elif set(["lat", "lon"]).issubset(set(dset.dims)):
        # LOCA data has lat, lon
        dset = dset.weighted(weights).mean("lat").mean("lon")
    return dset


def read_csv_file(rel_path, index_col=None):
    return pd.read_csv(
        _package_file_path(rel_path),
        index_col=index_col,
    )


def _package_file_path(rel_path):
    return os.path.normpath(os.path.join(os.path.dirname(__file__), "..", rel_path))


def get_closest_gridcell(data, lat, lon, print_coords=True):
    """From input gridded data, get the closest gridcell to a lat, lon coordinate pair.

    This function first transforms the lat,lon coords to the gridded data’s projection.
    Then, it uses xarray’s built in method .sel to get the nearest gridcell.

    Parameters
    -----------
    data: xr.DataArray or xr.Dataset
        Gridded data
    lat: float
        Latitude of coordinate pair
    lon: float
        Longitude of coordinate pair
    print_coords: bool, optional
        Print closest coorindates?
        Default to True. Set to False for backend use.

    Returns
    --------
    xr.DataArray
        Grid cell closest to input lat,lon coordinate pair

    See also
    --------
    xarray.DataArray.sel
    """
    # Make Transformer object
    lat_lon_to_model_projection = pyproj.Transformer.from_crs(
        crs_from="epsg:4326",  # Lat/lon
        crs_to=data.rio.crs,  # Model projection
        always_xy=True,
    )

    # Convert coordinates to x,y
    x, y = lat_lon_to_model_projection.transform(lon, lat)

    # Get closest gridcell
    closest_gridcell = data.sel(x=x, y=y, method="nearest")

    # Output information
    if print_coords:
        print(
            "Input coordinates: (%.2f, %.2f)" % (lat, lon)
            + "\nNearest grid cell coordinates: (%.2f, %.2f)"
            % (closest_gridcell.lat.values.item(), closest_gridcell.lon.values.item())
        )
    return closest_gridcell


def julianDay_to_str_date(julday, leap_year=True, str_format="%b-%d"):
    """Convert julian day of year to string format
    i.e. if str_format = "%b-%d", the output will be Mon-Day ("Jan-01")

    Parameters
    -----------
    julday: int
        Julian day
    leap_year: boolean
        leap year? (default to True)
    str_format: str
        string format of output date

    Returns
    --------
    date: str
        Julian day in the input format month-day (i.e. "Jan-01")
    """
    if leap_year:
        year = "2024"
    else:
        year = "2023"
    date = datetime.datetime.strptime(year + "." + str(julday), "%Y.%j").strftime(
        str_format
    )
    return date


def readable_bytes(B):
    """
    Return the given bytes as a human friendly KB, MB, GB, or TB string.
    Code from stackoverflow: https://stackoverflow.com/questions/12523586/python-format-size-application-converting-b-to-kb-mb-gb-tb
    """
    B = float(B)
    KB = float(1024)
    MB = float(KB**2)  # 1,048,576
    GB = float(KB**3)  # 1,073,741,824
    TB = float(KB**4)  # 1,099,511,627,776

    if B < KB:
        return "{0} {1}".format(B, "bytes")
    elif KB <= B < MB:
        return "{0:.2f} KB".format(B / KB)
    elif MB <= B < GB:
        return "{0:.2f} MB".format(B / MB)
    elif GB <= B < TB:
        return "{0:.2f} GB".format(B / GB)
    elif TB <= B:
        return "{0:.2f} TB".format(B / TB)


def read_ae_colormap(cmap="ae_orange", cmap_hex=False):
    """Read in AE colormap by name

    Parameters
    -----------
    cmap: str
        one of ["ae_orange","ae_blue","ae_diverging"]
    cmap_hex: boolean
        return RGB or hex colors?

    Returns
    --------
    one of either

    cmap_data: matplotlib.colors.LinearSegmentedColormap
        used for matplotlib (if cmap_hex == False)
    cmap_data: list
        used for hvplot maps (if cmap_hex == True)

    """

    if cmap == "ae_orange":
        cmap_data = ae_orange
    elif cmap == "ae_diverging":
        cmap_data = ae_diverging
    elif cmap == "ae_blue":
        cmap_data = ae_blue
    elif cmap == "ae_diverging_r":
        cmap_data = ae_diverging_r
    elif cmap == "categorical_cb":
        cmap_data = categorical_cb

    # Load text file
    cmap_np = np.loadtxt(_package_file_path(cmap_data), dtype=float)

    # RBG to hex
    if cmap_hex:
        cmap_data = [matplotlib.colors.rgb2hex(color) for color in cmap_np]
    else:
        cmap_data = mcolors.LinearSegmentedColormap.from_list(cmap, cmap_np, N=256)
    return cmap_data


def reproject_data(xr_da, proj="EPSG:4326", fill_value=np.nan):
    """Reproject xr.DataArray using rioxarray.

    Parameters
    -----------
    xr_da: xr.DataArray
        2-or-3-dimensional DataArray, with 2 spatial dimensions
    proj: str
        proj to use for reprojection (default to "EPSG:4326"-- lat/lon coords)
    fill_value: float
        fill value (default to np.nan)

    Returns
    --------
    data_reprojected: xr.DataArray
        2-or-3-dimensional reprojected DataArray

    Raises
    ------
    ValueError
        if input data does not have spatial coords x,y
    ValueError
        if input data has more than 5 dimensions

    """

    def _reproject_data_4D(data, reproject_dim, proj="EPSG:4326", fill_value=np.nan):
        """Reproject 4D xr.DataArray across an input dimension

        Parameters
        -----------
        data: xr.DataArray
            4-dimensional DataArray, with 2 spatial dimensions
        reproject_dim: str
            name of dimensions to use
        proj: str
            proj to use for reprojection (default to "EPSG:4326"-- lat/lon coords)
        fill_value: float
            fill value (default to np.nan)

        Returns
        --------
        data_reprojected: xr.DataArray
            4-dimensional reprojected DataArray

        """
        rp_list = []
        for i in range(len(data[reproject_dim])):
            dp_i = data[i].rio.reproject(
                proj, nodata=fill_value
            )  # Reproject each index in that dimension
            rp_list.append(dp_i)
        data_reprojected = xr.concat(
            rp_list, dim=reproject_dim
        )  # Concat along reprojection dim to get entire dataset reprojected
        return data_reprojected

    def _reproject_data_5D(data, reproject_dim, proj="EPSG:4326", fill_value=np.nan):
        """Reproject 5D xr.DataArray across two input dimensions

        Parameters
        -----------
        data: xr.DataArray
            5-dimensional DataArray, with 2 spatial dimensions
        reproject_dim: list
            list of str dimension names to use
        proj: str
            proj to use for reprojection (default to "EPSG:4326"-- lat/lon coords)
        fill_value: float
            fill value (default to np.nan)

        Returns
        --------
        data_reprojected: xr.DataArray
            5-dimensional reprojected DataArray

        """
        rp_list_j = []
        reproject_dim_j = reproject_dim[0]
        for j in range(len(data[reproject_dim_j])):
            rp_list_i = []
            reproject_dim_i = reproject_dim[1]
            for i in range(len(data[reproject_dim_i])):
                dp_i = data[j, i].rio.reproject(
                    proj, nodata=fill_value
                )  # Reproject each index in that dimension
                rp_list_i.append(dp_i)
            data_reprojected_i = xr.concat(
                rp_list_i, dim=reproject_dim_i
            )  # Concat along reprojection dim to get entire dataset reprojected
            rp_list_j.append(data_reprojected_i)
        data_reprojected = xr.concat(rp_list_j, dim=reproject_dim_j)
        return data_reprojected

    # Raise error if data doesn't have spatial dimensions x,y
    if not set(["x", "y"]).issubset(xr_da.dims):
        raise ValueError(
            (
                "Input DataArray cannot be reprojected because it"
                " does not contain spatial dimensions x,y"
            )
        )

    # Drop non-dimension coords. Will cause error with rioxarray
    coords = [coord for coord in xr_da.coords if coord not in xr_da.dims]
    data = xr_da.drop_vars(coords)

    # Re-write crs to data using original dataset
    data = data.rio.write_crs(xr_da.rio.crs)

    # Get non-spatial dimensions
    non_spatial_dims = [dim for dim in data.dims if dim not in ["x", "y"]]

    # 2 or 3D DataArray
    if len(data.dims) <= 3:
        data_reprojected = data.rio.reproject(proj, nodata=fill_value)
    # 4D DataArray
    elif len(data.dims) == 4:
        data_reprojected = _reproject_data_4D(
            data=data,
            reproject_dim=non_spatial_dims[0],
            proj=proj,
            fill_value=fill_value,
        )
    # 5D DataArray
    elif len(data.dims) == 5:
        data_reprojected = _reproject_data_5D(
            data=data,
            reproject_dim=non_spatial_dims[:-1],
            proj=proj,
            fill_value=fill_value,
        )
    else:
        raise ValueError(
            ("DataArrays with dimensions greater" " than 5 are not currently supported")
        )

    # Reassign attribute to reflect reprojection
    data_reprojected.attrs["grid_mapping"] = proj
    return data_reprojected


## DFU notebook-specific functions, flexible for all notebooks
def compute_annual_aggreggate(data, name, num_grid_cells):
    """Calculates the annual sum of HDD and CDD"""
    annual_ag = data.squeeze().groupby("time.year").sum(["time"])  # Aggregate annually
    annual_ag = annual_ag / num_grid_cells  # Divide by number of gridcells
    annual_ag.name = name  # Give new name to dataset
    return annual_ag


def compute_multimodel_stats(data):
    """Calculates model mean, min, max, median across simulations"""
    # Compute mean across simulation dimensions and add is as a coordinate
    sim_mean = (
        data.mean(dim="simulation")
        .assign_coords({"simulation": "simulation mean"})
        .expand_dims("simulation")
    )

    # Compute multimodel min
    sim_min = (
        data.min(dim="simulation")
        .assign_coords({"simulation": "simulation min"})
        .expand_dims("simulation")
    )

    # Compute multimodel max
    sim_max = (
        data.max(dim="simulation")
        .assign_coords({"simulation": "simulation max"})
        .expand_dims("simulation")
    )

    # Compute multimodel median
    sim_median = (
        data.median(dim="simulation")
        .assign_coords({"simulation": "simulation median"})
        .expand_dims("simulation")
    )

    # Add to main dataset
    stats_concat = xr.concat(
        [data, sim_mean, sim_min, sim_max, sim_median], dim="simulation"
    )
    return stats_concat


def trendline(data, kind="mean"):
    """Calculates treadline of the multi-model mean or median.

    Parameters
    ----------
    data: xr.Dataset
    kind: str (optional)
        Options are 'mean' and 'median'

    Returns
    -------
    trendline: xr.Dataset

    Note
    ----
    1. Development note: If an additional option to trendline 'kind' is required,
    compute_multimodel_stats must be modified to update optionality.
    """
    if kind == "mean":
        if "simulation mean" not in data.simulation:
            raise Exception(
                "Invalid data provdied, please pass the multimodel stats from compute_multimodel_stats"
            )

        data_sim_mean = data.sel(simulation="simulation mean")
        m, b = data_sim_mean.polyfit(dim="year", deg=1).polyfit_coefficients.values
        trendline = m * data_sim_mean.year + b  # y = mx + b

    elif kind == "median":
        if "simulation median" not in data.simulation:
            raise Exception(
                "Invalid data provided, please pass the multimodel stats from compute_multimodel_stats"
            )

        data_sim_med = data.sel(simulation="simulation median")
        m, b = data_sim_med.polyfit(dim="year", deg=1).polyfit_coefficients.values
        trendline = m * data_sim_med.year + b  # y = mx + b
    trendline.name = "trendline"
    return trendline


def combine_hdd_cdd(data):
    """Drops specific unneeded coords from HDD/CDD data, independent of station or gridded data source"""
    if data.name not in [
        "Annual Heating Degree Days (HDD)",
        "Annual Cooling Degree Days (CDD)",
        "Heating Degree Hours",
        "Cooling Degree Hours",
    ]:
        raise Exception(
            "Invalid data provided, please pass cooling/heating degree data"
        )

    to_drop = ["scenario", "Lambert_Conformal", "variable"]
    for coord in to_drop:
        if coord in data.coords:
            data = data.drop(coord)

    return data


## DFU plotting functions
def hdd_cdd_lineplot(annual_data, trendline, title="title"):
    """Plots annual CDD/HDD with trendline provided"""
    return annual_data.hvplot.line(
        x="year",
        by="simulation",
        width=800,
        height=350,
        title=title,
        yformatter="%.0f",  # Remove scientific notation
    ) * trendline.hvplot.line(  # Add trendline
        x="year", color="black", line_dash="dashed", label="trendline"
    )


def hdh_cdh_lineplot(data):
    """Plots HDH/CDH"""
    return data.hvplot.line(
        x="time", by="simulation", title=data.name, ylabel=data.name + " (degF)"
    )


## Heat Index summary table helper
def summary_table(data):
    """Helper function to organize dataset object into a pandas dataframe for ease.

    Parameters
    ----------
    data: xr.Dataset

    Returns
    -------
    df: pd.DataFrame
        df is organized so that the simulations are stacked in individual columns by year/time
    """

    # Identify whether the temporal dimension is "time" or "year"
    if "time" in data.dims:
        df = data.drop(
            ["lakemask", "landmask", "lat", "lon", "Lambert_Conformal", "x", "y"]
        ).to_dataframe(dim_order=["time", "scenario", "simulation"])

        df = df.unstack().unstack()
        df = df.sort_values(by=["time"])

    elif "year" in data.dims:
        df = data.drop(
            ["lakemask", "landmask", "lat", "lon", "Lambert_Conformal", "x", "y"]
        ).to_dataframe(dim_order=["year", "scenario", "simulation"])

        df = df.unstack().unstack()
        df = df.sort_values(by=["year"])

    return df


def convert_to_local_time(data, selections, lat, lng) -> xr.Dataset:
    """
    Converts the inputted data to the local time of the selection.
    """
    # 1. Find the other data
    start, end = selections.time_slice
    tz_selections = copy.copy(selections)
    tz_selections.time_slice = (
        end + 1,
        end + 1,
    )  # This is assuming selections passed with be negative UTC time. Also to get the next year of data.

    tz_data = tz_selections.retrieve()

    # 2. Combine the data
    total_data = xr.concat([data, tz_data], dim="time")

    # 3. Change datetime objects to local time
    tf = TimezoneFinder()
    local_tz = tf.timezone_at(lng=lng, lat=lat)
    new_time = (
        pd.DatetimeIndex(total_data.time)
        .tz_localize("UTC")
        .tz_convert(local_tz)
        .tz_localize(None)
        .astype("datetime64[ns]")
    )
    total_data["time"] = new_time

    # 4. Subset the data by the initial time
    start = data.time[0]
    end = data.time[-1]
    sliced_data = total_data.sel(time=slice(start, end))

    print("Data converted to local timezone!")

    return sliced_data
