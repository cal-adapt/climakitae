import copy
import datetime
import os
from typing import Iterable, Union

import geopandas as gpd
import intake_esm
import numpy as np
import pandas as pd
import pyproj
import rioxarray as rio
from shapely.geometry import mapping
from typing import Any
import xarray as xr
import intake
from timezonefinder import TimezoneFinder

from climakitae.core.constants import SSPS, UNSET, WARMING_LEVELS

# from climakitae.core.data_interface import DataParameters
from climakitae.core.paths import (
    data_catalog_url,
    stations_csv_path,
)


def downscaling_method_as_list(downscaling_method: str) -> list[str]:
    """Function to convert string based radio button values to python list.

    Parameters
    ----------
    downscaling_method: str
        one of "Dynamical", "Statistical", or "Dynamical+Statistical"

    Returns
    -------
    method_list: list
        one of ["Dynamical"], ["Statistical"], or ["Dynamical","Statistical"]
    """
    method_list = []
    if downscaling_method == "Dynamical+Statistical":
        method_list = ["Dynamical", "Statistical"]
    else:
        method_list = [downscaling_method]
    return method_list


def area_average(dset: xr.Dataset) -> xr.Dataset:
    """Weighted area-average

    Parameters
    ----------
    dset: xr.Dataset
        one dataset from the catalog

    Returns
    -------
    xr.Dataset
        sub-setted output data
    """
    weights = np.cos(np.deg2rad(dset.lat))
    if set(["x", "y"]).issubset(set(dset.dims)):
        # WRF data has x,y
        dset = dset.weighted(weights).mean(["x", "y"])
    elif set(["lat", "lon"]).issubset(set(dset.dims)):
        # LOCA data has lat, lon
        dset = dset.weighted(weights).mean(["lat", "lon"])
    return dset


def read_csv_file(
    rel_path: str, index_col: str = UNSET, parse_dates: bool = False
) -> pd.DataFrame:
    """Read CSV file into pandas DataFrame

    Parameters
    ----------
    rel_path: str
        path to CSV file relative to this util python file
    index_col: str
        CSV column to index DataFrame on
    parse_dates: boolean
        Whether to have pandas parse the date strings

    Returns
    -------
    pd.DataFrame
    """
    return pd.read_csv(
        _package_file_path(rel_path),
        index_col=None if index_col is UNSET else index_col,
        parse_dates=parse_dates,
        na_values=[
            "",
            "#N/A",
            "#N/A N/A",
            "#NA",
            "-1.#IND",
            "-1.#QNAN",
            "-NaN",
            "-nan",
            "1.#IND",
            "1.#QNAN",
            "<NA>",
            "N/A",
            "NA",
            "NULL",
            "NaN",
            "n/a",
            "nan",
            "null ",
        ],
        keep_default_na=False,
    )


def write_csv_file(df: pd.DataFrame, rel_path: str) -> None:
    """Write CSV file from pandas DataFrame

    Parameters
    ----------
    df: pd.DataFrame
        pandas DataFrame to write out
    rel_path: str
        path to CSV file relative to this util python file

    Returns
    -------
    None
    """
    return df.to_csv(_package_file_path(rel_path))


def _package_file_path(rel_path: str) -> str:
    """Find OS full path name given relative path

    Parameters
    ----------
    rel_path: str
        path to file relative to this util python file

    Returns
    -------
    str
    """
    return os.path.normpath(os.path.join(os.path.dirname(__file__), "..", rel_path))


def get_closest_gridcell(
    data: xr.Dataset | xr.DataArray, lat: float, lon: float, print_coords: bool = True
) -> xr.DataArray | None:
    """From input gridded data, get the closest gridcell to a lat, lon coordinate pair.

    This function first transforms the lat,lon coords to the gridded data’s projection.
    Then, it uses xarray’s built in method .sel to get the nearest gridcell.

    Parameters
    ----------
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
    xr.DataArray or None
        Grid cell closest to input lat,lon coordinate pair

    See also
    --------
    xr.DataArray.sel
    """

    # Use data cellsize as tolerance for selecting nearest
    # Using this method to guard against single row|col
    # Assumes data is from climakitae retrieve
    km_num = int(data.resolution.split(" km")[0])
    # tolerance = int(data.resolution.split(" km")[0]) * 1000

    if "x" in data.dims and "y" in data.dims:
        # Make Transformer object
        lat_lon_to_model_projection = pyproj.Transformer.from_crs(
            crs_from="epsg:4326",  # Lat/lon
            crs_to=data.rio.crs,  # Model projection
            always_xy=True,
        )

        # Convert coordinates to x,y
        x, y = lat_lon_to_model_projection.transform(lon, lat)

    # Get closest gridcell using tolerance
    # If input point outside of dataset by greater than one
    # grid cell, then None is returned
    try:
        if "x" in data.dims and "y" in data.dims:
            tolerance = km_num * 1000  # Converting km to m
            closest_gridcell = data.sel(x=x, y=y, method="nearest", tolerance=tolerance)
        elif "lat" in data.dims and "lon" in data.dims:
            tolerance = km_num / 111  # Rough translation of km to degrees
            closest_gridcell = data.sel(
                lat=lat, lon=lon, method="nearest", tolerance=tolerance
            )

    except KeyError:
        print(
            f"Input coordinates: ({lat:.2f}, {lon:.2f}) OUTSIDE of data extent by more than one cell. Returning None"
        )
        return None

    # Output information
    if print_coords:
        print(
            "Input coordinates: (%.2f, %.2f)" % (lat, lon)
            + "\nNearest grid cell coordinates: (%.2f, %.2f)"
            % (closest_gridcell.lat.values.item(), closest_gridcell.lon.values.item())
        )
    return closest_gridcell


def get_closest_gridcells(
    data: xr.Dataset, lats: Iterable[float] | float, lons: Iterable[float] | float
) -> xr.Dataset | None:
    """
    Find the nearest grid cell(s) for given latitude and longitude coordinates.

    If the dataset uses (x, y) coordinates, lat/lon values are transformed to match its projection.
    The function then selects the closest grid cell using `sel()` or `get_indexer()`, ensuring
    the selection is within an appropriate tolerance.

    Parameters
    ----------
    data : xr.DataArray or xr.Dataset
        Gridded dataset with (x, y) or (lat, lon) dimensions.
    lats : float or array-like
        Latitude coordinate(s).
    lons : float or array-like
        Longitude coordinate(s).

    Returns
    -------
    xr.Dataset | xr.DataArray | None
        Nearest grid cell(s) or `None` if no valid match is found.

    Notes
    -----
    - If (x, y) dimensions exist, lat/lon coordinates are projected using `pyproj.Transformer`.
    - The search tolerance is derived from the dataset resolution.
    - Returns `None` if no grid cells are within tolerance.

    See Also
    --------
    xr.DataArray.sel, pyproj.Transformer
    """
    # Use data cellsize as tolerance for selecting nearest
    # Using this method to guard against single row|col
    # Assumes data is from climakitae retrieve
    km_num = int(data.resolution.split(" km")[0])
    # tolerance = int(data.resolution.split(" km")[0]) * 1000

    if "x" and "y" in data.dims:
        # Make Transformer object
        lat_lon_to_model_projection = pyproj.Transformer.from_crs(
            crs_from="epsg:4326",  # Lat/lon
            crs_to=data.rio.crs,  # Model projection
            always_xy=True,
        )

        # Convert coordinates to x,y
        xs, ys = lat_lon_to_model_projection.transform(lons, lats)

    # Get closest gridcell using tolerance
    def find_closest_valid_gridcells(
        data: xr.DataArray | xr.Dataset,
        dim1_name: str,
        dim2_name: str,
        coords1: float | Iterable[float],
        coords2: float | Iterable[float],
        tolerance: float,
    ) -> xr.Dataset | xr.DataArray | None:
        """
        Find the nearest valid grid cells within a given tolerance.

        Uses `get_indexer()` to find the closest grid cell indices along two spatial dimensions,
        ensuring they are within the dataset bounds and tolerance.

        Parameters
        ----------
        data : xr.DataArray or xr.Dataset
            Gridded dataset with spatial dimensions.
        dim1_name : str
            First spatial dimension (e.g., 'x' or 'lat').
        dim2_name : str
            Second spatial dimension (e.g., 'y' or 'lon').
        coords1 : float or array-like
            Coordinates along `dim1_name`.
        coords2 : float or array-like
            Coordinates along `dim2_name`.
        tolerance : float
            Maximum allowed distance from the nearest grid cell.

        Returns
        -------
        xr.DataArray or None
            Nearest grid cell(s) or `None` if out of bounds.

        See Also
        --------
        xr.DataArray.get_indexer, xr.DataArray.isel
        """
        dim1_idx = data[dim1_name].to_index().get_indexer(coords1, method="nearest")
        dim2_idx = data[dim2_name].to_index().get_indexer(coords2, method="nearest")

        dim1_valid = (dim1_idx != -1) & (
            np.abs(data[dim1_name][dim1_idx] - coords1) <= tolerance
        )
        dim2_valid = (dim2_idx != -1) & (
            np.abs(data[dim2_name][dim2_idx] - coords2) <= tolerance
        )

        if not (dim1_valid.all() and dim2_valid.all()):
            print(
                "One or more coordinates are OUTSIDE of data extent by more than one cell. Returning None."
            )
            closest_gridcells = None
        else:
            closest_gridcells = data.isel(
                {
                    dim1_name: xr.DataArray(dim1_idx, dims="points"),
                    dim2_name: xr.DataArray(dim2_idx, dims="points"),
                }
            )

        return closest_gridcells

    # If input point outside of dataset by greater than one
    # grid cell, then None is returned
    if "x" and "y" in data.dims:
        tolerance = km_num * 1000  # Converting km to m
        closest_gridcells = find_closest_valid_gridcells(
            data, "x", "y", xs, ys, tolerance
        )

    elif "lat" and "lon" in data.dims:
        tolerance = km_num / 111  # Rough translation of km to degrees
        closest_gridcells = find_closest_valid_gridcells(
            data, "lat", "lon", lats, lons, tolerance
        )

    return closest_gridcells


def julianDay_to_date(
    julday: int, year: int = None, return_type: str = "str", str_format: str = "%b-%d"
) -> Union[str, datetime.datetime, datetime.date]:
    """Convert julian day of year to a date object or formatted string.

    Parameters
    ----------
    julday: int
        Julian day (day of year)
    year: int, optional
        Year to use. If None, uses current year or a leap year (2024) based on needs.
        Default is None.
    return_type: str, optional
        Type of return value:
        - "str": formatted string (default)
        - "datetime": datetime object
        - "date": date object
    str_format: str, optional
        String format of output date when return_type is "str".
        Default is "%b-%d" which outputs format like "Jan-01".

    Returns
    -------
    date: str, datetime.datetime, or datetime.date
        Julian day converted to specified format or object

    Examples
    --------
    >>> julianDay_to_date(1)
    'Jan-01'
    >>> julianDay_to_date(32, year=2023, return_type="date")
    datetime.date(2023, 2, 1)
    >>> julianDay_to_date(60, year=2024, str_format="%Y-%m-%d")
    '2024-02-29'
    """
    # Determine which year to use
    if year is None:
        year = datetime.datetime.now().year

    # Create datetime object from julian day
    date_obj = datetime.datetime.strptime(f"{year}.{julday}", "%Y.%j")

    # Return appropriate type
    match return_type:
        case "str":
            return date_obj.strftime(str_format)
        case "datetime":
            return date_obj
        case "date":
            return date_obj.date()
        case _:
            raise ValueError("return_type must be 'str', 'datetime', or 'date'")


def readable_bytes(b: int) -> str:
    """Return the given bytes as a human friendly KB, MB, GB, or TB string.

    Parameters
    ----------
    B: byte

    Returns
    -------
    str

    Code from stackoverflow: https://stackoverflow.com/questions/12523586/python-format-size-application-converting-b-to-kb-mb-gb-tb
    """
    b = float(b)
    kb = 1024
    mb = kb**2  # 1,048,576
    gb = kb**3  # 1,073,741,824
    tb = kb**4  # 1,099,511,627,776

    match b:
        case _ if b < kb:
            return f"{b} bytes"
        case _ if kb <= b < mb:
            return f"{b / kb:.2f} KB"
        case _ if mb <= b < gb:
            return f"{b / mb:.2f} MB"
        case _ if gb <= b < tb:
            return f"{b / gb:.2f} GB"
        case _ if tb <= b:
            return f"{b / tb:.2f} TB"


def reproject_data(
    xr_da: xr.DataArray, proj: str = "EPSG:4326", fill_value: float = np.nan
) -> xr.DataArray:
    """Reproject xr.DataArray using rioxarray.

    Parameters
    ----------
    xr_da: xr.DataArray
        2-or-3-dimensional DataArray, with 2 spatial dimensions
    proj: str
        proj to use for reprojection (default to "EPSG:4326"-- lat/lon coords)
    fill_value: float
        fill value (default to np.nan)

    Returns
    -------
    data_reprojected: xr.DataArray
        2-or-3-dimensional reprojected DataArray

    Raises
    ------
    ValueError
        if input data does not have spatial coords x,y
    ValueError
        if input data has more than 5 dimensions
    """

    def _reproject_data_4D(
        data: xr.DataArray,
        reproject_dim: str,
        proj: str = "EPSG:4326",
        fill_value: float = np.nan,
    ) -> xr.DataArray:
        """Reproject 4D xr.DataArray across an input dimension

        Parameters
        ----------
        data: xr.DataArray
            4-dimensional DataArray, with 2 spatial dimensions
        reproject_dim: str
            name of dimensions to use
        proj: str
            proj to use for reprojection (default to "EPSG:4326"-- lat/lon coords)
        fill_value: float
            fill value (default to np.nan)

        Returns
        -------
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

    def _reproject_data_5D(
        data: xr.DataArray,
        reproject_dim: list[str],
        proj: str = "EPSG:4326",
        fill_value: float = np.nan,
    ) -> xr.DataArray:
        """Reproject 5D xr.DataArray across two input dimensions

        Parameters
        ----------
        data: xr.DataArray
            5-dimensional DataArray, with 2 spatial dimensions
        reproject_dim: list
            list of str dimension names to use
        proj: str
            proj to use for reprojection (default to "EPSG:4326"-- lat/lon coords)
        fill_value: float
            fill value (default to np.nan)

        Returns
        -------
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
    data.rio.write_crs(xr_da.rio.crs, inplace=True)

    # Get non-spatial dimensions
    non_spatial_dims = [dim for dim in data.dims if dim not in ["x", "y"]]

    # test for different dims
    numofdims = len(data.dims)
    # 2 or 3D DataArray
    match numofdims:
        case numofdims if numofdims <= 3:
            data_reprojected = data.rio.reproject(proj, nodata=fill_value)
        # 4D DataArray
        case 4:
            data_reprojected = _reproject_data_4D(
                data=data,
                reproject_dim=non_spatial_dims[0],
                proj=proj,
                fill_value=fill_value,
            )
        # 5D DataArray
        case 5:
            data_reprojected = _reproject_data_5D(
                data=data,
                reproject_dim=non_spatial_dims[:-1],
                proj=proj,
                fill_value=fill_value,
            )
        case _:
            raise ValueError(
                "DataArrays with dimensions greater than 5 are not currently supported"
            )

    # Reassign attribute to reflect reprojection
    data_reprojected.attrs["grid_mapping"] = proj
    return data_reprojected


## DFU notebook-specific functions, flexible for all notebooks
def compute_annual_aggreggate(
    data: xr.DataArray, name: str, num_grid_cells: int
) -> xr.DataArray:
    """Calculates the annual sum of HDD and CDD

    Parameters
    ----------
    data: xr.DataArray
    name: str
    num_grid_cells: int

    Returns
    -------
    annual_ag: xr.DataArray
    """
    annual_ag = data.squeeze().groupby("time.year").sum(["time"])  # Aggregate annually
    annual_ag = annual_ag / num_grid_cells  # Divide by number of gridcells
    annual_ag.name = name  # Give new name to dataset
    return annual_ag


def compute_multimodel_stats(data: xr.DataArray) -> xr.DataArray:
    """Calculates model mean, min, max, median across simulations

    Parameters
    ----------
    data: xr.DataArray

    Returns
    -------
    stats_concat: xr.DataArray
    """
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


def trendline(data: xr.Dataset, kind: str = "mean") -> xr.Dataset:
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
    ret_trendline = xr.Dataset()
    match kind:
        case "mean":
            if "simulation mean" not in data.simulation:
                raise ValueError(
                    "Invalid data provided, please pass the multimodel stats from compute_multimodel_stats"
                )

            data_sim_mean = data.sel(simulation="simulation mean")
            m, b = data_sim_mean.polyfit(dim="year", deg=1).polyfit_coefficients.values
            ret_trendline = m * data_sim_mean.year + b  # y = mx + b

        case "median":
            if "simulation median" not in data.simulation:
                raise ValueError(
                    "Invalid data provided, please pass the multimodel stats from compute_multimodel_stats"
                )

            data_sim_med = data.sel(simulation="simulation median")
            m, b = data_sim_med.polyfit(dim="year", deg=1).polyfit_coefficients.values
            ret_trendline = m * data_sim_med.year + b  # y = mx + b
        case _:
            raise ValueError(
                "Invalid kind provided, please pass either 'mean' or 'median' as the kind"
            )
    ret_trendline.name = "trendline"
    return ret_trendline


def combine_hdd_cdd(data: xr.DataArray) -> xr.DataArray:
    """Drops specific unneeded coords from HDD/CDD data, independent of station or gridded data source

    Parameters
    ----------
    data: xr.DataArray

    Returns
    -------
    data: xr.DataArray
    """
    if data.name not in [
        "Annual Heating Degree Days (HDD)",
        "Annual Cooling Degree Days (CDD)",
        "Heating Degree Hours",
        "Cooling Degree Hours",
    ]:
        raise ValueError(
            "Invalid data provided, please pass cooling/heating degree data"
        )

    to_drop = ["scenario", "Lambert_Conformal", "variable"]
    for coord in to_drop:
        if coord in data.coords:
            data = data.drop_vars(coord)

    return data


def summary_table(data: xr.Dataset) -> pd.DataFrame:
    """
    Helper function to organize dataset object into a pandas dataframe for ease.

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
        df = data.drop_vars(
            ["lakemask", "landmask", "lat", "lon", "Lambert_Conformal", "x", "y"]
        ).to_dataframe(dim_order=["time", "scenario", "simulation"])

        df = df.unstack().unstack()
        df = df.sort_values(by=["time"])

    elif "year" in data.dims:
        df = data.drop_vars(
            ["lakemask", "landmask", "lat", "lon", "Lambert_Conformal", "x", "y"]
        ).to_dataframe(dim_order=["year", "scenario", "simulation"])

        df = df.unstack().unstack()
        df = df.sort_values(by=["year"])

    return df


def convert_to_local_time(data: xr.DataArray, selections) -> xr.DataArray:
    """
    Convert time dimension from UTC to local time for the grid or station.

    Args:
        data (xarray.DataArray): Input data.
        selections: DataParameters object containing selection details.

    Returns:
        xarray.DataArray: Data with converted time coordinate.
    """

    # If timescale is not hourly, no need to convert
    if selections.timescale in ["monthly", "daily"]:
        print(
            "You've selected a timescale that doesn't require any timezone shifting, due to its timescale not being granular enough (hourly). Please pass in more granular level data if you want to adjust its local timezone."
        )
        return data

    # 1. Get the time slice from selections
    start, end = selections.time_slice

    # Default lat/lon values in case other methods fail
    lat = None
    lon = None

    # Get latitude/longitude information
    if selections.data_type == "Stations":
        # Read stations database
        stations_df = read_csv_file(stations_csv_path)
        stations_df = stations_df.drop(columns=["Unnamed: 0"])

        # Filter by selected station(s) - assume first station if multiple
        selected_station = selections.stations[0]
        station_data = stations_df[stations_df["station"] == selected_station]
        lat = station_data["LAT_Y"].values[0]
        lon = station_data["LON_X"].values[0]

    elif selections.area_average == "Yes":
        # For area average, use the mean lat/lon
        lat = np.mean(selections.latitude)
        lon = np.mean(selections.longitude)

    elif selections.data_type == "Gridded" and selections.area_subset == "lat/lon":
        # Finding avg. lat/lon coordinates from all grid-cells
        lat = data.lat.mean().item()
        lon = data.lon.mean().item()

    elif selections.data_type == "Gridded" and selections.area_subset != "none":
        # Find the avg. lat/lon coordinates from entire geometry within an area subset
        boundaries = selections._geographies

        # Making mapping for different geographies to different polygons
        mapping = {
            "CA counties": (
                boundaries._ca_counties,
                boundaries._get_ca_counties(),
            ),
            "CA Electric Balancing Authority Areas": (
                boundaries._ca_electric_balancing_areas,
                boundaries._get_electric_balancing_areas(),
            ),
            "CA Electricity Demand Forecast Zones": (
                boundaries._ca_forecast_zones,
                boundaries._get_forecast_zones(),
            ),
            "CA Electric Load Serving Entities (IOU & POU)": (
                boundaries._ca_utilities,
                boundaries._get_ious_pous(),
            ),
            "CA watersheds": (
                boundaries._ca_watersheds,
                boundaries._get_ca_watersheds(),
            ),
        }

        # Finding the center point of the gridded WRF area
        center_pt = (
            mapping[selections.area_subset][0]
            .loc[mapping[selections.area_subset][1][selections.cached_area[0]]]
            .geometry.centroid
        )
        lat = center_pt.y
        lon = center_pt.x

    # Check if we were able to get valid coordinates
    if lat is None or lon is None:
        # Default to a reasonable timezone (UTC)
        local_tz = "UTC"
        print("Could not determine location coordinates, defaulting to UTC timezone.")
    else:
        # Find timezone for the coordinates
        tf = TimezoneFinder()
        local_tz = tf.timezone_at(lng=lon, lat=lat)

    # Condition if timezone adjusting is happening at the end of `Historical Reconstruction`
    if selections.scenario_historical == ["Historical Reconstruction"] and end == 2022:
        print(
            "Adjusting timestep but not appending data, as there is no more ERA5 data after 2022."
        )
        total_data = data

    # Condition if selected data is at the end of possible data time interval
    elif end < 2100:
        # Use selections object to retrieve new data for timezone shifting
        tz_selections = copy.copy(selections)
        tz_selections.time_slice = (end + 1, end + 1)
        tz_data = tz_selections.retrieve()

        if tz_data.time.size == 0:
            print(
                "You've selected a time slice that will additionally require a selected SSP. Please select an SSP in your selections and re-run this function."
            )
            return data

        # Combine the data
        total_data = xr.concat([data, tz_data], dim="time")

    else:  # 2100 or any years greater that the user has input
        print(
            "Adjusting timestep but not appending data, as there is no more data after 2100."
        )
        total_data = data

    # Change datetime objects to local time
    new_time = (
        pd.DatetimeIndex(total_data.time)
        .tz_localize("UTC")
        .tz_convert(local_tz)
        .tz_localize(None)
        .astype("datetime64[ns]")
    )
    total_data["time"] = new_time

    # Subset the data by the initial time
    start_slice = data.time[0]
    end_slice = data.time[-1]
    sliced_data = total_data.sel(time=slice(start_slice, end_slice))

    print(f"Data converted to {local_tz} timezone.")

    # Add timezone attribute to data
    sliced_data = sliced_data.assign_attrs({"timezone": local_tz})

    # Reset selections object to what it was originally (if we changed it)
    if end < 2100:
        selections.time_slice = (start, end)

    return sliced_data


def add_dummy_time_to_wl(wl_da: xr.DataArray) -> xr.DataArray:
    """
    Replace the `[hours/days/months]_from_center` or `time_delta` dimension in a DataArray returned from WarmingLevels with a dummy time index for calculations with tools that require a `time` dimension.

    Parameters
    ----------
    wl_da : xarray.DataArray
        The input Warming Levels DataArray. It is expected to have a time-based dimension which typically includes "from_center"
        in its name or `time_delta` indicating the time dimension in relation to the year that the given warming level is reached per simulation.

    Returns
    -------
    xarray.DataArray
        A modified version of the input DataArray with the original time dimension replaced by a dummy time series. The new dimension
        will be named "time".

    Notes
    -----
    - The function looks for the dimension name containing "from_center" to identify the time-based dimension.
    - It supports creating dummy time series with frequencies of hours, days, or months, based on the prefix of the dimension name.
    - The dummy time series starts from "2000-01-01".
    """
    # Adjusting the time index into dummy time-series for counting
    # Finding time-based dimension
    wl_time_dim = ""

    for dim in wl_da.dims:
        if dim == "time_delta":
            wl_time_dim = "time_delta"
        elif "from_center" in dim:
            wl_time_dim = dim

    if wl_time_dim == "":
        raise ValueError(
            "DataArray does not contain necessary warming level information."
        )

    # Finding time frequency and
    # Creating map from frequency name to freq var needed for pandas date range
    if wl_time_dim == "time_delta":
        time_freq_name = wl_da.frequency
        name_to_freq = {"hourly": "h", "daily": "D", "monthly": "ME"}
    else:
        time_freq_name = wl_time_dim.split("_")[0]
        name_to_freq = {"hours": "h", "days": "D", "months": "ME"}

    # Creating dummy timestamps
    timestamps = pd.date_range(
        "2000-01-01",
        periods=len(wl_da[wl_time_dim]),
        freq=name_to_freq[time_freq_name],
    )

    # Replacing WL timestamps with dummy timestamps so that calculations from tools like `thresholds_tools`
    # can be computed on a DataArray with a time dimension
    wl_da = wl_da.assign_coords({wl_time_dim: timestamps}).rename({wl_time_dim: "time"})
    return wl_da


def downscaling_method_to_activity_id(
    downscaling_method: str, reverse: bool = False
) -> str:
    """Convert downscaling method to activity id to match catalog names

    Parameters
    -----------
    downscaling_method: str
    reverse: boolean, optional
        Set reverse=True to get downscaling method from input activity_id
        Default to False

    Returns
    --------
    str
    """
    downscaling_dict = {"Dynamical": "WRF", "Statistical": "LOCA2"}

    if reverse:
        downscaling_dict = {v: k for k, v in downscaling_dict.items()}
    return downscaling_dict[downscaling_method]


def resolution_to_gridlabel(resolution: str, reverse: bool = False) -> str:
    """Convert resolution format to grid_label format matching catalog names.

    Parameters
    -----------
    resolution: str
    reverse: boolean, optional
        Set reverse=True to get resolution format from input grid_label.
        Default to False

    Returns
    -------
    str

    """
    res_dict = {"45 km": "d01", "9 km": "d02", "3 km": "d03"}

    if reverse:
        res_dict = {v: k for k, v in res_dict.items()}
    return res_dict[resolution]


def timescale_to_table_id(timescale: str, reverse: bool = False) -> str:
    """Convert resolution format to table_id format matching catalog names.

    Parameters
    ----------
    timescale : str
    reverse : boolean, optional
        Set reverse=True to get resolution format from input table_id.
        Default to False

    Returns
    -------
    str

    """
    # yearly max is not an option in the Selections GUI, but its included here to make parsing through the data easier for the non-GUI data access/view options
    timescale_dict = {
        "monthly": "mon",
        "daily": "day",
        "hourly": "1hr",
        "yearly_max": "yrmax",
    }

    if reverse:
        timescale_dict = {v: k for k, v in timescale_dict.items()}
    return timescale_dict[timescale]


def scenario_to_experiment_id(scenario: str, reverse: bool = False) -> str:
    """
    Convert scenario format to experiment_id format matching catalog names.

    Parameters
    ----------
    scenario: str
    reverse: boolean, optional
        Set reverse=True to get scenario format from input experiement_id.
        Default to False

    Returns
    -------
    str

    """
    scenario_dict = {
        "Historical Reconstruction": "reanalysis",
        "Historical Climate": "historical",
        "SSP 2-4.5": "ssp245",
        "SSP 5-8.5": "ssp585",
        "SSP 3-7.0": "ssp370",
    }

    if reverse:
        scenario_dict = {v: k for k, v in scenario_dict.items()}
    return scenario_dict[scenario]


# cannot import DataParameters due to circular import issue
def _get_cat_subset(
    selections,
) -> intake_esm.source.ESMDataSource:  #! selections: DataParameters
    """For an input set of data selections, get the catalog subset.

    Parameters
    ----------
    selections: DataParameters
        object holding user's selections

    Returns
    -------
    cat_subset: intake_esm.source.ESMDataSource
        catalog subset
    """

    scenario_ssp, scenario_historical = _get_scenario_from_selections(selections)

    scenario_selections = scenario_ssp + scenario_historical

    method_list = downscaling_method_as_list(selections.downscaling_method)

    # If the variable is a derived variable, get the catalog subset for the first variable dependency
    if "_derived" in selections.variable_id[0]:
        var_descrip_df = selections._variable_descriptions
        first_dependency_var_id = (
            var_descrip_df[var_descrip_df["variable_id"] == selections.variable_id[0]][
                "dependencies"
            ]
            .values[0]
            .split(",")[0]
        )
        variable_id = [first_dependency_var_id]
    else:  # Otherwise, just use the variable id
        variable_id = selections.variable_id

    # Get catalog keys
    # Convert user-friendly names to catalog names (i.e. "45 km" to "d01")
    activity_id = [downscaling_method_to_activity_id(dm) for dm in method_list]
    table_id = timescale_to_table_id(selections.timescale)
    grid_label = resolution_to_gridlabel(selections.resolution)
    experiment_id = [scenario_to_experiment_id(x) for x in scenario_selections]
    source_id = selections.simulation

    cat_subset = selections._data_catalog.search(
        activity_id=activity_id,
        table_id=table_id,
        grid_label=grid_label,
        variable_id=variable_id,
        experiment_id=experiment_id,
        source_id=source_id,
    )

    # Get just data that's on the LOCA grid
    # This will include LOCA data and WRF data on the LOCA native grid
    # Both datasets are tagged with UCSD as the institution_id, so we can use "UCSD" to further subset the catalog data
    if "Statistical" in selections.downscaling_method:
        cat_subset = cat_subset.search(institution_id="UCSD")
    # If only dynamical is selected, we need to remove UCSD from the WRF query
    else:
        wrf_on_native_grid = [
            institution
            for institution in selections._data_catalog.df.institution_id.unique()
            if institution != "UCSD"
        ]
        cat_subset = cat_subset.search(institution_id=wrf_on_native_grid)

    return cat_subset


def _get_scenario_from_selections(selections) -> tuple[list[str], list[str]]:
    """Get scenario from DataParameters object
    This needs to be handled differently due to warming levels retrieval method,
    which sets scenario to "n/a" for both historical and ssp.

    Parameters
    ----------
    selections: DataParameters
        object holding user's selections

    Returns
    -------
    scenario_ssp: list of str
    scenario_historical: list of str

    """

    match selections.approach:
        case "Time":
            scenario_ssp = selections.scenario_ssp
            scenario_historical = selections.scenario_historical
        case "Warming Level":
            # Need all scenarios for warming level approach
            scenario_ssp = SSPS
            scenario_historical = ["Historical Climate"]
        case _:
            raise ValueError('approach needs to be either "Time" or "Warming Level"')
    return scenario_ssp, scenario_historical


def stack_sims_across_locs(ds):
    # Renaming gridcell so that it can be concatenated with other lat/lon gridcells
    ds["simulation"] = [
        "{}_{}_{}".format(
            sim_name,
            ds.lat.compute().item(),
            ds.lon.compute().item(),
        )
        for sim_name in ds["simulation"]
    ]
    return ds


def clip_to_shapefile(
    data: xr.Dataset | xr.DataArray,
    shapefile_path: str,
    feature: tuple[str, Any] = (),
    name: str = "user-defined",
    **kwargs,
) -> xr.Dataset | xr.DataArray:
    """Use a shapefile to select an area subset of AE data.

    By default, this function will clip the data to the area covered by all features in
    the shapefile. To clip to specific features, use the `feature` keyword.

    Parameters
    ----------
    data : xr.Dataset | xr.DataArray
        Data to be clipped.
    shapefile_path: str
        Filepath to shapefile. Shapefile must include valid CRS.
    feature: tuple(str, str | int | float | list)
        Tuple containing attribute name and value(s) for target feature(s) (optional).
    name: str
        Location name to record in data attributes if 'feature' parameter is not set (optional).
    **kwargs
        Additional arguments to pass to the rioxarray clip function

    Returns
    -------
    clipped : xr.Dataset | xr.DataArray
        Returns same type as 'data', but grid is clipped to shapefile feature(s).
    """
    if data.rio.crs is None:
        raise RuntimeError(
            "No CRS found on input parameter 'data'. Use rioxarray write_crs() method to set CRS."
        )

    region = gpd.read_file(shapefile_path)

    if region.crs is None:
        raise RuntimeError(
            "No CRS found on data read from shapefile. Verify that shapefile contains valid CRS information."
        )

    # Select only user provided feature
    if feature:
        try:
            print("Selecting feature", feature)
            if isinstance(feature[1], list):
                region = region[region[feature[0]].isin(feature[1])]
            else:
                region = region[region[feature[0]] == feature[1]]
            if len(region) == 0:  # No features found
                raise ValueError("None of the requested features were found.")
        except ValueError as err:
            raise err
        except Exception as err:
            raise RuntimeError(
                "Could not select one or more feature(s) {0} in {1} ".format(
                    feature, shapefile_path
                )
            ) from err

    try:
        clipped = data.rio.clip(
            region.geometry.apply(mapping), region.crs, drop=True, **kwargs
        )
    except rio.exceptions.NoDataInBounds as err:
        msg = "Can't clip feature. Your grid resolution may be too low for your shapefile feature, or your shapefile's CRS may be incorrectly set."
        raise RuntimeError(msg) from err
    except Exception as err:
        raise err

    if feature:
        if isinstance(feature[1], list):
            location = [str(item) for item in feature[1]]
        else:
            location = [str(feature[1])]
    else:
        location = [name]
    clipped.attrs["location_subset"] = location

    return clipped
