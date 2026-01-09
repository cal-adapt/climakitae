import calendar
import datetime
import os
import warnings
from typing import Any, Iterable, Union

import geopandas as gpd
import intake_esm
import numpy as np
import pandas as pd
import pyproj
import rioxarray as rio
import xarray as xr
from shapely.geometry import Point, mapping
from timezonefinder import TimezoneFinder

from climakitae.core.constants import (
    LOCA_END_YEAR,
    LOCA_START_YEAR,
    SSPS,
    UNSET,
    WRF_END_YEAR,
    WRF_START_YEAR,
)

# from climakitae.core.data_interface import DataParameters
from climakitae.core.paths import DATA_CATALOG_URL, STATIONS_CSV_PATH


def downscaling_method_as_list(downscaling_method: str) -> list[str]:
    """Function to convert string based radio button values to python list.

    Parameters
    ----------
    downscaling_method : str
        one of "Dynamical", "Statistical", or "Dynamical+Statistical"

    Returns
    -------
    method_list : list
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
    dset : xr.Dataset
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
    rel_path : str
        path to CSV file relative to this util python file
    index_col : str
        CSV column to index DataFrame on
    parse_dates : boolean
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
    df : pd.DataFrame
        pandas DataFrame to write out
    rel_path : str
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
    rel_path : str
        path to file relative to this util python file

    Returns
    -------
    str

    """
    return os.path.normpath(os.path.join(os.path.dirname(__file__), "..", rel_path))


def _check_has_valid_data(test_data: xr.Dataset | xr.DataArray) -> bool:
    """Check if test_data has any valid (non-null) values.

    This helper handles both eager (numpy) and lazy (dask) arrays by explicitly
    computing only the validity check, not the entire dataset.

    Parameters
    ----------
    test_data : xr.Dataset | xr.DataArray
        Data to check for valid values. Should already be subset to a single
        gridcell (no spatial dimensions).

    Returns
    -------
    bool
        True if any data variable contains valid (non-null) data, False otherwise.

    """
    if isinstance(test_data, xr.Dataset):
        # For Dataset, check if any variable has valid data
        for var in test_data.data_vars:
            is_all_null = test_data[var].isnull().all()
            # Handle dask arrays - compute only the boolean result
            if hasattr(is_all_null.data, "compute"):
                is_all_null = is_all_null.compute()
            if not is_all_null:
                return True
        return False
    else:
        # For DataArray
        is_all_null = test_data.isnull().all()
        if hasattr(is_all_null.data, "compute"):
            is_all_null = is_all_null.compute()
        return not is_all_null


def _get_coord_value(coord: xr.DataArray) -> float:
    """Extract a scalar coordinate value, handling both eager and lazy arrays.

    Parameters
    ----------
    coord : xr.DataArray
        Coordinate array (should be scalar or 0-d).

    Returns
    -------
    float
        The scalar coordinate value.

    """
    # If it's a dask array, compute it first
    if hasattr(coord.data, "compute"):
        coord = coord.compute()
    # Now get the scalar value
    if hasattr(coord, "item"):
        return coord.item()
    return float(coord.values)


def _get_spatial_dims(data: xr.Dataset | xr.DataArray) -> tuple[str, str]:
    """Identify the spatial dimension names in the data.

    Parameters
    ----------
    data : xr.Dataset | xr.DataArray
        Gridded data with spatial dimensions.

    Returns
    -------
    tuple[str, str]
        Tuple of (lat_dim, lon_dim) where lat_dim is 'y' or 'lat' (north-south)
        and lon_dim is 'x' or 'lon' (east-west).

    """
    lat_dim = [d for d in data.dims if d in ["y", "lat"]][0]
    lon_dim = [d for d in data.dims if d in ["x", "lon"]][0]
    return lat_dim, lon_dim


def _transform_coords_to_data_crs(
    data: xr.Dataset | xr.DataArray,
    lat: float,
    lon: float,
    lat_dim: str,
    lon_dim: str,
) -> tuple[float, float]:
    """Transform lat/lon coordinates to the data's coordinate reference system.

    If the input coordinates appear to already be in the data's CRS (detected by
    magnitude - projection coordinates in meters are much larger than lat/lon),
    they are returned as-is.

    Parameters
    ----------
    data : xr.Dataset | xr.DataArray
        Gridded data with CRS information (via rioxarray).
    lat : float
        Latitude coordinate (or y coordinate if already in projection CRS).
    lon : float
        Longitude coordinate (or x coordinate if already in projection CRS).
    lat_dim : str
        Name of the latitude/y spatial dimension ('y' or 'lat').
    lon_dim : str
        Name of the longitude/x spatial dimension ('x' or 'lon').

    Returns
    -------
    tuple[float, float]
        Coordinates in the data's CRS as (lat_coord, lon_coord).

    """
    # Detect if user passed lat/lon or x/y projection coordinates
    # Lat/lon values are always <= 360, projection coords (meters) are much larger
    user_passed_latlon = abs(lat) <= 360 and abs(lon) <= 360

    if lat_dim == "y" and lon_dim == "x" and user_passed_latlon:
        # Transform lat/lon to projection coordinates (x, y)
        transformer = pyproj.Transformer.from_crs(
            crs_from="epsg:4326",
            crs_to=data.rio.crs,
            always_xy=True,
        )
        # transform returns (x, y) with always_xy=True
        # We return (y, x) to match (lat_coord, lon_coord) order
        x, y = transformer.transform(lon, lat)
        return y, x
    return lat, lon


def _reduce_to_single_point(
    gridcell: xr.Dataset | xr.DataArray,
    dim1_name: str,
    dim2_name: str,
) -> xr.Dataset | xr.DataArray:
    """Reduce a gridcell to a single point by selecting first index of non-spatial dims.

    This is used for validity checking - we only need to test one time slice
    to know if the gridcell has valid data.

    Parameters
    ----------
    gridcell : xr.Dataset | xr.DataArray
        Data at a single gridcell (spatial dims already removed via isel).
    dim1_name : str
        Name of the first spatial dimension.
    dim2_name : str
        Name of the second spatial dimension.

    Returns
    -------
    xr.Dataset | xr.DataArray
        Data reduced to a single point for validity testing.

    """
    test_data = gridcell
    for dim in gridcell.dims:
        if dim not in [dim1_name, dim2_name]:
            mid_idx = gridcell.sizes[dim] // 2
            test_data = test_data.isel({dim: mid_idx})
    return test_data


def _print_closest_coords(
    closest_gridcell: xr.Dataset | xr.DataArray,
    input_lat: float,
    input_lon: float,
    lat_dim: str,
    lon_dim: str,
    is_averaged: bool = False,
    coord_values: tuple[float, float] | None = None,
) -> None:
    """Print the closest gridcell coordinates in the appropriate coordinate system.

    Parameters
    ----------
    closest_gridcell : xr.Dataset | xr.DataArray
        The closest gridcell data.
    input_lat : float
        The user's input latitude (or y coordinate).
    input_lon : float
        The user's input longitude (or x coordinate).
    lat_dim : str
        Name of the latitude/y spatial dimension.
    lon_dim : str
        Name of the longitude/x spatial dimension.
    is_averaged : bool, optional
        Whether the result was averaged over nearby valid gridcells.
        Default is False.
    coord_values : tuple[float, float] | None, optional
        Pre-computed coordinate values (lat_val, lon_val) for averaged case.
        If None, coordinates are extracted from closest_gridcell.

    """
    suffix = " (averaged over nearby valid gridcells)" if is_averaged else ""
    # Detect if user passed lat/lon or x/y projection coordinates
    # Lat/lon values are always <= 360, projection coords (meters) are much larger
    user_passed_latlon = abs(input_lat) <= 360 and abs(input_lon) <= 360

    if user_passed_latlon:
        # Report in lat/lon regardless of internal data structure
        if lat_dim == "y" and lon_dim == "x" and "lat" in closest_gridcell.coords:
            # For projected data with lat/lon auxiliary coords, use those
            print_lat = _get_coord_value(closest_gridcell["lat"])
            print_lon = _get_coord_value(closest_gridcell["lon"])
        elif coord_values is not None:
            # Use pre-computed values for averaged case
            print_lat = (
                coord_values[0]
                if np.isscalar(coord_values[0])
                else float(coord_values[0])
            )
            print_lon = (
                coord_values[1]
                if np.isscalar(coord_values[1])
                else float(coord_values[1])
            )
        else:
            print_lat = _get_coord_value(closest_gridcell[lat_dim])
            print_lon = _get_coord_value(closest_gridcell[lon_dim])

        if is_averaged:
            print(
                f"Closest gridcell to lat: {input_lat}, lon: {input_lon} "
                f"is at lat: {print_lat:.4g}, lon: {print_lon:.4g}{suffix}"
            )
        else:
            print(
                f"Closest gridcell to lat: {input_lat}, lon: {input_lon} "
                f"is at lat: {print_lat}, lon: {print_lon}"
            )
    else:
        # User passed projection coordinates (y, x), report in y/x
        if coord_values is not None:
            print_y = (
                coord_values[0]
                if np.isscalar(coord_values[0])
                else float(coord_values[0])
            )
            print_x = (
                coord_values[1]
                if np.isscalar(coord_values[1])
                else float(coord_values[1])
            )
        else:
            print_y = _get_coord_value(closest_gridcell[lat_dim])
            print_x = _get_coord_value(closest_gridcell[lon_dim])

        if is_averaged:
            print(
                f"Closest gridcell to y: {input_lat}, x: {input_lon} "
                f"is at y: {print_y:.4g}, x: {print_x:.4g}{suffix}"
            )
        else:
            print(
                f"Closest gridcell to y: {input_lat}, x: {input_lon} "
                f"is at y: {print_y}, x: {print_x}"
            )


def _search_nearby_valid_gridcells(
    data: xr.Dataset | xr.DataArray,
    center_idx1: int,
    center_idx2: int,
    dim1_name: str,
    dim2_name: str,
    window_size: int = 1,
) -> list:
    """Search a window around the center index for valid gridcells.

    Parameters
    ----------
    data : xr.Dataset | xr.DataArray
        The full gridded dataset.
    center_idx1 : int
        Index of center gridcell along dim1.
    center_idx2 : int
        Index of center gridcell along dim2.
    dim1_name : str
        Name of the first spatial dimension.
    dim2_name : str
        Name of the second spatial dimension.
    window_size : int, optional
        Half-width of the search window (1 = 3x3 window). Default is 1.

    Returns
    -------
    list
        List of valid gridcells (xr.Dataset or xr.DataArray) found in the window.

    """
    valid_data = []
    dim1_size = data.sizes[dim1_name]
    dim2_size = data.sizes[dim2_name]

    for i in range(-window_size, window_size + 1):
        for j in range(-window_size, window_size + 1):
            new_idx1 = center_idx1 + i
            new_idx2 = center_idx2 + j

            # Skip if indices are out of bounds
            if not (0 <= new_idx1 < dim1_size and 0 <= new_idx2 < dim2_size):
                continue

            gridcell = data.isel({dim1_name: new_idx1, dim2_name: new_idx2})
            test_data = _reduce_to_single_point(gridcell, dim1_name, dim2_name)

            if _check_has_valid_data(test_data):
                valid_data.append(gridcell)

    return valid_data


def _average_gridcells_preserve_coords(
    valid_data: list,
    dim1_name: str,
    dim2_name: str,
) -> tuple[xr.Dataset | xr.DataArray, tuple[float, float]]:
    """Average multiple gridcells while preserving spatial coordinates.

    Parameters
    ----------
    valid_data : list
        List of valid gridcells to average.
    dim1_name : str
        Name of the first spatial dimension.
    dim2_name : str
        Name of the second spatial dimension.

    Returns
    -------
    tuple
        (averaged_gridcell, (coord1_val, coord2_val)) - the averaged data
        with spatial coordinates restored, and the coordinate values used.

    """
    # Store spatial coordinates before averaging (they will be lost during mean)
    # Use the first valid gridcell's coordinates as the representative location
    coord1 = valid_data[0].coords[dim1_name]
    coord2 = valid_data[0].coords[dim2_name]
    coord1_val = _get_coord_value(coord1) if len(coord1.shape) == 0 else coord1.values
    coord2_val = _get_coord_value(coord2) if len(coord2.shape) == 0 else coord2.values

    # Average the valid gridcells (this removes spatial coords)
    averaged = xr.concat(valid_data, dim="valid_points").mean(dim="valid_points")

    # Add back the spatial coordinates as scalars
    averaged = averaged.assign_coords({dim1_name: coord1_val, dim2_name: coord2_val})

    return averaged, (coord1_val, coord2_val)


def get_closest_gridcell(
    data: xr.Dataset | xr.DataArray, lat: float, lon: float, print_coords: bool = True
) -> xr.Dataset | xr.DataArray | None:
    """From input gridded data, get the closest VALID gridcell to a lat, lon coordinate pair.

    This function first transforms the lat,lon coords to the gridded data’s projection.
    Then, it uses xarray’s built in method .sel to get the nearest gridcell.

    Parameters
    ----------
    data : xr.DataArray | xr.Dataset
        Gridded data (can be backed by numpy or dask arrays)
    lat : float
        Latitude or y value of coordinate pair
    lon : float
        Longitude or x value of coordinate pair
    print_coords : bool, optional
        Print closest coorindates?
        Default to True. Set to False for backend use.

    Returns
    -------
    xr.Dataset | xr.DataArray | None
        Grid cell closest to input lat,lon coordinate pair, returns same type as input.
        The result preserves lazy evaluation if the input was lazy.

    See also
    --------
    xr.DataArray.isel

    """
    # Identify spatial dimensions and transform coordinates
    # lat_dim is y-axis (north-south), lon_dim is x-axis (east-west)
    lat_dim, lon_dim = _get_spatial_dims(data)
    lat_coord, lon_coord = _transform_coords_to_data_crs(
        data, lat, lon, lat_dim, lon_dim
    )

    # Find nearest indices
    lat_idx = data[lat_dim].to_index().get_indexer([lat_coord], method="nearest")[0]
    lon_idx = data[lon_dim].to_index().get_indexer([lon_coord], method="nearest")[0]

    if lat_idx == -1 or lon_idx == -1:
        print("Input coordinate is OUTSIDE of data extent. Returning None.")
        return None

    # Check for valid data at closest gridcell
    gridcell = data.isel({lat_dim: lat_idx, lon_dim: lon_idx})
    test_data = _reduce_to_single_point(gridcell, lat_dim, lon_dim)

    if _check_has_valid_data(test_data):
        if print_coords:
            _print_closest_coords(gridcell, lat, lon, lat_dim, lon_dim)
        return gridcell

    # If closest gridcell is all NaN, search in 3x3 window and average valid cells
    valid_data = _search_nearby_valid_gridcells(
        data, lat_idx, lon_idx, lat_dim, lon_dim
    )

    if len(valid_data) > 0:
        closest_gridcell, coord_values = _average_gridcells_preserve_coords(
            valid_data, lat_dim, lon_dim
        )
        if print_coords:
            _print_closest_coords(
                closest_gridcell,
                lat,
                lon,
                lat_dim,
                lon_dim,
                is_averaged=True,
                coord_values=coord_values,
            )
        return closest_gridcell

    return None


def _transform_coords_to_data_crs_vectorized(
    data: xr.Dataset | xr.DataArray,
    lats: np.ndarray,
    lons: np.ndarray,
    lat_dim: str,
    lon_dim: str,
) -> tuple[np.ndarray, np.ndarray]:
    """Transform arrays of lat/lon coordinates to the data's CRS (vectorized).

    Parameters
    ----------
    data : xr.Dataset | xr.DataArray
        Gridded data with CRS information (via rioxarray).
    lats : np.ndarray
        Array of latitude coordinates.
    lons : np.ndarray
        Array of longitude coordinates.
    lat_dim : str
        Name of the latitude/y spatial dimension ('y' or 'lat').
    lon_dim : str
        Name of the longitude/x spatial dimension ('x' or 'lon').

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        Arrays of coordinates in the data's CRS as (lat_coords, lon_coords).

    """
    # Detect if user passed lat/lon or x/y projection coordinates
    # Lat/lon values are always <= 360, projection coords (meters) are much larger
    user_passed_latlon = np.all(np.abs(lats) <= 360) and np.all(np.abs(lons) <= 360)

    if lat_dim == "y" and lon_dim == "x" and user_passed_latlon:
        # Transform lat/lon to projection coordinates (x, y) - vectorized
        transformer = pyproj.Transformer.from_crs(
            crs_from="epsg:4326",
            crs_to=data.rio.crs,
            always_xy=True,
        )
        # transform returns (x, y) with always_xy=True
        # We return (y, x) to match (lat_coord, lon_coord) order
        x_coords, y_coords = transformer.transform(lons, lats)
        return np.asarray(y_coords), np.asarray(x_coords)
    return lats, lons


def get_closest_gridcells(
    data: xr.Dataset | xr.DataArray,
    lats: Iterable[float] | float,
    lons: Iterable[float] | float,
    print_coords: bool = True,
    bbox_buffer: int = 5,
) -> xr.Dataset | xr.DataArray | None:
    """Find the nearest grid cell(s) for given latitude and longitude coordinates.

    This function uses vectorized operations to efficiently find closest gridcells
    for multiple coordinate pairs at once. For a single point, it delegates to
    get_closest_gridcell.

    Parameters
    ----------
    data : xr.DataArray | xr.Dataset
        Gridded dataset with (x, y) or (lat, lon) dimensions.
    lats : float | Iterable[float]
        Latitude coordinate(s).
    lons : float | Iterable[float]
        Longitude coordinate(s).
    print_coords : bool, optional
        Print closest coordinates for each point. Default is True.
        Note: For large numbers of points, printing is automatically suppressed.
    bbox_buffer : int, optional
        Number of grid cells to add as buffer around the bounding box when
        pre-clipping large datasets. Default is 5.

    Returns
    -------
    xr.Dataset | xr.DataArray | None
        Nearest grid cell(s) or None if no valid match is found.
        If multiple coordinates are provided, results are concatenated along 'points' dimension.

    See Also
    --------
    get_closest_gridcell

    Notes
    -----
    For large datasets with many target points, this function first clips the data
    to a bounding box around the target points. This dramatically reduces the Dask
    task graph complexity and improves performance for downstream operations.

    """
    # Convert single values to arrays for uniform handling
    lats_arr = np.atleast_1d(np.asarray(lats))
    lons_arr = np.atleast_1d(np.asarray(lons))

    # Ensure lats and lons have the same length
    if len(lats_arr) != len(lons_arr):
        raise ValueError(
            f"lats and lons must have the same length, got {len(lats_arr)} and {len(lons_arr)}"
        )

    n_points = len(lats_arr)
    print(f"Processing {n_points} coordinate pair(s)...")

    # Suppress per-point printing for large numbers of points
    if n_points > 10:
        print_coords = False

    # Identify spatial dimensions and transform all coordinates at once
    lat_dim, lon_dim = _get_spatial_dims(data)
    print(f"  Spatial dimensions: {lat_dim}, {lon_dim}")

    lat_coords, lon_coords = _transform_coords_to_data_crs_vectorized(
        data, lats_arr, lons_arr, lat_dim, lon_dim
    )

    # Get coordinate arrays from data
    lat_index = data[lat_dim].to_index()
    lon_index = data[lon_dim].to_index()
    print(f"  Data grid size: {len(lat_index)} x {len(lon_index)}")

    # OPTIMIZATION: Pre-clip to bounding box to reduce Dask task graph complexity
    # This is critical for large datasets with many scattered points
    lat_min_idx = lat_index.get_indexer([lat_coords.min()], method="nearest")[0]
    lat_max_idx = lat_index.get_indexer([lat_coords.max()], method="nearest")[0]
    lon_min_idx = lon_index.get_indexer([lon_coords.min()], method="nearest")[0]
    lon_max_idx = lon_index.get_indexer([lon_coords.max()], method="nearest")[0]

    # Add buffer and clamp to valid range
    lat_min_idx = max(0, min(lat_min_idx, lat_max_idx) - bbox_buffer)
    lat_max_idx = min(len(lat_index) - 1, max(lat_min_idx, lat_max_idx) + bbox_buffer)
    lon_min_idx = max(0, min(lon_min_idx, lon_max_idx) - bbox_buffer)
    lon_max_idx = min(len(lon_index) - 1, max(lon_min_idx, lon_max_idx) + bbox_buffer)

    bbox_lat_size = lat_max_idx - lat_min_idx + 1
    bbox_lon_size = lon_max_idx - lon_min_idx + 1
    original_size = len(lat_index) * len(lon_index)
    bbox_size = bbox_lat_size * bbox_lon_size

    # Only pre-clip if it reduces spatial size significantly (>50% reduction)
    if bbox_size < original_size * 0.5:
        print(
            f"  Pre-clipping to bounding box: {bbox_lat_size} x {bbox_lon_size} "
            f"({bbox_size / original_size * 100:.1f}% of original)"
        )
        data = data.isel(
            {
                lat_dim: slice(lat_min_idx, lat_max_idx + 1),
                lon_dim: slice(lon_min_idx, lon_max_idx + 1),
            }
        )
        # Update indices to reference the clipped data
        lat_index = data[lat_dim].to_index()
        lon_index = data[lon_dim].to_index()

    # Find nearest indices for all points at once (vectorized)
    lat_indices = lat_index.get_indexer(lat_coords, method="nearest")
    lon_indices = lon_index.get_indexer(lon_coords, method="nearest")

    # Check for out-of-bounds points
    valid_mask = (lat_indices != -1) & (lon_indices != -1)
    if not np.any(valid_mask):
        print("All input coordinates are OUTSIDE of data extent. Returning None.")
        return None

    if not np.all(valid_mask):
        n_invalid = np.sum(~valid_mask)
        print(
            f"  Warning: {n_invalid} point(s) are outside data extent and will be excluded."
        )

    # Filter to valid indices only
    valid_lat_indices = lat_indices[valid_mask]
    valid_lon_indices = lon_indices[valid_mask]
    valid_lats = lats_arr[valid_mask]
    valid_lons = lons_arr[valid_mask]
    n_valid = len(valid_lat_indices)
    print(f"  Found {n_valid} valid point(s) within data extent")

    # Check for invalid (ocean/masked) points using landmask BEFORE extracting data
    # This avoids loading the full dataset just to check for NaNs
    # SKIP expensive validity check for large point sets (>100 points) - the compute
    # call triggers the entire Dask task graph which is extremely slow
    needs_nan_handling = []
    skip_validity_check = n_valid > 100

    if skip_validity_check:
        print(f"  Skipping validity check for {n_valid} points (too expensive)")
    elif "landmask" in data.coords or "landmask" in getattr(data, "data_vars", {}):
        print("  Checking landmask for valid land points...")
        landmask = data["landmask"]
        # Compute if dask-backed (this is just 2D, so cheap)
        if hasattr(landmask.data, "compute"):
            landmask = landmask.compute()
        landmask_values = landmask.values

        # Check which target points are on land vs water
        is_land = np.array(
            [
                landmask_values[lat_idx, lon_idx] == 1
                for lat_idx, lon_idx in zip(valid_lat_indices, valid_lon_indices)
            ]
        )
        needs_nan_handling = list(np.where(~is_land)[0])
        print(f"  {np.sum(is_land)}/{n_valid} points are on land")
    else:
        # No landmask available - create one from a single time slice
        # Only do this for small point sets since it requires compute()
        print("  No landmask found, checking single time slice for validity...")
        if isinstance(data, xr.Dataset):
            first_var = list(data.data_vars)[0]
            check_data = data[first_var]
        else:
            check_data = data

        # Get a single time slice to create validity mask
        if "time" in check_data.dims:
            single_slice = check_data.isel(time=0)
        else:
            single_slice = check_data

        # Reduce any remaining non-spatial dims
        for dim in list(single_slice.dims):
            if dim not in [lat_dim, lon_dim]:
                single_slice = single_slice.isel({dim: 0})

        # Compute the validity mask (2D)
        if hasattr(single_slice.data, "compute"):
            single_slice = single_slice.compute()

        validity_mask = ~np.isnan(single_slice.values)

        # Check which target points are valid
        is_valid = np.array(
            [
                validity_mask[lat_idx, lon_idx]
                for lat_idx, lon_idx in zip(valid_lat_indices, valid_lon_indices)
            ]
        )
        needs_nan_handling = list(np.where(~is_valid)[0])
        print(f"  {np.sum(is_valid)}/{n_valid} points have valid data")

    # For invalid points, find valid neighbor indices using the 2D mask
    # This is done BEFORE extracting the full timeseries data
    final_lat_indices = valid_lat_indices.copy()
    final_lon_indices = valid_lon_indices.copy()

    if len(needs_nan_handling) > 0:
        print(f"  {len(needs_nan_handling)} point(s) need neighbor search...")

        # Get the validity mask for neighbor searching
        if "landmask" in data.coords or "landmask" in getattr(data, "data_vars", {}):
            mask_2d = landmask_values == 1
        else:
            mask_2d = validity_mask

        lat_size = data.sizes[lat_dim]
        lon_size = data.sizes[lon_dim]
        points_fixed = 0

        for point_idx in needs_nan_handling:
            lat_idx = int(valid_lat_indices[point_idx])
            lon_idx = int(valid_lon_indices[point_idx])

            # Search 3x3 window for valid neighbor using the 2D mask
            found_valid = False
            for di in range(-1, 2):
                for dj in range(-1, 2):
                    ni, nj = lat_idx + di, lon_idx + dj
                    if 0 <= ni < lat_size and 0 <= nj < lon_size:
                        if mask_2d[ni, nj]:
                            # Found a valid neighbor - use its indices
                            final_lat_indices[point_idx] = ni
                            final_lon_indices[point_idx] = nj
                            found_valid = True
                            points_fixed += 1
                            break
                if found_valid:
                    break

        print(
            f"  Fixed {points_fixed}/{len(needs_nan_handling)} points with valid neighbors"
        )

    # Now extract the full timeseries data using the corrected indices
    print("  Extracting gridcells...")

    # Use vectorized advanced indexing with DataArray indexers
    # This is more efficient than stack+isel because it doesn't create a complex MultiIndex
    lat_indexer = xr.DataArray(final_lat_indices, dims=["points"])
    lon_indexer = xr.DataArray(final_lon_indices, dims=["points"])

    # Select all points at once using vectorized indexing
    result = data.isel({lat_dim: lat_indexer, lon_dim: lon_indexer})

    # Add coordinate information for each point
    actual_lats = data[lat_dim].values[final_lat_indices]
    actual_lons = data[lon_dim].values[final_lon_indices]

    # Assign coordinates to the result
    result = result.assign_coords(
        {
            lat_dim: ("points", actual_lats),
            lon_dim: ("points", actual_lons),
        }
    )

    # Print coordinates if requested
    if print_coords:
        for i in range(n_valid):
            print(
                f"  Point {i+1}: ({valid_lats[i]:.4f}, {valid_lons[i]:.4f}) -> "
                f"({actual_lats[i]:.4f}, {actual_lons[i]:.4f})"
            )

    # Reorder dimensions to put 'points' at the end
    if isinstance(result, xr.Dataset):
        first_var = list(result.data_vars)[0]
        all_dims = list(result[first_var].dims)
    else:
        all_dims = list(result.dims)

    if "points" in all_dims:
        all_dims.remove("points")
        all_dims.append("points")
        result = result.transpose(*all_dims)

    print(f"Done! Extracted {n_valid} gridcell(s)")
    return result


def julianDay_to_date(
    julday: int, year: int = None, return_type: str = "str", str_format: str = "%b-%d"
) -> Union[str, datetime.datetime, datetime.date]:
    """Convert julian day of year to a date object or formatted string.

    Parameters
    ----------
    julday : int
        Julian day (day of year)
    year : int, optional
        Year to use. If None, uses current year or a leap year (2024) based on needs.
        Default is None.
    return_type : str, optional
        Type of return value:
        - "str": formatted string (default)
        - "datetime": datetime object
        - "date": date object
    str_format : str, optional
        String format of output date when return_type is "str".
        Default is "%b-%d" which outputs format like "Jan-01".

    Returns
    -------
    date : str, datetime.datetime, or datetime.date
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
    B : byte

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
    xr_da : xr.DataArray
        2-or-3-dimensional DataArray, with 2 spatial dimensions
    proj : str
        proj to use for reprojection (default to "EPSG:4326"-- lat/lon coords)
    fill_value : float
        fill value (default to np.nan)

    Returns
    -------
    data_reprojected : xr.DataArray
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
        data : xr.DataArray
            4-dimensional DataArray, with 2 spatial dimensions
        reproject_dim : str
            name of dimensions to use
        proj : str
            proj to use for reprojection (default to "EPSG:4326"-- lat/lon coords)
        fill_value : float
            fill value (default to np.nan)

        Returns
        -------
        data_reprojected : xr.DataArray
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
        data : xr.DataArray
            5-dimensional DataArray, with 2 spatial dimensions
        reproject_dim : list
            list of str dimension names to use
        proj : str
            proj to use for reprojection (default to "EPSG:4326"-- lat/lon coords)
        fill_value : float
            fill value (default to np.nan)

        Returns
        -------
        data_reprojected : xr.DataArray
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
    data : xr.DataArray
    name : str
    num_grid_cells : int

    Returns
    -------
    annual_ag : xr.DataArray

    """
    annual_ag = data.squeeze().groupby("time.year").sum(["time"])  # Aggregate annually
    annual_ag = annual_ag / num_grid_cells  # Divide by number of gridcells
    annual_ag.name = name  # Give new name to dataset
    return annual_ag


def compute_multimodel_stats(data: xr.DataArray) -> xr.DataArray:
    """Calculates model mean, min, max, median across simulations

    Parameters
    ----------
    data : xr.DataArray

    Returns
    -------
    stats_concat : xr.DataArray

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
    data : xr.Dataset
    kind : str , optional
        Options are 'mean' and 'median'

    Returns
    -------
    trendline : xr.Dataset

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
    data : xr.DataArray

    Returns
    -------
    data : xr.DataArray

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
    """Helper function to organize dataset object into a pandas dataframe for ease.

    Parameters
    ----------
    data : xr.Dataset

    Returns
    -------
    df : pd.DataFrame
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


def convert_to_local_time(
    data: xr.DataArray | xr.Dataset,
    lon: float = UNSET,
    lat: float = UNSET,
) -> xr.DataArray | xr.Dataset:
    """Convert time dimension from UTC to local time for the grid or station.

    Parameters
    ----------
        data : xr.DataArray | xr.Dataset
            Input data.
        grid_lon : float
            Mean longitude of dataset if no lat/lon coordinates
        grid_lat : float
            Mean latitude of dataset if no lat/lon coordinates

    Returns
    -------
        xr.DataArray | xr.Dataset
            Data with converted time coordinate.

    """

    # Only converting hourly data
    if not (frequency := data.attrs.get("frequency", None)):
        # Make a guess at frequency
        timestep = pd.Timedelta(
            data.time[1].item() - data.time[0].item()
        ).total_seconds()
        match timestep:
            case 3600:
                frequency = "hourly"
            case 86400:
                frequency = "daily"
            case _ if timestep > 86400:
                frequency = "monthly"

    # If timescale is not hourly, no need to convert
    if frequency != "hourly":
        print(
            "This dataset's timescale is not granular enough to covert to local time. Local timezone conversion requires hourly data."
        )
        return data

    # Find out if Stations or Gridded type
    if not (data_type := data.attrs.get("data_type", None)):
        if isinstance(data, xr.core.dataarray.DataArray):
            print(
                "Data Array attribute 'data_type' not found. Please set 'data_type' to 'Stations' or 'Gridded'."
            )
            return data
        else:
            try:
                # Grab from one of data arrays in dataset
                variable = list(data.keys())[0]
                data_type = data[variable].attrs["data_type"]
            except KeyError:
                print(
                    f"Could not find attribute 'data_type' attribute set in {variable} attributes. Please set data_type attribute."
                )
                return data

    # Get latitude/longitude information
    match data_type:
        case "Stations":
            # Read stations database
            stations_df = read_csv_file(STATIONS_CSV_PATH)
            stations_df = stations_df.drop(columns=["Unnamed: 0"])

            # Filter by selected station(s) - assume first station if multiple
            match data:
                case xr.DataArray():
                    station_name = data.name
                case xr.Dataset():
                    # Grab first one
                    station_name = list(data.keys())[0]
                case _:
                    print(
                        f"Invalid data type {type(data)}. Please provide xarray DataArray or Dataset."
                    )
                    return data
            station_data = stations_df[stations_df["station"] == station_name]
            if len(station_data) == 0:
                print(
                    f"Station {data.name} not found in Stations CSV. Please set Data Array name to valid station name."
                )
                return data
            lat = station_data["LAT_Y"].values[0]
            lon = station_data["LON_X"].values[0]

        case "Gridded":
            # if both lat and lon are set, can move on to timezone finding.
            if (lat is UNSET) or (lon is UNSET):
                try:
                    # Finding avg. lat/lon coordinates from all grid-cells
                    lat = data.lat.mean().item()
                    lon = data.lon.mean().item()
                except AttributeError:
                    print(
                        "lat/lon coordinates not found in data. Please pass in data with 'lon' and 'lat' coordinates or set both 'lon' and 'lat' arguments."
                    )
                    return data

        case _:
            print(
                "Invalid data type attribute. Data type should be 'Stations' or 'Gridded'."
            )
            return data

    # Find timezone for the coordinates
    tf = TimezoneFinder()
    local_tz = tf.timezone_at(lng=lon, lat=lat)

    # Change datetime objects to local time
    new_time = (
        pd.DatetimeIndex(data.time)
        .tz_localize("UTC")
        .tz_convert(local_tz)
        .tz_localize(None)
        .astype("datetime64[ns]")
    )
    data["time"] = new_time

    print(f"Data converted to {local_tz} timezone.")

    # Add timezone attribute to data
    match data:
        case xr.DataArray():
            data = data.assign_attrs({"timezone": local_tz})
        case xr.Dataset():
            variables = list(data.keys())
            for variable in variables:
                data[variable] = data[variable].assign_attrs({"timezone": local_tz})
        case _:
            print(f"Invalid data type {type(data)}. Could not set timezone attribute.")

    return data


def add_dummy_time_to_wl(wl_da: xr.DataArray, freq_name="daily") -> xr.DataArray:
    """Replace the [hours/days/months]_from_center or time_delta dimension in a DataArray returned from WarmingLevels with a dummy time index for calculations with tools that require a time dimension.

    Parameters
    ----------
    wl_da : xr.DataArray
        The input Warming Levels DataArray. It is expected to have a time-based dimension which typically includes "from_center"
        in its name or time_delta indicating the time dimension in relation to the year that the given warming level is reached per simulation.
    freq_name : str, optional
        The frequency name to use when time_delta is the time dimension. Options are "hourly", "daily", or "monthly". Default is "daily".

    Returns
    -------
    xr.DataArray
        A modified version of the input DataArray with the original time dimension replaced by a dummy time series. The new dimension
        will be named "time".

    Notes
    -----
    - The function looks for the dimension name containing "from_center" to identify the time-based dimension.
    - It supports creating dummy time series with frequencies of hours, days, or months, based on the prefix of the dimension name.
    - The dummy time series starts from "2000-01-01".

    """
    ### Adjusting the time index into a dummy time-series for counting

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

    # Determine time frequency name and pandas freq string mapping
    if wl_time_dim == "time_delta":

        try:
            time_freq_name = wl_da.frequency
        except AttributeError:
            time_freq_name = freq_name

        name_to_freq = {
            "1hr": "h",
            "hourly": "h",
            "daily": "D",
            "day": "D",
            "monthly": "MS",
        }

    else:
        time_freq_name = wl_time_dim.split("_")[0]
        name_to_freq = {"hours": "h", "days": "D", "months": "MS"}

    freq = name_to_freq[time_freq_name]

    # Number of time units per normal year
    num_time_units_per_year = {"h": 8760, "D": 365, "MS": 12}

    # Calculate total number of units in wl_da along wl_time_dim
    len_time = len(wl_da[wl_time_dim])

    # Calculate approximate number of years spanned by data
    years_span = len_time / num_time_units_per_year[freq]
    start_year = 2000
    end_year = int(start_year + years_span - 1)

    # Calculate total leap days in the period
    total_leap_days = sum(
        calendar.isleap(year) for year in range(start_year, end_year + 1)
    )

    # Adjust number of periods to add leap day hours if hourly, else add leap days as periods
    extra_periods = total_leap_days * 24 if freq == "h" else total_leap_days

    # Edge cases:
    # if total time passed in is less than 60 days (when Feb 29th is), then don't add extra_periods
    # if we're looking at monthly data, then don't add extra_periods
    if (
        (freq == "h" and len_time < 24 * 60)
        or (freq == "D" and len_time < 60)
        or (freq == "MS")
    ):
        extra_periods = 0

    # Create the dummy timestamps including leap day adjustments
    timestamps = pd.date_range(
        start="2000-01-01", periods=len_time + extra_periods, freq=freq
    )

    # Filter out leap days (Feb 29)
    timestamps = timestamps[~((timestamps.month == 2) & (timestamps.day == 29))]

    # Replacing WL timestamps with dummy timestamps so that calculations from tools like thresholds_tools
    # can be computed on a DataArray with a time dimension
    wl_da = wl_da.assign_coords({wl_time_dim: timestamps}).rename({wl_time_dim: "time"})
    return wl_da


def downscaling_method_to_activity_id(
    downscaling_method: str, reverse: bool = False
) -> str:
    """Convert downscaling method to activity id to match catalog names

    Parameters
    ----------
    downscaling_method : str
        Downscaling method
    reverse : boolean, optional
        Set reverse=True to get downscaling method from input activity_id
        Default to False

    Returns
    -------
    str

    """
    downscaling_dict = {"Dynamical": "WRF", "Statistical": "LOCA2"}

    if reverse:
        downscaling_dict = {v: k for k, v in downscaling_dict.items()}
    return downscaling_dict[downscaling_method]


def resolution_to_gridlabel(resolution: str, reverse: bool = False) -> str:
    """Convert resolution format to grid_label format matching catalog names.

    Parameters
    ----------
    resolution : str
        resolution
    reverse : boolean, optional
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
    """Convert scenario format to experiment_id format matching catalog names.

    Parameters
    ----------
    scenario : str
    reverse : boolean, optional
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
    selections : DataParameters
        object holding user's selections

    Returns
    -------
    cat_subset : intake_esm.source.ESMDataSource
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
    selections : DataParameters
        object holding user's selections

    Returns
    -------
    scenario_ssp : list of str
    scenario_historical : list of str

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
    the shapefile. To clip to specific features, use the feature keyword.

    Parameters
    ----------
    data : xr.Dataset | xr.DataArray
        Data to be clipped.
    shapefile_path : str
        Filepath to shapefile. Shapefile must include valid CRS.
    feature : tuple(str, str | int | float | list)
        Tuple containing attribute name and value(s) for target feature(s) (optional).
    name : str
        Location name to record in data attributes if 'feature' parameter is not set (optional).
    **kwargs
        Additional arguments to pass to the rioxarray clip function

    Returns
    -------
    clipped: xr.Dataset | xr.DataArray
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


def clip_gpd_to_shapefile(
    gdf: gpd.GeoDataFrame,
    shapefile: gpd.GeoDataFrame,
) -> gpd.GeoDataFrame:
    """Use a shapefile to select an area subset of a geodataframe.
    Used to subset stationlist to shapefile area.

    Parameters
    ----------
    gdf : gpd.GeoDataFrame
        Data to be clipped.
    shapefile : gpd.GeoDataFrame
        Shapefile must include valid CRS.

    Returns
    -------
    clipped : gpd.GeoDataFrame
        Subsetted geodataframe within shapefile area of interest.
    """

    # Adds coordinates
    geom = gpd.points_from_xy(gdf["longitude"], gdf["latitude"])
    sub_gdf = gpd.GeoDataFrame(gdf, geometry=geom).set_crs(
        crs="EPSG:3857", allow_override=True
    )

    # Check CRS
    if sub_gdf is None or shapefile.crs is None:
        raise RuntimeError(
            "Both input GeoDataFrame and shapefile must have a defined CRS."
        )

    if sub_gdf.crs != shapefile.crs:
        shapefile = shapefile.to_crs(sub_gdf.crs)

    # Subset for stations within area boundaries
    clipped = sub_gdf[sub_gdf.geometry.intersects(shapefile.unary_union)]

    if clipped.empty:
        raise RuntimeError(
            "Clipping returned an empty GeoDataFrame; check geometries and CRS."
        )

    # Useful information
    print(f"Number of stations within area: {len(clipped)}")

    return clipped


def _determine_is_complete_wl(
    start_year: int,
    end_year: int,
    simulation_name: str,
    downscaling_method: str,
    level: float,
) -> bool:
    """
    Determine if a complete warming level slice can be created for the given start and end years.
    This checks if the years fall within valid ranges based on the downscaling method.

    Parameters
    ----------
    start_year : int
        The starting year of the warming level slice.
    end_year : int
        The ending year of the warming level slice.
    simulation_name : str
        The name of the simulation being evaluated.
    downscaling_method : str
        The downscaling method used ("Statistical" or "Dynamical").
    level : float
        The warming level being evaluated.

    Returns
    -------
    bool
        True if a complete warming level slice can be created, False otherwise.
    """
    valid_years = {
        "LOCA2": (LOCA_START_YEAR, LOCA_END_YEAR),
        "Statistical": (LOCA_START_YEAR, LOCA_END_YEAR),
        "WRF": (WRF_START_YEAR, WRF_END_YEAR),
        "Dynamical": (WRF_START_YEAR, WRF_END_YEAR),
    }

    min_year, max_year = valid_years.get(downscaling_method, (None, None))

    if min_year and (start_year < min_year or end_year > max_year):
        warnings.warn(
            f"\n\nIncomplete warming level for {simulation_name} at {level}C. "
            "\nSkipping this warming level."
        )
        return False

    return True
