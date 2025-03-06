import os
import numpy as np
import datetime
import xarray as xr
import pyproj
import rioxarray as rio
import pandas as pd
import copy
import intake
from timezonefinder import TimezoneFinder
from climakitae.core.paths import data_catalog_url, stations_csv_path
from climakitae.core.constants import SSPS


def downscaling_method_as_list(downscaling_method):
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


def area_average(dset):
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
        dset = dset.weighted(weights).mean("x").mean("y")
    elif set(["lat", "lon"]).issubset(set(dset.dims)):
        # LOCA data has lat, lon
        dset = dset.weighted(weights).mean("lat").mean("lon")
    return dset


def read_csv_file(rel_path, index_col=None, parse_dates=False):
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
        index_col=index_col,
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


def write_csv_file(df, rel_path):
    """Write CSV file from pandas DataFrame

    Parameters
    ----------
    df: pd.DataFrame
        pandas DataFrame to write out
    rel_path: str
        path to CSV file relative to this util python file

    Returns
    -------
    None or str
    """
    return df.to_csv(_package_file_path(rel_path))


def _package_file_path(rel_path):
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


def get_closest_gridcell(data, lat, lon, print_coords=True):
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

    if "x" and "y" in data.dims:
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
        if "x" and "y" in data.dims:
            tolerance = km_num * 1000  # Converting km to m
            closest_gridcell = data.sel(x=x, y=y, method="nearest", tolerance=tolerance)
        elif "lat" and "lon" in data.dims:
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


def julianDay_to_str_date(julday, leap_year=True, str_format="%b-%d"):
    """Convert julian day of year to string format
    i.e. if str_format = "%b-%d", the output will be Mon-Day ("Jan-01")

    Parameters
    ----------
    julday: int
        Julian day
    leap_year: boolean
        leap year? (default to True)
    str_format: str
        string format of output date

    Returns
    -------
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
    """Return the given bytes as a human friendly KB, MB, GB, or TB string.

    Parameters
    ----------
    B: byte

    Returns
    -------
    str

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


def reproject_data(xr_da, proj="EPSG:4326", fill_value=np.nan):
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

    def _reproject_data_4D(data, reproject_dim, proj="EPSG:4326", fill_value=np.nan):
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

    def _reproject_data_5D(data, reproject_dim, proj="EPSG:4326", fill_value=np.nan):
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
    data = data.rio.write_crs(xr_da.rio.crs, inplace=True)

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


def compute_multimodel_stats(data):
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
        raise Exception(
            "Invalid data provided, please pass cooling/heating degree data"
        )

    to_drop = ["scenario", "Lambert_Conformal", "variable"]
    for coord in to_drop:
        if coord in data.coords:
            data = data.drop(coord)

    return data


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


def add_dummy_time_to_wl(wl_da):
    """
    Replace the `[hours/days/months]_from_center` dimension in a DataArray returned from WarmingLevels with a dummy time index for calculations with tools that require a `time` dimension.

    Parameters
    ----------
    wl_da : xarray.DataArray
        The input Warming Levels DataArray. It is expected to have a time-based dimension which typically includes "from_center"
        in its name indicating the time dimension in relation to the year that the given warming level is reached per simulation.

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
    # Creating map from frequency name to freq var needed for pandas date range
    name_to_freq = {"hourly": "h", "daily": "D", "monthly": "ME"}

    # Creating dummy timestamps
    timestamps = pd.date_range(
        "2000-01-01",
        periods=len(wl_da["time_delta"]),
        freq=name_to_freq[wl_da.frequency],
    )

    # Replacing WL timestamps with dummy timestamps so that calculations from tools like `thresholds_tools`
    # can be computed on a DataArray with a time dimension
    wl_da = wl_da.assign_coords({"time_delta": timestamps}).rename(
        {"time_delta": "time"}
    )
    return wl_da


def downscaling_method_to_activity_id(downscaling_method, reverse=False):
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

    if reverse == True:
        downscaling_dict = {v: k for k, v in downscaling_dict.items()}
    return downscaling_dict[downscaling_method]


def resolution_to_gridlabel(resolution, reverse=False):
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

    if reverse == True:
        res_dict = {v: k for k, v in res_dict.items()}
    return res_dict[resolution]


def timescale_to_table_id(timescale, reverse=False):
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

    if reverse == True:
        timescale_dict = {v: k for k, v in timescale_dict.items()}
    return timescale_dict[timescale]


def scenario_to_experiment_id(scenario, reverse=False):
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

    if reverse == True:
        scenario_dict = {v: k for k, v in scenario_dict.items()}
    return scenario_dict[scenario]


def _get_cat_subset(selections):
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


def _get_scenario_from_selections(selections):
    """Get scenario from DataParameters object
    This needs to be handled differently due to warming levels retrieval method, which sets scenario to "n/a" for both historical and ssp.

    Parameters
    ----------
    selections: DataParameters
        object holding user's selections

    Returns
    -------
    scenario_ssp: list of str
    scenario_historical: list of str

    """

    if selections.approach == "Time":
        scenario_ssp = selections.scenario_ssp
        scenario_historical = selections.scenario_historical

    elif selections.approach == "Warming Level":
        # Need all scenarios for warming level approach
        scenario_ssp = SSPS
        scenario_historical = ["Historical Climate"]

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
