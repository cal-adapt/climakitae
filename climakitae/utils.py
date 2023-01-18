import numpy as np
import datetime
import xarray as xr
import pyproj
import rioxarray as rio
import pandas as pd
import s3fs
import intake
import matplotlib.colors as mcolors
import matplotlib
import pkg_resources
import warnings
from .selectors import Boundaries


# Read colormap text files
ae_orange = pkg_resources.resource_filename("climakitae", "data/cmaps/ae_orange.txt")
ae_diverging = pkg_resources.resource_filename(
    "climakitae", "data/cmaps/ae_diverging.txt"
)
ae_blue = pkg_resources.resource_filename("climakitae", "data/cmaps/ae_blue.txt")
ae_diverging_r = pkg_resources.resource_filename(
    "climakitae", "data/cmaps/ae_diverging_r.txt"
)


def get_closest_gridcell(data, lat, lon):
    """From input gridded data, get the closest gridcell to a lat, lon coordinate pair.
    This function first transforms the lat,lon coords to the gridded data’s projection.
    Then, it uses xarray’s built in method .sel to get the nearest gridcell.

    Args:
        data (xr.DataArray): gridded data
        lat (float): latitude
        lon (float): longitude

    Returns:
        closest_gridcell (xr.DataArray): grid cell closest to input lat,lon

    """
    print(
        "WARNING: Due to the inconsistency between a station and an area-average, when comparing a grid cell with historical observed station data, consider using a bias-correction function for that location instead.\n"
    )

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
    print(
        "Input coordinates: (%.2f, %.2f)" % (lat, lon)
        + "\nNearest grid cell coordinates: (%.2f, %.2f)"
        % (closest_gridcell.lat.item(), closest_gridcell.lon.item())
    )
    return closest_gridcell


def julianDay_to_str_date(julday, leap_year=True, str_format="%b-%d"):
    """Convert julian day of year to string format
    i.e. if str_format = "%b-%d", the output will be Mon-Day ("Jan-01")

    Args:
        julday (int): Julian day
        leap_year (boolean): leap year? (default to True)
        str_format (str): string format of output date

    Return:
        date (str): Julian day in the input format month-day (i.e. "Jan-01")
    """
    if leap_year:
        year = "2024"
    else:
        year = "2023"
    date = datetime.datetime.strptime(year + "." + str(julday), "%Y.%j").strftime(
        str_format
    )
    return date


def _readable_bytes(B):
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


def _read_ae_colormap(cmap="ae_orange", cmap_hex=False):
    """Read in AE colormap by name

    Args:
        cmap (str): one of ["ae_orange","ae_blue","ae_diverging"]
        cmap_hex (boolean): return RGB or hex colors?

    Returns: one of either
        cmap_data (matplotlib.colors.LinearSegmentedColormap): used for
            matplotlib (if cmap_hex == False)
        cmap_data (list): used for hvplot maps (if cmap_hex == True)

    """

    if cmap == "ae_orange":
        cmap_data = ae_orange
    elif cmap == "ae_diverging":
        cmap_data = ae_diverging
    elif cmap == "ae_blue":
        cmap_data = ae_blue
    elif cmap == "ae_diverging_r":
        cmap_data = ae_diverging_r

    # Load text file
    cmap_np = np.loadtxt(cmap_data, dtype=float)

    # RBG to hex
    if cmap_hex:
        cmap_data = [matplotlib.colors.rgb2hex(color) for color in cmap_np]
    else:
        cmap_data = mcolors.LinearSegmentedColormap.from_list(cmap, cmap_np, N=256)
    return cmap_data


def _reproject_data(xr_da, proj="EPSG:4326", fill_value=np.nan):
    """Reproject xr.DataArray using rioxarray.
    Raises ValueError if input data does not have spatial coords x,y
    Raises ValueError if input data has more than 5 dimensions

    Args:
        xr_da (xr.DataArray): 2-or-3-dimensional DataArray, with 2 spatial dimensions
        proj (str): proj to use for reprojection (default to "EPSG:4326"-- lat/lon coords)
        fill_value (float): fill value (default to np.nan)

    Returns:
        data_reprojected (xr.DataArray): 2-or-3-dimensional reprojected DataArray

    """

    def _reproject_data_4D(data, reproject_dim, proj="EPSG:4326", fill_value=np.nan):
        """Reproject 4D xr.DataArray across an input dimension

        Args:
            data (xr.DataArray): 4-dimensional DataArray, with 2 spatial dimensions
            reproject_dim (str): name of dimensions to use
            proj (str): proj to use for reprojection (default to "EPSG:4326"-- lat/lon coords)
            fill_value (float): fill value (default to np.nan)

        Returns:
            data_reprojected (xr.DataArray): 4-dimensional reprojected DataArray

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

        Args:
            data (xr.DataArray): 5-dimensional DataArray, with 2 spatial dimensions
            reproject_dim (list): list of str dimension names to use
            proj (str): proj to use for reprojection (default to "EPSG:4326"-- lat/lon coords)
            fill_value (float): fill value (default to np.nan)

        Returns:
            data_reprojected (xr.DataArray): 5-dimensional reprojected DataArray

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


### some utils for generating warming level reference data in ../data/ ###
def write_gwl_files():
    """Call everything needed to write the global warming level reference files
    for all of the currently downscaled GCMs."""

    # Connect to AWS S3 storage
    fs = s3fs.S3FileSystem(anon=True)

    df = pd.read_csv("https://cmip6-pds.s3.amazonaws.com/pangeo-cmip6.csv")
    df_subset = df[
        (df.table_id == "Amon")
        & (df.variable_id == "tas")
        & (df.experiment_id == "historical")
    ]

    def build_timeseries(variable, model, ens_mem, scenarios):
        """Builds an xarray Dataset with only a time dimension, for the appended
        historical+ssp timeseries for all the scenarios of a particular
        model/variant combo. Works for all of the models(/GCMs) in the list
        models_for_now, which appear in the current data catalog of WRF
        downscaling."""
        scenario = "historical"
        data_historical = xr.Dataset()
        df_scenario = df_subset[(df.source_id == model) & (df.member_id == ens_mem)]
        with xr.open_zarr(fs.get_mapper(df_scenario.zstore.values[0])) as temp:
            weightlat = np.sqrt(np.cos(np.deg2rad(temp.lat)))
            weightlat = weightlat / np.sum(weightlat)
            data_historical = (temp[variable] * weightlat).sum("lat").mean("lon")
            if model == "FGOALS-g3":
                data_historical = data_historical.isel(time=slice(0, -12 * 2))

        data_one_model = xr.Dataset()
        for scenario in scenarios:
            df_scenario = df[
                (df.table_id == "Amon")
                & (df.variable_id == variable)
                & (df.experiment_id == scenario)
                & (df.source_id == model)
                & (df.member_id == ens_mem)
            ]
            with xr.open_zarr(fs.get_mapper(df_scenario.zstore.values[0])) as temp:
                weightlat = np.sqrt(np.cos(np.deg2rad(temp.lat)))
                weightlat = weightlat / np.sum(weightlat)
                timeseries = (temp[variable] * weightlat).sum("lat").mean("lon")
                timeseries = timeseries.sortby("time")  # needed for MPI-ESM1-2-LR
                data_one_model[scenario] = xr.concat(
                    [data_historical, timeseries], dim="time"
                )
        return data_one_model

    def get_gwl(smoothed, degrees):
        """Takes a smoothed timeseries of global mean temperature for multiple
        scenarios, and returns a small table of the timestamp that a given
        global warming level is reached."""
        gwl = smoothed.sub(degrees).abs().idxmin()
        # make sure it's not just choosing one of the final timestamps just
        # because it's the highest warming despite being nowhere close to
        # (much less than) the target value:
        for scenario in smoothed:
            if smoothed[scenario].sub(degrees).abs().min() > 0.01:
                gwl[scenario] = np.NaN
        return gwl

    def get_gwl_table(
        variable, model, scenarios, start_year="18500101", end_year="19000101"
    ):
        """Loops through global warming levels, and returns an aggregate table
        for all warming levels (1.5, 2, 3, and 4 degrees) for all scenarios of
        the model/variant requested."""
        ens_mem = models[model]
        data_one_model = build_timeseries(variable, model, ens_mem, scenarios)
        anom = (
            data_one_model - data_one_model.sel(time=slice(start_year, end_year)).mean()
        )
        smoothed = anom.rolling(time=20 * 12, center=True).mean("time")
        one_model = (
            smoothed.to_array(dim="scenario", name=model).dropna("time").to_pandas()
        )
        gwlevels = pd.DataFrame()
        for level in [1.5, 2, 3, 4]:
            gwlevels[level] = get_gwl(one_model.T, level)
        return gwlevels

    models_WRF = {
        "ACCESS-CM2": "",
        "CanESM5": "",
        "CESM2": "r11i1p1f1",
        "CNRM-ESM2-1": "r1i1p1f2",
        "EC-Earth3": "",
        "EC-Earth3-Veg": "r1i1p1f1",
        "FGOALS-g3": "r1i1p1f1",
        "MPI-ESM1-2-LR": "r7i1p1f1",
        "UKESM1-0-LL": "",
    }
    models_for_now = {
        "CESM2": "r11i1p1f1",
        "CNRM-ESM2-1": "r1i1p1f2",
        "EC-Earth3-Veg": "r1i1p1f1",
        "FGOALS-g3": "r1i1p1f1",
        "MPI-ESM1-2-LR": "r7i1p1f1",
    }
    models = models_for_now

    variable = "tas"
    scenarios = ["ssp585", "ssp370", "ssp245"]
    all_gw_levels = pd.concat(
        [get_gwl_table(variable, model, scenarios) for model in list(models.keys())],
        keys=list(models.keys()),
    )
    all_gw_levels.to_csv("../data/gwl_1850-1900ref.csv")

    start_year = "19810101"
    end_year = "20101231"
    all_gw_levels2 = pd.concat(
        [
            get_gwl_table(variable, model, scenarios, start_year, end_year)
            for model in list(models.keys())
        ],
        keys=list(models.keys()),
    )
    all_gw_levels2.to_csv("../data/gwl_1981-2010ref.csv")


### utils for uncertainty notebooks
class CmipOpt:
    """A class for holding the following data options for input to cmip_clip:
        variable: variable name, cf-compliant (or cmip6 variable name)
        area_subset: geographic boundary name (states/counties)
        location: geographic area name (name of county/state)
        timescale: frequency of data
        area_average (bool): average computed across domain
    """
    def __init__(
        self,
        variable="tas",
        area_subset="states",
        location="California",
        timescale="monthly",
        area_average=True,
    ):
        self.variable = variable
        self.area_subset = area_subset
        self.location = location
        self.area_average = area_average
        self.timescale = timescale

    def cmip_clip(self, ds):
    """CMIP6 function to subset dataset based on the data selection options.
    Args:
        (1) ds (xr.Dataset)
    """
        variable = self.variable
        location = self.location
        area_average = self.area_average
        area_subset = self.area_subset
        timescale = self.timescale

        to_drop = [v for v in list(ds.data_vars) if v != variable]
        ds = ds.drop_vars(to_drop)
        ds = _clip_region(ds, area_subset, location)
        if area_average:
            ds = _area_wgt_average(ds)
        return ds


def cf_to_dt(ds):
    """CMIP6 function to convert non-standard calendars using cftime to pandas datetime
    Args:
        (1) ds (xr.Dataset): CMIP6 dataset
    """
    if type(ds.indexes["time"]) not in [pd.core.indexes.datetimes.DatetimeIndex]:
        datetimeindex = ds.indexes["time"].to_datetimeindex()
        ds["time"] = datetimeindex
    return ds


def calendar_align(ds):
    """CMIP6 function to set the day for all monthly values to the the 1st of
    each month, as models have different calendars (e.g., no leap day, 360 days),
    which can result in a large dataset with empty values in time when concatenated.
    WARNING: can impact functions which use number of days in each month (e.g.,
    precipitation flux to total monthly accumulation).
    Args:
        (1) ds (xr.Dataset): CMIP6 dataset
    """
    ds["time"] = pd.to_datetime(ds.time.dt.strftime("%Y-%m"))
    return ds


def clip_region(ds, area_subset, location):
    """Clips CMIP6 dataset using a polygon.
    Args:
        (1) ds (xr.Dataset): input dataset
        (2) area_subset (string): "counties"/"states" as options
        (3) location (string): county/state name
    Optional:
        (1) all_touched (bool): include all cells that intersect boundary.
        Default is False.
    """
    geographies = Boundaries()
    us_states = geographies._us_states
    us_counties = geographies._ca_counties
    ds = ds.rio.write_crs(4326)

    if "counties" in area_subset:
        ds_region = us_counties[us_counties.NAME == location].geometry
    elif "states" in area_subset:
        ds_region = us_states[us_states.NAME == location].geometry

    try:
        ds = ds.rio.clip(geometries=ds_region, crs=4326, drop=True, all_touched=False)
    except:
        # if no grid centers in region instead select all cells which
        # intersect the region
        print("selecting all cells which intersect region")
        ds = ds.rio.clip(geometries=ds_region, crs=4326, drop=True, all_touched=True)
    return ds


def _area_wgt_average(ds):
    """Calculates the weighted area average for CMIP6 model input.
    Args:
        (1) ds (xr.Dataset)
    """
    weights = np.cos(np.deg2rad(ds.y))
    weights.name = "weights"
    ds_weighted = ds.weighted(weights)
    ds = ds_weighted.mean(("x", "y"))
    return ds


def drop_member_id(dset_dict):
    """Drop member_id coordinate/dimensions
    Args:
        dset_dict (dict): dictionary in the format {dataset_name:xr.Dataset}
    """
    for dname, dset in dset_dict.items():
        if "member_id" in dset.coords:
            dset = dset.isel(member_id=0).drop("member_id")  # Drop coord
            dset_dict.update({dname: dset})  # Update dataset in dictionary
    return dset_dict


def cmip_annual(ds):
    """Calculates the annual average temperature across CMIP6 models
    and converts from Kelvin to degC.
    Args:
        ds (xr.Dataset): dataset of CMIP6 models
    """
    ds_degC = ds - 273.15  # convert to degC
    ds_degC = ds_degC.groupby("time.year").mean(dim=["time"])
    return ds_degC


def calc_anom(ds_yr, base_start, base_end):
    """Calculates the temperature change relative to a historical baseline
    (1850-1900) for each model. Returns the difference from the annual
    timeseries and the respective model baseline.
    Args:
        (1) ds_yr (xr.Dataset): must be the output from cmip_annual
        (2) base_start (int): start year of baseline to calculate
        (3) base_end (int): end year of the baseline to calculate
    """
    mdl_baseline = ds_yr.sel(year=slice(base_start, base_end)).mean("year")
    mdl_temp_anom = ds_yr - mdl_baseline
    return mdl_temp_anom


def cmip_mmm(ds):
    """Calculate the CMIP6 multi-model mean by collapsing across simulations.
    Args:
        (1) ds (xr.Dataset): dataset of CMIP6 model output
    """
    ds_mmm = ds.mean("simulation")
    return ds_mmm


def compute_vmin_vmax(da_min, da_max):
    """Computes min, max, and center for plotting.
    Args:
        (1) da_min (xr.Dataset): data input to calculate the minimum
        (2) da_max (xr.Dataset): data input to calculate the maximum
    Returns:
        (1) vmin: minimum value
        (2) vmax: maximum value
        (3) sopt (bool): indicates symmetry if vmin and vmax have opposite signs
    """
    vmin = np.nanpercentile(da_min, 1)
    vmax = np.nanpercentile(da_max, 99)
    # define center for diverging symmetric data
    if (vmin < 0) and (vmax > 0):
        sopt = True
    else:
        sopt = None
    return vmin, vmax, sopt
