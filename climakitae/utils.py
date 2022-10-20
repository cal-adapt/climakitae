import numpy as np
import xarray as xr
import rioxarray as rio
import numpy as np 
import pandas as pd
import s3fs
import intake
import matplotlib.colors as mcolors
import matplotlib
import pkg_resources

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

    cmap_filename = cmap+".txt" # Filename of colormap
    cmap_pkg_data = pkg_resources.resource_filename("climakitae", "data/cmaps/" + cmap_filename) # Read package data
    cmap_np = np.loadtxt(cmap_pkg_data, dtype = float)

    # RBG to hex
    if cmap_hex:
        cmap_data = [matplotlib.colors.rgb2hex(color) for color in cmap_np]
    else:
        cmap_data = mcolors.LinearSegmentedColormap.from_list(cmap, cmap_np, N = 256)

    return cmap_data

def _reproject_data(xr_da, proj = "EPSG:4326", fill_value = np.nan):
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

    def _reproject_data_4D(data, reproject_dim, proj = "EPSG:4326", fill_value = np.nan):
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
            dp_i = data[i].rio.reproject(proj, nodata = fill_value) # Reproject each index in that dimension
            rp_list.append(dp_i)
        data_reprojected = xr.concat(rp_list, dim = reproject_dim) # Concat along reprojection dim to get entire dataset reprojected
        return data_reprojected

    def _reproject_data_5D(data, reproject_dim, proj = "EPSG:4326", fill_value = np.nan):
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
                dp_i = data[j, i].rio.reproject(proj, nodata = fill_value) # Reproject each index in that dimension
                rp_list_i.append(dp_i)
            data_reprojected_i = xr.concat(rp_list_i, dim = reproject_dim_i) # Concat along reprojection dim to get entire dataset reprojected 
            rp_list_j.append(data_reprojected_i)
        data_reprojected = xr.concat(rp_list_j, dim = reproject_dim_j)
        return data_reprojected

    # Raise error if data doesn't have spatial dimensions x,y
    if not set(["x", "y"]).issubset(xr_da.dims): 
        raise ValueError("Input DataArray cannot be reprojected because it does not contain spatial dimensions x,y")

    # Drop non-dimension coords. Will cause error with rioxarray
    coords = [coord for coord in xr_da.coords if coord not in xr_da.dims]
    data = xr_da.drop_vars(coords)

    # Re-write crs to data using original dataset
    data = data.rio.write_crs(xr_da.rio.crs)

    # Get non-spatial dimensions
    non_spatial_dims = [dim for dim in data.dims if dim not in ["x", "y"]]

    # 2 or 3D DataArray
    if len(data.dims) <= 3:
        data_reprojected = data.rio.reproject(proj, nodata = fill_value)
    # 4D DataArray
    elif len(data.dims) == 4:
        data_reprojected = _reproject_data_4D(
            data = data,
            reproject_dim = non_spatial_dims[0],
            proj = proj,
            fill_value = fill_value
        )
    # 5D DataArray
    elif len(data.dims) == 5:
        data_reprojected = _reproject_data_5D(
            data = data,
            reproject_dim = non_spatial_dims[: -1],
            proj = proj,
            fill_value = fill_value
        )
    else:
        raise ValueError("DataArrays with dimensions greater than 5 are not currently supported")

    # Reassign attribute to reflect reprojection
    data_reprojected.attrs["grid_mapping"] = proj

    return data_reprojected

# Read csv file containing variable information as dictionary
def _read_var_csv(
    csv_file,
    index_col = "name",
    usecols = [
        "name",
        "description",
        "extended_description",
        "native_unit",
        "alt_unit_options",
        "default_cmap"
    ]
):
    """Read in variable descriptions csv file as a dictionary

    Args:
        csv_file (str): Local path to variable csv file
        index_col (str): Column in csv to use as keys in dictionary

    Returns:
        descrip_dict (dictionary): Dictionary containing index_col as keys and additional columns as values

    """
    # Print warning if user inputs invalid index column
    if index_col in ["native_unit", "alt_unit_options", "default_cmap"]:
        print("Index column must have unique values. Cannot set index_col to " + index_col + ". Setting index to 'name'.")
        index_col = "name"

    # Read in csv and return as dictionary
    csv = pd.read_csv(csv_file, index_col = index_col, usecols = usecols)
    descrip_dict = csv.to_dict(orient = "index")
    return descrip_dict

### some utils for generating warming level reference data in ../data/ ###
def write_gwl_files():
    """Call everything needed to write the global warming level reference files
    for all of the currently downscaled GCMs."""

    # Connect to AWS S3 storage
    fs = s3fs.S3FileSystem(anon = True)

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
                data_historical = data_historical.isel(time = slice(0, -12 * 2))

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
                timeseries = timeseries.sortby("time") # needed for MPI-ESM1-2-LR
                data_one_model[scenario] = xr.concat(
                    [data_historical, timeseries], dim = "time"
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
        variable, model, scenarios, start_year = "18500101", end_year = "19000101"
    ):
        """Loops through global warming levels, and returns an aggregate table
        for all warming levels (1.5, 2, 3, and 4 degrees) for all scenarios of
        the model/variant requested."""
        ens_mem = models[model]
        data_one_model = build_timeseries(variable, model, ens_mem, scenarios)
        anom = (
            data_one_model - data_one_model.sel(time = slice(start_year, end_year)).mean()
        )
        smoothed = anom.rolling(time = 20 * 12, center = True).mean("time")
        one_model = (
            smoothed.to_array(dim = "scenario", name = model).dropna("time").to_pandas()
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
        keys = list(models.keys()),
    )
    all_gw_levels.to_csv("../data/gwl_1850-1900ref.csv")

    start_year = "19810101"
    end_year = "20101231"
    all_gw_levels2 = pd.concat(
        [
            get_gwl_table(variable, model, scenarios, start_year, end_year)
            for model in list(models.keys())
        ],
        keys = list(models.keys()),
    )
    all_gw_levels2.to_csv("../data/gwl_1981-2010ref.csv")
