import datetime

import intake
import numpy as np
import pandas as pd
import rioxarray as rio
import xarray as xr
from scipy import stats
from xmip.preprocessing import rename_cmip6

from climakitae.core.data_interface import DataInterface, DataParameters
from climakitae.core.data_load import area_subset_geometry
from climakitae.core.paths import gwl_1850_1900_file, gwl_1981_2010_file
from climakitae.util.utils import read_csv_file


### Utility functions for uncertainty analyses and notebooks
class CmipOpt:
    """A class for holding relevant data options for cmip preprocessing

    Parameters
    ----------
    variable: str
        variable name, cf-compliant (or cmip6 variable name)
    area_subset: str
        geographic boundary name (states/counties)
    location: str
        geographic area name (name of county/state)
    timescale: str
        frequency of data
    area_average: bool
        average computed across domain

    Methods
    -------
    _cmip_clip
        CMIP6-specific subsetting
    """

    def __init__(
        self,
        variable: str = "tas",  ## set up for temp uncertainty notebook
        area_subset: str = "states",
        location: str = "California",
        timescale: str = "monthly",
        area_average: bool = True,
    ) -> None:
        self.variable = variable
        self.area_subset = area_subset
        self.location = location
        self.timescale = timescale
        self.area_average = area_average

    def _cmip_clip(self, ds: xr.Dataset) -> xr.Dataset:
        """CMIP6 function to subset dataset based on the data selection options.

        Parameters
        ----------
        ds: xr.Dataset
            Input data

        Returns
        -------
        ds: xr.Dataset
            Subsetted data, area-weighting applied if area_average is true
        """
        to_drop = [v for v in list(ds.data_vars) if v != self.variable]
        ds = ds.drop_vars(to_drop)
        ds = _clip_region(ds, self.area_subset, self.location)
        if self.variable == "pr":
            ds = _precip_flux_to_total(ds)
        if self.area_average:
            ds = _area_wgt_average(ds)
        return ds


def _cf_to_dt(ds: xr.Dataset) -> xr.Dataset:
    """Converts non-standard calendars using cftime to pandas datetime

    Parameters
    ----------
    ds: xr.Dataset
        Input data

    Returns
    -------
    ds: xr.Dataset
        Converted calendar data
    """
    if type(ds.indexes["time"]) not in [pd.core.indexes.datetimes.DatetimeIndex]:
        datetimeindex = ds.indexes["time"].to_datetimeindex()
        ds["time"] = datetimeindex
    return ds


def _calendar_align(ds: xr.Dataset) -> xr.Dataset:
    """Aligns calendars for consistent matching

    CMIP6 function to set the day for all monthly values to the the 1st of
    each month, as models have different calendars (e.g., no leap day, 360 days),
    which can result in a large dataset with empty values in time when concatenated.

    WARNING: can impact functions which use number of days in each month (e.g.,
    precipitation flux to total monthly accumulation).

    Parameters
    ----------
    ds: xr.Dataset
        Input data

    Returns
    -------
    ds: xr.Dataset
        Calendar-aligned data
    """
    ds["time"] = pd.to_datetime(ds.time.dt.strftime("%Y-%m"))
    return ds


def _clip_region(ds: xr.Dataset, area_subset: list, location: str) -> xr.Dataset:
    """Clips CMIP6 dataset using a polygon.

    Parameters
    ----------
    ds: xr.Dataset
        Input data
    area_subset: DataParameters.area_subset
        "counties"/"states" as options
    location: str
        county/state name

    Returns
    -------
    xr.Dataset
        Clipped dataset to region of interest
    """
    data_interface = DataInterface()
    geographies = data_interface.geographies
    us_states = geographies._us_states
    us_counties = geographies._ca_counties
    ds = ds.rio.write_crs("epsg:4326", inplace=True)

    match area_subset:
        case area_subset if "counties" in area_subset:
            ds_region = us_counties[us_counties.NAME == location].geometry
        case area_subset if "states" in area_subset:
            ds_region = us_states[us_states.NAME == location].geometry
        case _:
            raise ValueError('area_subset needs to be either "counties" or "states"')

    try:
        ds = ds.rio.clip(geometries=ds_region, crs=4326, drop=True, all_touched=False)
    except:
        # if no grid centers in region instead select all cells which
        # intersect the region
        print("selecting all cells which intersect region")
        ds = ds.rio.clip(geometries=ds_region, crs=4326, drop=True, all_touched=True)
    return ds


def _standardize_cmip6_data(ds: xr.Dataset) -> xr.Dataset:
    """Pre-processing wrapper function.

    First, updates cmip6 dataset names and calendars for consistency.
    Drops latitude, longitude, and height variables, and assigns coordiates.

    Parameters
    ----------
    ds: xr.Dataset
        Input data

    Returns
    -------
    ds: xr.Dataset
        CMIP6 data with consistent dimensions, names, and calendars.
    """

    ds_simulation = ds.attrs["source_id"]
    ds_scenario = ds.attrs["experiment_id"]
    ds_freq = ds.attrs["frequency"]

    ds = rename_cmip6(ds)
    ds = _cf_to_dt(ds)
    if ds_freq in ("mon"):
        ds = _calendar_align(ds)
    ds = ds.drop_vars(["lon", "lat", "height"], errors="ignore")
    ds = ds.assign_coords({"simulation": ds_simulation, "scenario": ds_scenario})
    ds = ds.squeeze(drop=True)
    return ds


def _area_wgt_average(ds: xr.Dataset) -> xr.Dataset:
    """Calculates the weighted area average for CMIP6 model input.

    Parameters
    ----------
    ds: xr.Dataset
        Input data

    Returns
    -------
    ds: xr.Dataset
        Area-averaged data by weights
    """
    weights = np.cos(np.deg2rad(ds.y))
    weights.name = "weights"
    ds_weighted = ds.weighted(weights)
    ds = ds_weighted.mean(("x", "y"))
    return ds


def _drop_member_id(dset_dict: dict) -> xr.Dataset:
    """Drop member_id coordinate/dimensions

    Parameters
    ----------
    dset_dict: dict
        dictionary in the format {dataset_name:xr.Dataset}

    Returns
    -------
    dset_dict: xr.Dataset
        Data, with member_id dim removed
    """
    for dname, dset in dset_dict.items():
        if "member_id" in dset.coords:
            dset = dset.isel(member_id=0).drop_vars("member_id")  # Drop coord
            dset_dict.update({dname: dset})  # Update dataset in dictionary
    return dset_dict


def _precip_flux_to_total(ds: xr.Dataset) -> xr.Dataset:
    """
    converts precip flux units
    (kg m-2 s-1) to total precip
    per month (mm)
    NOTE: assumes regular calendar

    Parameters
    ----------
    ds: xarray.Dataset
        CMIP6 output with data variable 'pr'
    Returns
    -------
    ds: xr.Dataset
        Data with converted precipitation units
    """
    ds_attrs = ds.attrs
    days_month = ds.time.dt.days_in_month
    seconds_month = 86400 * days_month
    ds = ds * seconds_month
    ds = ds.clip(0.1)
    ds.attrs = ds_attrs
    ds.pr.attrs["units"] = "mm"
    return ds


def _grab_ensemble_data_by_experiment_id(
    variable: str, cmip_names: list[str], experiment_id: str
) -> list[xr.Dataset]:
    """Grab CMIP6 ensemble data

    Parameters
    -----------
    variable: str
        Name of variable
    cmip_names: list of str
        Name of CMIP6 simulations
    experiment_id: scenario, one of "historical" or "ssp375"

    Returns
    -------
    list of xr.Dataset
    """

    # Open AE data catalog for regridded CMIP6 data
    col = intake.open_esm_datastore(
        "https://cadcat.s3.amazonaws.com/tmp/cmip6-regrid.json"
    )
    # Get subset of catalog corresponding to user inputs
    col_subset = col.search(
        table_id="Amon",
        variable_id=variable,
        experiment_id=experiment_id,
        source_id=cmip_names,
    )
    # Read data from catalog
    data_dict = col_subset.to_dataset_dict(
        zarr_kwargs={"consolidated": True},
        storage_options={"anon": True},
        preprocess=_standardize_cmip6_data,  # Preprocess function to perform on each DataArray
        progressbar=False,  # Don't show a progress bar in notebook
    )
    return list(data_dict.values())


## Grab data - model uncertainty analysis
def grab_multimodel_data(copt: CmipOpt, alpha_sort: bool = False) -> xr.Dataset:
    """Returns processed data from multiple CMIP6 models for uncertainty analysis.

    Searches the CMIP6 data catalog for data from models that have specific
    ensemble member id in the historical and ssp370 runs. Preprocessing includes
    subsetting for specific location and dropping the member_id for easier
    analysis.

    Parameters
    ----------
    copt: CmipOpt
        Selections: variable, area_subset, location, area_average, timescale
    alpha_sort: bool, default=False
        Set to True if sorting model names alphabetically is desired

    Returns
    -------
    mdls_ds: xr.Dataset
        Processed CMIP6 models concatenated into a single ds
    """
    col = intake.open_esm_datastore(
        "https://cadcat.s3.amazonaws.com/tmp/cmip6-regrid.json"
    )  # data catalog

    # searches catalog for data from the cmip6 archive using our specific data options
    cat = col.search(
        table_id=copt.timescale,
        variable_id=copt.variable,
        experiment_id=[
            "historical",
            "ssp370",
        ],  # identifies models that have both historical and ssp3-7.0 runs
        member_id="r1i1p1f1",  # ensures specific ensemble member 1
        require_all_on="source_id",
    ).search(activity_id=["CMIP", "ScenarioMIP"])

    # grabs the data from the catalog, and processes it using the wrapper function defined above
    dsets = cat.to_dataset_dict(
        zarr_kwargs={"consolidated": True},
        storage_options={"anon": True},
        preprocess=_standardize_cmip6_data,
    )

    # searches the catalog for the additional cal-adapt simulations
    if "pr" in copt.variable:
        paths = [
            "CESM2.*r11i1p1f1",
            "CNRM-ESM2-1.*r1i1p1f2",
            "MPI-ESM1-2-LR.*r7i1p1f1",
        ]  # note, three of the Cal-Adapt models (precip only) use a different ensemble member
    else:
        paths = [
            "CESM2.*r11i1p1f1",
            "CNRM-ESM2-1.*r1i1p1f2",
        ]  # note, two of the Cal-Adapt models (temperature) use a different ensemble member

    cat = col.search(
        table_id=copt.timescale,
        variable_id=copt.variable,
        path=paths,
        activity_id=["CMIP", "ScenarioMIP"],
    )

    # grabs the cal-adapt simulations from the catalog, and processes it using the wrapper function
    cal_dsets = cat.to_dataset_dict(
        zarr_kwargs={"consolidated": True},
        storage_options={"anon": True},
        preprocess=_standardize_cmip6_data,
    )

    # subsets the cmip6 and cal-adapt models in the historical period
    hist_dsets = {key: val for key, val in dsets.items() if "historical" in key}
    cal_hist_dsets = {key: val for key, val in cal_dsets.items() if "historical" in key}

    # subsets the cmip6 and cal-adapt models in the future (ssp370) period
    ssp_dsets = {key: val for key, val in dsets.items() if "ssp370" in key}
    cal_ssp_dsets = {key: val for key, val in cal_dsets.items() if "ssp370" in key}

    # drop member id, in order to ensure that merging is along the same data axis
    hist_dsets = _drop_member_id(hist_dsets)
    cal_hist_dsets = _drop_member_id(cal_hist_dsets)
    ssp_dsets = _drop_member_id(ssp_dsets)
    cal_ssp_dsets = _drop_member_id(cal_ssp_dsets)

    # merge datasets together
    all_hist_mdls = hist_dsets | cal_hist_dsets
    all_ssp_mdls = ssp_dsets | cal_ssp_dsets

    if alpha_sort:
        # sort models alphabetically
        all_hist_mdls = dict(
            sorted(all_hist_mdls.items(), key=lambda x: x[0].split(".")[2])
        )
        all_ssp_mdls = dict(
            sorted(all_ssp_mdls.items(), key=lambda x: x[0].split(".")[2])
        )

    # concatenate historical data based on the model, and subset for California
    hist_ds = xr.concat(list(all_hist_mdls.values()), dim="simulation").squeeze()
    hist_ds = copt._cmip_clip(
        hist_ds.sel(time=slice("1850", "2014"))
    )  # time slice ensures the same historical timeframe
    ssp_ds = xr.concat(list(all_ssp_mdls.values()), dim="simulation").squeeze()
    ssp_ds = copt._cmip_clip(ssp_ds)

    # concatenate all data together based on model
    mdls_ds = xr.concat(
        [hist_ds, ssp_ds],
        dim="time",
        coords="minimal",
        compat="override",
        join="inner",
    )

    return mdls_ds


## Grab data - internal variability analysis
def get_ensemble_data(
    variable: str,
    selections: DataParameters,
    cmip_names: list[str],
    warm_level: float = 3.0,
):
    """Returns processed data from multiple CMIP6 models for uncertainty analysis.

    Searches the CMIP6 data catalog for data from models that have specific
    ensemble member id in the historical and ssp370 runs. Preprocessing includes
    subsetting for specific location and dropping the member_id for easier
    analysis.

    Get's future data at warming level range. Slices historical period to 1981-2010.

    Parameters
    -----------
    variable: str
        Name of variable
    selections: _DataSelector
        Data and location settings
    cmip_names: list of str
        Name of CMIP6 simulations
    warm_level: float, optional
        Global warming level to use, default to 3.0

    Returns
    -------
    hist_ds, warm_ds: list of xr.Dataset

    """
    # Get a list of datasets, each with one simulation (i.e. one dataset with several member_id values for CESM2, etc)
    ssp_list = _grab_ensemble_data_by_experiment_id(variable, cmip_names, "ssp370")
    hist_list = _grab_ensemble_data_by_experiment_id(variable, cmip_names, "historical")

    # Reorder lists to match order of cmip_names
    hist_list_reordered = [
        ds for sim in cmip_names for ds in hist_list if ds.simulation.item() == sim
    ]
    ssp_list_reordered = [
        ds for sim in cmip_names for ds in ssp_list if ds.simulation.item() == sim
    ]

    # Get each simulation/member_id unique combo
    warm_ravel, hist_ravel = [], []
    for hist_ds, ssp_ds in zip(hist_list_reordered, ssp_list_reordered):
        # First, get each dataset by one simulation and one member ID for the historical data

        # Next, using the SSP dataset, computing the data at a particular warming level, for each simulation/member_id combo
        warm_ravel += [
            get_warm_level(
                warm_level, ssp_ds.sel(member_id=m), multi_ens=True, ipcc=False
            )
            for m in ssp_ds.member_id.values
        ]
        hist_ravel += [hist_ds.sel(member_id=m) for m in ssp_ds.member_id.values]

    # Concatenate the lists along the member_id dimension to get a single xr.Dataset
    hist_ds = xr.concat(hist_ravel, dim="member_id")

    # print(warm_ravel)
    warm_ravel = list(filter(lambda item: item is not None, warm_ravel))
    warm_ds = xr.concat(warm_ravel, dim="member_id")

    # ensure that we have the same members for both ds
    warm_m_ids = warm_ds.member_id.values
    warm_sim_mem = list(zip(warm_ds.simulation.values, warm_m_ids))
    warm_combo_ids = [s + m for s, m in warm_sim_mem]
    # list of unique identifiers
    # this is needed to take the difference between projected and historical

    hist_sim_mem = list(zip(hist_ds.simulation.values, hist_ds.member_id.values))
    hist_combo_ids = [s + m for s, m in hist_sim_mem]
    hist_ds.coords["member_id"] = hist_combo_ids
    # assigns unique identifiers to hist_ds

    hist_ds = hist_ds.sel(member_id=warm_combo_ids)
    hist_ds.coords["member_id"] = warm_m_ids
    # reassigns the old (normal) member_ids

    # Time slice historical period
    hist_ds = hist_ds.sel(time=slice("1981", "2010"))

    # Post-processing functions to perform on both datasets
    def _postprocess(ds, selections, variable):
        """Subset the dataset by an input location, convert variables, perform area averaging"""
        # Perform area subsetting
        ds_region = area_subset_geometry(selections)
        ds = ds.rio.write_crs("epsg:4326", inplace=True)
        ds = ds.rio.clip(geometries=ds_region, crs=4326, drop=True)

        # Convert to mm/mon for precip data
        if variable == "pr":
            ds = _precip_flux_to_total(ds)

        # Perform area averaging
        if selections.area_average == "Yes":
            ds = _area_wgt_average(ds)
        return ds

    hist_ds = _postprocess(hist_ds, selections, variable)
    warm_ds = _postprocess(warm_ds, selections, variable)

    return hist_ds, warm_ds


## -----------------------------------------------------------------------------
## Useful individual analysis functions


def weighted_temporal_mean(ds: xr.DataArray) -> xr.DataArray:
    """weight by days in each month

    Function for calculating annual averages pulled + adapted from NCAR
    Link: https://ncar.github.io/esds/posts/2021/yearly-averages-xarray/

    Parameters
    ----------
    ds: xarray.DataArray

    Returns
    -------
    obs_sum / ones_out : xarray.Dataset
    """

    # Determine the month length
    month_length = ds.time.dt.days_in_month

    # Calculate the weights
    wgts = month_length.groupby("time.year") / month_length.groupby("time.year").sum()

    # Make sure the weights in each year add up to 1
    np.testing.assert_allclose(wgts.groupby("time.year").sum(xr.ALL_DIMS), 1.0)

    # Setup our masking for nan values
    cond = ds.isnull()
    ones = xr.where(cond, 0.0, 1.0)

    # Calculate the numerator
    obs_sum = (ds * wgts).resample(time="YS").sum(dim="time")

    # Calculate the denominator
    ones_out = (ones * wgts).resample(time="YS").sum(dim="time")

    # Calculate weighted average
    weighted_avg = obs_sum / ones_out

    # Setting time array to the year
    weighted_avg["time"] = weighted_avg.time.dt.year

    return weighted_avg


def calc_anom(ds_yr: xr.Dataset, base_start: int, base_end: int) -> xr.Dataset:
    """Calculates the difference relative to a historical baseline.

    First calculates a baseline per simulation using input (base_start, base_end).
    Then calculates the anomaly from baseline per simulation.

    Parameters
    ----------
    ds_yr: xr.Dataset
        must be the output from cmip_annual
    base_start: int
        start year of baseline to calculate
    base_end: int
        end year of the baseline to calculate

    Returns
    -------
    mdl_temp_anom: xr.Dataset
        Anomaly data calculated with input baseline start and end
    """
    mdl_baseline = ds_yr.sel(time=slice(base_start, base_end)).mean("time")
    mdl_temp_anom = ds_yr - mdl_baseline
    return mdl_temp_anom


def cmip_mmm(ds: xr.Dataset) -> xr.Dataset:
    """Calculate the CMIP6 multi-model mean by collapsing across simulations.

    Parameters
    ----------
    ds: xr.Dataset
        Input data, multiple simulations

    Returns
    -------
    ds_mmm: xr.Dataset
        Mean across input data taken on simulation dim
    """
    ds_mmm = ds.mean("simulation")
    return ds_mmm


# TODO check whether this function actually works (it seems like it doesn't)
def get_ks_pval_df(
    sample1: xr.Dataset, sample2: xr.Dataset, sig_lvl: float = 0.05
) -> pd.DataFrame:
    """Performs a Kolmogorov-Smirnov test at all lat, lon points

    Parameters
    ----------
    sample1: xr.Dataset
        first sample for comparison
    sample2: xr.Dataset
        sample against which to compare sample1
    sig_lvl: float
        alpha level for statistical significance

    Returns
    -------
    p_df: pd.DataFrame
        columns are lat, lon, and p_value;
        only retains spatial points where
        p_value < sig_lvl
    """

    sample1 = sample1.stack(allpoints=["y", "x"]).squeeze().groupby("allpoints")
    sample2 = sample2.stack(allpoints=["y", "x"]).squeeze().groupby("allpoints")

    def ks_stat_2sample(sample1, sample2):
        try:
            ks = stats.kstest(sample1, sample2)
            d_statistic = ks[0]
            p_value = ks[1]
        except (ValueError, ZeroDivisionError):
            d_statistic = np.nan
            p_value = np.nan

        return d_statistic, p_value

    _, p_value = xr.apply_ufunc(
        ks_stat_2sample,
        sample1,
        sample2,
        input_core_dims=[["index"], ["index"]],
        exclude_dims=set(("index",)),
        output_core_dims=[[], []],
    )

    p_df = p_value.dropna(dim="allpoints")
    p_df = p_value.rename("p_value")
    p_df = p_df.unstack("allpoints")
    p_df = p_df.to_dataframe().reset_index()
    p_df = p_df[["lat", "lon", "p_value"]]
    p_df = p_df.loc[:, ["lon", "lat", "p_value"]]
    p_df = p_df[p_df["p_value"] < sig_lvl]

    return p_df


def get_warm_level(
    warm_level: float | int, ds: xr.Dataset, multi_ens: bool = False, ipcc: bool = True
) -> xr.Dataset:
    """Subsets projected data centered to the year
    that the selected warming level is reached
    for a particular simulation/member_id

    Parameters
    ----------
    warm_level : float or int
        options: 1.5, 2.0, 3.0, 4.0
    ds : xr.Dataset
        Can only have one 'simulation' coordinate
    multi_ens : bool, default=False
        Set to True if passing a simulation with multiple member_id
    ipcc : bool, default=True
        Set to False if not performing warming level analysis with
        respect to IPCC standard baseline (1850-1900)

    Returns
    -------
    xr.Dataset
        Subset of projected data -14/+15 years from warming level threshold
    """
    try:
        warm_level = float(warm_level)
    except ValueError:
        raise ValueError("Please specify warming level as an integer or float.")

    if warm_level not in [1.5, 2.0, 3.0, 4.0]:
        raise ValueError(
            "Specified warming level is not valid. Options are: 1.5, 2.0, 3.0, 4.0"
        )

    if ipcc:
        gwl_file = gwl_1850_1900_file
        gwl_times = read_csv_file(gwl_file, index_col=[0, 1, 2])
    else:
        gwl_file_all = gwl_1981_2010_file
        gwl_times_all = read_csv_file(gwl_file_all)
        # TODO Add information on a more complete list of ensemble members of
        # EC-Earth3 to cover internal variability notebook needs
        gwl_file_ece3 = "data/gwl_1981-2010ref_EC-Earth3_ssp370.csv"
        gwl_times_ece3 = read_csv_file(gwl_file_ece3)
        gwl_times = (
            pd.concat([gwl_times_all, gwl_times_ece3])
            .drop_duplicates()
            .set_index(["GCM", "run", "scenario"])
        )

    # grab the ensemble members specific to our needs here
    sim_idx = []
    scenario = "ssp370"
    model = str(ds.simulation.values)
    if model in gwl_times.index:
        if multi_ens:
            member_id = str(ds["member_id"].values)
        else:
            match model:
                case "CESM2":
                    member_id = "r11i1p1f1"
                case "CNRM-ESM2-1":
                    member_id = "r1i1p1f2"
                case _:
                    member_id = "r1i1p1f1"
        sim_idx = (model, member_id, scenario)

        # identify the year that the selected warming level is reached for each ensemble member
        year_warmlevel_reached = str(gwl_times[str(warm_level)].loc[sim_idx])[:4]
        if len(year_warmlevel_reached) != 4:
            print(
                "{}°C warming level not reached for ensemble member {} of model {}".format(
                    warm_level, member_id, model
                )
            )
        else:
            if (int(year_warmlevel_reached) + 15) > 2100:
                print(
                    "End year for SSP time slice occurs after 2100;"
                    + " skipping ensemble member {} of model {}".format(
                        member_id, model
                    )
                )
            else:
                year0 = str(int(year_warmlevel_reached) - 14)
                year1 = str(int(year_warmlevel_reached) + 15)
                return ds.sel(time=slice(year0, year1))
