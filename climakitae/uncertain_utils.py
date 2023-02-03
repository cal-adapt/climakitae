import numpy as np
import datetime
import xarray as xr
import pyproj
import rioxarray as rio
import pandas as pd
import intake
import warnings
from .selectors import Boundaries
from cmip6_preprocessing.preprocessing import rename_cmip6


### Utility functions for uncertainty analyses and notebooks
class CmipOpt:
    """A class for holding relevant data options for cmip preprocessing

    Attributes
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
    cmip_clip
        CMIP6-specific subsetting
    """

    def __init__(
        self,
        variable="tas",  ## set up for temp uncertainty notebook
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

    def _cmip_clip(self, ds):
        """CMIP6 function to subset dataset based on the data selection options.

        Parameters
        ----------
        ds: xr.Dataset
            Input data

        Returns
        -------
        xr.Dataset
            Subsetted data, area-weighting applied if area_average is true
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


def _cf_to_dt(ds):
    """Converts non-standard calendars using cftime to pandas datetime

    Parameters
    ----------
    ds: xr.Dataset
        Input data

    Returns
    -------
    xr.Dataset
        Converted calendar data
    """
    if type(ds.indexes["time"]) not in [pd.core.indexes.datetimes.DatetimeIndex]:
        datetimeindex = ds.indexes["time"].to_datetimeindex()
        ds["time"] = datetimeindex
    return ds


def _calendar_align(ds):
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
    xr.Dataset
        Calendar-aligned data
    """
    ds["time"] = pd.to_datetime(ds.time.dt.strftime("%Y-%m"))
    return ds


def _clip_region(ds, area_subset, location):
    """Clips CMIP6 dataset using a polygon.

    Parameters
    ----------
    ds: xr.Dataset
        Input data
    area_subset: LocSelectorArea
        "counties"/"states" as options
    location: LocSelectorArea
        county/state name
    all_touched: bool, optional
        Include all cells that intersect boundary, default is false

    Returns
    -------
    xr.Dataset
        Clipped dataset to region of interest
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


def _wrapper(ds):
    """Pre-processing wrapper function.

    First, updates cmip6 dataset names and calendars for consistency.
    Drops latitude, longitude, and height variables, and assigns coordiates.

    Parameters
    ----------
    ds: xr.Dataset
        Input data

    Returns
    -------
    xr.Dataset
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


def _area_wgt_average(ds):
    """Calculates the weighted area average for CMIP6 model input.

    Parameters
    ----------
    ds: xr.Dataset
        Input data

    Returns
    -------
    xr.Dataset
        Area-averaged data by weights
    """
    weights = np.cos(np.deg2rad(ds.y))
    weights.name = "weights"
    ds_weighted = ds.weighted(weights)
    ds = ds_weighted.mean(("x", "y"))
    return ds


def _drop_member_id(dset_dict):
    """Drop member_id coordinate/dimensions

    Parameters
    ----------
    dset_dict: dict
        dictionary in the format {dataset_name:xr.Dataset}

    Returns
    -------
    xr.Dataset
        Data, with member_id dim removed
    """
    for dname, dset in dset_dict.items():
        if "member_id" in dset.coords:
            dset = dset.isel(member_id=0).drop("member_id")  # Drop coord
            dset_dict.update({dname: dset})  # Update dataset in dictionary
    return dset_dict


## Grab temperature data - model uncertainty analysis (explore_model_uncertainty nb)
def grab_temp_data(copt):
    """Returns processed data from multiple CMIP6 models for uncertainty analysis.

    Searches the CMIP6 data catalog for data from models that have specific
    ensemble member id in the historical and ssp370 runs. Preprocessing includes
    subsetting for specific location and dropping the member_id for easier
    analysis.

    Attributes
    ----------
    copt: CmipOpt
        Selections for variable, area_subset, location, area_average, timescale

    Returns
    -------
    xr.Dataset
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
        preprocess=_wrapper,
    )

    # searches the catalog for the additional cal-adapt simulations
    paths = [
        "CESM2.*r11i1p1f1",
        "CNRM-ESM2-1.*r1i1p1f2",
    ]  # note, two of the Cal-Adapt models use a different ensemble member
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
        preprocess=_wrapper,
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


## -----------------------------------------------------------------------------
## Useful individual analysis functions


def cmip_annual(ds):
    """Calculates the annual average temperature timeseries in degC from monthly data.

    Note: as is, is specified for temperature in order to convert units. Can be
    generalized.

    Parameters
    ----------
    ds: xr.Dataset
        Input data, default temperature unit is K

    Returns
    -------
    xr.Dataset
        Annual temperature timeseries in degC
    """
    ds_degC = ds - 273.15  # convert to degC
    ds_degC = ds_degC.groupby("time.year").mean(dim=["time"])
    return ds_degC


def calc_anom(ds_yr, base_start, base_end):
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
    xr.Dataset
        Anomaly data calculated with input baseline start and end
    """
    mdl_baseline = ds_yr.sel(year=slice(base_start, base_end)).mean("year")
    mdl_temp_anom = ds_yr - mdl_baseline
    return mdl_temp_anom


def cmip_mmm(ds):
    """Calculate the CMIP6 multi-model mean by collapsing across simulations.

    Parameters
    ----------
    ds: xr.Dataset
        Input data, multiple simulations

    Returns
    -------
    xr.Dataset
        Mean across input data taken on simulation dim
    """
    ds_mmm = ds.mean("simulation")
    return ds_mmm


def compute_vmin_vmax(da_min, da_max):
    """Computes min, max, and center for plotting.

    Parameters
    ----------
    da_min: xr.Dataset
        data input to calculate the minimum
    da_max: xr.Dataset
        data input to calculate the maximum

    Returns
    -------
    int
        minimum value
    int
        maximum value
    bool
        indicates symmetry if vmin and vmax have opposite signs
    """
    vmin = np.nanpercentile(da_min, 1)
    vmax = np.nanpercentile(da_max, 99)
    # define center for diverging symmetric data
    if (vmin < 0) and (vmax > 0):
        sopt = True
    else:
        sopt = None
    return vmin, vmax, sopt
