"""
Calculates the Average Meterological Year (AMY) and Severe Meteorological Year (SMY) for the Cal-Adapt: Analytics Engine using a standard climatological period (1981-2010) for the historical baseline, and uses a 30-year window around when a designated warming level is exceeded for the SSP3-7.0 future scenario for 1.5°C, 2°C, and 3°C.
The AMY is comparable to a typical meteorological year, but not quite the same full methodology.
"""

## PROCESS: average meteorological year
# for each hour, the average variable over the whole climatological period is determined
# data for that hour that has the value most closely equal (smalleset absolute difference)
# to the hourly average over the whole measurement period is chosen as the AMY data for that hour
# process is repeated for each hour in the year
# repeat values (where multiple years have the same smallest abs value) are
# removed, earliest occurence selected for AMY
# hours are added together to provide a full year of hourly samples

## Produces 3 different kinds of AMY
## 1: Absolute/unbias corrected raw AMY, either historical or warming level-centered future
## 2: Future-minus-historical warming level AMY (see warming_levels)
## 3: Severe AMY based upon historical baseline and a designated threshold/percentile

import numpy as np
import pandas as pd
import xarray as xr
from tqdm.auto import tqdm  # Progress bar

from climakitae.core.data_interface import DataParameters
from climakitae.core.data_load import read_catalog_from_select
from climakitae.util.utils import julianDay_to_date

xr.set_options(keep_attrs=True)  # Keep attributes when mutating xr objects


# =========================== HELPER FUNCTIONS: DATA RETRIEVAL ==============================


def _set_amy_year_inputs(year_start: int, year_end: int) -> tuple[int, int]:
    """
    Helper function for _retrieve_meteo_yr_data.
    Checks that the user has input valid values.
    Sets year end if it hasn't been set; default is 30 year range (year_start + 30). Minimum is 5 year range.
    """
    match year_end:
        case None:
            year_end = (
                year_start + 30 if (year_start + 30 < 2100) else 2100
            )  # Default is +30 years
        case _ if year_end > 2100:
            print("Your end year cannot exceed 2100. Resetting end year to 2100.")
            year_end = 2100
    if year_end - year_start < 5:
        raise ValueError(
            """To compute an Average Meteorological Year, you must input a date range with a difference
            of at least 5 years, where the end year is no later than 2100 and the start year is no later than
            2095."""
        )
    if year_start < 1980:
        raise ValueError(
            """You've input an invalid start year. The start year must be 1980 or later."""
        )
    return (year_start, year_end)


def retrieve_meteo_yr_data(
    data_params: DataParameters,
    ssp: str = None,
    year_start: int = 2015,
    year_end: int = None,
) -> xr.DataArray:
    """Backend function for retrieving data needed for computing a meteorological year.

    Reads in the hourly ensemble means instead of the hourly data.
    Reads in future SSP data, historical climate data, or a combination
    of both, depending on year_start and year_end

    Parameters
    ----------
    self: AverageMetYearParameters
    ssp: str
        one of "SSP 2-4.5", "SSP 3-7.0", "SSP 5-8.5"
        Shared Socioeconomic Pathway. Defaults to SSP 3-7.0
    year_start: int, optional
        Year between 1980-2095. Default to 2015
    year_end: int, optional
        Year between 1985-2100. Default to year_start+30

    Returns
    -------
    xr.DataArray
        Hourly ensemble means from year_start-year_end for the ssp specified.
    """
    # Ensure only WRF data is being used
    data_params.downscaling_method = "Dynamical"

    # Save units. Sometimes they get lost.
    units = data_params.units

    # Check year start and end inputs
    year_start, year_end = _set_amy_year_inputs(year_start, year_end)

    # Set scenario selections
    if (ssp is not None) and (year_end >= 2015):
        data_params.scenario_ssp = [ssp]
    if year_end < 2015:
        data_params.scenario_ssp = []
    elif (year_end >= 2015) and (data_params.scenario_ssp) == []:
        data_params.scenario_ssp = ["SSP 3-7.0"]  # Default
    if year_start < 2015:  # Append historical data
        data_params.scenario_historical = ["Historical Climate"]
    else:
        data_params.scenario_historical = []
    if len(data_params.scenario_ssp) > 1:
        # If multiple SSPs are selected, only use the first one
        data_params.scenario_ssp = [data_params.scenario_ssp[0]]

    # Set other data parameters
    data_params.simulation = ["ensmean"]
    data_params.time_slice = (year_start, year_end)
    data_params.area_average = "Yes"
    data_params.timescale = "hourly"
    data_params.units = units

    # Grab data from the catalog
    amy_data = read_catalog_from_select(data_params)
    if amy_data is None:
        # Catch small spatial resolutions
        raise ValueError(
            "COULD NOT RETRIEVE DATA: For the provided data selections, there is not sufficient data to retrieve. Try selecting a larger spatial area, or a higher resolution. Returning None."
        )
    return amy_data.isel(scenario=0, simulation=0)


# =========================== HELPER FUNCTIONS: AMY/TMY CALCULATION ==============================


def _format_meteo_yr_df(df: pd.DataFrame) -> pd.DataFrame:
    """Format dataframe output from compute_amy and compute_severe_yr"""
    ## Re-order columns for PST, with easy to read time labels
    cols = df.columns.tolist()
    cols = cols[7:] + cols[:7]
    df = df[cols]

    n_col_lst = []
    for ampm in ["am", "pm"]:
        hr_lst = []
        for hr in range(1, 13, 1):
            hr_lst.append(str(hr) + ampm)
        hr_lst = hr_lst[-1:] + hr_lst[:-1]
        n_col_lst = n_col_lst + hr_lst
    df.columns = n_col_lst
    df.columns.name = "Hour"

    # Convert Julian date index to Month-Day format
    # Use 2024 as year if we have 366 days (leap year), otherwise use 2023
    year = 2024 if len(df) == 366 else 2023
    new_index = [
        julianDay_to_date(julday, year=year, str_format="%b-%d") for julday in df.index
    ]
    df.index = pd.Index(new_index, name="Day of Year")
    return df


def compute_amy(data: xr.DataArray, days_in_year: int = 366) -> pd.DataFrame:
    """Calculates the average meteorological year based on a designated period of time

    Applicable for both the historical and future periods.

    Parameters
    ----------
    data: xr.DataArray
        Hourly data for one variable
    days_in_year: int, optional
        Either 366 or 365, depending on whether or not the year is a leap year.
        Default to 366 days (leap year)

    Returns
    -------
    pd.DataFrame
        Average meteorological year table, with days of year as
        the index and hour of day as the columns.
    """

    def _closest_to_mean(dat: xr.DataArray) -> xr.DataArray:
        """Find the value closest to the mean of the data."""
        stacked = dat.stack(allofit=list(dat.dims))
        index = abs(stacked - stacked.mean("allofit")).argmin(dim="allofit").values
        return xr.DataArray(stacked.isel(allofit=index).values)

    def _return_diurnal(y: xr.DataArray) -> xr.DataArray:
        """Return the diurnal cycle of the data."""
        return y.groupby("time.hour").map(_closest_to_mean)

    hourly_da = data.groupby("time.dayofyear").map(_return_diurnal)

    # Funnel data into pandas DataFrame object
    df_amy = pd.DataFrame(
        hourly_da,  # hourly DataArray,
        columns=np.arange(1, 25, 1),
        index=np.arange(1, days_in_year + 1, 1),
    )
    # Format dataframe
    df_amy = _format_meteo_yr_df(df_amy)
    return df_amy


def compute_severe_yr(data, days_in_year=366):
    """Calculate the severe meteorological year based on the 90th percentile of data.

    Applicable for both the historical and future periods.

    Parameters
    ----------
    data: xr.DataArray
        Hourly data for one variable
    days_in_year: int, optional
        Either 366 or 365, depending on whether or not the year is a leap year.
        Default to 366 days (leap year)
    show_pbar: bool, optional
        Show progress bar? Default to false.
        Progress bar is nice for using this function within a notebook.

    Returns
    -------
    pd.DataFrame
        Severe meteorological year table, with days of year as
        the index and hour of day as the columns.
    """

    def closest_to_quantile(dat):
        stacked = dat.stack(allofit=list(dat.dims))
        index = (
            abs(stacked - stacked.quantile(q=0.90, dim="allofit"))
            .argmin(dim="allofit")
            .values
        )
        return xr.DataArray(stacked.isel(allofit=index).values)

    def return_diurnal(y):
        return y.groupby("time.hour").map(closest_to_quantile)

    hourly_da = data.groupby("time.dayofyear").map(return_diurnal)

    ## Funnel data into pandas DataFrame object
    df_severe_yr = pd.DataFrame(
        hourly_da,
        columns=np.arange(1, 25, 1),
        index=np.arange(1, days_in_year + 1, 1),
    )

    # Format dataframe
    df_severe_yr = _format_meteo_yr_df(df_severe_yr)
    return df_severe_yr


# =========================== HELPER FUNCTIONS: MISC ==============================


def compute_mean_monthly_meteo_yr(tmy_df, col_name="mean_value"):
    """Compute mean monthly values for input meteorological year data.

    Parameters
    ----------
    tmy_df: pd.DataFrame
        Matrix with day of year as index and hour as columns
        Output of either compute_severe_yr or compute_meteo_yr
    col_name: str, optional
        Name to give single output column
        It may be informative to assign this to the name of the data variable

    Returns
    -------
    pd.DataFrame
        Table with month as index and monthly mean as column
    """
    # Convert from matrix --> hour and data as individual columns
    tmy_stacked = (
        pd.DataFrame(tmy_df.stack()).rename(columns={0: col_name}).reset_index()
    )
    # Combine Hour and Day of Year to get combined date. Assign as index
    tmy_stacked["Date"] = tmy_stacked["Day of Year"] + " " + tmy_stacked["Hour"]
    tmy_stacked = tmy_stacked.drop(columns=["Day of Year", "Hour"]).set_index("Date")

    # Reformat index to datetime so that you can resample the data monthly
    reformatted_idx = pd.to_datetime(
        ["2024." + idx for idx in tmy_stacked.index], format="%Y.%b-%d %I%p"
    )
    tmy_monthly_mean = tmy_stacked.set_index(reformatted_idx).resample("MS").mean()

    # Reset index to be user-friendly month strings
    tmy_monthly_mean = tmy_monthly_mean.set_index(tmy_monthly_mean.index.strftime("%b"))
    tmy_monthly_mean.index.name = "Month"
    return tmy_monthly_mean
