"""Backend for agnostic tools."""

import numpy as np
import pandas as pd
from dask import compute
import xarray as xr
import intake
from climakitae.core.data_interface import (
    DataInterface,
    DataParameters,
    _get_variable_options_df,
    _get_user_options,
)
from climakitae.util.utils import read_csv_file, get_closest_gridcell
from climakitae.core.paths import variable_descriptions_csv_path, data_catalog_url
from climakitae.util.unit_conversions import get_unit_conversion_options
from typing import Union, Tuple
from climakitae.core.data_load import load
from climakitae.util.logger import logger
from climakitae.core.constants import SSPS
import warnings

warnings.simplefilter(action="ignore", category=FutureWarning)


def create_lookup_tables():
    """Create lookup tables for converting between warming level and time.

    Returns
    -------
    dict of pandas.DataFrame
        A dictionary containing two dataframes: "time lookup table" which maps
        warming levels to their occurence times for each GCM simulation we
        catalog, and "warming level lookup table" which contains yearly warming
        levels for those simulations.
    """
    # Find the names of all the GCMs that we catalog
    data_interface = DataInterface()
    gcms = data_interface.data_catalog.df.source_id.unique()

    time_df = _create_time_lut(gcms)
    warm_df = _create_warm_level_lut(gcms)

    return {"time lookup table": time_df, "warming level lookup table": warm_df}


def _create_time_lut(gcms):
    """Prepare lookup table for converting warming levels to times."""
    # Read in simulation vs warming levels table
    df = read_csv_file("data/gwl_1850-1900ref.csv")
    # Subset to cataloged GCMs
    df = df[df["GCM"].isin(gcms)]

    df.dropna(
        axis="rows", how="all", subset=["1.5", "2.0", "3.0", "4.0"], inplace=True
    )  # model EC-Earth3 runs/simulations
    return df


def _create_warm_level_lut(gcms):
    """Prepare lookup table for converting times to warming levels."""
    # Read in time vs simulation table
    df = read_csv_file(
        "data/gwl_1850-1900ref_timeidx.csv", index_col="time", parse_dates=True
    )
    # Subset time to 2021-2089
    subset_rows = (df.index.year > 2020) & (df.index.year < 2090)
    # Subset to cataloged GCMs and scenario "ssp370"
    subset_columns = [
        col
        for col in df.columns
        if col.split("_")[0] in gcms and col.endswith("ssp370")
    ]
    df = df.loc[subset_rows, subset_columns]

    df.dropna(
        axis="columns", how="all", inplace=True
    )  # model EC-Earth3 runs/simulations

    year_df = df.resample("Y").mean()
    year_df.index = year_df.index.year  # Drop 12-31
    return year_df


def warm_level_to_month(time_df, scenario, warming_level):
    """Given warming level, give month."""
    med_date = (
        time_df[time_df["scenario"] == scenario][warming_level]
        .astype("datetime64[ns]")
        .quantile(0.5, interpolation="midpoint")
    )
    return med_date.strftime("%Y-%m")


def year_to_warm_levels(warm_df, scenario, year):
    """Given year, give warming levels and their median."""
    warm_levels = warm_df.loc[year]
    med_level = np.quantile(warm_levels, 0.5, interpolation="midpoint")
    return warm_levels, med_level


##### TASK 2 #####


def _round_to_nearest_half(number):
    return round(number * 2) / 2


def _get_var_info(variable, downscaling_method, wrf_timescale="monthly"):
    """Gets the variable info for the specific variable name and downscaling method"""
    var_desc_df = read_csv_file(variable_descriptions_csv_path)
    _validate_timescale(wrf_timescale)
    timescale = wrf_timescale if downscaling_method == "Dynamical" else "monthly"
    return var_desc_df[
        (var_desc_df["display_name"] == variable)
        & (var_desc_df["timescale"].str.contains(timescale))
        & (var_desc_df["downscaling_method"] == downscaling_method)
    ]


def get_available_units(variable, downscaling_method, wrf_timescale="monthly"):
    """Get other available units available for the given unit"""
    # Select your desired units
    var_info_df = _get_var_info(variable, downscaling_method, wrf_timescale)
    if var_info_df.empty:
        raise ValueError(
            "Please input a valid variable for the given downscaling method."
        )
    available_units = get_unit_conversion_options()[var_info_df["unit"].item()]
    return available_units


def _complete_selections(selections, variable, units, years, wrf_timescale="monthly"):
    """Completes the attributes for the `selections` objects from `create_lat_lon_select` and `create_cached_area_select`."""
    metric_info_df = _get_var_info(variable, selections.downscaling_method)
    selections.data_type = "Gridded"
    selections.variable = metric_info_df["display_name"].item()
    selections.scenario_historical = ["Historical Climate"]

    # If we want to allow users to select on criteria beyond just the metric and downscaling (i.e. also timescale and resolution), then the following line will be useful to present users
    # print(variable_description_df[['display_name', 'downscaling_method', 'timescale']].to_string())
    match selections.downscaling_method:
        case "Statistical":
            selections.timescale = "monthly"
        case "Dynamical":
            selections.timescale = wrf_timescale
        case _:
            raise ValueError(
                'downscaling_method needs to be either "Statistical" or "Dynamical"'
            )
    selections.resolution = "3 km"
    selections.units = units
    selections.time_slice = years
    return selections


def _create_lat_lon_select(
    lat, lon, variable, downscaling_method, units, years, wrf_timescale="monthly"
):
    """Creates a selection object for the given lat/lon parameters."""
    # Creates a selection object
    selections = DataParameters()
    selections.area_subset = "lat/lon"
    match lat:
        case float() | int():
            # Creating a box around which to find the nearest gridcell for compute
            selections.latitude = (lat - 0.05, lat + 0.05)
            selections.longitude = (lon - 0.05, lon + 0.05)
        case tuple():
            selections.latitude = lat
            selections.longitude = lon
            selections.area_average = "Yes"
        case _:
            raise Exception(
                "lat coordinate not of the correct type of float, int, or tuple"
            )
    selections.downscaling_method = downscaling_method

    # Add attributes for the rest of the selections object
    selections = _complete_selections(selections, variable, units, years, wrf_timescale)
    return selections


def _create_cached_area_select(
    area_subset,
    cached_area,
    variable,
    downscaling_method,
    units,
    years,
    wrf_timescale="monthly",
):
    """Creates a selection object for the given cached area parameters."""
    # Creates a selection object for area subsetting simulations
    selections = DataParameters()
    selections.area_subset = area_subset
    selections.cached_area = [cached_area]
    selections.downscaling_method = downscaling_method
    selections.area_average = "Yes"

    # Add attributes for the rest of the selections object
    selections = _complete_selections(selections, variable, units, years, wrf_timescale)

    return selections


def _compute_results(selections, agg_func, years, months):
    """
    Retrieves selections object data from all SSP 2-4.5, 3-7.0, 5-8.5 pathways, aggregates the simulations together,
    and computes the passed in metric on all the simulations.
    """
    # Aggregating all simulations across all SSP pathways
    all_data = []

    # V0.1: Only allow specific SSPs for different `3 km` applications.
    available_ssps = {
        "Statistical": SSPS,
        "Dynamical": [
            "SSP 3-7.0",
        ],
    }
    logger.debug("Retrieving datasets")
    for ssp in available_ssps[selections.downscaling_method]:
        selections.scenario_ssp = [ssp]
        selections.time_slice = years  # Must re-instantiate `time_slice` with every `scenario_ssp` change because `time_slice` gets reset.

        # Retrieve data
        ssp_data = selections.retrieve()

        # Renaming simulations so that they can be concatenated together correctly
        ssp_data["simulation"] = (
            ssp_data.simulation.astype("object")
            + ", "
            + ssp_data.scenario.item().split("--")[0].split("+")[1].strip()
        )

        # Combining different SSP's simulations together to be aggregated into one xr.DataArray
        all_data.append(ssp_data.squeeze())

    data = xr.concat(all_data, dim="simulation")
    data = data.sel(time=data["time"][data.time.dt.month.isin(months)])

    # Dealing with different cases of downscaling method and lat/lon types
    if selections.area_subset == "lat/lon":
        if selections.area_average == "Yes":  # Means lat/lon range was given
            data.attrs["lat"] = selections.latitude
            data.attrs["lon"] = selections.longitude

        else:  # Otherwise, find the closest gridcell
            if "Statistical" in selections.downscaling_method:
                # Manually finding nearest gridcell for LOCA data, or data with lat/lon coords already.
                data = data.sel(
                    lat=np.mean(selections.latitude),
                    lon=np.mean(selections.longitude),
                    method="nearest",
                )
            else:
                data = get_closest_gridcell(
                    data, np.mean(selections.latitude), np.mean(selections.longitude)
                )

    # Calculate the given metric on the data
    calc_vals = data.groupby("simulation").map(agg_func).chunk(chunks="auto")

    # Sorting sims and getting metrics
    logger.debug("Loading data and computing aggregation")
    loaded_sims = load(calc_vals)
    sorted_sims = loaded_sims.sortby(loaded_sims)
    # sorted_sims = load(calc_vals.sortby(calc_vals)[0])  # Need all the values in order to create histogram + return values

    return sorted_sims, data


def _split_stats(sims, data):
    """
    Takes calculated simulations and creates different dictionaries to describe statistics about the simulations
    Ex. single model compute = min, q1, median, q3, max
    multi-model = middle 10%
    """
    single_model_names = {
        "min": sims[sims.argmin()].simulation.item(),
        "q1": sims.loc[sims == sims.quantile(0.25, method="nearest")].simulation.item(),
        "median": sims.loc[
            sims == sims.quantile(0.5, method="nearest")
        ].simulation.item(),
        "q3": sims.loc[sims == sims.quantile(0.75, method="nearest")].simulation.item(),
        "max": sims[sims.argmax()].simulation.item(),
    }

    # Multiple model statistics can depend on the number of available simulations. i.e. No middle 10% of sims for 4 WRF sims, so fallback functionality to getting the median simulation.
    if len(sims) < 10:
        middle_10 = single_model_names["median"]
    else:
        middle_10 = sims[
            round(len(sims) * 0.45) - 1 : round(len(sims) * 0.55) - 1
        ].simulation.values

    multiple_model_names = {"middle 10%": middle_10}

    # Creating a dictionary of single stats names to the initial models from the dataset
    single_model_stats = dict(
        zip(
            list(single_model_names.keys()),
            data.sel(simulation=list(single_model_names.values())),
        )
    )

    # Creating a dictionary of stats that return multiple models to the initial models from the dataset
    multiple_model_stats = {
        k: data.sel(simulation=v) for k, v in multiple_model_names.items()
    }

    return (
        single_model_stats,
        multiple_model_stats,
    )


def _compute_selections_and_stats(selections, agg_func, years, months):
    """
    Aggregates the selections data across SSPs and computes statistics from the results
    """
    # Compute results on selections object
    results, data = _compute_results(selections, agg_func, years, months)

    # Compute statistics to extract from results
    single_stats, multiple_stats = _split_stats(results, data)

    # Return single statistic simulations, multiple statistic simulations, and the sorted simulations computed.
    return single_stats, multiple_stats, results


def _validate_lat_lon(lat, lon):
    """Validates the lat/lon values input by the user"""
    if lat and lon:  # Only validating lat/lon inputs for `agg_lat_lon_sims`
        if (type(lat) != float and type(lat) != tuple and type(lat) != int) or (
            type(lon) != float and type(lon) != tuple and type(lon) != int
        ):
            raise ValueError(
                "Error: Please enter either a tuple or a float or an int for your each of your lat/lon coordinates."
            )
        elif type(lat) != type(lon):
            raise ValueError(
                "Error: Please enter lat/lon coordinates both as a float, tuple, or int types."
            )
    else:
        raise ValueError("Error: Please enter valid lat/lon coordinates.")


def _validate_inputs(
    year_range, variable, downscaling_method, units, wrf_timescale="monthly"
):
    """Validates all the user inputs"""
    if variable not in set(show_available_vars(downscaling_method, wrf_timescale)):
        raise ValueError(
            "Error: Please enter an available variable for the given downscaling method."
        )
    if units not in get_available_units(variable, downscaling_method, wrf_timescale):
        raise ValueError(
            "Error: Please enter a unit type that is available for your selected variable."
        )
    if year_range[0] < 1950 or year_range[1] > 2100:
        raise ValueError("Error: Please enter a year range from 1950-2100.")


def _validate_timescale(timescale):
    """Validates the user input timescale"""
    if timescale not in ["monthly", "daily", "hourly"]:
        raise ValueError(
            "Please enter a valid timescale between 'monthly', 'daily', and 'hourly'."
        )


def show_available_vars(downscaling_method, wrf_timescale="monthly"):
    """Function that shows the available variables based on the input downscaling method."""
    _validate_timescale(wrf_timescale)

    # Read in catalogs
    data_catalog = intake.open_esm_datastore(data_catalog_url)
    var_desc = read_csv_file(variable_descriptions_csv_path)

    # Get available variable IDs
    match downscaling_method:
        case "Statistical":
            timescale = "monthly"
        case "Dynamical":
            timescale = wrf_timescale
        case _:
            raise ValueError(
                'downscaling_method needs to be either "Statistical" or "Dynamical"'
            )
    available_vars = _get_user_options(
        data_catalog,
        downscaling_method,
        timescale=timescale,
        resolution="3 km",  # Hard-coded to only accept `monthly` and `3 km` options for now.
    )[2]

    # Get variable names in written form
    var_opts = _get_variable_options_df(
        var_desc, available_vars, downscaling_method, timescale=timescale
    )["display_name"].to_list()

    return var_opts


def agg_lat_lon_sims(
    lat: Union[float, Tuple[float, float]],
    lon: Union[float, Tuple[float, float]],
    downscaling_method,
    variable,
    agg_func,
    units,
    years,
    months=list(range(1, 13)),
    wrf_timescale="monthly",
):
    """
    Gets aggregated WRF or LOCA simulation data for a lat/lon coordinate or lat/lon range for a given metric and timeframe (years, months).
    It combines all selected simulation data that is filtered by lat/lon, years, and specific months across SSP pathways
    and runs the passed in metric on all of the data. The results are then returned in ascending order,
    along with dictionaries mapping specific statistic names to the simulation objects themselves.

    Parameters
    ----------
    lat: float
        Latitude for specific location of interest.
    lon: float
        Longitude for specific location of interest.
    agg_func: str
        The function to aggregate the simulations by.
    years: tuple
        The lower and upper year bounds (inclusive) to subset simulation data by.
    months: list, optional
        Specific months of interest. The default is all months.

    Returns
    -------
    single_stats: dict of str: xr.DataArray
        Dictionary mapping string names of statistics to single simulation xr.DataArray objects.
    multiple_stats: dict of str: xr.DataArray
        Dictionary mapping string names of statistics to multiple simulations xr.DataArray objects.
    results: xr.DataArray
        Aggregated results of running the given aggregation function on the lat/lon gridcell of interest. Results are also sorted in ascending order.
    """
    # Validating if inputs are correct (lat/lon is appropriate types and variable is available for selected downscaling method)
    logger.debug("Validating inputs")
    _validate_lat_lon(lat, lon)
    _validate_inputs(years, variable, downscaling_method, units)
    # Create selections object

    logger.debug("Selecting data")
    selections = _create_lat_lon_select(
        lat, lon, variable, downscaling_method, units, years, wrf_timescale
    )
    # Runs calculations and derives statistics on simulation data pulled via selections object
    return _compute_selections_and_stats(selections, agg_func, years, months)


def agg_area_subset_sims(
    area_subset,
    cached_area,
    downscaling_method,
    variable,
    agg_func,
    units,
    years,
    months=list(range(1, 13)),
    wrf_timescale="monthly",
):
    """
    This function combines all available WRF or LOCA simulation data that is filtered on the `area_subset` (a string
    from existing keys in Boundaries.boundary_dict()) and on one of the areas of the values in that
    `area_subset` (`cached_area`). It then extracts this data across all SSP pathways for specific years/months,
    and runs the passed in `agg_func` on all of this data. The results are then returned in 3 values, the first
    as a dict of statistic names to xr.DataArray single simulation objects (i.e. median),
    the second as a dict of statistic names to xr.DataArray objects consisting of multiple simulation objects (i.e. middle 10%),
    and the last as a xr.DataArray of simulations' aggregated values sorted in ascending order.

    Parameters
    ----------
    area_subset: str
        Describes the category of the boundaries of interest (i.e. "CA Electric Load Serving Entities (IOU & POU)")
    cached_area: str
        Describes the specific area of interest (i.e. "Southern California Edison")
    agg_func: str
        The metric to aggregate the simulations by.
    years: tuple
        The lower and upper year bounds (inclusive) to extract simulation data by.
    months: list, optional
        Specific months of interest. The default is all months.

    Returns
    -------
    single_stats: dict of str: xr.DataArray
        Dictionary mapping string names of statistics to single simulation xr.DataArray objects.
    multiple_stats: dict of str: xr.DataArray
        Dictionary mapping string names of statistics to multiple simulations xr.DataArray objects.
    results: xr.DataArray
        Aggregated results of running the given aggregation function on the lat/lon gridcell of interest. Results are also sorted in ascending order.
    """
    # Validating if variable is available for the given downscaling method
    _validate_inputs(years, variable, downscaling_method, units, wrf_timescale)
    # Creates the selections object
    selections = _create_cached_area_select(
        area_subset,
        cached_area,
        variable,
        downscaling_method,
        units,
        years,
        wrf_timescale,
    )
    # Runs calculations and derives statistics on simulation data pulled via selections object
    return _compute_selections_and_stats(selections, agg_func, years, months)
