"""Backend for agnostic tools."""

import numpy as np
import pandas as pd
from dask import compute
import xarray as xr
import seaborn as sns
import matplotlib.pyplot as plt
import panel as pn
import intake
from climakitae.core.data_interface import (
    Select,
    DataInterface,
    _get_variable_options_df,
    _get_user_options,
)
from climakitae.util.utils import read_csv_file, get_closest_gridcell, area_average
from climakitae.core.paths import variable_descriptions_csv_path, data_catalog_url
from climakitae.util.unit_conversions import get_unit_conversion_options
from typing import Union, Tuple

sns.set_style("whitegrid")


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
    # Read in simulation vs warming levels (1.5, 2, 3, 4) table
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


def _warm_level_to_years_plot(time_df, scenario, warming_level):
    """Given warming level, plot histogram of years and label median year."""
    datetimes = time_df[time_df["scenario"] == scenario][warming_level].astype(
        "datetime64[ns]"
    )
    med_datetime = datetimes.quantile(0.5, interpolation="midpoint")
    years = datetimes.dt.year
    med_year = med_datetime.year

    fig, ax = plt.subplots()
    _plot_years(ax, years, med_year, warming_level)


def _plot_years(ax, years, med_year, warming_level):
    """Plot histogram of years and label median year, on axis."""
    n = len(years)
    sns.histplot(ax=ax, data=years)
    ax.set_title(
        f"Years when each of all {n} model simulations reach "
        f"{warming_level} °C warming level"
    )
    ax.set_xlabel("Year")
    ax.set_ylabel("Number of simulations")
    ax.axvline(
        med_year, color="black", linestyle="--", label=f"Median year is {med_year}"
    )
    ax.legend()
    return ax


def _warm_level_to_month(time_df, scenario, warming_level):
    """Given warming level, give month."""
    med_date = (
        time_df[time_df["scenario"] == scenario][warming_level]
        .astype("datetime64[ns]")
        .quantile(0.5, interpolation="midpoint")
    )
    return med_date.strftime("%Y-%m")


def _plot_warm_levels(fig, ax, levels, year):
    """Plot histogram of warming levels, on axis."""
    n = len(levels)
    sns.histplot(ax=ax, data=levels, binwidth=0.25)
    ax.set_title(f"Warming levels reached in {year} by all {n} model simulations")
    ax.set_xlabel("Warming level (°C)")
    ax.set_ylabel("Number of simulations")
    return ax


def _year_to_warm_levels(warm_df, scenario, year):
    """Given year, give warming levels and their median."""
    warm_levels = warm_df.loc[year]
    med_level = np.quantile(warm_levels, 0.5, interpolation="midpoint")
    return warm_levels, med_level


def _round_to_nearest_half(number):
    return round(number * 2) / 2


def find_wl_or_time(lookup_tables, scenario="ssp370", warming_level=None, year=None):
    """
    Given either a warming level or a time, find information about the other.

    If given a `warming_level`, the function looks up the times when simulations
    under the specified `scenario` reach the warming level. It returns the
    median month across the simulations. It also plots a histogram of the years
    with a label for the median year. The lookup is based on the
    "time lookup table" in `lookup_tables`.

    If given a `year`, the function looks up the warming levels reached in the
    year by simulations under the specified `scenario`. It plots a histogram of
    the warming levels. It also calculates the median warming level across the
    simulations and prints the major warming level nearest to the median.
    Guidance provided include a list of major warming levels considered (1.0,
    1.5, 2.0, 2.5, 3.0, 3.5, 4.0, and 4.5°C), and the 0.5 interval the median
    is in.

    Parameters
    ----------
    lookup_tables : dict of pandas.DataFrame
        Lookup tables as output from the `create_lookup_tables` function. It
        is a dictionary with a "time lookup table" and a "warming level lookup
        table".
    scenario : str, optional
        The scenario to consider. The default is "ssp370".
    warming_level : str, optional
        The warming level to analyze ("1.5", "2.0", "3.0"). The default is None.
    year : int, optional
        The year to analyze. Must be between 2021 and 2089. The default is None.

    Returns
    -------
    str or None
        Given a warming level, returns a string representing the median month.
        Given a year, returns None.
        None is also returned if neither warming level nor year is given, or
        if both are given.
    """
    if not warming_level and not year:
        print("Pass in either a warming level or a year.")

    elif warming_level is not None and year is not None:
        print("Pass in either a warming level or a year, but not both.")

    else:
        if scenario != "ssp370":
            return print("Scenarios other than ssp370 are under development.")

        # Given warming level, plot years and find median month
        if warming_level is not None and year is None:
            allowed_warm_level = ["1.5", "2.0", "3.0"]
            if warming_level not in allowed_warm_level:
                return print(
                    f"Please choose a warming level among {allowed_warm_level}"
                )

            lookup_df = lookup_tables["time lookup table"]
            _warm_level_to_years_plot(lookup_df, scenario, warming_level)
            return _warm_level_to_month(lookup_df, scenario, warming_level)

        # Given year, plot warming levels, find median, and guide interpretation
        elif warming_level is None and year is not None:
            min_year, max_year = 2021, 2089
            if not (min_year <= year and year <= max_year):
                return print(
                    f"Please provide a year between {min_year} and {max_year}."
                )

            lookup_df = lookup_tables["warming level lookup table"]
            warm_levels, med_level = _year_to_warm_levels(lookup_df, scenario, year)
            major_levels = np.arange(1, 4.51, 0.5)

            fig, ax = plt.subplots()
            _plot_warm_levels(fig, ax, warm_levels, year)
            if med_level in major_levels:
                return print(f"The median projected warming level is {med_level}°C. \n")
            else:
                major_level = _round_to_nearest_half(med_level)
                print(
                    (
                        "The major warming level nearest to the median "
                        f"projected warming level is {major_level}°C."
                    )
                )
                if med_level < major_level:
                    lower_level = major_level - 0.5
                    upper_level = major_level
                elif med_level > major_level:
                    lower_level = major_level
                    upper_level = major_level + 0.5
                return print(
                    (
                        "The actual median projected warming level is between "
                        f"{lower_level} and {upper_level}°C.\n"
                        "Major warming levels considered include 1.0, 1.5, 2.0, "
                        "2.5, 3.0, 3.5, 4.0, and 4.5°C.\n"
                    )
                )


def create_conversion_function(lookup_tables):
    """
    Create a function that converts between warming level and time.

    Parameters
    ----------
    lookup_tables : dict of pandas.DataFrame
        Lookup tables for the conversions as output from the
        `create_lookup_tables` function. It is a dictionary with a "time
        lookup table" and a "warming level lookup table".

    Returns
    -------
    function
        The `find_wl_or_time` function preloaded with the given `lookup_tables`.
        Given either a warming level or a time, the function uses
        `lookup_tables` to find information about the other. Please see
        `find_wl_or_time` for details.

    Notes
    -----
    This saves time otherwise needed to remake the lookup tables for each call.
    """
    return lambda scenario="ssp370", warming_level=None, year=None: find_wl_or_time(
        lookup_tables, scenario, warming_level, year
    )


##### TASK 2 #####


def _get_var_info(variable, downscaling_method):
    """Gets the variable info for the specific variable name and downscaling method"""
    var_desc_df = read_csv_file(variable_descriptions_csv_path)
    return var_desc_df[
        (var_desc_df["display_name"] == variable)
        & (var_desc_df["timescale"].str.contains("monthly"))
        & (var_desc_df["downscaling_method"] == downscaling_method)
    ]


def get_available_units(variable, downscaling_method):
    """Get other available units available for the given unit"""
    # Select your desired units
    var_info_df = _get_var_info(variable, downscaling_method)
    if var_info_df.empty:
        raise ValueError(
            "Please input a valid variable for the given downscaling method."
        )
    available_units = get_unit_conversion_options()[var_info_df["unit"].item()]
    return available_units


def _complete_selections(selections, variable, units, years):
    """Completes the attributes for the `selections` objects from `create_lat_lon_select` and `create_cached_area_select`."""
    metric_info_df = _get_var_info(variable, selections.downscaling_method)
    selections.data_type = "Gridded"
    selections.variable = metric_info_df["display_name"].item()
    selections.scenario_historical = ["Historical Climate"]

    # If we want to allow users to select on criteria beyond just the metric and downscaling (i.e. also timescale and resolution), then the following line will be useful to present users
    # print(variable_description_df[['display_name', 'downscaling_method', 'timescale']].to_string())
    selections.timescale = "monthly"
    selections.resolution = "3 km"
    selections.units = units
    selections.time_slice = years
    return selections


def _create_lat_lon_select(lat, lon, variable, downscaling_method, units, years):
    """Creates a selection object for the given lat/lon parameters."""
    # Creates a selection object
    selections = Select()
    selections.area_subset = "lat/lon"
    if (
        type(lat) == float
    ):  # Creating a box around which to find the nearest gridcell for compute
        selections.latitude = (lat - 0.05, lat + 0.05)
        selections.longitude = (lon - 0.05, lon + 0.05)
    elif type(lat) == tuple:
        selections.latitude = lat
        selections.longitude = lon
        selections.area_average = "Yes"
    selections.downscaling_method = downscaling_method

    # Add attributes for the rest of the selections object
    selections = _complete_selections(selections, variable, units, years)
    return selections


def _create_cached_area_select(
    area_subset, cached_area, variable, downscaling_method, units, years
):
    """Creates a selection object for the given cached area parameters."""
    # Creates a selection object for area subsetting simulations
    selections = Select()
    selections.area_subset = area_subset
    selections.cached_area = [cached_area]
    selections.downscaling_method = downscaling_method

    # Add attributes for the rest of the selections object
    selections = _complete_selections(selections, variable, units, years)

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
        "Statistical": [
            "SSP 3-7.0 -- Business as Usual",
            "SSP 2-4.5 -- Middle of the Road",
            "SSP 5-8.5 -- Burn it All",
        ],
        "Dynamical": [
            "SSP 3-7.0 -- Business as Usual",
        ],
    }
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

    # Retrieving closest grid-cell's data for lat/lon area subsetting IF the given lat/lon coordinates are not tuple ranges
    if selections.area_subset == "lat/lon" and selections.area_average != "Yes":

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
    sorted_sims = compute(calc_vals.sortby(calc_vals))[
        0
    ]  # Need all the values in order to create histogram + return values

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
        middle_10 = single_model_names['median']
    else:
        middle_10 = sims[
            round(len(sims) * 0.45) - 1 : round(len(sims) * 0.55) - 1
        ].simulation.values
        
    multiple_model_names = {
        "middle 10%": middle_10
    }

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


def _validate_variable(variable, downscaling_method):
    if variable not in set(show_available_vars(downscaling_method)):
        raise ValueError(
            "Error: Please enter an available variable for the given downscaling method."
        )


def _validate_lat_lon(lat, lon):
    if lat and lon:  # Only validating lat/lon inputs for `agg_lat_lon_sims`
        if (type(lat) != float and type(lon) != tuple) or (
            type(lon) != float and type(lon) != tuple
        ):
            raise ValueError(
                "Error: Please enter either a tuple or a float for your each of your lat/lon coordinates."
            )
        elif type(lat) != type(lon):
            raise ValueError(
                "Error: Please enter lat/lon coordinates both as a float or tuple types."
            )
    else:
        raise ValueError("Error: Please enter valid lat/lon coordinates.")


def _validate_units(variable, downscaling_method, units):
    if units not in get_available_units(variable, downscaling_method):
        raise ValueError(
            "Error: Please enter a unit type that is available for your selected variable."
        )


def show_available_vars(downscaling_method):
    """Function that shows the available variables based on the input downscaling method."""

    # Read in catalogs
    data_catalog = intake.open_esm_datastore(data_catalog_url)
    var_desc = read_csv_file(variable_descriptions_csv_path)

    # Get available variable IDs
    available_vars = _get_user_options(
        data_catalog,
        downscaling_method,
        timescale="monthly",
        resolution="3 km",  # Hard-coded to only accept `monthly` and `3 km` options for now.
    )[2]

    # Get variable names in written form
    var_opts = _get_variable_options_df(
        var_desc, available_vars, downscaling_method, timescale="monthly"
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
    _validate_lat_lon(lat, lon)
    _validate_variable(variable, downscaling_method)
    _validate_units(variable, downscaling_method, units)
    # Create selections object
    selections = _create_lat_lon_select(
        lat, lon, variable, downscaling_method, units, years
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
    _validate_variable(variable, downscaling_method)
    _validate_units(variable, downscaling_method, units)
    # Creates the selections object
    selections = _create_cached_area_select(
        area_subset, cached_area, variable, downscaling_method, units, years
    )
    # Runs calculations and derives statistics on simulation data pulled via selections object
    return _compute_selections_and_stats(selections, agg_func, years, months)


def plot_LOCA(sim_vals, selected_val, time_slice, stats):
    """Creates a histogram for LOCA simulations aggregated from `agg_lat_lon_sims` or `agg_area_subset_sims`."""
    # Finding the proper title for the plot
    area_text = ""
    if sim_vals.location_subset == ["coordinate selection"]:
        area_text = "given lat/lon"
    else:
        area_text = sim_vals.location_subset[0]

    # Creating the histogram
    plt.figure(figsize=(10, 5))
    ax = sns.histplot(
        sim_vals,
        edgecolor="white",
        bins=np.linspace(min(sim_vals).item(), max(sim_vals).item(), 12),
    )
    ax.set_title(
        "Histogram of {} from {} of all {} LOCA sims for {}".format(
            selected_val, time_slice, len(sim_vals), area_text
        )
    )
    ax.set_xlabel("Monthly " + str(sim_vals.units).capitalize())
    ax.set_ylabel("Count of Simulations")

    # Creating pairings between simulations and calculated values
    sim_stats_names = list([sim.simulation.item() for sim in stats.values()])
    sim_val_pairings = list(
        zip(
            stats.keys(),
            sim_stats_names,
            sim_vals.sel(simulation=sim_stats_names).values,
        )
    )
    color_mapping = {
        "min": "red",
        "q1": "blue",
        "median": "black",
        "q3": "blue",
        "max": "red",
    }

    # Plotting vertical lines
    for item in sim_val_pairings:
        stat, name, val = item
        # Plotting vertical lines for individual statistics
        ax.axvline(
            val,
            color=color_mapping[stat],
            linestyle="--",
            label="{} sim: {}".format(stat.capitalize(), name[6:]),
        )

    plt.legend(fontsize=9.5)


def plot_WRF(sim_vals, variable, agg_func):
    """Bar plot of WRF models with their aggregated values from `agg_lat_lon_sims` or `agg_area_subset_sims`."""
    sims = [name.split(",")[0] for name in list(sim_vals.simulation.values)]
    sims = [name[4:] for name in sims]
    sims = ['\n'.join(sim_name.split('_')) for sim_name in sims]
    vals = sim_vals.values

    fig, ax = plt.subplots()
    ax.bar(sims, vals)
    ax.set_xlabel("WRF Simulation, Emission Scenario 3-7.0", labelpad=15, fontsize=12)
    ax.set_ylabel(f"{variable} ({sim_vals.units})", labelpad=10, fontsize=12)
    ax.set_ylim(bottom=min(sim_vals) - 5, top=max(sim_vals) + 5)

    if sim_vals.location_subset == ["coordinate selection"]:
        location = (round(sim_vals.lat.item(), 2), round(sim_vals.lon.item(), 2))
    else:
        location = sim_vals.location_subset[0]

    plt.title(
        "{} of {} at {}".format(str(agg_func.__name__).capitalize(), variable, location)
    )
    plt.show()


def plot_climate_response_WRF(var1, var2):
    """
    Scatter plot of two climate variables from WRF models with their aggregated values from `agg_lat_lon_sims` or `agg_area_subset_sims`.
    """
    # Make sure that the two variables are the same length and have the same simulation names
    if (len(var1) != len(var2)) & (
        set(var1.simulation.values) != set(var2.simulation.values)
    ):
        raise IndexError(
            "The two variables must have the same length of simulations and have the same simulation names."
        )

    var1 = var1.sortby("simulation")
    var2 = var2.sortby("simulation")
    fig, ax = plt.subplots(figsize=(7, 5))

    # Get sim names
    sims = [name.split(",")[0] for name in list(var1.simulation.values)]
    sims = [name[4:] for name in sims]
    sims = ['\n'.join(sim_name.split('_')) for sim_name in sims]

    # Plot points and add labels
    for idx in range(len(var1.simulation)):
        ax.scatter(var1[idx], var2[idx], label=sims[idx])
    ax.set_title(
        "WRF CA Metrics: CA Statewide Average", fontsize=12
    )  # Specifically supporting visualizing CA statewide average (for current applications)
    ax.set_xlabel(f"{var1.name} ({var1.units})", labelpad=10, fontsize=12)
    ax.set_ylabel(f"{var2.name} ({var2.units})", labelpad=10, fontsize=12)

    # Add point annotations
    for i, txt in enumerate(sims):
        ax.annotate(
            txt,
            (var1[i], var2[i]),
            va="center",
            textcoords="offset points",
            xytext=(7, 0),
        )
        
    x_padding = (max(var1.values) - min(var1.values)) * 0.2  # 10% padding
    y_padding = (max(var2.values) - min(var2.values)) * 0.2  # 10% padding

    # Adjust limits with padding
    plt.xlim(min(var1.values) - x_padding, max(var1.values) + x_padding)
    plt.ylim(min(var2.values) - y_padding, max(var2.values) + y_padding)
    plt.grid(True)
    plt.gca().set_aspect("equal")
    plt.show()