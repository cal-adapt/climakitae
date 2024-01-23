"""Backend for agnostic tools."""
import numpy as np
import pandas as pd
from dask import compute
import xarray as xr
import seaborn as sns
import matplotlib.pyplot as plt
import panel as pn
from climakitae.core.data_interface import Select

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
    from climakitae.core.data_interface import DataInterface

    data_interface = DataInterface()
    gcms = data_interface.data_catalog.df.source_id.unique()

    time_df = _create_time_lut(gcms)
    warm_df = _create_warm_level_lut(gcms)

    return {"time lookup table": time_df, "warming level lookup table": warm_df}


def _create_time_lut(gcms):
    """Prepare lookup table for converting warming levels to times."""
    # Read in simulation vs warming levels (1.5, 2, 3, 4) table
    df = pd.read_csv("~/src/climakitae/climakitae/data/gwl_1850-1900ref (2).csv")
    # Subset to cataloged GCMs
    df = df[df["GCM"].isin(gcms)]

    df.dropna(
        axis="rows", how="all", subset=["1.5", "2.0", "3.0", "4.0"], inplace=True
    )  # model EC-Earth3 runs/simulations
    return df


def _create_warm_level_lut(gcms):
    """Prepare lookup table for converting times to warming levels."""
    # Read in time vs simulation table
    df = pd.read_csv(
        "~/src/climakitae/climakitae/data/gwl_1850-1900ref_timeidx.csv",
        index_col="time",
        parse_dates=True,
    )

    month_df = df.groupby(
        [df.index.year, df.index.month]
    ).mean()  # This ignores NaN and gets the only value in each month
    # MultiIndex to DatetimeIndex
    month_df.index = pd.to_datetime(["-".join(map(str, idx)) for idx in month_df.index])
    # Subset time to 2021-2089
    subset_rows = (month_df.index.year > 2020) & (month_df.index.year < 2090)
    # Subset to cataloged GCMs and scenario "ssp370"
    subset_columns = [
        col
        for col in month_df.columns
        if col.split("_")[0] in gcms and col.endswith("ssp370")
    ]
    month_df = month_df.loc[subset_rows, subset_columns]

    month_df.dropna(
        axis="columns", how="all", inplace=True
    )  # model EC-Earth3 runs/simulations

    year_df = month_df.resample("Y").mean()
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
    return None


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

# Making pre-determined metrics
metrics = {
    "Average Air Temperature (2030-2059)": {
        "var": "Maximum air temperature at 2m",
        "time_slice": (2030, 2059),
        "agg": np.mean,
        "units": "degF",
    },
    "Average Annual Total Precipitation (2030-2059)": {
        "var": "Precipitation (total)",
        "time_slice": (2030, 2059),
        "agg": np.mean,
        "units": "inches",
    },
}


def get_all_loca_cached_area(area_subset, cached_area, selected_val):
    """
    Given a lat and lon, return statistics of predetermined metric and parameters of all simulations in SSP 3-7.0.
    """
    ### Creating selection object for each SSP
    total_data = []

    for ssp in [
        "SSP 3-7.0 -- Business as Usual",
        "SSP 2-4.5 -- Middle of the Road",
        "SSP 5-8.5 -- Burn it All",
    ]:
        # Creating Select object
        selections = Select()

        # Getting warming data for all LOCA models at given location on monthly time-scale at 3km resolution
        selections.area_subset = area_subset
        selections.cached_area = [cached_area]
        selections.scenario_ssp = [ssp]
        selections.data_type = "Gridded"
        selections.variable = metrics[selected_val]["var"]
        selections.scenario_historical = ["Historical Climate"]
        selections.downscaling_method = ["Statistical"]
        selections.timescale = "monthly"
        selections.resolution = "3 km"
        selections.units = metrics[selected_val]["units"]
        selections.time_slice = metrics[selected_val]["time_slice"]

        # Retrieve data
        data = selections.retrieve()
        total_data.append(data)

    ### Calculate metric on each SSP subset of data
    all_calc_vals = []
    for ssp_data in total_data:
        calc_vals = (
            ssp_data.squeeze()
            .groupby("simulation")
            .map(metrics[selected_val]["agg"])
            .chunk(chunks="auto")
        )
        calc_vals["simulation"] = (
            calc_vals.simulation
            + ", "
            + calc_vals.scenario.item().split("--")[0].split("+")[1].strip()
        )
        calc_vals = calc_vals.drop("scenario")
        all_calc_vals.append(calc_vals)
    all_calc_vals = xr.concat(all_calc_vals, dim="simulation")

    # Sorting sims and getting metrics
    sorted_sims = compute(all_calc_vals.sortby("simulation"))[
        0
    ]  # Need all the values in order to create histogram + return values

    # TODO: Re-write this, Dask runs 3x times for the quantile methods
    single_model_compute = {
        "min": sorted_sims[0].simulation.item(),
        "q1": sorted_sims.loc[
            sorted_sims == sorted_sims.quantile(0.25, method="nearest")
        ].simulation.item(),
        "median": sorted_sims.loc[
            sorted_sims == sorted_sims.quantile(0.5, method="nearest")
        ].simulation.item(),
        "q3": sorted_sims.loc[
            sorted_sims == sorted_sims.quantile(0.75, method="nearest")
        ].simulation.item(),
        "max": sorted_sims[-1].simulation.item(),
    }

    multiple_model_compute = {
        "middle 10%": sorted_sims[
            round(len(sorted_sims + 1) * 0.45)
            - 1 : round(len(sorted_sims + 1) * 0.55)
            - 1
        ].simulation.values
    }

    # Creating a dictionary of single stats names to the initial models from the dataset
    single_model_stats = dict(
        zip(
            list(single_model_compute.keys()),
            all_calc_vals.sel(
                simulation=list(single_model_compute.values())
            ),  # TODO: Fix
        )
    )

    # Creating a dictionary of stats that return multiple models to the initial models from the dataset
    # multiple_model_stats = {k: data.squeeze().sel(simulation=v) for k, v in multiple_model_compute.items()} # TODO: This line doesn't work

    # Returning models from stats and aggregated results
    return (single_model_stats, sorted_sims)


def get_all_models_and_stats(lat, lon, selected_val):
    """
    Given a lat and lon, return statistics of predetermined metric and parameters of all simulations in SSP 3-7.0.
    """
    # Making pre-determined metrics
    metrics = {
        "Average Air Temperature (2030-2059)": {
            "var": "Maximum air temperature at 2m",
            "time_slice": (2030, 2059),
            "agg": np.mean,
            "units": "degF",
        },
        "Average Annual Total Precipitation (2030-2059)": {
            "var": "Precipitation (total)",
            "time_slice": (2030, 2059),
            "agg": np.mean,
            "units": "inches",
        },
    }

    # Creating Select object
    selections = Select()

    # Getting warming data for all LOCA models at given location on monthly time-scale at 3km resolution
    selections.area_subset = "lat/lon"
    selections.data_type = "Gridded"
    selections.variable = metrics[selected_val]["var"]
    selections.scenario_historical = ["Historical Climate"]
    selections.downscaling_method = ["Statistical"]
    selections.scenario_ssp = ["SSP 3-7.0 -- Business as Usual"]
    selections.timescale = "monthly"
    selections.resolution = "3 km"
    selections.units = metrics[selected_val]["units"]
    selections.time_slice = metrics[selected_val]["time_slice"]
    selections.latitude = (lat - 0.1, lat + 0.1)
    selections.longitude = (lon - 0.1, lon + 0.1)

    # Retrieve data
    data = selections.retrieve()

    # Retrieving closest grid-cell's data
    subset_data = data.sel(lat=lat, lon=lon, method="nearest").sel(
        scenario="Historical + SSP 3-7.0 -- Business as Usual"
    )

    # Calculate the given metric on the data
    calc_vals = (
        subset_data.squeeze()
        .groupby("simulation")
        .map(metrics[selected_val]["agg"])
        .chunk(chunks="auto")
    )
    # heat_vals = subset_data.squeeze().groupby('simulation').map(np.mean).chunk(chunks='auto')

    # Sorting sims and getting metrics
    sorted_sims = compute(calc_vals.sortby("simulation"))[
        0
    ]  # Need all the values in order to create histogram + return values

    # TODO: Re-write this, Dask runs 3x times for the quantile methods
    single_model_compute = {
        "min": sorted_sims[0].simulation.item(),
        "q1": sorted_sims.loc[
            sorted_sims == sorted_sims.quantile(0.25, method="nearest")
        ].simulation.item(),
        "median": sorted_sims.loc[
            sorted_sims == sorted_sims.quantile(0.5, method="nearest")
        ].simulation.item(),
        "q3": sorted_sims.loc[
            sorted_sims == sorted_sims.quantile(0.75, method="nearest")
        ].simulation.item(),
        "max": sorted_sims[-1].simulation.item(),
    }

    multiple_model_compute = {
        "middle 10%": sorted_sims[
            round(len(sorted_sims + 1) * 0.45)
            - 1 : round(len(sorted_sims + 1) * 0.55)
            - 1
        ].simulation.values
    }

    # Creating a dictionary of single stats names to the initial models from the dataset
    single_model_stats = dict(
        zip(
            list(single_model_compute.keys()),
            subset_data.sel(simulation=list(single_model_compute.values())),
        )
    )

    # Creating a dictionary of stats that return multiple models to the initial models from the dataset
    multiple_model_stats = {
        k: subset_data.sel(simulation=v) for k, v in multiple_model_compute.items()
    }

    # Returning models from stats and aggregated results
    return (single_model_stats, multiple_model_stats, sorted_sims)


def plot_sims(sim_vals, selected_val):
    """
    Creates resulting plot figures.
    """
    plt.figure(figsize=(10, 5))
    ax = sns.histplot(sim_vals, edgecolor="white", linewidth=0.3, alpha=0.5)
    ax.set_title(
        "Histogram of {} of all {} LOCA sims for SCE service territory".format(
            selected_val, len(sim_vals)
        )
    )
    ax.set_xlabel("Monthly " + str(sim_vals.units).capitalize())
    ax.set_ylabel("Count of Simulations")

    # Finding different simulations and labeling them on the chart
    min_sim_name = sim_vals.isel(simulation=[sim_vals.argmin()]).simulation.item()
    q1_sim_name = sim_vals.loc[
        sim_vals == sim_vals.quantile(0.25, method="nearest")
    ].simulation.item()
    med_sim_name = sim_vals.loc[
        sim_vals == sim_vals.quantile(0.5, method="nearest")
    ].simulation.item()
    q3_sim_name = sim_vals.loc[
        sim_vals == sim_vals.quantile(0.75, method="nearest")
    ].simulation.item()
    max_sim_name = sim_vals.isel(simulation=[sim_vals.argmax()]).simulation.item()

    # Making the vertical lines
    ax.axvline(
        np.min(sim_vals),
        color="red",
        linestyle="--",
        label="Min sim: {}".format(min_sim_name[6:]),
    )
    ax.axvline(
        sim_vals.quantile(0.25, method="nearest"),
        color="blue",
        linestyle="--",
        label="Q1 sim: {}".format(q1_sim_name[6:]),
    )
    ax.axvline(
        sim_vals.quantile(0.5, method="nearest"),
        color="black",
        linestyle="--",
        label="Median sim: {}".format(med_sim_name[6:]),
    )
    ax.axvline(
        sim_vals.quantile(0.75, method="nearest"),
        color="blue",
        linestyle="--",
        label="Q3 sim: {}".format(q3_sim_name[6:]),
    )
    ax.axvline(
        np.max(sim_vals),
        color="red",
        linestyle="--",
        label="Max sim: {}".format(max_sim_name[6:]),
    )

    plt.legend()


def create_interactive_panel():
    """
    Creates an interactive panel for users to interact with to specify what metrics they'd like calculated on all LOCA runs.
    """
    pn.extension()

    # Pre-included metrics for the dropdown
    dropdown_options = list(metrics.keys())

    def on_dropdown_change(event):
        selected_value = dropdown.value

    # Create a Panel column layout
    column_layout = pn.Column()

    # Create dropdown widget
    dropdown = pn.widgets.Select(
        options=dropdown_options,
        value=dropdown_options[0],
        name="Pre-calculated metrics",
        width=350,
    )
    dropdown.param.watch(on_dropdown_change, "value")

    # Add the dropdown to the column layout
    column_layout.append(dropdown)

    # Display the Panel app
    return column_layout.servable(), dropdown
