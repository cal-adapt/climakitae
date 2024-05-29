import xarray as xr
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import panel as pn
from climakitae.core.data_interface import Select
import panel as pn
from climakitae.explore.agnostic import (
    warm_level_to_month,
    year_to_warm_levels,
    _round_to_nearest_half,
)

sns.set_style("whitegrid")


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
            return warm_level_to_month(lookup_df, scenario, warming_level)

        # Given year, plot warming levels, find median, and guide interpretation
        elif warming_level is None and year is not None:
            min_year, max_year = 2021, 2089
            if not (min_year <= year and year <= max_year):
                return print(
                    f"Please provide a year between {min_year} and {max_year}."
                )

            lookup_df = lookup_tables["warming level lookup table"]
            warm_levels, med_level = year_to_warm_levels(lookup_df, scenario, year)
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


def _plot_warm_levels(fig, ax, levels, year):
    """Plot histogram of warming levels, on axis."""
    n = len(levels)
    sns.histplot(ax=ax, data=levels, binwidth=0.25)
    ax.set_title(f"Warming levels reached in {year} by all {n} model simulations")
    ax.set_xlabel("Warming level (°C)")
    ax.set_ylabel("Number of simulations")
    return ax


def plot_WRF(sim_vals, agg_func, years):
    """
    Visualizes a barplot of WRF simulations that are aggregated from `agg_lat_lon_sims` or `agg_area_subset_sims`.
    Used with `results_gridcell` or `results_area` as inputs, as well as the aggregated function and time slice, all predefined within `agnostic_tools.ipynb`.

    Parameters
    ----------
    sim_vals: xr.DataArray
        DataArray of the aggregated results of a climate variable.
    agg_func: Function
        Function that takes in a series of values and returns a statistic, like np.mean
    time_slice: tuple
        Years of interest

    Returns
    -------
    None
    """
    sims = [name.split(",")[0] for name in list(sim_vals.simulation.values)]
    sims = [name[4:] for name in sims]
    sims = ["\n".join(sim_name.split("_")) for sim_name in sims]
    vals = sim_vals.values

    # Allowing WRF labels to be plotted when visualizing 4 or 8 sims
    if len(sims) == 8:
        figsize = (12, 3)
    else:
        figsize = None

    fig, ax = plt.subplots(figsize=figsize)
    ax.bar(sims, vals)
    ax.set_xlabel("WRF Simulation, Emission Scenario 3-7.0", labelpad=15, fontsize=12)
    ax.set_ylabel(f"{sim_vals.name} ({sim_vals.units})", labelpad=10, fontsize=12)
    ax.set_ylim(bottom=min(sim_vals) - 5, top=max(sim_vals) + 5)

    if sim_vals.location_subset == ["coordinate selection"]:
        if (
            "lat" in sim_vals.attrs
        ):  # Determine if lat/lon was manually written onto DataArray because of area averaging or not
            location = f"lat: {sim_vals.lat}, lon: {sim_vals.lon}"
        else:
            location = (round(sim_vals.lat.item(), 2), round(sim_vals.lon.item(), 2))
    else:
        location = sim_vals.location_subset[0]

    plt.title(
        f"{str(agg_func.__name__).capitalize()} of {sim_vals.name} at {location} from {years}"
    )
    plt.show()


def plot_LOCA(sim_vals, agg_func, time_slice, stats):
    """
    Visualizes a histogram of LOCA simulations that are aggregated from `agg_lat_lon_sims` or `agg_area_subset_sims`.
    Used with `results_gridcell` or `results_area` as inputs, as well as the aggregated function, time slice, and
    simulation stats all predefined within `agnostic_tools.ipynb`.

    Parameters
    ----------
    sim_vals: xr.DataArray
        DataArray of the aggregated results of a climate variable.
    agg_func: Function
        Function that takes in a series of values and returns a statistic, like np.mean
    time_slice: tuple
        Years of interest
    stats: dict
        Statistics that are returned from `single_stats_gridcell` or `single_stats_area` in `agnostic_tools.ipynb`

    Returns
    -------
    None
    """
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
        f"Histogram of {str(agg_func.__name__).capitalize()} {sim_vals.name} from {time_slice} of all {len(sim_vals)} LOCA sims for {area_text}"
    )
    ax.set_xlabel(f"{sim_vals.name} ({sim_vals.units})")
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
            label=f"{stat.capitalize()} sim: {name[6:]}",
        )

    plt.legend(fontsize=9.5)


def plot_climate_response_WRF(var1, var2):
    """
    Visualizes a scatterplot of two aggregated WRF climate variables from `agg_lat_lon_sims` or `agg_area_subset_sims`.
    Used with `results_gridcell` or `results_area` as inputs, as seen within `agnostic_tools.ipynb`.

    Parameters
    ----------
    var1: xr.DataArray
        DataArray of the first climate variable, with simulation, name, and units attributes.
    var2: xr.DataArray
        DataArray of the second climate variable, with simulation, name, and units attributes.

    Returns
    -------
    None
    """
    # Make sure that the two variables are the same length and have the same simulation names
    if (len(var1) != len(var2)) & (
        set(var1.simulation.values) != set(var2.simulation.values)
    ):
        raise IndexError(
            "The two variables must have the same length of simulations and have the same simulation names."
        )

    merged_results = xr.merge([var1, var2])
    plot = merged_results.hvplot.scatter(
        x=var1.name,
        y=var2.name,
        by="simulation",
        title=f"WRF results for {var1.location_subset[0]}: \n{var1.name} vs {var2.name}",
    )

    # Changing legend location depending on number of simulations
    if len(merged_results.simulation) == 4:
        legend_offset = (10, 128)
    elif len(merged_results.simulation) == 8:
        legend_offset = (10, 30)
    else:
        legend_offset = (10, 0)

    plot = plot.opts(
        legend_position="right", legend_offset=legend_offset, width=800, height=350
    )
    return pn.panel(plot)


def plot_climate_response_LOCA(var1, var2):
    """
    Visualizes a scatterplot of two aggregated LOCA climate variables from `agg_lat_lon_sims` or `agg_area_subset_sims`.
    Used with `results_gridcell` or `results_area` as inputs, as seen within `agnostic_tools.ipynb`.

    Parameters
    ----------
    var1: xr.DataArray
        DataArray of the first climate variable, with simulation, name, and units attributes.
    var2: xr.DataArray
        DataArray of the second climate variable, with simulation, name, and units attributes.

    Returns
    -------
    None
    """
    # Make sure that the two variables are the same length and have the same simulation names
    if (len(var1) != len(var2)) & (
        set(var1.simulation.values) != set(var2.simulation.values)
    ):
        raise IndexError(
            "The two variables must have the same length of simulations and have the same simulation names."
        )

    merged_results = xr.merge([var1, var2])

    # Finding GCM names for simulations
    sims = [name.split(",")[0] for name in list(merged_results.simulation.values)]
    sims = [name[6:].split("_")[0] for name in sims]
    merged_results["simulation"] = sims
    merged_results = merged_results.rename({"simulation": "Global Climate Model"})

    plot = merged_results.hvplot.scatter(
        x=var1.name, y=var2.name, by="Global Climate Model", size=45
    )
    plot = plot.opts(
        legend_position="right",
        legend_offset=(10, 25),
        width=800,
        height=500,
        title=f"LOCA2 results for {var1.location_subset[0]}: \n{var1.name} vs {var2.name}",
    )
    return pn.panel(plot)
