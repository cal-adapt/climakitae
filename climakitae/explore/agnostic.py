"""Backend for agnostic tools."""
import numpy as np
import pandas as pd
from dask import compute
import xarray as xr
import seaborn as sns
import matplotlib.pyplot as plt
import panel as pn
from climakitae.core.data_interface import Select


def create_lookup_table():
    """
    Create a warming level table that is subsetting only on GCMs that we have data for.
    """
    # Finding the names of all the GCMs that we catalog
    from climakitae.core.data_interface import DataInterface

    data_interface = DataInterface()
    gcms = data_interface.data_catalog.df.source_id.unique()

    # Reading GCM warming levels 1850-1900 table
    temp_df = pd.read_csv(
        "~/climakitae/climakitae/data/gwl_1850-1900ref_1to4deg_per05.csv"
    )
    # Clean long float column names
    temp_df.columns = np.append(
        temp_df.columns[:3], np.round(temp_df.columns[3:].astype(float), 2).astype(str)
    )

    # Subsetting on the table for only the GCMs that we track
    df = temp_df[temp_df["GCM"].isin(gcms)]
    return df


def warm_level_to_years(warm_df, scenario, warming_level):
    """Given warming level, plot histogram of years and label median year."""
    datetimes = warm_df[warm_df["scenario"] == scenario][warming_level].astype(
        "datetime64[ns]"
    )
    years = datetimes.dt.year
    med_datetime = datetimes.quantile(0.5, interpolation="midpoint")
    med_year = med_datetime.year
    fig, ax = plt.subplots()
    plot_years(ax, years, med_year)
    return None


def plot_years(ax, years, med_year):
    """Plot histogram of years and label median year, on axis."""
    n = len(years)
    ax.hist(years)
    ax.set_title(
        f"Years when each of all {n} model simulations reach the warming level"
    )
    ax.set_xlabel("Year")
    ax.set_ylabel("Number of simulations")
    ax.axvline(
        med_year, color="black", linestyle="--", label=f"Median year is {med_year}"
    )
    ax.legend()
    return ax


def warm_level_to_year(warm_df, scenario, warming_level):
    """Given warming level, give year."""
    med_date = (
        warm_df[warm_df["scenario"] == scenario][warming_level]
        .astype("datetime64[ns]")
        .quantile(0.5, interpolation="midpoint")
    )
    return med_date.year


def warm_level_to_month(warm_df, scenario, warming_level):
    """Given warming level, give month."""
    med_date = (
        warm_df[warm_df["scenario"] == scenario][warming_level]
        .astype("datetime64[ns]")
        .quantile(0.5, interpolation="midpoint")
    )
    return med_date.strftime("%Y-%m")


def plot_warm_levels(fig, ax, levels, med_level):
    """Plot histogram of warming levels, on axis."""
    n = len(levels)
    ax.hist(levels)
    fig.suptitle(f"Warming levels reached in the year by {n} model simulations")
    ax.set_title("(out of 80 simulations)", fontsize=10)
    ax.set_xlabel("Warming level (°C)")
    ax.set_ylabel("Number of simulations")
    # ax.axvline(
    #     med_level, color='black', linestyle='--',
    #     label=f'Median warming level is {med_level}'
    # )
    # ax.legend()
    return ax


def year_to_warm_levels(warm_df, scenario, year):
    """Given year, give warming levels and their median."""
    warm_levels = np.arange(1, 4.01, 0.05).round(2).astype(str)
    date_df = warm_df[warm_df["scenario"] == scenario][warm_levels]

    # Creating new counts dataframe
    counts_df = pd.DataFrame()
    years = pd.to_datetime(date_df.values.flatten()).year
    years_idx = set(years[pd.notna(years)].sort_values())
    counts_df.index = years_idx

    # Creating counts by warming level and year
    for level in warm_levels:
        counts_df = counts_df.merge(
            pd.to_datetime(date_df[level]).dt.year.value_counts(),
            left_index=True,
            right_index=True,
            how="outer",
        )
        counts_df = counts_df.fillna(0)
    counts_df = counts_df.T

    # Find the warming levels and their median
    levels = counts_df[year]
    expanded_levels = [
        float(value) for value, count in levels.items() for _ in range(int(count))
    ]
    med_level = np.quantile(expanded_levels, 0.5, interpolation="nearest")
    return expanded_levels, med_level


def round_to_nearest_half(number):
    # TODO: what to do with .25
    return round(number * 2) / 2


def find_warm_index(warm_df, scenario="ssp370", warming_level=None, year=None):
    """
    Given a scenario and either a warming level or a time, return either the median warming level or month from all simulations within this scenario.

    Note the median warming level is rounded to the nearest half.

    Parameters
    ----------
    scenario: string ('ssp370')
    warming_level: string ('1.5', '2.0', '3.0')
    year: int
        Must be in [2001, 2090].

    """
    if not warming_level and not year:
        print("Pass in either a warming level or a year.")

    elif warming_level is not None and year is not None:
        print("Pass in either a warming level or a year, but not both.")

    else:
        if scenario != "ssp370":
            return print("Scenarios other than ssp370 are under development.")

        # Given warming level, plot years and find month
        if warming_level is not None and year is None:
            allowed_warm_level = ["1.5", "2.0", "3.0"]
            if warming_level not in allowed_warm_level:
                return print(
                    f"Please choose a warming level among {allowed_warm_level}"
                )
            warm_level_to_years(warm_df, scenario, warming_level)
            return warm_level_to_month(warm_df, scenario, warming_level)

        # Given year, plot warming levels and find median
        elif warming_level is None and year is not None:
            min_year, max_year = 2001, 2090  # years with 10+ simulations
            if not (min_year <= year and year <= max_year):
                return print(
                    f"Please provide a year between {min_year} and {max_year}."
                )

            warm_levels, med_level = year_to_warm_levels(warm_df, scenario, year)
            major_levels = np.arange(1, 4.01, 0.5)

            fig, ax = plt.subplots()
            plot_warm_levels(fig, ax, warm_levels, med_level)
            if med_level in major_levels:
                return print(f"The median projected warming level is {med_level}°C. \n")
            else:
                major_level = round_to_nearest_half(med_level)
                print(
                    (
                        "The major warming level nearest to the median "
                        f"projected warming level is {major_level}°C."
                    )
                )
                if med_level < major_level:
                    # TODO: unknown lower_level when major_level == 1
                    lower_level = major_level - 0.5
                    upper_level = major_level
                elif med_level > major_level:
                    # TODO: unknown upper_level when major_level == 4
                    lower_level = major_level
                    upper_level = major_level + 0.5
                return print(
                    (
                        "The actual median projected warming level is between "
                        f"{lower_level} and {upper_level}°C.\n"
                        "Major warming levels considered include 1.0, 1.5, 2.0, "
                        "2.5, 3.0, 3.5 and 4.0°C.\n"
                        f"{med_level}"
                    )
                )


# Lambda function to pass in a created warming level table rather than remaking it with every call.
def create_conversion_function(warm_df):
    return lambda scenario="ssp370", warming_level=None, year=None: find_warm_index(
        warm_df, scenario, warming_level, year
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