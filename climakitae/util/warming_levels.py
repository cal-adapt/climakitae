"""Helper functions related to applying a warming levels approach to a data object"""

import calendar
import numpy as np
import pandas as pd
import xarray as xr
import intake

from typing import Union
from climakitae.util.utils import (
    scenario_to_experiment_id,
    _get_cat_subset,
    resolution_to_gridlabel,
    read_csv_file,
)
from climakitae.core.paths import (
    ssp119_file,
    ssp126_file,
    ssp245_file,
    ssp370_file,
    ssp585_file,
    hist_file,
    data_catalog_url,
    GWL_1850_1900_TIMEIDX_FILE,
    gwl_1850_1900_file,
)


def calculate_warming_level(warming_data, gwl_times, level, months, window):
    """Perform warming level computation for a single warming level.
    Assumes the data has already been stacked by simulation and scenario to create a MultiIndex dimension "all_sims" and that the invalid simulations have been removed such that the gwl_times table can be adequately parsed.
    Internal function only; see the function _apply_warming_levels_approach for more documentation on how this function is applied internally.
    Appropriate attributes for new dimensions are applied by the retrieval function (not here).

    Parameters
    ----------
    warming_data: xr.DataArray
        Data object returned by _get_data_one_var, stacked by simulation/scenario, and then with invalid simulations removed.
    gwl_times: pd.DataFrame
        Global warming levels table indicating when each unique model/run/scenario (simulation) reaches each warming level.
    level: float
        Warming level. Must be a valid column in gwl_times table.
    months: list of int
        Months of the year (in integers) to compute function for.
        i.e. for a full year, months = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
    window: int
        Years around Global Warming Level (+/-) \n (e.g. 15 means a 30yr window)

    Returns
    -------
    warming_data: xr.DataArray

    """
    # Raise error if proper processing has not been performed on the data before calling the function
    if "all_sims" not in warming_data.dims:
        raise AttributeError(
            "Missing an `all_sims` dimension on the dataset. Create `all_sims` with .stack on `simulation` and `scenario`."
        )

    # Apply _get_sliced_data function by simulation dimension
    warming_data = warming_data.groupby("all_sims").map(
        _get_sliced_data, level=level, gwl_times=gwl_times, months=months, window=window
    )

    warming_data = warming_data.expand_dims({"warming_level": [level]})

    # Check that there exist simulations that reached this warming level before cleaning. Otherwise, don't modify anything.
    if not (warming_data.centered_year.isnull()).all():
        # Removing simulations where this warming level is not crossed. (centered_year)
        warming_data = warming_data.sel(all_sims=~warming_data.centered_year.isnull())

    return warming_data


def _get_sliced_data(y, level, gwl_times, months, window):
    """Calculate warming level anomalies.
    Warming level is computed for each individual simulation/scenario.

    Parameters
    ----------
    y: xr.DataArray
        Data to compute warming level anomolies, one simulation at a time via groupby
    gwl_times: pd.DataFrame
        Global warming levels table indicating when each unique model/run/scenario (simulation) reaches each warming level.
    level: float
        Warming level. Must be a valid column in gwl_times table.
    months: list of int
        Months of the year (in integers) to compute function for.
        i.e. for a full year, months = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
    window: int
        Years around Global Warming Level (+/-) \n (e.g. 15 means a 30yr window)

    Returns
    --------
    anomaly_da: xr.DataArray
    """
    # Get the years when the global warming level is reached for all levels available in the gwl_times dataframe
    gwl_times_subset = gwl_times.loc[_extract_string_identifiers(y)]

    # Checking if the centered year is null, if so, return dummy DataArray
    center_time = gwl_times_subset.loc[str(float(level))]

    # Dropping leap days before slicing time dimension because the window size can affect number of leap days per slice
    y = y.loc[~((y.time.dt.month == 2) & (y.time.dt.day == 29))]

    if not pd.isna(center_time):

        # Find the centered year
        centered_year = pd.to_datetime(center_time).year
        start_year = centered_year - window
        end_year = centered_year + (window - 1)

        sliced = y.sel(time=slice(str(start_year), str(end_year)))

        # Creating a mask for timestamps that are within the desired months
        valid_months_mask = sliced.time.dt.month.isin([months])

        # Resetting and renaming time index for each data array so they can overlap and save storage space.
        expected_counts = {
            "monthly": window * 2 * 12,
            "daily": window * 2 * 365,
            "hourly": window * 2 * 8760,
        }
        # There may be missing time for time slices that exceed the 2100 year bound. If that is the case, only return a warming slice for the amount of valid data available AND correctly center `time_from_center` values.
        # Otherwise, if no time is missing, then the warming slice will just center the center year.
        sliced["time"] = np.arange(
            -expected_counts[y.frequency] / 2,
            expected_counts[y.frequency] / 2
            - (expected_counts[y.frequency] - len(sliced)),
        )

        # Removing data not in the desired months (in this new time dimension)
        sliced = sliced.sel(time=valid_months_mask)

        # Assigning `centered_year` as a coordinate to the DataArray
        sliced = sliced.assign_coords({"centered_year": centered_year})

    else:

        # This clause creates an empty DataArray with similar shape to real WL slices
        # to get dropped after the `.groupby` method is finished.

        # Get number of days per month for non-leap year
        days_per_month = {i: calendar.monthrange(2001, i)[1] for i in np.arange(1, 13)}

        # This creates an approximately appropriately sized DataArray to be dropped later
        match y.frequency:
            case "monthly":
                time_freq = len(months)
            case "daily":
                time_freq = sum([days_per_month[month] for month in months])
            case "hourly":
                time_freq = sum([days_per_month[month] for month in months]) * 24
            case _:
                raise ValueError(
                    'frequency needs to be either "hourly", "daily", or "monthly"'
                )
        y = y.isel(
            time=slice(0, window * 2 * time_freq)
        )  # This is to create a dummy slice that conforms with other data structure. Can be re-written to something more elegant.

        # Creating attributes
        y["time"] = np.arange(-len(y.time) / 2, len(y.time) / 2)
        y["centered_year"] = np.nan

        # Returning DataArray of NaNs to be dropped later.
        sliced = xr.full_like(y, np.nan)

    return sliced


def _extract_string_identifiers(da):
    """
    Extract string identifiers from DataArray coordinate.
    Function returns the simulation, the ensemble, and the scenario (in the format of experiement_id which can be used to search the catalog)

    Parameters
    ----------
    da: xr.DataArray
        Catalog data in the format as returned by data_load (with simulation and scenario as coordinates)

    Returns
    -------
    tuple
        Simulation, ensemble, and scenario, as string values
    """
    simulation = da.simulation.item()
    scenario = scenario_to_experiment_id(da.scenario.item().split("+")[1].strip())
    downscaling_method, sim_str, ensemble = simulation.split("_")
    return (sim_str, ensemble, scenario)


def drop_invalid_sims(ds, selections):
    """
    As part of the warming levels calculation, the data is stacked by simulation and scenario, creating some empty values for that coordinate.
    Here, we remove those empty coordinate values.

    Parameters
    ----------
    ds : xr.Dataset
        The dataset must have a
        dimension `all_sims` that results from stacking `simulation` and
        `scenario`.
    data_catalog: pd.DataFrame
        intake catalog, loaded as a pandas dataframe

    Returns
    -------
    xr.Dataset
        The dataset with only valid simulations retained.

    Raises
    ------
    AttributeError
        If the dataset does not have an `all_sims` dimension.
    """
    df = _get_cat_subset(selections).df

    # Just trying to see simulations across SSPs, not including historical period
    filter_df = df[df["experiment_id"] != "historical"]

    # Creating a valid simulation list to filter the original dataset from
    valid_sim_list = list(
        zip(
            filter_df["activity_id"]
            + "_"
            + filter_df["source_id"]
            + "_"
            + filter_df["member_id"],
            filter_df["experiment_id"].apply(
                lambda val: f"Historical + {scenario_to_experiment_id(val, reverse=True)}"
            ),
        )
    )
    return ds.sel(all_sims=valid_sim_list)


def read_warming_level_csvs() -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Reads two CSV files containing global warming level (GWL) data.

    Returns:
        tuple:
            - df (pd.DataFrame): Time-indexed DataFrame (time as index, simulations as columns).
            - other_df (pd.DataFrame): DataFrame with warming levels per simulation (no datetime index).
    """
    df = read_csv_file(GWL_1850_1900_TIMEIDX_FILE, index_col="time", parse_dates=True)
    other_df = read_csv_file(gwl_1850_1900_file)
    return df, other_df


def get_wl_timestamp(series: pd.Series, degree: float) -> Union[str, float]:
    """
    Finds the first timestamp when the series crosses the specified warming level.

    Parameters:
        series (pd.Series): A time-indexed warming level series.
        degree (float): Target warming level.

    Returns:
        str or float: Timestamp as string if crossed, else np.nan.
    """
    if any(series >= degree):
        return series[series >= degree].index[0].strftime("%Y-%m-%d %H:%M")
    return np.nan


def create_new_warming_level_table(warming_level: float) -> pd.DataFrame:
    """
    Returns a table of timestamps when each simulation reaches the given warming level.

    Parameters:
        warming_level (float): New WL to retrieve WL timing for.

    Returns:
        pd.DataFrame: Same DataFrame as `data/gwl_1850-1900ref.csv`, just with a new WL columns with the `warming_level` arg passed.
    """
    df, other_df = read_warming_level_csvs()

    # Map each simulation to its crossing timestamp for the given warming level
    wl_timestamps = {
        col: get_wl_timestamp(df[col], warming_level) for col in df.columns
    }

    result = other_df.copy(deep=True)
    result["sim"] = result["GCM"] + "_" + result["run"] + "_" + result["scenario"]
    timestamp_series = pd.Series(wl_timestamps)

    result[str(warming_level)] = result["sim"].map(timestamp_series)
    result = result.drop(columns="sim")
    result = result.set_index(["GCM", "run", "scenario"])

    return result


def filter_warming_trajectories_to_ae(
    simulations_df: pd.DataFrame,
    warming_trajectories: pd.DataFrame,
    downscaling_method: str,
) -> pd.DataFrame:
    """
    Filters all simulations in `warming_trajectories` to only the ones we have on AE (`simulations_df`).
    Does this filtering by `downscaling_method` as well.

    Parameters:
        simulations_df (pd.DataFrame): Complete simulation dataframe of all simulations in GWL tables.
        warming_trajectories (pd.DataFrame): Full warming trajectory DataFrame, computed from `read_warming_level_csvs`.
        downscaling_method (str): Downscaling method to filter DataFrame by ('LOCA' or 'WRF').

    Returns:
        pd.DataFrame: Filtered `simulations_df` to only simulations accessible on AE.
    """
    columns_to_keep = []
    activity_simulations = simulations_df[
        simulations_df["activity_id"] == downscaling_method
    ]

    for _, row in activity_simulations.iterrows():
        pattern = f"{row['source_id']}_{row['member_id']}_{row['experiment_id']}"
        matches = [col for col in warming_trajectories.columns if pattern in col]
        columns_to_keep.extend(matches)

    return warming_trajectories[columns_to_keep]


def create_ae_warming_trajectories(
    resolution: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Creates warming trajectories for all AE simulations based on a given resolution.
    This resolution is an important parameter because not all resolutions have the same number of WRF simulations (i.e. 3km has 8 but 9km has 10).

    Parameters:
        resolution (str): Grid resolution (e.g., "6km", "12km").

    Returns:
        tuple:
            - LOCA2 warming trajectories (pd.DataFrame)
            - WRF warming trajectories (pd.DataFrame)
    """
    df = intake.open_esm_datastore(data_catalog_url).df
    grid_label = resolution_to_gridlabel(resolution)

    # Only select simulations with the given grid label, since WRF has a different number of simulations depending on the spatial resolution
    select_sims = df[df["grid_label"] == grid_label]

    simulations_df = (
        select_sims[["activity_id", "source_id", "experiment_id", "member_id"]]
        .drop_duplicates()
        .reset_index(drop=True)
    )
    warming_trajectories, _ = read_warming_level_csvs()

    loca2 = filter_warming_trajectories_to_ae(
        simulations_df, warming_trajectories, "LOCA2"
    )
    wrf = filter_warming_trajectories_to_ae(simulations_df, warming_trajectories, "WRF")

    return loca2, wrf


def generate_ssp_dict() -> dict[str, pd.DataFrame]:
    """
    Loads historical and SSP scenario CSVs into one dictionary.

    Returns:
        Dict[str, pd.DataFrame]: A dictionary mapping scenario names to their
        pandas DataFrames, indexed by year.
    """
    files_dict = {
        "Historical": hist_file,
        "SSP 1-1.9": ssp119_file,
        "SSP 1-2.6": ssp126_file,
        "SSP 2-4.5": ssp245_file,
        "SSP 3-7.0": ssp370_file,
        "SSP 5-8.5": ssp585_file,
    }
    return {
        ssp_str: read_csv_file(filename, index_col="Year")
        for ssp_str, filename in files_dict.items()
    }


def get_gwl_at_year(year: int, ssp: str = "all") -> pd.DataFrame:
    """
    Retrieve estimated Global Warming Level (GWL) statistics for a given year.

    Parameters
    ----------
    year : int
        The year for which to retrieve GWL estimates.
    ssp : str, default='all'
        The SSP scenario to use. Use 'all' to retrieve results for all SSPs.

    Returns
    -------
    pd.DataFrame
        A DataFrame with SSPs as rows and '5%', 'Mean', and '95%' as columns,
        containing the warming level estimates for the specified year.
    """
    ssp_dict = generate_ssp_dict()
    wl_timing_df = pd.DataFrame(columns=["5%", "Mean", "95%"])

    if year >= 2015:
        ssp_list = (
            ["SSP 1-1.9", "SSP 1-2.6", "SSP 2-4.5", "SSP 3-7.0", "SSP 5-8.5"]
            if ssp == "all"
            else [ssp]
        )
        # Find the data for the given year and different scenarios
        for scenario in ssp_list:
            data = ssp_dict.get(scenario)
            if year not in data.index:
                print(f"Year {year} not found in {scenario}")
                wl_timing_df.loc[scenario] = [np.nan, np.nan, np.nan]
            else:
                wl_timing_df.loc[scenario] = data.loc[year]

    else:
        # Finding the data from the historical period
        if ssp != "all":
            print(f"Year {year} before 2015, using Historical data")
        hist_data = ssp_dict["Historical"]

        if year not in hist_data.index:
            print(f"Year {year} not found in Historical")
            wl_timing_df.loc["Historical"] = [np.nan, np.nan, np.nan]
        else:
            wl_timing_df.loc["Historical"] = hist_data.loc[year]

    return wl_timing_df


def get_year_at_gwl(gwl: Union[float, int], ssp: str = "all") -> pd.DataFrame:
    """
    Retrieve the year when a given Global Warming Level (GWL) is reached for each SSP scenario.

    Parameters
    ----------
    gwl : float or int
        The Global Warming Level to check (e.g., 1.5, 2.0).
    ssp : str, default='all'
        The SSP scenario to evaluate. Use 'all' to check across all SSPs and the Historical period.

    Returns
    -------
    pd.DataFrame
        A DataFrame with SSPs as rows and columns ['5%', 'Mean', '95%'] indicating the years
        when each warming level threshold is crossed for the respective uncertainty bounds.
        NaN indicates the level was not reached by 2100.
    """
    ssp_dict = generate_ssp_dict()
    wl_timing_df = pd.DataFrame(columns=["5%", "Mean", "95%"])

    ssp_list = (
        ["Historical", "SSP 1-1.9", "SSP 1-2.6", "SSP 2-4.5", "SSP 3-7.0", "SSP 5-8.5"]
        if ssp == "all"
        else [ssp]
    )

    for ssp in ssp_list:
        ssp_selected = ssp_dict[ssp]

        mean_mask = ssp_selected["Mean"] > gwl
        upper_mask = ssp_selected["95%"] > gwl
        lower_mask = ssp_selected["5%"] > gwl

        def first_wl_year(one_ssp: pd.Series, mask: pd.Series) -> Union[int, float]:
            """Return the first year where the pd.Series mask is True, or NaN if none."""
            return one_ssp.index[mask][0] if mask.any() else np.nan

        # Only add data for a scenario if the mean and upper bound of uncertainty reach the gwl
        if mean_mask.any() and upper_mask.any():
            x_5 = first_wl_year(ssp_selected, upper_mask)
            x_95 = first_wl_year(ssp_selected, lower_mask)
            year_gwl_reached = first_wl_year(ssp_selected, mean_mask)

        else:
            x_5 = x_95 = year_gwl_reached = np.nan

        wl_timing_df.loc[ssp] = [x_5, year_gwl_reached, x_95]

    return wl_timing_df
