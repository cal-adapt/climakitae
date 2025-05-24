"""Helper functions related to applying a warming levels approach to a data object"""

import xarray as xr
import numpy as np
import pandas as pd
import calendar
from climakitae.util.utils import scenario_to_experiment_id
from climakitae.util.utils import _get_cat_subset


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
