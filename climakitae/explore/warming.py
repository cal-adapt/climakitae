"""
Helper functions for performing analyses related to global warming levels, along with
backend code for building the warming levels GUI
"""

import calendar

# Silence warnings
import logging
from typing import Iterable, List

import numpy as np
import pandas as pd
import param
import xarray as xr
from tqdm.auto import tqdm

from climakitae.core.constants import SSPS, WARMING_LEVELS
from climakitae.core.data_interface import DataParameters
from climakitae.core.data_load import load
from climakitae.core.paths import gwl_1850_1900_file, gwl_1981_2010_file
from climakitae.util.utils import (
    _get_cat_subset,
    read_csv_file,
    scenario_to_experiment_id,
)

# warnings.simplefilter(action="ignore", category=FutureWarning)


logging.getLogger("param").setLevel(logging.CRITICAL)
xr.set_options(keep_attrs=True)  # Keep attributes when mutating xr objects
# Remove param's parameter descriptions from docstring because
# ANSI escape sequences in them complicate their rendering
param.parameterized.docstring_describe_params = False
# Docstring signatures are also hard to read and therefore removed
param.parameterized.docstring_signature = False


class WarmingLevels:
    """
    A container for all of the warming levels-related functionality:
    - A pared-down Select panel, under "choose_data"
    - a "calculate" step where most of the waiting occurs
    - an optional "visualize" panel, as an instance of WarmingLevelVisualize
    - postage stamps from visualize "main" tab are accessible via "gwl_snapshots"
    - data sliced around gwl window retrieved from "sliced_data"
    """

    catalog_data = xr.DataArray()
    sliced_data = xr.DataArray()
    gwl_snapshots = xr.DataArray()

    def __init__(self):
        self.wl_params = WarmingLevelChoose()
        # self.warming_levels = ["0.8", "1.2", "1.5", "2.0", "3.0", "4.0"]
        self.warming_levels = _check_available_warming_levels()
        self.gwl_times = None  # Placeholder for the warming level times
        self.catalog_data = None  # Placeholder for the catalog data

    def find_warming_slice(self, level: str, gwl_times: pd.DataFrame) -> xr.DataArray:
        """
        Find the warming slice data for the current level from the catalog data.

        Parameters
        ----------
        level: str
            The warming level to find the slice for.
        gwl_times: pd.DataFrame
            The DataFrame containing the warming level times.

        Returns
        -------
        xr.DataArray
            The warming slice data for the specified level.
        """
        warming_data = self.catalog_data.groupby("all_sims").map(
            get_sliced_data,
            level=level,
            years=gwl_times,
            months=self.wl_params.months,
            window=self.wl_params.window,
            anom=self.wl_params.anom,
        )
        warming_data = warming_data.expand_dims({"warming_level": [level]})
        warming_data = warming_data.assign_attrs(
            window=self.wl_params.window, months=self.wl_params.months
        )

        # Cleaning data
        warming_data = clean_warm_data(warming_data)

        # Relabeling `all_sims` dimension
        new_warm_data = warming_data.drop_vars("all_sims")
        new_warm_data["all_sims"] = relabel_axis(warming_data["all_sims"])
        return new_warm_data

    def calculate(self):
        """
        Calculate the warming levels for the selected parameters.

        This function retrieves the data from the catalog, slices it according to the
        warming levels, and stores the results in the `sliced_data` and `gwl_snapshots`
        attributes.
        """
        # manually reset to all SSPs, in case it was inadvertently changed by
        # temporarily have ['Dynamical','Statistical'] for downscaling_method
        self.wl_params.scenario_ssp = SSPS

        # Postage data and anomalies
        self.catalog_data = self.wl_params.retrieve()
        self.catalog_data = self.catalog_data.stack(all_sims=["simulation", "scenario"])

        # Dropping invalid simulations that come up from stacking scenarios and simulations together
        self.catalog_data = _drop_invalid_sims(self.catalog_data, self.wl_params)

        if self.wl_params.anom == "Yes":
            self.gwl_times = read_csv_file(gwl_1981_2010_file, index_col=[0, 1, 2])
        else:
            self.gwl_times = read_csv_file(gwl_1850_1900_file, index_col=[0, 1, 2])
        self.gwl_times = self.gwl_times.dropna(how="all")
        self.catalog_data = clean_list(self.catalog_data, self.gwl_times)

        self.sliced_data = {}
        self.gwl_snapshots = {}
        for level in tqdm(
            self.wl_params.warming_levels, desc="Computing each warming level"
        ):
            warm_slice = self.find_warming_slice(level, self.gwl_times)
            if self.wl_params.load_data:
                warm_slice = load(warm_slice, progress_bar=True)

            # Add GWL snapshots
            self.gwl_snapshots[level] = warm_slice.mean(dim="time", skipna=True)

            # Renaming time dimension for warming slice once "time" is all computed on
            freq_strs = {"monthly": "months", "daily": "days", "hourly": "hours"}
            warm_slice = warm_slice.rename(
                {"time": f"{freq_strs[warm_slice.frequency]}_from_center"}
            )
            self.sliced_data[level] = warm_slice

        self.gwl_snapshots = xr.concat(self.gwl_snapshots.values(), dim="warming_level")


def relabel_axis(all_sims_dim: Iterable) -> List[str]:
    """
    Converts an iterable of tuples into a list of strings by concatenating the first two elements
    of each tuple with an underscore (`_`).

    This function is designed to simplify dimension names, particularly for compatibility with
    plotting libraries like `hvplot`, which may not handle tuple-based dimension names well.

    Parameters
    ----------
    all_sims_dim : Iterable
        An iterable containing elements that can be converted into tuples. Each element is expected
        to have a `.values.item()` method to extract the tuple.

    Returns
    -------
    List[str]
        A list of strings where each string is formed by concatenating the first two elements of
        the tuples in `all_sims_dim` with an underscore (`_`).

    Raises
    ------
    AttributeError
        If an element in `all_sims_dim` does not have a `.values.item()` method.
    IndexError
        If a tuple in `all_sims_dim` does not have at least two elements.

    Examples
    --------
    >>> import xarray as xr
    >>> # The input `all_sims_dim` is typically an xarray.DataArray
    >>> # representing a coordinate, often created by stacking dimensions.
    >>> # For example, if `ds` is a Dataset with 'simulation' and 'scenario'
    >>> # coordinates, then `ds.stack(all_sims=('simulation', 'scenario'))['all_sims']`
    >>> # would be such an input.
    >>>
    >>> # Create an example of such an xarray.DataArray:
    >>> simulation_scenario_pairs = [
    ...     ('ModelA_run1', 'SSP1-2.6'),
    ...     ('ModelB_run2', 'SSP5-8.5')
    ... ]
    >>> # This DataArray holds the coordinate values (the tuples).
    >>> all_sims_coordinate = xr.DataArray(
    ...     data=simulation_scenario_pairs,
    ...     dims=['all_sims'],
    ...     name='all_sims_stacked_coordinate'
    ... )
    >>> # The function iterates over `all_sims_coordinate`. Each element `one`
    >>> # (as in the function's loop) is a 0-D xarray.DataArray containing one tuple.
    >>> # For the first pair, `one.values.item()` would yield ('ModelA_run1', 'SSP1-2.6').
    >>> relabel_axis(all_sims_coordinate)
    ['ModelA_run1_SSP1-2.6', 'ModelB_run2_SSP5-8.5']
    """
    new_arr = []
    for one in all_sims_dim:
        temp = list(one.values.item())
        a = temp[0] + "_" + temp[1]
        new_arr.append(a)
    return new_arr


def process_item(y: xr.DataArray) -> tuple[str, str, str]:
    """
    Extracts and processes simulation metadata from an xarray DataArray.

    This function retrieves identifiers for a simulation, including the simulation string,
    ensemble, and scenario, and returns them as a tuple. The scenario string is processed
    using the `scenario_to_experiment_id` function to standardize its format.

    Parameters
    ----------
    y : xr.DataArray
        An xarray DataArray containing metadata about a simulation. It is expected to have
        `simulation` and `scenario` attributes that can be accessed using `.item()`.

    Returns
    -------
    tuple[str, str, str]
        A tuple containing:
        - `sim_str` (str): The second part of the `simulation` string.
        - `ensemble` (str): The third part of the `simulation` string.
        - `scenario` (str): The processed scenario identifier.

    Raises
    ------
    AttributeError
        If `y` does not have `simulation` or `scenario` attributes.
    ValueError
        If the `simulation` string cannot be split into three parts.

    Examples
    --------
    >>> y = xr.DataArray(attrs={
    ...     "simulation": "Dynamical_sim1_ensemble1",
    ...     "scenario": "Historical + ssp585"
    ... })
    >>> process_item(y)
    ('sim1', 'ensemble1', 'ssp585')
    """
    simulation = y.simulation.item()
    scenario = scenario_to_experiment_id(y.scenario.item().split("+")[1].strip())
    _, sim_str, ensemble = simulation.split("_")
    return (sim_str, ensemble, scenario)


def clean_list(data: xr.Dataset, gwl_times: pd.DataFrame) -> xr.Dataset:
    """
    Filters an xarray dataset to retain only simulations with valid warming level data.

    This function removes simulations from the dataset that do not have corresponding entries
    in the provided lookup table (`gwl_times`). It ensures that only valid simulations are
    included for further analysis.

    Parameters
    ----------
    data : xr.Dataset
        An xarray dataset containing a dimension `all_sims`, which represents simulation metadata.
    gwl_times : pd.DataFrame
        A pandas DataFrame acting as a lookup table. Its index should contain valid simulation
        metadata (e.g., simulation string, ensemble, and scenario).

    Returns
    -------
    xr.Dataset
        A filtered xarray dataset containing only simulations with valid warming level data.

    Raises
    ------
    AttributeError
        If `data` does not have a dimension named `all_sims`.
    KeyError
        If `process_item` fails to find a simulation in the `gwl_times` index.
    """
    # Create a list of all simulation identifiers
    keep_list = list(data.all_sims.values)
    # Iterate over each simulation and check if it exists in the lookup table
    for sim in data.all_sims:
        if process_item(data.sel(all_sims=sim)) not in list(gwl_times.index):
            keep_list.remove(sim.item())
    # Filter the dataset to retain only valid simulations
    return data.sel(all_sims=keep_list)


def clean_warm_data(warm_data: xr.DataArray) -> xr.DataArray:
    """
    Cleans warming level data by removing invalid simulations and timestamps.

    This function performs the following cleaning steps:
    1. Removes simulations where the warming level is not crossed (i.e., `centered_year` is null).
    2. (Optional) Removes timestamps at the end to account for leap years.
    3. (Optional) Removes simulations that exceed the year 2100.

    Parameters
    ----------
    warm_data : xr.DataArray
        An xarray DataArray containing warming level data. It is expected to have a `centered_year`
        attribute and dimensions like `all_sims` and `time`.

    Returns
    -------
    xr.DataArray
        The cleaned xarray DataArray with invalid simulations and timestamps removed.

    Raises
    ------
    AttributeError
        If `warm_data` does not have the required attributes or dimensions.
    """
    # Check that there exist simulations that reached this warming level before cleaning. Otherwise, don't modify anything.
    if not (warm_data.centered_year.isnull()).all():

        # Cleaning #1
        if not (warm_data.centered_year.isnull()).all():
            # Use .values to get numpy array of booleans instead of DataArray
            warm_data = warm_data.sel(all_sims=~warm_data.centered_year.isnull().values)

        # Cleaning #2
        # warm_data = warm_data.isel(
        #     time=slice(0, len(warm_data.time) - 1)
        # )  # -1 is just a placeholder for 30 year window, this could be more specific.

        # Cleaning #3
        # warm_data = warm_data.dropna(dim="all_sims")

    return warm_data


def get_sliced_data(
    y: xr.DataArray,
    level: str,
    years: pd.DataFrame,
    months: Iterable = np.arange(1, 13),
    window: int = 15,
    anom: str = "No",
) -> xr.DataArray:
    """Calculating warming level anomalies.

    Parameters
    ----------
    y: xr.DataArray
        Data to compute warming level anomolies, one simulation at a time via groupby
    level: str
        Warming level amount
    years: pd.DataFrame
        Lookup table for the date a given simulation reaches each warming level.
    months: np.ndarray
        Months to include in a warming level slice.
    window: int, optional
        Number of years to generate time window for. Default to 15 years.
        For example, a 15 year window would generate a window of 15 years in the past from the central warming level date, and 15 years into the future. I.e. if a warming level is reached in 2030, the window would be (2015,2045).
    scenario: str, one of "ssp370", "ssp585", "ssp245"
        Shared Socioeconomic Pathway. Default to SSP 3-7.0

    Returns
    --------
    anomaly_da: xr.DataArray
    """
    gwl_times_subset = years.loc[process_item(y)]

    # Checking if the centered year is null, if so, return dummy DataArray
    center_time = gwl_times_subset.loc[level]

    # Dropping leap days before slicing time dimension because the window size can affect number of leap days per slice
    y = y.loc[~((y.time.dt.month == 2) & (y.time.dt.day == 29))]

    if not pd.isna(center_time):

        # Find the centered year
        centered_year = pd.to_datetime(center_time).year
        start_year = centered_year - window
        end_year = centered_year + (window - 1)

        if anom == "Yes":
            # Find the anomaly
            anom_val = y.sel(time=slice("1981", "2010")).mean(
                "time"
            )  # Calvin- this line is run 3-4x the number of times it actually needs to be run. Each simulation gets this value calculated for each warming level, so there is no need to calculate this 3-4x when it only needs to be calculated once.
            sliced = y.sel(time=slice(str(start_year), str(end_year))) - anom_val
        else:
            # Finding window slice of data
            sliced = y.sel(time=slice(str(start_year), str(end_year)))

        # Creating a mask for timestamps that are within the desired months
        valid_months_mask = sliced.time.dt.month.isin([months])

        ### Resetting and renaming time index for each data array so they can overlap and save storage space.
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

        # Add user warning if the total slice is missing any time that exceeds the 2100 year bound.
        if len(sliced["time"]) < expected_counts[sliced.frequency]:
            try:
                print(
                    f"\nWarming Level data for {sliced.simulation[0].item()} is not completely available, since the warming level slice's center year is towards the end of the century. All other valid data is returned.\n"
                )
            except:
                print(
                    "\nWarming Level data for a simulation is not completely available, since the warming level slice's center year is towards the end of the century. All other valid data is returned.\n"
                )

        # Removing data not in the desired months (in this new time dimension)
        sliced = sliced.sel(time=valid_months_mask)

        # Assigning `centered_year` as a coordinate to the DataArray
        sliced = sliced.assign_coords({"centered_year": centered_year})

        return sliced

    else:

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
                    f"Invalid frequency '{y.frequency}'. Expected 'monthly', 'daily', or 'hourly'."
                )
        y = y.isel(
            time=slice(0, window * 2 * time_freq)
        )  # This is to create a dummy slice that conforms with other data structure. Can be re-written to something more elegant.

        # Creating attributes
        y["time"] = np.arange(-len(y.time) / 2, len(y.time) / 2)
        y["centered_year"] = np.nan

        # Returning DataArray of NaNs to be dropped later.
        return xr.full_like(y, np.nan)


class WarmingLevelChoose(DataParameters):
    """
    Class for selecting data at specific warming levels in climate datasets.
    This class extends DataParameters to provide functionality for choosing and analyzing
    data around specific global warming levels (GWLs). It allows users to specify a time window
    around the warming level and whether to return anomalies relative to a historical reference period.
    Attributes:
        window (param.Integer): Size of the time window (in years) around the global warming level.
                                The default is 15 years (i.e., a 30-year window centered on the GWL).
        anom (param.Selector): Whether to return data as anomalies (difference from historical
                                reference period). Options are "Yes" or "No".
        warming_levels (list): Available warming levels for selection.
        months (numpy.ndarray): Available months (1-12) for selection.
        load_data (bool): Whether to load data as it's being computed. Setting to False allows
                         for batch processing or working with smaller chunks of data.
    Methods:
        _anom_allowed(): Controls whether the anomaly option is required based on the
                        downscaling method.
    """

    window = param.Integer(
        default=15,
        bounds=(5, 25),
        doc="Years around Global Warming Level (+/-) \n (e.g. 15 means a 30yr window)",
    )

    anom = param.Selector(
        default="Yes",
        objects=["Yes", "No"],
        doc="Return a delta signal \n(difference from historical reference period)?",
    )

    def __init__(self, *args, **params):
        super().__init__(*args, **params)
        self.downscaling_method = "Dynamical"
        self.scenario_historical = ["Historical Climate"]
        self.area_average = "No"
        self.resolution = "45 km"
        self.scenario_ssp = SSPS
        self.time_slice = (1980, 2100)
        self.timescale = "monthly"
        self.variable = "Air Temperature at 2m"

        # Choosing specific warming levels
        self.warming_levels = [str(x) for x in WARMING_LEVELS]
        self.months = np.arange(1, 13)

        # Location defaults
        self.area_subset = "states"
        self.cached_area = ["CA"]

        # Toggle whether or not data is loaded in as it is being computed
        # This may be set to False if you are interested in loading smaller chunks of
        # warming level data at a time, or in batch computing a series of warming level
        # data points by creating all the xarray DataArrays first before loading them
        # all in.
        self.load_data = True

    @param.depends("downscaling_method", watch=True)
    def _anom_allowed(self):
        """
        Require 'anomaly' for non-bias-corrected data.
        """
        if self.downscaling_method == "Dynamical":
            self.param["anom"].objects = ["Yes", "No"]
            self.anom = "Yes"
        else:
            self.param["anom"].objects = ["Yes", "No"]
            self.anom = "Yes"


def _drop_invalid_sims(ds: xr.Dataset, selections: DataParameters) -> xr.Dataset:
    """
    As part of the warming levels calculation, the data is stacked by simulation and
    scenario, creating some empty values for that coordinate.
    Here, we remove those empty coordinate values.

    Parameters
    ----------
    ds : xr.Dataset
        The dataset must have a
        dimension `all_sims` that results from stacking `simulation` and
        `scenario`.
    selections: DataParameters
        The selections made in the GUI, which are used to filter the
        dataset.

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


def _check_available_warming_levels() -> List[float]:
    gwl_times = read_csv_file(gwl_1850_1900_file)
    available_warming_levels = list(
        gwl_times.columns.drop(["GCM", "run", "scenario"]).values
    )
    available_warming_levels = [float(w) for w in available_warming_levels]
    return available_warming_levels
