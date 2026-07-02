"""Global warming level (GWL) analysis utilities.

Provides functions and classes for slicing climate data around global warming
level thresholds, computing anomalies, and supporting the warming levels GUI
workflow.

Classes
-------
WarmingLevels
    High-level container that drives the end-to-end warming levels workflow.
WarmingLevelChoose
    Param-based parameter selection panel for the warming levels GUI.

Functions
---------
relabel_axis
    Flatten stacked ``(simulation, scenario)`` coordinate tuples to strings.
process_item
    Extract standardised ``(sim, ensemble, scenario)`` metadata from a
    single simulation DataArray.
clean_list
    Remove simulations that have no entry in the GWL lookup table.
clean_warm_data
    Drop simulations where the warming level was never reached.
get_sliced_data
    Slice a single simulation to a symmetric window around its GWL crossing
    date, with optional anomaly computation.
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
from climakitae.core.paths import GWL_1850_1900_FILE, GWL_1981_2010_FILE
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
    """High-level container for the warming levels analysis workflow.

    Orchestrates data retrieval, GWL-window slicing, optional anomaly
    computation, and optional in-memory loading for all available warming
    levels.

    Attributes
    ----------
    wl_params : WarmingLevelChoose
        Parameter panel that controls variable, resolution, window size, and
        other query options.
    warming_levels : list of float
        Warming level thresholds derived from the GWL lookup table.
    gwl_times : pandas.DataFrame or None
        Lookup table mapping each simulation/scenario combination to the
        calendar year it crosses each warming level.  Populated by
        :py:meth:`calculate`.
    catalog_data : xarray.DataArray or None
        Raw stacked data retrieved from the catalog.  Populated by
        :py:meth:`calculate`.
    sliced_data : dict of {str: xarray.DataArray}
        Mapping from warming level string (e.g. ``"1.5"``) to the
        corresponding time-windowed DataArray.  Populated by
        :py:meth:`calculate`.
    gwl_snapshots : xarray.DataArray
        Time-mean snapshots for every warming level, concatenated along a
        ``warming_level`` dimension.  Populated by :py:meth:`calculate`.

    Examples
    --------
    >>> wl = WarmingLevels()
    >>> wl.wl_params.variable = "Maximum air temperature at 2m"
    >>> wl.calculate()  # blocks until all levels are computed
    >>> snapshot_2deg = wl.gwl_snapshots.sel(warming_level="2.0")
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
        """Slice catalog data to the window around a single warming level.

        Groups ``catalog_data`` by ``all_sims``, calls :py:func:`get_sliced_data`
        for every simulation, then cleans and relabels the result.

        Parameters
        ----------
        level : str
            Warming level threshold to slice around, e.g. ``"1.5"``.
        gwl_times : pandas.DataFrame
            GWL lookup table produced by :py:meth:`calculate`.  Rows are
            indexed by ``(sim_str, ensemble, scenario)`` tuples; columns are
            warming level strings.

        Returns
        -------
        xarray.DataArray
            DataArray with an expanded ``warming_level`` dimension set to
            ``[level]``.  The ``all_sims`` coordinate is relabelled from
            tuples to ``"simulation_scenario"`` strings.
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
        """Run the full warming levels pipeline and populate instance attributes.

        Retrieves catalog data using ``wl_params``, stacks simulations and
        scenarios, removes invalid combinations, then iterates over every
        warming level to produce time-windowed slices and time-mean snapshots.

        Results are stored in-place:

        - ``catalog_data`` — raw stacked DataArray
        - ``sliced_data``  — dict mapping level → windowed DataArray
        - ``gwl_snapshots`` — concatenated time-mean DataArray

        Notes
        -----
        Progress is reported via a :py:class:`tqdm` progress bar, one tick per
        warming level.  If ``wl_params.load_data`` is ``True``, each slice is
        loaded into memory before moving to the next level.
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
            self.gwl_times = read_csv_file(GWL_1981_2010_FILE, index_col=[0, 1, 2])
        else:
            self.gwl_times = read_csv_file(GWL_1850_1900_FILE, index_col=[0, 1, 2])
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
    """Convert stacked simulation/scenario coordinate tuples to flat strings.

    Joins the first two elements of each coordinate tuple with an underscore,
    producing labels of the form ``"<simulation>_<scenario>"``.

    This is required before handing data to plotting libraries such as
    ``hvplot`` that do not support tuple-valued coordinates.

    Parameters
    ----------
    all_sims_dim : Iterable
        Iterable of 0-D ``xarray.DataArray`` objects, typically obtained via
        ``ds.stack(all_sims=("simulation", "scenario"))["all_sims"]``.  Each
        element must expose a ``.values.item()`` method that returns a tuple.

    Returns
    -------
    list of str
        One string per element, formed by joining the first two tuple elements
        with ``"_"``.

    Examples
    --------
    >>> import xarray as xr
    >>> pairs = [('ModelA_r1i1p1f1', 'ssp245'), ('ModelB_r1i1p1f1', 'ssp585')]
    >>> coord = xr.DataArray(pairs, dims=['all_sims'])
    >>> relabel_axis(coord)
    ['ModelA_r1i1p1f1_ssp245', 'ModelB_r1i1p1f1_ssp585']
    """
    new_arr = []
    for one in all_sims_dim:
        temp = list(one.values.item())
        a = temp[0] + "_" + temp[1]
        new_arr.append(a)
    return new_arr


def process_item(y: xr.DataArray) -> tuple[str, str, str]:
    """Extract standardised simulation metadata from a single-simulation DataArray.

    Parses the ``simulation`` coordinate (``"<activity>_<source>_<member>"``)
    and the ``scenario`` coordinate (``"Historical + <ssp>"``), returning the
    three fields used as the row key in the GWL lookup table.

    Parameters
    ----------
    y : xarray.DataArray
        A DataArray representing one simulation.  Must have ``simulation`` and
        ``scenario`` scalar coordinates accessible via ``.item()``.

    Returns
    -------
    sim_str : str
        Source model identifier (second ``"_"``-delimited field of
        ``simulation``).
    ensemble : str
        Ensemble member identifier (third field of ``simulation``).
    scenario : str
        CMIP6 experiment ID derived from the scenario string, e.g.
        ``"ssp245"``.

    Raises
    ------
    AttributeError
        If ``y`` does not expose ``simulation`` or ``scenario`` coordinates.
    ValueError
        If the ``simulation`` string does not contain exactly three
        ``"_"``-delimited fields.

    Examples
    --------
    >>> import xarray as xr
    >>> y = xr.DataArray(
    ...     0,
    ...     coords={"simulation": "WRF_ACCESS-CM2_r1i1p1f1",
    ...             "scenario": "Historical + SSP 2-4.5"}
    ... )
    >>> process_item(y)
    ('ACCESS-CM2', 'r1i1p1f1', 'ssp245')
    """
    simulation = y.simulation.item()
    scenario = scenario_to_experiment_id(y.scenario.item().split("+")[1].strip())
    _, sim_str, ensemble = simulation.split("_")
    return (sim_str, ensemble, scenario)


def clean_list(data: xr.Dataset, gwl_times: pd.DataFrame) -> xr.Dataset:
    """Remove simulations absent from the GWL lookup table.

    Iterates over every element of the ``all_sims`` dimension and discards
    any simulation whose ``(sim_str, ensemble, scenario)`` key does not
    appear in ``gwl_times.index``.

    Parameters
    ----------
    data : xarray.Dataset
        Dataset with an ``all_sims`` stacked dimension produced by
        ``ds.stack(all_sims=("simulation", "scenario"))``.
    gwl_times : pandas.DataFrame
        GWL lookup table whose index is a MultiIndex of
        ``(sim_str, ensemble, scenario)`` tuples.

    Returns
    -------
    xarray.Dataset
        Dataset containing only the subset of simulations present in
        ``gwl_times``.
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
    """Drop simulations that never reach the target warming level.

    Removes entries from the ``all_sims`` dimension whose ``centered_year``
    coordinate is ``NaN``, indicating that the simulation did not cross the
    requested warming level threshold within the available data.

    Parameters
    ----------
    warm_data : xarray.DataArray
        Output of :py:func:`get_sliced_data` mapped over ``all_sims``.  Must
        have a ``centered_year`` coordinate along the ``all_sims`` dimension.

    Returns
    -------
    xarray.DataArray
        DataArray with non-crossing simulations removed.  Returned unchanged
        if *all* simulations have a null ``centered_year`` (i.e. no simulation
        reached the level).
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
    """Slice a single simulation to a symmetric window around its GWL crossing year.

    Intended to be called via ``xr.DataArray.groupby("all_sims").map(...)``
    so that each simulation is processed independently.

    Leap days are dropped before slicing to ensure a consistent number of
    time steps across all simulations.  The ``time`` dimension is re-indexed
    to integer offsets centred on zero so that slices from different
    simulations can be compared directly.

    If the simulation does not cross ``level`` (i.e. the crossing year is
    ``NaN``), a same-shaped DataArray of ``NaN`` values is returned with
    ``centered_year`` set to ``NaN``; these are removed by
    :py:func:`clean_warm_data`.

    Parameters
    ----------
    y : xarray.DataArray
        Single-simulation DataArray with a ``time`` dimension and a
        ``frequency`` attribute (``"monthly"``, ``"daily"``, or
        ``"hourly"``).
    level : str
        Warming level threshold to slice around, e.g. ``"2.0"``.
    years : pandas.DataFrame
        GWL lookup table indexed by ``(sim_str, ensemble, scenario)``
        tuples; columns are warming level strings.
    months : array-like of int, optional
        Calendar months to retain in the slice.  Defaults to all twelve
        months (``np.arange(1, 13)``).
    window : int, optional
        Half-width of the time window in years.  The returned slice spans
        ``[crossing_year - window, crossing_year + window - 1]``.  Defaults
        to ``15`` (a 30-year window).
    anom : {"Yes", "No"}, optional
        If ``"Yes"``, subtract the 1981-2010 mean before returning.  Defaults
        to ``"No"``.

    Returns
    -------
    xarray.DataArray
        Time-sliced DataArray with integer ``time`` offsets.  Carries a
        ``centered_year`` scalar coordinate.  Filtered to the requested
        ``months``.
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
    """Param-based parameter panel for warming level data selection.

    Extends :py:class:`~climakitae.core.data_interface.DataParameters` with
    GWL-specific controls: the window half-width, anomaly toggle, target
    warming levels, months filter, and a flag controlling whether data is
    eagerly loaded.

    Attributes
    ----------
    window : param.Integer
        Half-width of the time window in years (default ``15``).  The full
        window spans ``2 * window`` years centred on the GWL crossing date.
    anom : param.Selector
        Whether to express results as anomalies relative to the 1981-2010
        baseline.  Accepts ``"Yes"`` or ``"No"`` (default ``"Yes"``).
    warming_levels : list of str
        Warming level thresholds to compute, e.g. ``["1.5", "2.0", "3.0"]``.
        Defaults to all levels present in the GWL lookup file.
    months : numpy.ndarray
        Calendar months (1-12) to include in each warming level slice.
        Defaults to all twelve months.
    load_data : bool
        If ``True`` (default), each warming level slice is loaded into memory
        immediately after it is computed.  Set to ``False`` to defer loading,
        which is useful when building a collection of lazy DataArrays for
        batch processing.
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
        """Enforce anomaly setting when the downscaling method changes.

        Called automatically by ``param`` whenever ``downscaling_method`` is
        updated.  Currently resets ``anom`` to ``"Yes"`` for all methods;
        the branch structure is retained to support future per-method logic.
        """
        if self.downscaling_method == "Dynamical":
            self.param["anom"].objects = ["Yes", "No"]
            self.anom = "Yes"
        else:
            self.param["anom"].objects = ["Yes", "No"]
            self.anom = "Yes"


def _drop_invalid_sims(ds: xr.Dataset, selections: DataParameters) -> xr.Dataset:
    """Remove empty ``all_sims`` coordinates introduced by stacking.

    When simulation and scenario dimensions are stacked, the Cartesian product
    includes combinations that do not exist in the catalog (e.g. a model that
    only ran ``ssp245`` will have empty entries for ``ssp585``).  This function
    filters ``ds`` to the set of ``(simulation, scenario)`` pairs that are
    actually present in the catalog subset for ``selections``.

    Parameters
    ----------
    ds : xarray.Dataset
        Dataset with an ``all_sims`` stacked dimension.
    selections : DataParameters
        Current GUI selections used to query the catalog subset.

    Returns
    -------
    xarray.Dataset
        Dataset containing only valid ``(simulation, scenario)`` combinations.
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
    """Read the warming level thresholds available in the GWL lookup file.

    Parses the column headers of the 1850-1900 reference GWL CSV file to
    derive the list of supported thresholds, excluding the ``GCM``, ``run``,
    and ``scenario`` metadata columns.

    Returns
    -------
    list of float
        Warming level thresholds as floats, e.g. ``[0.8, 1.0, 1.2, 1.5,
        2.0, 2.5, 3.0, 4.0]``.
    """
    gwl_times = read_csv_file(GWL_1850_1900_FILE)
    available_warming_levels = list(
        gwl_times.columns.drop(["GCM", "run", "scenario"]).values
    )
    available_warming_levels = [float(w) for w in available_warming_levels]
    return available_warming_levels
