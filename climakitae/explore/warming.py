"""Helper functions for performing analyses related to global warming levels, along with backend code for building the warming levels GUI"""

import xarray as xr
import numpy as np
import pandas as pd
import param
import dask

from climakitae.core.data_interface import DataParameters
from climakitae.core.data_load import load
from climakitae.core.paths import (
    gwl_1981_2010_file,
    gwl_1850_1900_file,
)
from climakitae.util.utils import (
    read_csv_file,
    scenario_to_experiment_id,
)
from climakitae.util.colormap import read_ae_colormap

# Silence warnings
import logging

logging.getLogger("param").setLevel(logging.CRITICAL)
xr.set_options(keep_attrs=True)  # Keep attributes when mutating xr objects


class WarmingLevels:
    """A container for all of the warming levels-related functionality:
    - A pared-down Select panel, under "choose_data"
    - a "calculate" step where most of the waiting occurs
    - an optional "visualize" panel, as an instance of WarmingLevelVisualize
    - postage stamps from visualize "main" tab are accessible via "gwl_snapshots"
    - data sliced around gwl window retrieved from "sliced_data"
    """

    catalog_data = xr.DataArray()
    sliced_data = xr.DataArray()
    gwl_snapshots = xr.DataArray()
    gwl_times = pd.DataFrame()
    warming_levels = []

    def __init__(self, **params):
        self.wl_params = WarmingLevelDataParameters()
        self.warming_levels = ["1.5", "2.0", "3.0", "4.0"]


    def calculate(self):
        def _clean_list(data, gwl_times):
            # this is necessary because there are two simulations that
            # lack data for any warming level in the lookup table
            keep_list = list(data.all_sims.values)
            for sim in data.all_sims:
                if _process_item(data.sel(all_sims=sim)) not in list(gwl_times.index):
                    keep_list.remove(sim.item())
            return data.sel(all_sims=keep_list)
        # manually reset to all SSPs, in case it was inadvertently changed by
        # temporarily have ['Dynamical','Statistical'] for downscaling_method
        self.wl_params.scenario_ssp = [
            "SSP 3-7.0 -- Business as Usual",
            "SSP 2-4.5 -- Middle of the Road",
            "SSP 5-8.5 -- Burn it All",
        ]
        # Postage data and anomalies
        self.catalog_data = self.wl_params.retrieve()
        self.catalog_data = self.catalog_data.stack(
            all_sims=["simulation", "scenario"]
        ).squeeze()
        self.catalog_data = self.catalog_data.dropna(dim="all_sims", how="all")
        if self.wl_params.anom == "Yes":
            self.gwl_times = read_csv_file(gwl_1981_2010_file, index_col=[0, 1, 2])
        else:
            self.gwl_times = read_csv_file(gwl_1850_1900_file, index_col=[0, 1, 2])
        self.gwl_times = self.gwl_times.dropna(how="all")
        self.catalog_data = _clean_list(self.catalog_data, self.gwl_times)

        def _find_warming_slice(self, level):
            """
            Find the warming slice data for the current level from the catalog data.
            """
            warming_data = self.catalog_data.groupby("all_sims").map(
                _get_sliced_data,
                level=level,
                years=self.gwl_times,
                window=self.wl_params.window,
                anom=self.wl_params.anom,
            )
            warming_data = warming_data.expand_dims({"warming_level": [level]})
            warming_data = warming_data.assign_attrs(window=self.wl_params.window)

            def _clean_warm_data(warm_data):
                """
                Cleaning the warming levels data in 3 parts:
                1. Removing simulations where this warming level is not crossed. (centered_year)
                2. Removing timestamps at the end to account for leap years (time)
                3. Removing simulations that go past 2100 for its warming level window (all_sims)
                """
                # Check that there exist simulations that reached this warming level before cleaning. Otherwise, don't modify anything.
                if not (warm_data.centered_year.isnull()).all():

                    # Cleaning #1
                    warm_data = warm_data.sel(all_sims=~warm_data.centered_year.isnull())

                    # Cleaning #2
                    # warm_data = warm_data.isel(
                    #     time=slice(0, len(warm_data.time) - 1)
                    # )  # -1 is just a placeholder for 30 year window, this could be more specific.

                    # Cleaning #3
                    # warm_data = warm_data.dropna(dim="all_sims")

                return warm_data

            # Cleaning data
            warming_data = _clean_warm_data(warming_data)

            # Relabeling `all_sims` dimension
            new_warm_data = warming_data.drop("all_sims")
            new_warm_data["all_sims"] = relabel_axis(warming_data["all_sims"])
            return new_warm_data

        self.sliced_data = {}
        self.gwl_snapshots = {}
        for level in self.warming_levels:
            # Assign warming slices to dask computation graph
            warm_slice = load(_find_warming_slice(self, level))
            # Dropping simulations that only have NaNs
            warm_slice = warm_slice.dropna(dim="all_sims", how="all")
            self.gwl_snapshots[level] = warm_slice.reduce(np.nanmean, "time")

            # Renaming time dimension for warming slice once "time" is all computed on
            freq_strs = {"monthly": "months", "daily": "days", "hourly": "hours"}
            warm_slice = warm_slice.rename(
                {"time": f"{freq_strs[warm_slice.frequency]}_from_center"}
            )
            self.sliced_data[level] = warm_slice

        self.gwl_snapshots = xr.concat(self.gwl_snapshots.values(), dim="warming_level")

        def _get_cmap(wl_params):
            """Set colormap depending on variable"""
            if (
                wl_params.variable == "Air Temperature at 2m"
                or wl_params.variable == "Dew point temperature"
            ):
                cmap_name = "ae_orange"
            else:
                cmap_name = "ae_diverging"

            # Read colormap hex
            cmap = read_ae_colormap(cmap=cmap_name, cmap_hex=True)
            return cmap

        self.cmap = _get_cmap(self.wl_params)


def relabel_axis(all_sims_dim):
    # so that hvplot doesn't complain about the all_sims dimension names being tuples:
    new_arr = []
    for one in all_sims_dim:
        temp = list(one.values.item())
        a = temp[0] + "_" + temp[1]
        new_arr.append(a)
    return new_arr


def _process_item(y):
    # get a tuple of identifiers for the lookup table from DataArray indexers
    simulation = y.simulation.item()
    scenario = scenario_to_experiment_id(y.scenario.item().split("+")[1].strip())
    downscaling_method, sim_str, ensemble = simulation.split("_")
    return (sim_str, ensemble, scenario)


def _get_sliced_data(y, level, years, window=15, anom="Yes"):
    """Calculating warming level anomalies mapping function.

    Parameters
    ----------
    y: xr.DataArray
        Data to compute warming level anomolies, one simulation at a time via groupby
    level: str
        Warming level amount
    years: pd.DataFrame
        Lookup table for the date a given simulation reaches each warming level.
    window: int, optional
        Number of years to generate time window for. Default to 15 years.
        For example, a 15 year window would generate a window of 15 years in the past from the central warming level date, and 15 years into the future. I.e. if a warming level is reached in 2030, the window would be (2015,2045).
    scenario: str, one of "ssp370", "ssp585", "ssp245"
        Shared Socioeconomic Pathway. Default to SSP 3-7.0

    Returns
    --------
    anomaly_da: xr.DataArray
        Warming level anomalies at all warming levels for a scenario
    """
    gwl_times_subset = years.loc[_process_item(y)]

    # Checking if the centered year is null, if so, return dummy DataArray
    center_time = gwl_times_subset.loc[level]

    # Dropping leap days before slicing time dimension because the window size can affect number of leap days per slice
    y = y.loc[~((y.time.dt.month == 2) & (y.time.dt.day == 29))]

    if not pd.isna(center_time):
        # Find the centered year
        centered_year = pd.to_datetime(center_time).year
        start_year = centered_year - window
        end_year = centered_year + (window - 1)

        # TODO: This method will create incorrect data for daily data selection, as leap days create different time indices to be merged on. This will create weird time indices that can be visualized with the `out.plot.line` method in `warming_levels.ipynb`.

        if anom == "Yes":
            # Find the anomaly
            anom_val = y.sel(time=slice("1981", "2010")).mean("time")
            sliced = y.sel(time=slice(str(start_year), str(end_year))) - anom_val
        else:
            # Finding window slice of data
            sliced = y.sel(time=slice(str(start_year), str(end_year)))

        # Resetting and renaming time index for each data array so they can overlap and save storage space
        sliced["time"] = np.arange(-len(sliced.time) / 2, len(sliced.time) / 2)

        # Assigning `centered_year` as a coordinate to the DataArray
        sliced = sliced.assign_coords({"centered_year": centered_year})

        return sliced

    else:

        # This creates an approximately appropriately sized DataArray to be dropped later
        if y.frequency == "monthly":
            time_freq = 12
        elif y.frequency == "daily":
            time_freq = 365
        elif y.frequency == "hourly":
            time_freq = 8760

        y = y.isel(
            time=slice(0, window * 2 * time_freq)
        )  # This is to create a dummy slice that conforms with other data structure. Can be re-written to something more elegant.

        # Creating attributes
        y["time"] = np.arange(-len(y.time) / 2, len(y.time) / 2)
        y["centered_year"] = np.nan

        # Returning DataArray of NaNs to be dropped later.
        return xr.full_like(y, np.nan)


def _select_one_gwl(one_gwl, snapshots):
    """
    This needs to happen in two places. You have to drop the sims
    which are nan because they don't reach that warming level, else the
    plotting functions and cross-sim statistics will get confused.
    But it's important that you drop it from a copy, or it may modify the
    original data.
    """
    all_plot_data = snapshots.sel(warming_level=one_gwl).copy()
    all_plot_data = all_plot_data.dropna("all_sims", how="all")
    return all_plot_data


class WarmingLevelDataParameters(DataParameters):
    window = param.Integer(
        default=15,
        bounds=(5, 25),
        doc="Years around Global Warming Level (+/-) \n (e.g. 15 means a 30yr window)",
    )

    anom = param.Selector(
        default="Yes",
        objects=["Yes"],
        doc="Return an anomaly \n(difference from historical reference period)?",
    )

    def __init__(self, **params):
        super().__init__(**params)
        self.downscaling_method = "Dynamical"
        self.scenario_historical = ["Historical Climate"]
        self.area_average = "No"
        self.resolution = "45 km"
        self.scenario_ssp = [
            "SSP 3-7.0 -- Business as Usual",
            "SSP 2-4.5 -- Middle of the Road",
            "SSP 5-8.5 -- Burn it All",
        ]
        self.time_slice = (1980, 2100)
        self.timescale = "monthly"
        self.variable = "Air Temperature at 2m"

        # Location defaults
        self.area_subset = "states"
        self.cached_area = ["CA"]

    @param.depends("downscaling_method", watch=True)
    def _anom_allowed(self):
        """
        Require 'anomaly' for non-bias-corrected data.
        """
        if self.downscaling_method == "Dynamical":
            self.param["anom"].objects = ["Yes"]
            self.anom = "Yes"
        else:
            self.param["anom"].objects = ["Yes", "No"]
            self.anom = "Yes"
