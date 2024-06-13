"""Helper functions for performing analyses related to global warming levels, along with backend code for building the warming levels GUI"""

import xarray as xr
import numpy as np
import pandas as pd
import param
import dask

from climakitae.core.data_load import read_catalog_from_select
from climakitae.util.utils import (
    read_csv_file,
    area_average,
    scenario_to_experiment_id,
)
from climakitae.util.colormap import read_ae_colormap
from climakitae.core.paths import (
    gwl_1981_2010_file,
    gwl_1850_1900_file,
    ssp119_file,
    ssp126_file,
    ssp245_file,
    ssp370_file,
    ssp585_file,
    hist_file,
)
from climakitae.core.data_interface import DataParameters
from climakitae.explore import threshold_tools
from scipy.stats import pearson3
from climakitae.core.data_load import load

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

    def __init__(self, **params):
        self.wl_params = WarmingLevelChoose()
        self.warming_levels = ["1.5", "2.0", "3.0", "4.0"]

    def find_warming_slice(self, level, gwl_times):
        """
        Find the warming slice data for the current level from the catalog data.
        """
        warming_data = self.catalog_data.groupby("all_sims").map(
            get_sliced_data,
            level=level,
            years=gwl_times,
            window=self.wl_params.window,
            anom=self.wl_params.anom,
        )
        warming_data = warming_data.expand_dims({"warming_level": [level]})
        warming_data = warming_data.assign_attrs(window=self.wl_params.window)

        # Cleaning data
        warming_data = clean_warm_data(warming_data)

        # Relabeling `all_sims` dimension
        new_warm_data = warming_data.drop("all_sims")
        new_warm_data["all_sims"] = relabel_axis(warming_data["all_sims"])
        return new_warm_data

    def calculate(self):
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
        self.catalog_data = clean_list(self.catalog_data, self.gwl_times)

        self.sliced_data = {}
        self.gwl_snapshots = {}
        for level in self.warming_levels:
            # Assign warming slices to dask computation graph
            warm_slice = load(self.find_warming_slice(level, self.gwl_times))
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
        self.cmap = _get_cmap(self.wl_params)
        self.wl_viz = WarmingLevelVisualize(
            gwl_snapshots=self.gwl_snapshots,
            wl_params=self.wl_params,
            cmap=self.cmap,
            warming_levels=self.warming_levels,
        )
        self.wl_viz.compute_stamps()


def relabel_axis(all_sims_dim):
    # so that hvplot doesn't complain about the all_sims dimension names being tuples:
    new_arr = []
    for one in all_sims_dim:
        temp = list(one.values.item())
        a = temp[0] + "_" + temp[1]
        new_arr.append(a)
    return new_arr


def process_item(y):
    # get a tuple of identifiers for the lookup table from DataArray indexers
    simulation = y.simulation.item()
    scenario = scenario_to_experiment_id(y.scenario.item().split("+")[1].strip())
    downscaling_method, sim_str, ensemble = simulation.split("_")
    return (sim_str, ensemble, scenario)


def clean_list(data, gwl_times):
    # this is necessary because there are two simulations that
    # lack data for any warming level in the lookup table
    keep_list = list(data.all_sims.values)
    for sim in data.all_sims:
        if process_item(data.sel(all_sims=sim)) not in list(gwl_times.index):
            keep_list.remove(sim.item())
    return data.sel(all_sims=keep_list)


def clean_warm_data(warm_data):
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


def get_sliced_data(y, level, years, window=15, anom="Yes"):
    """Calculating warming level anomalies.

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


class WarmingLevelChoose(DataParameters):
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

    def __init__(self, *args, **params):
        super().__init__(*args, **params)
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


class WarmingLevelVisualize(param.Parameterized):
    """Create Warming Levels panel GUI"""

    ## Intended to be accessed through WarmingLevels class.
    ## Allows the user to toggle between several data options.
    ## Produces dynamically updating gwl snapshot maps.

    # Read in GMT context plot data
    ssp119_data = read_csv_file(ssp119_file, index_col="Year")
    ssp126_data = read_csv_file(ssp126_file, index_col="Year")
    ssp245_data = read_csv_file(ssp245_file, index_col="Year")
    ssp370_data = read_csv_file(ssp370_file, index_col="Year")
    ssp585_data = read_csv_file(ssp585_file, index_col="Year")
    hist_data = read_csv_file(hist_file, index_col="Year")

    warmlevel = param.Selector(
        default=1.5, objects=[1.5, 2, 3, 4], doc="Warming level in degrees Celcius."
    )
    ssp = param.Selector(
        default="All",
        objects=[
            "All",
            "SSP 1-1.9 -- Very Low Emissions Scenario",
            "SSP 1-2.6 -- Low Emissions Scenario",
            "SSP 2-4.5 -- Middle of the Road",
            "SSP 3-7.0 -- Business as Usual",
            "SSP 5-8.5 -- Burn it All",
        ],
        doc="Shared Socioeconomic Pathway.",
    )

    def __init__(self, gwl_snapshots, wl_params, cmap, warming_levels):
        """
        Two things are passed in where this is initialized, and come in through
        *args, and **params
            wl_params: an instance of WarmingLevelParameters
            gwl_snapshots: xarray DataArray -- anomalies at each warming level
        """
        # super().__init__(*args, **params)
        super().__init__()
        self.gwl_snapshots = gwl_snapshots
        self.wl_params = wl_params
        self.warming_levels = warming_levels
        self.cmap = cmap
        some_dims = self.gwl_snapshots.dims  # different names depending on WRF/LOCA
        some_dims = list(some_dims)
        some_dims.remove("warming_level")
        self.mins = self.gwl_snapshots.min(some_dims).compute()
        self.maxs = self.gwl_snapshots.max(some_dims).compute()

    def compute_stamps(self):
        self.main_stamps = GCM_PostageStamps_MAIN_compute(self)
        self.stats_stamps = GCM_PostageStamps_STATS_compute(self)

    @param.depends("warmlevel", watch=True)
    def GCM_PostageStamps_MAIN(self):
        return self.main_stamps[str(float(self.warmlevel))]

    @param.depends("warmlevel", watch=True)
    def GCM_PostageStamps_STATS(self):
        return self.stats_stamps[str(float(self.warmlevel))]

    @param.depends("warmlevel", "ssp", watch=False)
    def GMT_context_plot(self):
        """Display GMT plot using package data that updates whenever the warming level or SSP is changed by the user."""
        ## Plot dimensions
        width = 575
        height = 300

        ## Plot figure
        hist_t = np.arange(1950, 2015, 1)
        cmip_t = np.arange(2015, 2100, 1)

        ## https://pyam-iamc.readthedocs.io/en/stable/tutorials/ipcc_colors.html
        c119 = "#00a9cf"
        c126 = "#003466"
        c245 = "#f69320"
        c370 = "#df0000"
        c585 = "#980002"

        ipcc_data = self.hist_data.hvplot(
            y="Mean", color="k", label="Historical", width=width, height=height
        ) * self.hist_data.hvplot.area(
            x="Year",
            y="5%",
            y2="95%",
            alpha=0.1,
            color="k",
            ylabel="°C",
            xlabel="",
            ylim=[-1, 5],
            xlim=[1950, 2100],
        )
        if self.ssp == "All":
            ipcc_data = (
                ipcc_data
                * self.ssp119_data.hvplot(y="Mean", color=c119, label="SSP1-1.9")
                * self.ssp126_data.hvplot(y="Mean", color=c126, label="SSP1-2.6")
                * self.ssp245_data.hvplot(y="Mean", color=c245, label="SSP2-4.5")
                * self.ssp370_data.hvplot(y="Mean", color=c370, label="SSP3-7.0")
                * self.ssp585_data.hvplot(y="Mean", color=c585, label="SSP5-8.5")
            )
        elif self.ssp == "SSP 1-1.9 -- Very Low Emissions Scenario":
            ipcc_data = ipcc_data * self.ssp119_data.hvplot(
                y="Mean", color=c119, label="SSP1-1.9"
            )
        elif self.ssp == "SSP 1-2.6 -- Low Emissions Scenario":
            ipcc_data = ipcc_data * self.ssp126_data.hvplot(
                y="Mean", color=c126, label="SSP1-2.6"
            )
        elif self.ssp == "SSP 2-4.5 -- Middle of the Road":
            ipcc_data = ipcc_data * self.ssp245_data.hvplot(
                y="Mean", color=c245, label="SSP2-4.5"
            )
        elif self.ssp == "SSP 3-7.0 -- Business as Usual":
            ipcc_data = ipcc_data * self.ssp370_data.hvplot(
                y="Mean", color=c370, label="SSP3-7.0"
            )
        elif self.ssp == "SSP 5-8.5 -- Burn it All":
            ipcc_data = ipcc_data * self.ssp585_data.hvplot(
                y="Mean", color=c585, label="SSP5-8.5"
            )

        # SSP intersection lines
        cmip_t = np.arange(2015, 2101, 1)

        # Warming level connection lines & additional labeling
        warmlevel_line = hv.HLine(self.warmlevel).opts(
            color="black", line_width=1.0
        ) * hv.Text(
            x=1964,
            y=self.warmlevel + 0.25,
            text=".    " + str(self.warmlevel) + "°C warming level",
        ).opts(
            style=dict(text_font_size="8pt")
        )

        # Create plot
        to_plot = ipcc_data * warmlevel_line

        if self.ssp != "All":
            # Label to give addional plot info
            info_label = "Intersection information"

            # Add interval line and shading around selected SSP
            ssp_dict = {
                "SSP 1-1.9 -- Very Low Emissions Scenario": (self.ssp119_data, c119),
                "SSP 1-2.6 -- Low Emissions Scenario": (self.ssp126_data, c126),
                "SSP 2-4.5 -- Middle of the Road": (self.ssp245_data, c245),
                "SSP 3-7.0 -- Business as Usual": (self.ssp370_data, c370),
                "SSP 5-8.5 -- Burn it All": (self.ssp585_data, c585),
            }

            ssp_selected = ssp_dict[self.ssp][0]  # data selected
            ssp_color = ssp_dict[self.ssp][1]  # color corresponding to ssp selected

            # Shading around selected SSP
            ci_label = "90% interval"
            ssp_shading = ssp_selected.hvplot.area(
                x="Year", y="5%", y2="95%", alpha=0.28, color=ssp_color, label=ci_label
            )
            to_plot = to_plot * ssp_shading

            # If the mean/upperbound/lowerbound does not cross threshold,
            # set to 2100 (not visible)
            if (np.argmax(ssp_selected["Mean"] > self.warmlevel)) > 0:
                # Add dashed line
                label1 = "Warming level reached"
                year_warmlevel_reached = (
                    ssp_selected.where(ssp_selected["Mean"] > self.warmlevel)
                    .dropna()
                    .index[0]
                )
                ssp_int = hv.Curve(
                    [[year_warmlevel_reached, -2], [year_warmlevel_reached, 10]],
                    label=label1,
                ).opts(color=ssp_color, line_dash="dashed", line_width=1)
                ssp_int = ssp_int * hv.Text(
                    x=year_warmlevel_reached - 2,
                    y=4.5,
                    text=str(int(year_warmlevel_reached)),
                    rotation=90,
                    label=label1,
                ).opts(style=dict(text_font_size="8pt", color=ssp_color))
                to_plot *= ssp_int  # Add to plot

            if (np.argmax(ssp_selected["95%"] > self.warmlevel)) > 0 and (
                np.argmax(ssp_selected["5%"] > self.warmlevel)
            ) > 0:
                # Make 95% CI line
                x_95 = cmip_t[0] + np.argmax(ssp_selected["95%"] > self.warmlevel)
                ssp_firstdate = hv.Curve([[x_95, -2], [x_95, 10]], label=ci_label).opts(
                    color=ssp_color, line_width=1
                )
                to_plot *= ssp_firstdate

                # Make 5% CI line
                x_5 = cmip_t[0] + np.argmax(ssp_selected["5%"] > self.warmlevel)
                ssp_lastdate = hv.Curve([[x_5, -2], [x_5, 10]], label=ci_label).opts(
                    color=ssp_color, line_width=1
                )
                to_plot *= ssp_lastdate

                ## Bar to connect firstdate and lastdate of threshold cross
                bar_y = -0.5
                yr_len = [(x_95, bar_y), (x_5, bar_y)]
                yr_rng = np.argmax(ssp_selected["5%"] > self.warmlevel) - np.argmax(
                    ssp_selected["95%"] > self.warmlevel
                )
                if yr_rng > 0:
                    interval = hv.Curve(
                        [[x_95, bar_y], [x_5, bar_y]], label=ci_label
                    ).opts(color=ssp_color, line_width=1) * hv.Text(
                        x=x_95 + 5,
                        y=bar_y + 0.25,
                        text=str(yr_rng) + "yrs",
                        label=ci_label,
                    ).opts(
                        style=dict(text_font_size="8pt", color=ssp_color)
                    )

                    to_plot *= interval

        to_plot.opts(
            opts.Overlay(
                title="Global mean surface temperature change relative to 1850-1900",
                fontsize=12,
            )
        )
        to_plot.opts(legend_position="bottom", fontsize=10)
        return to_plot
