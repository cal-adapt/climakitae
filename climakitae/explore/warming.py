"""Helper functions for performing analyses related to global warming levels, along with backend code for building the warming levels GUI"""

import hvplot.xarray
import hvplot.pandas
import xarray as xr
import holoviews as hv
from holoviews import opts
import numpy as np
import pandas as pd
import param
import panel as pn
import dask

from climakitae.core.data_load import read_catalog_from_select
from climakitae.core.data_interface import (
    DataParametersWithPanes,
    _selections_param_to_panel,
)
from climakitae.core.data_view import compute_vmin_vmax
from climakitae.util.utils import (
    read_csv_file,
    read_ae_colormap,
    area_average,
    scenario_to_experiment_id,
)
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

    def choose_data(self):
        return warming_levels_select(self.wl_params)

    # @dask.delayed
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

        # Cleaning data
        warming_data = clean_warm_data(warming_data)

        # Relabeling `all_sims` dimension
        new_warm_data = warming_data.drop("all_sims")
        new_warm_data["all_sims"] = relabel_axis(warming_data["all_sims"])

        # Adding warming data to dictionary of data
        # self.sliced_data[level] = warming_data
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
        warming_levels = ["1.5", "2.0", "3.0", "4.0"]
        for level in warming_levels:
            # Assign warming slices to dask computation graph
            self.sliced_data[level] = self.find_warming_slice(level, self.gwl_times)

        # self.gwl_snapshots = self.sliced_data.reduce(np.nanmean, "time")
        # self.gwl_snapshots = self.gwl_snapshots.compute()
        # self.cmap = _get_cmap(self.wl_params)
        # self.wl_viz = WarmingLevelVisualize(
        #     gwl_snapshots=self.gwl_snapshots, wl_params=self.wl_params, cmap=self.cmap
        # )

    def visualize(self):
        if self.wl_viz:
            return warming_levels_visualize(self.wl_viz)
        else:
            print("Please run 'calculate' first.")


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
    # Cleaning #1
    warm_data = warm_data.sel(all_sims=~warm_data.centered_year.isnull())

    # Cleaning #2
    warm_data = warm_data.isel(
        time=slice(0, len(warm_data.time) - 1)
    )  # -1 is just a placeholder for 30 year window, this could be more specific.

    # Cleaning #3
    warm_data = warm_data.dropna(dim="all_sims")

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
    attrs_temp = y.attrs
    dims_temp = y.dims

    # Checking if the centered year is null, if so, return dummy DataArray
    center_time = gwl_times_subset.loc[level]
    if not pd.isna(center_time):
        # Find the centered year
        centered_year = pd.to_datetime(center_time).year
        start_year = centered_year - window
        end_year = centered_year + (window - 1)

        if anom == "Yes":
            sliced = y.sel(time=slice(str(start_year), str(end_year))) - y.sel(
                time=slice("1981", "2010")
            ).mean("time")
        else:
            # Finding window slice of data
            sliced = y.sel(time=slice(str(start_year), str(end_year)))

            # Resetting time index for each data array so they can overlap and save storage space
            sliced["time"] = sliced.time - sliced.time[0]

            # Assigning `centered_year` as a coordinate to the DataArray
            sliced = sliced.assign_coords({"centered_year": centered_year})

        return sliced

    else:
        y = y.isel(
            time=slice(0, window * 2 * 365)
        )  # This is to create a dummy slice that conforms with other data structure. Can be re-written to something more elegant.

        # Creating attributes
        y["time"] = y.time - y.time[0]
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


class WarmingLevelChoose(DataParametersWithPanes):
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
        self.downscaling_method = ["Dynamical"]
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
        if self.downscaling_method == ["Dynamical"]:
            self.param["anom"].objects = ["Yes"]
            self.anom = "Yes"
        else:
            self.param["anom"].objects = ["Yes", "No"]
            self.anom = "Yes"


class WarmingLevelVisualize(param.Parameterized):
    """Generate warming levels panel GUI in notebook.

    Intended to be accessed through WarmingLevels class.
    Allows the user to toggle between several data options.
    Produces dynamically updating gwl snapshot maps.

    Attributes
    ----------
    warmlevel: param.Selector
        Warming level in degrees Celcius.
    ssp: param.Selector
        Shared socioeconomic pathway.
    """

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

    def __init__(self, *args, **params):
        """
        Two things are passed in where this is initialized, and come in through
        *args, and **params
            wl_params: an instance of WarmingLevelParameters
            gwl_snapshots: xarray DataArray -- anomalies at each warming level
        """
        super().__init__(*args, **params)
        some_dims = self.gwl_snapshots.dims  # different names depending on WRF/LOCA
        some_dims = list(some_dims)
        some_dims.remove("warming_level")
        self.mins = self.gwl_snapshots.min(some_dims).compute()
        self.maxs = self.gwl_snapshots.max(some_dims).compute()

    @param.depends("warmlevel", watch=True)
    def GCM_PostageStamps_MAIN(self):
        # Get data to plot
        one_warming_level = str(float(self.warmlevel))
        all_plot_data = _select_one_gwl(one_warming_level, self.gwl_snapshots)
        if all_plot_data.all_sims.size != 0:
            if self.wl_params.variable == "Relative humidity":
                all_plot_data = all_plot_data * 100

            # Set up plotting arguments
            width = 210
            height = 110
            clabel = (
                self.wl_params.variable + " (" + self.gwl_snapshots.attrs["units"] + ")"
            )
            vmin = self.mins.sel(warming_level=one_warming_level).values.item()
            vmax = self.maxs.sel(warming_level=one_warming_level).values.item()
            if (vmin < 0) and (vmax > 0):
                sopt = True
            else:
                sopt = None

            # now prepare the plot object:
            all_plots = all_plot_data.hvplot.image(
                by="all_sims",
                subplots=True,
                colorbar=False,
                clim=(vmin, vmax),
                clabel=clabel,
                cmap=self.cmap,
                symmetric=sopt,
                width=width,
                height=height,
                xaxis=False,
                yaxis=False,
                title="",
            ).cols(4)

            try:
                all_plots.opts(
                    title=self.wl_params.variable
                    + ": for "
                    + str(self.warmlevel)
                    + "°C Warming by Simulation"
                )  # Add title
            except:
                all_plots.opts(title=str(self.warmlevel) + "°C")  # Add shorter title

            all_plots.opts(toolbar="below")  # Set toolbar location
            all_plots.opts(hv.opts.Layout(merge_tools=True))  # Merge toolbar
            return all_plots
        else:
            return None

    @param.depends("warmlevel", watch=True)
    def GCM_PostageStamps_STATS(self):
        # Get data to plot
        one_warming_level = str(float(self.warmlevel))
        all_plot_data = _select_one_gwl(one_warming_level, self.gwl_snapshots)
        if all_plot_data.all_sims.size != 0:
            if self.wl_params.variable == "Relative humidity":
                all_plot_data = all_plot_data * 100

            # compute stats
            def get_name(simulation, my_func_name):
                method, GCM, run, scenario = simulation.split("_")
                return (
                    my_func_name
                    + ": \n"
                    + method
                    + " "
                    + GCM
                    + " "
                    + run
                    + "\n"
                    + scenario.split("+")[1]
                )

            def arg_median(data):
                """
                Returns the simulation closest to the median.
                """
                return data.loc[
                    data == data.quantile(0.5, "all_sims", method="nearest")
                ].all_sims.values.item()

            def find_sim(all_plot_data, area_avgs, stat_funcs, my_func):
                if my_func == "Median":
                    one_sim = all_plot_data.sel(all_sims=stat_funcs[my_func](area_avgs))
                else:
                    which_sim = area_avgs.reduce(stat_funcs[my_func], dim="all_sims")
                    one_sim = all_plot_data.isel(all_sims=which_sim)
                one_sim.all_sims.values = get_name(
                    one_sim.all_sims.values.item(),
                    my_func,
                )
                return one_sim

            area_avgs = area_average(all_plot_data)
            stat_funcs = {
                "Minimum": np.argmin,
                "Maximum": np.argmax,
                "Median": arg_median,
            }
            stats = xr.concat(
                [
                    find_sim(all_plot_data, area_avgs, stat_funcs, one_func)
                    for one_func in stat_funcs
                ],
                dim="all_sims",
            )
            stats.name = "Cross-simulation Statistics"

            # Set up plotting arguments
            width = 410
            height = 210
            clabel = (
                self.wl_params.variable + " (" + self.gwl_snapshots.attrs["units"] + ")"
            )
            vmin = self.mins.sel(warming_level=one_warming_level).values.item()
            vmax = self.maxs.sel(warming_level=one_warming_level).values.item()
            if (vmin < 0) and (vmax > 0):
                sopt = True
            else:
                sopt = None

            # Make plots
            plot_list = []
            for stat in stats:
                plot_list.append(
                    stat.squeeze()
                    .drop(["warming_level"])
                    .hvplot.image(
                        clabel=clabel,
                        cmap=self.cmap,
                        clim=(vmin, vmax),
                        symmetric=sopt,
                        width=width,
                        height=height,
                        xaxis=False,
                        yaxis=False,
                        title=stat.all_sims.values.item(),  # dim has been overwritten with nicer title
                    )
                )
            all_plots = plot_list[0] + plot_list[1] + plot_list[2]

            all_plots.opts(
                title=self.wl_params.variable
                + ": for "
                + str(self.warmlevel)
                + "°C Warming Across Models"
            )  # Add title
            return all_plots.cols(1)
        else:
            return None

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


def warming_levels_select(self):
    """
    An initial pared-down version of the Select panel, with fewer options exposed,
    to help the user select a variable and location for further warming level steps.
    """
    widgets = _selections_param_to_panel(self)

    data_choices = pn.Column(
        widgets["variable_text"],
        widgets["variable"],
        widgets["variable_description"],
        widgets["units_text"],
        widgets["units"],
        widgets["timescale_text"],
        widgets["timescale"],
        widgets["resolution_text"],
        widgets["resolution"],
        width=250,
    )

    col_1_location = pn.Column(
        self.map_view,
        widgets["area_subset"],
        widgets["cached_area"],
        widgets["latitude"],
        widgets["longitude"],
        width=220,
    )

    gwl_specific = pn.Row(
        pn.Column(
            pn.widgets.StaticText(
                value=self.param.window.doc,
                name="",
            ),
            pn.widgets.IntSlider.from_param(self.param.window, name=""),
            width=250,
        ),
        pn.Column(
            pn.widgets.StaticText(value=self.param.anom.doc, name=""),
            pn.widgets.RadioBoxGroup.from_param(self.param.anom, name="", inline=True),
            width=220,
        ),
    )

    most_things = pn.Row(data_choices, pn.layout.HSpacer(width=10), col_1_location)

    # Panel overall structure:
    all_things = pn.Column(
        pn.Row(
            pn.Column(
                widgets["downscaling_method_text"],
                widgets["downscaling_method"],
                width=270,
            ),
            pn.Column(
                widgets["data_warning"],
                width=120,
            ),
        ),
        pn.Spacer(background="black", height=1),
        most_things,
        pn.Spacer(background="black", height=1),
        gwl_specific,
    )

    return pn.Card(
        all_things,
        title="Choose Data to Explore at Global Warming Levels",
        collapsible=False,
    )


def warming_levels_visualize(wl_viz):
    # Create panel doodad!

    GMT_plot = pn.Card(
        pn.Column(
            (
                "Shading around selected global emissions scenario shows the 90% interval"
                " across different simulations. Dotted line indicates when the multi-model"
                " ensemble reaches the selected warming level, while solid vertical lines"
                " indicate when the earliest and latest simulations of that scenario reach"
                " the warming level. Figure and data are reproduced from the"
                " [IPCC AR6 Summary for Policymakers Fig 8]"
                "(https://www.ipcc.ch/report/ar6/wg1/figures/summary-for-policymakers/figure-spm-8/)."
            ),
            pn.widgets.Select.from_param(wl_viz.param.ssp, name="Scenario", width=250),
            wl_viz.GMT_context_plot,
        ),
        title="When do different scenarios reach the warming level?",
        collapsible=False,
        width=600,
        height=515,
    )

    postage_stamps_MAIN = pn.Column(
        pn.widgets.StaticText(
            value=(
                "Panels show the 30-year average centered on the year that each"
                "GCM run (each panel) reaches the specified warming level."
                "If you selected 'Yes' to return an anomaly, you will see the difference"
                "from average over the 1981-2010 historical reference period."
            ),
            width=800,
        ),
        pn.Row(
            wl_viz.GCM_PostageStamps_MAIN,
            pn.Column(
                pn.widgets.StaticText(value="<br><br><br>", width=150),
                pn.widgets.StaticText(
                    value=(
                        "<b>Tip</b>: There's a toolbar below the maps."
                        " Try clicking the magnifying glass to zoom in on a"
                        " particular region. You can also click the save button"
                        " to save a copy of the figure to your computer."
                    ),
                    width=150,
                    style={
                        "border": "1.2px red solid",
                        "padding": "5px",
                        "border-radius": "4px",
                        "font-size": "13px",
                    },
                ),
            ),
        ),
    )

    postage_stamps_STATS = pn.Column(
        pn.widgets.StaticText(
            value=(
                "Panels show the median, minimum, or maximum conditions"
                " across all models. These statistics are computed from the data"
                " in the first panel."
            ),
            width=800,
        ),
        wl_viz.GCM_PostageStamps_STATS,
    )

    map_tabs = pn.Card(
        pn.Row(
            pn.widgets.StaticText(name="", value="Warming level (°C)"),
            pn.widgets.RadioButtonGroup.from_param(wl_viz.param.warmlevel, name=""),
            width=230,
        ),
        pn.Tabs(
            ("Maps of individual simulations", postage_stamps_MAIN),
            (
                "Maps of cross-model statistics: median/max/min",
                postage_stamps_STATS,
            ),
        ),
        title="Regional response at selected warming level",
        width=850,
        height=600,
        collapsible=False,
    )

    warming_panel = pn.Column(GMT_plot, map_tabs)
    return warming_panel


def _make_hvplot(data, clabel, clim, cmap, sopt, title, width=225, height=210):
    """Make single map"""
    if len(data.x) > 1 and len(data.y) > 1:
        # If data has more than one grid cell, make a pretty map
        _plot = data.hvplot.image(
            x="x",
            y="y",
            grid=True,
            width=width,
            height=height,
            xaxis=None,
            yaxis=None,
            clabel=clabel,
            clim=clim,
            cmap=cmap,
            symmetric=sopt,
            title=title,
        )
    else:
        # Make a scatter plot if it's just one grid cell
        _plot = data.hvplot.scatter(
            x="x",
            y="y",
            hover_cols=data.name,
            grid=True,
            width=width,
            height=height,
            xaxis=None,
            yaxis=None,
            clabel=clabel,
            clim=clim,
            cmap=cmap,
            symmetric=sopt,
            title=title,
            s=150,  # Size of marker
        )
    return _plot
