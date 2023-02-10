"""Backend helper functions for performing analyses related to global warming levels."""

import hvplot.xarray
import hvplot.pandas
import xarray as xr
import holoviews as hv
from holoviews import opts
import numpy as np
import pandas as pd
import param
import panel as pn
import pkg_resources
from .utils import _read_ae_colormap
from .data_loaders import _read_catalog_from_select
from .catalog_convert import _scenario_to_experiment_id


# Silence warnings
import logging

logging.getLogger("param").setLevel(logging.CRITICAL)

xr.set_options(keep_attrs=True)  # Keep attributes when mutating xr objects

# Variable info
var_catalog_resource = pkg_resources.resource_filename(
    "climakitae", "data/variable_descriptions.csv"
)
var_catalog = pd.read_csv(var_catalog_resource, index_col="variable_id")

# Global warming levels file (years when warming level is reached)
gwl_file = pkg_resources.resource_filename("climakitae", "data/gwl_1981-2010ref.csv")
gwl_times = pd.read_csv(gwl_file).rename(
    columns={"Unnamed: 0": "simulation", "Unnamed: 1": "run"}
)

# Read in GMT context plot data
ssp119 = pkg_resources.resource_filename("climakitae", "data/tas_global_SSP1_1_9.csv")
ssp126 = pkg_resources.resource_filename("climakitae", "data/tas_global_SSP1_2_6.csv")
ssp245 = pkg_resources.resource_filename("climakitae", "data/tas_global_SSP2_4_5.csv")
ssp370 = pkg_resources.resource_filename("climakitae", "data/tas_global_SSP3_7_0.csv")
ssp585 = pkg_resources.resource_filename("climakitae", "data/tas_global_SSP5_8_5.csv")
hist = pkg_resources.resource_filename("climakitae", "data/tas_global_Historical.csv")
ssp119_data = pd.read_csv(ssp119, index_col="Year")
ssp126_data = pd.read_csv(ssp126, index_col="Year")
ssp245_data = pd.read_csv(ssp245, index_col="Year")
ssp370_data = pd.read_csv(ssp370, index_col="Year")
ssp585_data = pd.read_csv(ssp585, index_col="Year")
hist_data = pd.read_csv(hist, index_col="Year")


def _get_postage_data(selections, location, cat):
    """
    This function pulls data from the catalog and reads it into memory

    Args:
        selections (DataLoaders): object holding user's selections
        location (LocSelectorArea): location object containing boundary information
        cat (intake_esm.core.esm_datastore): catalog

    Returns:
        data (xr.DataArray): data to use for creating postage stamp data

    """
    # Read data from catalog
    data = _read_catalog_from_select(selections=selections, location=location, cat=cat)
    data = data.compute()  # Read into memory
    return data


def get_anomaly_data(data, warmlevel=3.0, scenario="ssp370"):
    """Calculating warming level anomalies.

    Parameters
    ----------
    data: xr.DataArray
        Data to compute warming level anomolies
    warmlevel: float, optional
        Warming level (in deg C) to use. Default to 3 degC
    scenario: str, one of "ssp370", "ssp585", "ssp245"
        Shared Socioeconomic Pathway. Default to SSP 3-7.0

    Returns
    --------
    xr.DataArray
        Warming level anomalies at the input warming level and scenario
    """
    sim_names = {
        "cesm2": "CESM2",
        "cnrm-esm2-1": "CNRM-ESM2-1",
        "ec-earth3-veg": "EC-Earth3-Veg",
        "fgoals-g3": "FGOALS-g3",
        "mpi-esm1-2-lr": "MPI-ESM1-2-LR",
    }
    sim_and_runs_dict = {
        "cesm2": "r11i1p1f1",
        "cnrm-esm2-1": "r1i1p1f2",
        "fgoals-g3": "r1i1p1f1",
        "ec-earth3-veg": "r1i1p1f1",
    }
    all_sims = xr.Dataset()
    all_sims.attrs = data.attrs
    central_year_l, year_start_l, year_end_l = [], [], []
    for simulation in data.simulation.values:
        one_ts = data.sel(simulation=simulation).squeeze()
        gwl_times_subset = gwl_times[
            (gwl_times["simulation"] == sim_names[simulation])
            & (gwl_times["run"] == sim_and_runs_dict[simulation])
            & (gwl_times["scenario"] == scenario)
        ]
        centered_time_pd = gwl_times_subset[str(float(warmlevel))]
        centered_time = pd.to_datetime(centered_time_pd.item()).year
        if not np.isnan(centered_time):
            start_year = centered_time - 15
            end_year = centered_time + 14
            anom = one_ts.sel(time=slice(str(start_year), str(end_year))).mean(
                "time"
            ) - one_ts.sel(time=slice("1981", "2010")).mean("time")
            all_sims[simulation] = anom

            # Append to list. Used to assign descriptive attributes & coordinates to final dataset
            central_year_l.append(centered_time)
            year_start_l.append(start_year)
            year_end_l.append(end_year)
    anomaly_da = all_sims.to_array("simulation")

    # Assign descriptivie coordinates
    anomaly_da = anomaly_da.assign_coords(
        {
            "window_year_center": ("simulation", central_year_l),
            "window_year_start": ("simulation", year_start_l),
            "window_year_end": ("simulation", year_end_l),
        }
    )
    # Assign descriptive attributes to new coordinates
    anomaly_da["window_year_center"].attrs[
        "description"
    ] = "year that defines the center of the 30-year window around which the anomaly was computed"
    anomaly_da["window_year_start"].attrs[
        "description"
    ] = "year that defines the start of the 30-year window around which the anomaly was computed"
    anomaly_da["window_year_end"].attrs[
        "description"
    ] = "year that defines the end of the 30-year window around which the anomaly was computed"
    anomaly_da.attrs["warming_level"] = warmlevel

    # Rename
    anomaly_da.name = data.name + " Anomalies"
    return anomaly_da


def _compute_vmin_vmax(da_min, da_max):
    """Compute min, max, and center for plotting"""
    vmin = np.nanpercentile(da_min, 1)
    vmax = np.nanpercentile(da_max, 99)
    # define center for diverging symmetric data
    if (vmin < 0) and (vmax > 0):
        # dabs = abs(vmax) - abs(vmin)
        sopt = True
    else:
        sopt = None
    return vmin, vmax, sopt


def _make_hvplot(data, clabel, clim, cmap, sopt, title, width=225, height=210):
    """Make single map"""
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
    return _plot


class _WarmingLevels(param.Parameterized):
    """Generate warming levels panel GUI in notebook.

    Intended to be accessed through app.explore.warming_levels()
    Allows the user to toggle between several data options.
    Produces dynamically updating postage stamp maps.

    Attributes
    ----------
    warmlevel: param.ObjectSelector
        Warming level in degrees Celcius.
    ssp: param.ObjectSelector
        Shared socioeconomic pathway.
    cmap: param.ObjectSelector
        Colormap used to color maps
    changed_loc_and_var: param.Boolean
        Has the location and variable been changed?
        If so, reload the warming level anomolies.
    """

    warmlevel = param.ObjectSelector(
        default=1.5, objects=[1.5, 2, 3, 4], doc="Warming level in degrees Celcius."
    )
    ssp = param.ObjectSelector(
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
    cmap = param.ObjectSelector(dict(), doc="Colormap")

    def __init__(self, *args, **params):
        super().__init__(*args, **params)

        # Selectors defaults
        self.selections.scenario_historical = ["Historical Climate"]
        self.selections.area_average = "No"
        self.selections.resolution = "45 km"
        self.selections.scenario_ssp = ["SSP 3-7.0 -- Business as Usual"]
        self.selections.time_slice = (1980, 2100)
        self.selections.timescale = "monthly"
        self.selections.variable = "Air Temperature at 2m"

        # Location defaults
        self.location.area_subset = "states"
        self.location.cached_area = "CA"

        # Postage data and anomalies defaults
        self.postage_data = _get_postage_data(
            selections=self.selections, location=self.location, cat=self.cat
        )
        self._warm_all_anoms = get_anomaly_data(data=self.postage_data, warmlevel=1.5)

        self.cmap = _read_ae_colormap(cmap="ae_orange", cmap_hex=True)

    # For reloading postage stamp data and plots
    reload_data = param.Action(
        lambda x: x.param.trigger("reload_data"), label="Reload Data", doc="Reload data"
    )
    changed_loc_and_var = param.Boolean(
        default=True,
        doc="Has the location and variable been changed? If so, reload the warming level anomolies.",
    )

    @param.depends("selections.variable", watch=True)
    def _update_cmap(self):
        """Set colormap depending on variable"""
        cmap_name = var_catalog[
            (var_catalog["display_name"] == self.selections.variable)
            & (var_catalog["timescale"] == "daily/monthly")
        ].colormap.item()

        # Colormap normalization for hvplot -- only for relative humidity!
        if self.selections.variable == "Relative Humidity":
            cmap_name = "ae_diverging"

        # Read colormap hex
        self.cmap = _read_ae_colormap(cmap=cmap_name, cmap_hex=True)

    @param.depends(
        "location.area_subset",
        "location.cached_area",
        "selections.variable",
        "selections.units",
        watch=True,
    )
    def _updated_bool_loc_and_var(self):
        """Update boolean if any changes were made to the location, variable, or unit"""
        self.changed_loc_and_var = True

    @param.depends("reload_data", watch=True)
    def _update_postage_data(self):
        """If the button was clicked and the location or variable was changed,
        reload the postage stamp data from AWS"""
        if self.changed_loc_and_var == True:
            self.postage_data = _get_postage_data(
                selections=self.selections, location=self.location, cat=self.cat
            )
            self.changed_loc_and_var = False
        self._warm_all_anoms = get_anomaly_data(
            data=self.postage_data, warmlevel=self.warmlevel
        )

    @param.depends("reload_data", watch=False)
    def _GCM_PostageStamps_MAIN(self):
        # Get plot data
        all_plot_data = self._warm_all_anoms
        if self.selections.variable == "Relative Humidity":
            all_plot_data = all_plot_data * 100

        # Get int number of simulations
        num_simulations = len(all_plot_data.simulation.values)

        # Set up plotting arguments
        clabel = (
            self.selections.variable + " (" + self.postage_data.attrs["units"] + ")"
        )

        # Compute 1% min and 99% max of all simulations
        vmin_l, vmax_l = [], []
        for sim in range(num_simulations):
            data = all_plot_data.isel(simulation=sim)
            vmin_i, vmax_i, sopt = _compute_vmin_vmax(data, data)
            vmin_l.append(vmin_i)
            vmax_l.append(vmax_i)
        vmin = min(vmin_l)
        vmax = max(vmax_l)

        # Make each plot
        all_plots = _make_hvplot(  # Need to make the first plot separate from the loop
            data=all_plot_data.isel(simulation=0),
            clabel=clabel,
            clim=(vmin, vmax),
            cmap=self.cmap,
            sopt=sopt,
            title=all_plot_data.isel(simulation=0).simulation.item(),
        )
        for sim_i in range(1, num_simulations):
            pl_i = _make_hvplot(
                data=all_plot_data.isel(simulation=sim_i),
                clabel=clabel,
                clim=(vmin, vmax),
                cmap=self.cmap,
                sopt=sopt,
                title=all_plot_data.isel(simulation=sim_i).simulation.item(),
            )
            all_plots += pl_i

        try:
            all_plots.cols(3)  # Organize into 3 columns
            all_plots.opts(
                title=self.selections.variable
                + ": Anomalies for "
                + str(self.warmlevel)
                + "°C Warming by Simulation"
            )  # Add title
        except:
            all_plots.opts(
                title=str(self.warmlevel) + "°C Anomalies"
            )  # Add shorter title

        all_plots.opts(toolbar="below")  # Set toolbar location
        all_plots.opts(hv.opts.Layout(merge_tools=True))  # Merge toolbar
        return all_plots

    @param.depends("reload_data", watch=False)
    def _GCM_PostageStamps_STATS(self):
        # Get plot data
        all_plot_data = self._warm_all_anoms
        if self.selections.variable == "Relative Humidity":
            all_plot_data = all_plot_data * 100

        # Compute stats
        min_data = all_plot_data.min(dim="simulation")
        max_data = all_plot_data.max(dim="simulation")
        med_data = all_plot_data.median(dim="simulation")
        mean_data = all_plot_data.mean(dim="simulation")

        # Set up plotting arguments
        width = 210
        height = 210
        clabel = (
            self.selections.variable + " (" + self.postage_data.attrs["units"] + ")"
        )
        vmin, vmax, sopt = _compute_vmin_vmax(min_data, max_data)

        # Make plots
        min_plot = _make_hvplot(
            data=min_data,
            clabel=clabel,
            cmap=self.cmap,
            clim=(vmin, vmax),
            sopt=sopt,
            title="Minimum",
            width=width,
            height=height,
        )
        max_plot = _make_hvplot(
            data=max_data,
            clabel=clabel,
            cmap=self.cmap,
            clim=(vmin, vmax),
            sopt=sopt,
            title="Maximum",
            width=width,
            height=height,
        )
        med_plot = _make_hvplot(
            data=med_data,
            clabel=clabel,
            cmap=self.cmap,
            clim=(vmin, vmax),
            sopt=sopt,
            title="Median",
            width=width,
            height=height,
        )
        mean_plot = _make_hvplot(
            data=mean_data,
            clabel=clabel,
            cmap=self.cmap,
            clim=(vmin, vmax),
            sopt=sopt,
            title="Mean",
            width=width,
            height=height,
        )

        all_plots = mean_plot + med_plot + min_plot + max_plot
        all_plots.opts(
            title=self.selections.variable
            + ": Anomalies for "
            + str(self.warmlevel)
            + "°C Warming Across Models"
        )  # Add title
        all_plots.opts(toolbar="below")  # Set toolbar location
        all_plots.opts(hv.opts.Layout(merge_tools=True))  # Merge toolbar
        return all_plots

    @param.depends("reload_data", watch=False)
    def _30_yr_window(self):
        """Create a dataframe to give information about the 30-yr anomalies window for each simulation used in the postage stamp maps."""
        anom = self._warm_all_anoms
        df = pd.DataFrame(
            {
                "simulation": anom.simulation.values,
                "30-yr window": zip(
                    anom.window_year_start.values, anom.window_year_end.values
                ),
                "central year": anom.window_year_center.values,
                "warming level": [anom.attrs["warming_level"]] * len(anom.simulation),
            }
        )
        df_pane = pn.pane.DataFrame(df, width=400, index=False)
        return df_pane

    @param.depends("warmlevel", "ssp", watch=True)
    def _GMT_context_plot(self):
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

        ipcc_data = hist_data.hvplot(
            y="Mean", color="k", label="Historical", width=width, height=height
        ) * hist_data.hvplot.area(
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
                * ssp119_data.hvplot(y="Mean", color=c119, label="SSP1-1.9")
                * ssp126_data.hvplot(y="Mean", color=c126, label="SSP1-2.6")
                * ssp245_data.hvplot(y="Mean", color=c245, label="SSP2-4.5")
                * ssp370_data.hvplot(y="Mean", color=c370, label="SSP3-7.0")
                * ssp585_data.hvplot(y="Mean", color=c585, label="SSP5-8.5")
            )
        elif self.ssp == "SSP 1-1.9 -- Very Low Emissions Scenario":
            ipcc_data = ipcc_data * ssp119_data.hvplot(
                y="Mean", color=c119, label="SSP1-1.9"
            )
        elif self.ssp == "SSP 1-2.6 -- Low Emissions Scenario":
            ipcc_data = ipcc_data * ssp126_data.hvplot(
                y="Mean", color=c126, label="SSP1-2.6"
            )
        elif self.ssp == "SSP 2-4.5 -- Middle of the Road":
            ipcc_data = ipcc_data * ssp245_data.hvplot(
                y="Mean", color=c245, label="SSP2-4.5"
            )
        elif self.ssp == "SSP 3-7.0 -- Business as Usual":
            ipcc_data = ipcc_data * ssp370_data.hvplot(
                y="Mean", color=c370, label="SSP3-7.0"
            )
        elif self.ssp == "SSP 5-8.5 -- Burn it All":
            ipcc_data = ipcc_data * ssp585_data.hvplot(
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
                "SSP 1-1.9 -- Very Low Emissions Scenario": (ssp119_data, c119),
                "SSP 1-2.6 -- Low Emissions Scenario": (ssp126_data, c126),
                "SSP 2-4.5 -- Middle of the Road": (ssp245_data, c245),
                "SSP 3-7.0 -- Business as Usual": (ssp370_data, c370),
                "SSP 5-8.5 -- Burn it All": (ssp585_data, c585),
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


def _display_warming_levels(warming_data, selections, location, map_view):
    # Create panel doodad!
    data_options = pn.Card(
        pn.Row(
            pn.Column(
                pn.widgets.StaticText(name="", value="Warming level (°C)"),
                pn.widgets.RadioButtonGroup.from_param(
                    warming_data.param.warmlevel, name=""
                ),
                selections.param.variable,
                pn.widgets.StaticText.from_param(
                    selections.param.extended_description, name=""
                ),
                pn.widgets.StaticText(name="", value="Variable Units"),
                pn.widgets.RadioButtonGroup.from_param(selections.param.units),
                pn.widgets.StaticText(name="", value="Model Resolution"),
                pn.widgets.RadioButtonGroup.from_param(selections.param.resolution),
                pn.widgets.StaticText(name="", value=""),
                pn.widgets.Button.from_param(
                    warming_data.param.reload_data,
                    button_type="primary",
                    width=150,
                    height=30,
                ),
                width=230,
            ),
            pn.Column(
                location.param.area_subset,
                location.param.latitude,
                location.param.longitude,
                location.param.cached_area,
                map_view.view,
                width=230,
            ),
        ),
        title="Data Options",
        collapsible=False,
        width=460,
        height=515,
    )

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
            pn.widgets.Select.from_param(
                warming_data.param.ssp, name="Scenario", width=250
            ),
            warming_data._GMT_context_plot,
        ),
        title="When do different scenarios reach the warming level?",
        collapsible=False,
        width=600,
        height=515,
    )

    postage_stamps_MAIN = pn.Column(
        pn.widgets.StaticText(
            value=(
                "Panels show the difference (anomaly) between the 30-year average"
                " centered on the year that each GCM (name of model titles each panel)"
                " reaches the specified warming level and the average from 1981-2010."
            ),
            width=800,
        ),
        pn.Row(
            warming_data._GCM_PostageStamps_MAIN,
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
                "Panels show the average, median, minimum, or maximum conditions"
                " across all models. These statistics are computed from the data"
                " in the first panel: the difference (anomaly) between the 30-year"
                " average centered on the year that each GCM reaches the specified"
                " warming level and the average from 1981-2010. Minimum and maximum"
                " values are calculated across simulations for each grid cell, so one"
                " map may contain grid cells from different simulations. Median and"
                " mean maps show those respective summaries across simulations at each grid cell."
            ),
            width=800,
        ),
        warming_data._GCM_PostageStamps_STATS,
    )

    window_df = pn.Column(
        pn.widgets.StaticText(
            value=(
                "This panel displays start and end years that define the 30-year"
                " window for which the anomalies were computed for each model. It"
                " also displays the year at which each model crosses the selected"
                " warming level, defined in the table below as the central year."
                " This information corresponds to the anomalies shown in the maps"
                " on the previous two tabs."
            ),
            width=800,
        ),
        warming_data._30_yr_window,
    )

    map_tabs = pn.Card(
        pn.Tabs(
            ("Maps of individual simulations", postage_stamps_MAIN),
            (
                "Maps of cross-model statistics: mean/median/max/min",
                postage_stamps_STATS,
            ),
            ("Anomaly computation details", window_df),
        ),
        title="Regional response at selected warming level",
        width=850,
        height=600,
        collapsible=False,
    )

    panel_doodad = pn.Column(pn.Row(data_options, GMT_plot), map_tabs)
    return panel_doodad
