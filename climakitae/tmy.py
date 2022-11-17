"""
Calculates the Average Meterological Year (TMY) for the Cal-Adapt: Analytics
Engine using a standard climatological period (1981-2010) for the historical
baseline, and uses a 30-year window around when a designated warming level is
exceeded for the SSP3-7.0 future scenario for 1.5°C, 2°C, and 3°C. Additional
functionality for 4°C is forthcoming. Working Group 4 (Aug 31, 2022) version
focuses on air temperature and relative humidity, with all variables being
available for Analytics Engine Beta Launch. The AMY is comparable to a typical
 meteorological year, but not quite the same full methodology.
"""

## PROCESS: average meteorological year
# for each hour, the average variable over the whole climatological period is determined
# data for that hour that has the value most closely equal (smalleset absolute difference)
# to the hourly average over the whole measurement period is chosen as the AMY data for that hour
# process is repeated for each hour in the year
# repeat values (where multiple years have the same smallest abs value) are
# removed, earliest occurence selected for AMY
# hours are added together to provide a full year of hourly samples

## Produces 3 different kinds of AMY
## 1: Absolute/unbias corrected raw AMY, either historical or warming level-centered future
## 2: Future-minus-historical warming level AMY (see warming_levels)
## 3: Severe AMY based upon historical baseline and a designated threshold/percentile

import cartopy.crs as ccrs
import hvplot.xarray
import hvplot.pandas
import xarray as xr
import holoviews as hv
from holoviews import opts
import numpy as np
import pandas as pd
import param
import panel as pn
import warnings
import pkg_resources
from .utils import _read_ae_colormap
from .catalog_convert import (
    _resolution_to_gridlabel,
    _timescale_to_table_id,
    _scenario_to_experiment_id
)
from .data_loaders import _read_from_catalog

import logging  # Silence warnings

logging.getLogger("param").setLevel(logging.CRITICAL)

# Variable info
var_catalog_resource = pkg_resources.resource_filename(
    "climakitae", "data/variable_catalog.csv"
)
var_catalog = pd.read_csv(var_catalog_resource, index_col="variable_id")

xr.set_options(keep_attrs=True)  # Keep attributes when mutating xr objects

def _get_historical_tmy_data(cat, selections, location):
    """Get historical data from AWS catalog"""
    selections.scenario = ["Historical Climate"]
    selections.time_slice = (1981, 2010)
    selections.append_historical = False
    selections.area_average = True
    selections.timescale = "hourly"
    historical_da_mean = _read_from_catalog(
        selections = selections, 
        location = location, 
        cat = cat, 
        source_id = "ensmean"
    ).isel(scenario = 0, simulation = 0)
    return historical_da_mean.compute()

def _get_future_heatmap_data(cat, selections, location, warmlevel):
    """Gets data from AWS catalog based upon desired warming level"""
    warming_year_average_range = {
        1.5: (2034, 2063),
        2: (2047, 2076),
        3: (2061, 2090),
    }
    selections.scenario = ["SSP 3-7.0 -- Business as Usual"]
    selections.time_slice = warming_year_average_range[warmlevel]
    selections.append_historical = False
    selections.area_average = True
    selections.timescale = "hourly"
    future_da_mean = _read_from_catalog(
        selections = selections, 
        location = location, 
        cat = cat, 
        source_id = "ensmean"
    ).isel(scenario = 0, simulation = 0)
    return future_da_mean.compute()

def remove_repeats(xr_data):
    """
    Remove hours that have repeats.
    This occurs if two hours have the same absolute difference from the mean.
    Returns numpy array
    """
    unq, unq_idx, unq_cnt = np.unique(
        xr_data.time.dt.hour.values, return_inverse=True, return_counts=True
    )
    cnt_mask = unq_cnt > 1
    (cnt_idx,) = np.nonzero(cnt_mask)
    idx_mask = np.in1d(unq_idx, cnt_idx)
    (idx_idx,) = np.nonzero(idx_mask)
    srt_idx = np.argsort(unq_idx[idx_mask])
    dup_idx = np.split(idx_idx[srt_idx], np.cumsum(unq_cnt[cnt_mask])[:-1])
    if len(dup_idx[0]) > 0:
        dup_idx_keep_first_val = np.concatenate(
            [dup_idx[x][1:] for x in range(len(dup_idx))], axis=0
        )
        cleaned_np = np.delete(xr_data.values, dup_idx_keep_first_val)
        return cleaned_np
    else:
        return xr_data.values


## Compute hourly AMY for each hour of the year
def tmy_calc(data, days_in_year = 366):
    """
    Calculates the average meteorological year based on a designated period of time.
    Applicable for both the historical and future periods.
    Returns: dataframe of amy
    """
    hourly_list = []
    for x in np.arange(1, days_in_year + 1, 1):
        data_on_day_x = data.where(data.time.dt.dayofyear == x, drop=True)
        data_grouped = data_on_day_x.groupby("time.hour")
        mean_by_hour = data_grouped.mean()
        min_diff = abs(data_grouped - mean_by_hour).groupby("time.hour").min()
        typical_hourly_data_on_day_x = data_on_day_x.where(
            abs(data_grouped - mean_by_hour).groupby("time.hour") == min_diff, drop=True
        ).sortby("time.hour")
        np_typical_hourly_data_on_day_x = remove_repeats(typical_hourly_data_on_day_x)
        hourly_list.append(np_typical_hourly_data_on_day_x)

    ## Funnel data into pandas DataFrame object
    df_amy = pd.DataFrame(
        hourly_list,
        columns=np.arange(1, 25, 1),
        index=np.arange(1, days_in_year + 1, 1),
    )

    ## Re-order columns for PST, with easy to read time labels
    cols = df_amy.columns.tolist()
    cols = cols[7:] + cols[:7]
    df_amy = df_amy[cols]

    n_col_lst = []
    for ampm in ["am", "pm"]:
        hr_lst = []
        for hr in range(1, 13, 1):
            hr_lst.append(str(hr) + ampm)
        hr_lst = hr_lst[-1:] + hr_lst[:-1]
        n_col_lst = n_col_lst + hr_lst
    df_amy.columns = n_col_lst

    return df_amy


class AverageMeteorologicalYear(param.Parameterized):
    """
    An object that holds the "Data Options" paramters for the
    explore.tmy panel.
    """

    # Create dictionary of TMY advanced options depending on TMY type
    tmy_advanced_options_dict = {
        "Absolute": {
            "default": "Historical",
            "objects": ["Historical", "Warming Level Future"],
        },
        "Difference": {
            "default": "Warming Level Future",
            "objects": ["Warming Level Future"],  # , "Severe AMY"]
        },
    }

    # Create a dictionary that briefly explains the computation being perfomed
    computatation_description_dict = {
        "Absolute": {
            "Historical": "AMY computed using the historical baseline for 1981-2010.",
            "Warming Level Future": (
                "AMY computed using the 30-year future period"
                " centered around when the selected warming level is reached."
            ),
        },
        "Difference": {
            "Warming Level Future": (
                "AMY computed by taking the difference between"
                " the 30-year future period centered around the selected warming"
                " level and the historical baseline."
            )
            # "Severe AMY": ("AMY computed by taking the difference between the 90th percentile of the 30-year future"
            #                " period centered around the selected warming level and the historical baseline.")
        },
    }

    # Define TMY params
    tmy_options = param.ObjectSelector(
        default="Absolute", objects=["Absolute", "Difference"]
    )

    # Define new advanced options param, that is dependent on the user selection in tmy_options
    tmy_advanced_options = param.ObjectSelector(objects=dict())

    # Define new computation description param
    # This will provide a string description of the computation option selected
    # and will update dynamically depending on the user selections
    tmy_computation_description = param.ObjectSelector(objects=dict())

    # Colormap
    cmap = param.ObjectSelector(objects=dict())

    # Warming level selection
    warmlevel = param.ObjectSelector(default=1.5, objects=[1.5, 2, 3])

    def __init__(self, *args, **params):
        super().__init__(*args, **params)

        # Location defaults
        self.location.area_subset = "CA counties"
        self.location.cached_area = "Los Angeles County"
        
        # Selectors defaults
        self.selections.append_historical = False
        self.selections.area_average = True
        self.selections.resolution = "45 km"
        self.selections.scenario = ["Historical Climate"]  # setting for historical
        self.selections.time_slice = (1981, 2010)
        self.selections.timescale = "hourly"
        self.selections.variable = "Air Temperature at 2m"

        # Initialze tmy_adanced_options param
        self.param["tmy_advanced_options"].objects = self.tmy_advanced_options_dict[
            self.tmy_options
        ]["objects"]
        self.tmy_advanced_options = self.tmy_advanced_options_dict[self.tmy_options][
            "default"
        ]

        # Initialize tmy_computation_description param
        self.tmy_computation_description = self.computatation_description_dict[
            self.tmy_options
        ][self.tmy_advanced_options]

        # Postage data and anomalies defaults
        self.historical_tmy_data = _get_historical_tmy_data(
            cat = self.cat,
            selections = self.selections,
            location = self.location,
        )
        self.future_tmy_data = _get_future_heatmap_data(
            cat = self.cat,
            selections = self.selections,
            location = self.location,
            warmlevel = 1.5
        )
        
        # Colormap 
        self.cmap = _read_ae_colormap(cmap="ae_orange", cmap_hex=True)

    # For reloading data and plots
    reload_data = param.Action(
        lambda x: x.param.trigger("reload_data"), label="Reload Data"
    )

    @param.depends("selections.variable", "tmy_options", watch=True)
    def _update_cmap(self):
        """Set colormap depending on variable"""
        cmap_name = var_catalog[
            (var_catalog["display_name"] == self.selections.variable)
            & (var_catalog["timescale"] == "hourly")
        ].colormap.item()

        # Set to diverging colormap if difference is selected
        if self.tmy_options == "Difference":
            cmap_name = "ae_diverging"

        # Read colormap hex
        self.cmap = _read_ae_colormap(cmap=cmap_name, cmap_hex=True)

    @param.depends("tmy_advanced_options", "reload_data", "warmlevel", watch=True)
    def _update_data_to_be_returned(self):
        """Update self.selections so that the correct data is returned by app.retrieve()"""
        if self.tmy_advanced_options == "Historical":
            self.selections.scenario = ["Historical Climate"]
            self.selections.time_slice = (
                1981,
                2010,
            )  # to match historical 30-year average

        elif self.tmy_advanced_options == "Warming Level Future":
            warming_year_average_range = {
                1.5: (2034, 2063),
                2: (2047, 2076),
                3: (2061, 2090),
            }
            self.selections.scenario = ["SSP 3-7.0 -- Business as Usual"]
            self.selections.time_slice = warming_year_average_range[self.warmlevel]

    @param.depends("reload_data", watch=True)
    def _update_tmy_data(self):
        """If the button was clicked and the location or variable was changed,
        reload the tmy data from AWS"""
        self.historical_tmy_data = _get_historical_tmy_data(
            cat = self.cat,
            selections = self.selections,
            location = self.location,
        )
        self.future_tmy_data = _get_future_heatmap_data(
            cat = self.cat,
            selections = self.selections,
            location = self.location,
            warmlevel = self.warmlevel
        )

    # Create a function that will update tmy_advanced_options when tmy_options is modified
    @param.depends("tmy_options", watch=True)
    def _update_tmy_advanced_options(self):
        self.param["tmy_advanced_options"].objects = self.tmy_advanced_options_dict[
            self.tmy_options
        ]["objects"]
        self.tmy_advanced_options = self.tmy_advanced_options_dict[self.tmy_options][
            "default"
        ]

    @param.depends("tmy_options", "tmy_advanced_options", watch=True)
    def _update_tmy_computatation_description(self):
        self.tmy_computation_description = self.computatation_description_dict[
            self.tmy_options
        ][self.tmy_advanced_options]

    @param.depends("reload_data", watch=False)
    def _tmy_hourly_heatmap(self):
        # update heatmap df and title with selections
        days_in_year = 366
        if self.tmy_options == "Absolute":
            if self.tmy_advanced_options == "Historical":
                df = tmy_calc(self.historical_tmy_data, days_in_year=days_in_year)
                title = "Average Meteorological Year: {}\nAbsolute {} Baseline".format(
                    self.location.cached_area, self.tmy_advanced_options
                )
                clabel = (
                    self.selections.variable
                    + " ("
                    + self.historical_tmy_data.attrs["units"]
                    + ")"
                )
            else:
                df = tmy_calc(self.future_tmy_data, days_in_year=days_in_year)
                title = "Average Meteorological Year: {}\nAbsolute {} at {}°C".format(
                    self.location.cached_area, self.tmy_advanced_options, self.warmlevel
                )
                clabel = self.selections.variable + " (" + self.selections.units + ")"
        elif self.tmy_options == "Difference":
            cmap = _read_ae_colormap("ae_diverging", cmap_hex=True)
            if self.tmy_advanced_options == "Warming Level Future":
                df = tmy_calc(
                    self.future_tmy_data, days_in_year=days_in_year
                ) - tmy_calc(self.historical_tmy_data, days_in_year=days_in_year)
                title = "Average Meteorological Year: {}\nDifference between {} at {}°C and Historical Baseline".format(
                    self.location.cached_area, self.tmy_advanced_options, self.warmlevel
                )
                clabel = self.selections.variable + " (" + self.selections.units + ")"
            else:  # placeholder for now for severe amy
                df = tmy_calc(
                    self.future_tmy_data, days_in_year=days_in_year
                ) - tmy_calc(self.historical_tmy_data, days_in_year=days_in_year)
                title = "Average Meteorological Year: {}\nDifference between {} at 90th percentile and Historical Baseline".format(
                    self.location.cached_area, self.tmy_advanced_options
                )
                clabel = self.selections.variable + " (" + self.selections.units + ")"
        else:
            title = "Average Meteorological Year\n{}".format(self.location.cached_area)

        heatmap = df.hvplot.heatmap(
            x="columns",
            y="index",
            title=title,
            cmap=self.cmap,
            xaxis="bottom",
            xlabel="Hour of Day (PST)",
            ylabel="Day of Year",
            clabel=clabel,
            rot=60,
            width=800,
            height=350,
        ).opts(fontsize={"title": 13, "xlabel": 12, "ylabel": 12}, toolbar="below")
        return heatmap


def _amy_visualize(tmy_ob, selections, location):
    """
    Creates a new AMY focus panel object to display user selections
    """
    user_options = pn.Card(
        pn.Row(
            pn.Column(
                pn.widgets.StaticText(
                    name="", value="Average Meteorological Year Type"
                ),
                pn.widgets.RadioButtonGroup.from_param(tmy_ob.param.tmy_options),
                pn.widgets.Select.from_param(
                    tmy_ob.param.tmy_advanced_options, name="Computation Options"
                ),
                pn.widgets.StaticText.from_param(
                    tmy_ob.param.tmy_computation_description, name=""
                ),
                pn.widgets.StaticText(name="", value="Warming level (°C)"),
                pn.widgets.RadioButtonGroup.from_param(tmy_ob.param.warmlevel),
                pn.widgets.Select.from_param(
                    selections.param.variable, name="Data variable"
                ),
                pn.widgets.StaticText.from_param(
                    selections.param.extended_description, name=""
                ),
                pn.widgets.StaticText(name="", value="Variable Units"),
                pn.widgets.RadioButtonGroup.from_param(selections.param.units),
                pn.widgets.StaticText(name = "", value = "Model Resolution"),
                pn.widgets.RadioButtonGroup.from_param(selections.param.resolution),
                width = 280
            ),
            pn.Column(
                location.param.area_subset,
                location.param.latitude,
                location.param.longitude,
                location.param.cached_area,
                location.view,
                pn.widgets.Button.from_param(
                    tmy_ob.param.reload_data,
                    button_type="primary",
                    width=150,
                    height=30,
                ),
                width=230,
            ),
        ),
        title=" How do you want to investigate AMY?",
        collapsible=False,
        width=510,
        height=600,
    )

    mthd_bx = pn.Column(
        pn.widgets.StaticText(
            value=(
                "An average meteorological year is calculated by selecting"
                " the 24 hours for every day that best represent multi-model mean"
                " conditions during a 30-year period – 1981-2010 for the historical"
                " baseline or centered on the year the warming level is reached."
                " Absolute average meteorolgoical year profiles represent data that"
                " is not bias corrected, please exercise caution when analyzing."
                " The 'severe' AMY is calculated using the 90th percentile of future"
                " warming level data at the selected warming level, and is compared"
                " to the historical baseline."
            ),
            width=400,
        ),
    )

    TMY = pn.Card(
        pn.widgets.StaticText(value="Absolute AMY", width=600, height=500),
        tmy_ob._tmy_hourly_heatmap,
    )

    tmy_tabs = pn.Card(
        pn.Tabs(("AMY Heatmap", tmy_ob._tmy_hourly_heatmap), ("Methodology", mthd_bx)),
        title=" Average Meteorological Year",
        width=850,
        height=600,
        collapsible=False,
    )

    tmy_panel = pn.Column(pn.Row(user_options, tmy_tabs))
    return tmy_panel
