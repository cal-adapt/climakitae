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

import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import datetime
import xarray as xr
import numpy as np
import pandas as pd
import param
import panel as pn
import warnings
import pkg_resources
from .utils import _read_ae_colormap, julianDay_to_str_date
from .catalog_convert import (
    _resolution_to_gridlabel,
    _timescale_to_table_id,
    _scenario_to_experiment_id,
)
from .data_loaders import _read_from_catalog
from tqdm.auto import tqdm  # Progress bar
import logging  # Silence warnings

logging.getLogger("param").setLevel(logging.CRITICAL)
xr.set_options(keep_attrs=True)  # Keep attributes when mutating xr objects

# Variable info
var_catalog_resource = pkg_resources.resource_filename(
    "climakitae", "data/variable_descriptions.csv"
)
var_catalog = pd.read_csv(var_catalog_resource, index_col="variable_id")


# =========================== HELPER FUNCTIONS: DATA RETRIEVAL ==============================


def _set_amy_year_inputs(year_start, year_end):
    """
    Helper function for retrieve_amy_data.
    Checks that the user has input valid values.
    Sets year end if it hasn't been set; default is 30 year range (year_start + 30). Minimum is 5 year range.
    """
    if year_end is None:
        year_end = (
            year_start + 30 if (year_start + 30 < 2100) else 2100
        )  # Default is +30 years
    elif year_end > 2100:
        print("Your end year cannot exceed 2100. Resetting end year to 2100.")
        year_end = 2100
    if year_end - year_start < 5:
        raise ValueError(
            """To compute an Average Meteorological Year, you must input a date range with a difference
            of at least 5 years, where the end year is no later than 2100 and the start year is no later than
            2095."""
        )
    if year_start < 1980:
        raise ValueError(
            """You've input an invalid start year. The start year must be 1980 or later."""
        )
    return (year_start, year_end)


def retrieve_amy_data(
    app=None, selections=None, location=None, _cat=None, year_start=2015, year_end=None
):
    """Get average meteorological year data.
    Input one of the two:
        (1) app: climakitae Applications object, or (user-facing)
        (2) all the following: selections, location, _cat (backend)

    Args:
        app (climakitae Application): user-facing object for using within a notebook environment
        selections (climakitae DataSelector)
        location (climakitae LocationSelector)
        _cat (intake catalog)
        year_start (int, optional): year between 1980-2095
        year_end (int, optional) year between 1985-2100

    Returns:
        amy_data (xr.DataArray)

    """

    # Deal with input issues
    if (app is None) and ((selections is None) or (location is None) or (_cat is None)):
        raise ValueError(
            """You must input one either one climakitae Application object or
            the following three objects: selections, location, and _cat"""
        )

    if app is not None:
        selections = app.selections
        location = app.location
        _cat = app._cat

    # Check year start and end inputs
    year_start, year_end = _set_amy_year_inputs(year_start, year_end)

    # Set scenario selections
    if year_end >= 2015:
        selections.scenario_ssp = ["SSP 3-7.0 -- Business as Usual"]
    else:
        selections.scenario_ssp = []
    if year_start < 2015:
        selections.scenario_historical = ["Historical Climate"]
    else:
        selections.scenario_historical = []

    # Set other data parameters
    selections.time_slice = (year_start, year_end)
    selections.area_average = "Yes"
    selections.timescale = "hourly"
    selections.simulation = ["ensmean"]  # Read from ensemble means

    # Grab data from the catalog
    amy_data = _read_from_catalog(
        selections=selections, location=location, cat=_cat
    ).isel(scenario=0, simulation=0)
    return amy_data


# =========================== HELPER FUNCTIONS: AMY/TMY CALCULATION ==============================


def _remove_repeats(xr_data):
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


# =========================== HELPER FUNCTIONS: AMY/TMY PLOTTING ==============================


def compute_amy(data, days_in_year=366, show_pbar=False):
    """
    Calculates the average meteorological year based on a designated period of time.
    Applicable for both the historical and future periods.
    Returns: dataframe of amy
    """
    hourly_list = []
    for x in tqdm(np.arange(1, days_in_year + 1, 1), disable=not show_pbar):
        data_on_day_x = data.where(data.time.dt.dayofyear == x, drop=True)
        data_grouped = data_on_day_x.groupby("time.hour")
        mean_by_hour = data_grouped.mean()
        min_diff = abs(data_grouped - mean_by_hour).groupby("time.hour").min()
        typical_hourly_data_on_day_x = data_on_day_x.where(
            abs(data_grouped - mean_by_hour).groupby("time.hour") == min_diff, drop=True
        ).sortby("time.hour")
        np_typical_hourly_data_on_day_x = _remove_repeats(typical_hourly_data_on_day_x)
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
    df_amy.columns.name = "Hour"

    # Convert Julian date index to Month-Day format
    if days_in_year == 366:
        leap_year = True
    else:
        leap_year = False
    new_index = [
        julianDay_to_str_date(julday, leap_year=leap_year, str_format="%b-%d")
        for julday in df_amy.index
    ]
    df_amy.index = pd.Index(new_index, name="Day of Year")
    return df_amy


def compute_severe_yr(data, days_in_year=366, show_pbar=False):
    """
    Calculates the severe meteorological year based on the 90th percentile of data.
    Applicable for both the historical and future periods.
    Returns: dataframe of severe meteorological year
    """
    hourly_list = []
    for x in tqdm(np.arange(1, days_in_year + 1, 1), disable=not show_pbar):
        data_on_day_x = data.where(data.time.dt.dayofyear == x, drop=True)
        data_grouped = data_on_day_x.groupby("time.hour")
        severe_by_hour = data_grouped.quantile(q=0.90)
        min_diff = abs(data_grouped - severe_by_hour).groupby("time.hour").min()
        typical_hourly_data_on_day_x = data_on_day_x.where(
            abs(data_grouped - severe_by_hour).groupby("time.hour") == min_diff, drop=True
        ).sortby("time.hour")
        np_typical_hourly_data_on_day_x = _remove_repeats(typical_hourly_data_on_day_x)
        hourly_list.append(np_typical_hourly_data_on_day_x)

    ## Funnel data into pandas DataFrame object
    df_severe_yr = pd.DataFrame(
        hourly_list,
        columns=np.arange(1, 25, 1),
        index=np.arange(1, days_in_year + 1, 1),
    )

    ## Re-order columns for PST, with easy to read time labels
    cols = df_severe_yr.columns.tolist()
    cols = cols[7:] + cols[:7]
    df_severe_yr = df_severe_yr[cols]

    n_col_lst = []
    for ampm in ["am", "pm"]:
        hr_lst = []
        for hr in range(1, 13, 1):
            hr_lst.append(str(hr) + ampm)
        hr_lst = hr_lst[-1:] + hr_lst[:-1]
        n_col_lst = n_col_lst + hr_lst
    df_severe_yr.columns = n_col_lst
    df_severe_yr.columns.name = "Hour"

    # Convert Julian date index to Month-Day format
    if days_in_year == 366:
        leap_year = True
    else:
        leap_year = False
    new_index = [
        julianDay_to_str_date(julday, leap_year=leap_year, str_format="%b-%d")
        for julday in df_severe_yr.index
    ]
    df_severe_yr.index = pd.Index(new_index, name="Day of Year")
    return df_severe_yr


def _amy_heatmap(amy_df, title=None, cmap="viridis", cbar_label=None):
    """Create AMY heatmap using matplotlib

    Args:
        amy_df (pd.DataFrame): AMY dataframe, with hour of day as columns and day of year as index
        title (str): title to give heatmap
        cmap (str): colormap
        cbar_label (str): name of variable being plotted

    Returns:
        fig (matplotlib.figure.Figure)

    """
    fig, ax = plt.subplots(1, 1, figsize=(9, 5))
    heatmap = ax.imshow(
        amy_df.values, cmap=cmap, aspect=0.03, origin="lower"  # Flip y axis
    )

    # Set xticks
    ax.set_xticks(np.arange(len(amy_df.columns)))
    ax.set_xticklabels(amy_df.columns.values, rotation=45)

    # Set yticks
    if len(amy_df.index) == 366:  # Leap year
        first_days_of_month = [0, 31, 60, 91, 121, 152, 182, 213, 244, 274, 305, 335]
    else:  # Not a leap year
        first_days_of_month = [0, 31, 59, 90, 120, 151, 181, 212, 243, 273, 304, 334]
    ax.set_yticks(first_days_of_month)
    ax.set_yticklabels(amy_df.index[first_days_of_month])

    # Set title and labels
    if title is not None:
        ax.set_title(title)
    ax.set_ylabel(amy_df.index.name)
    ax.set_xlabel(amy_df.columns.name)

    # Make colorbar
    cax = fig.add_axes([0.92, 0.24, 0.02, 0.53])
    fig.colorbar(heatmap, cax=cax, orientation="vertical", label=cbar_label)

    plt.close()  # Close figure
    return fig


def lineplot_from_amy_data(
    amy_data,
    computation_method=None,
    location_subset=None,
    warmlevel=None,
    variable=None,
):
    """Generate a lineplot of AMY data, with mon-day-hr on the x-axis

    Args:
        amy_data (pd.DataFrame): data in the format of the dataframe returned by _amy_calc
        computation_method (str): AMY computation method of data
        location_subset (str): location subset of data
        warmlevel (str): warming level used to generate data
        variable (str): Name of data variable

    Returns:
        fig (matplotlib.figure.Figure)

    """

    # Stack data
    amy_stacked = (
        pd.DataFrame(amy_data.stack()).rename(columns={0: "data"}).reset_index()
    )
    amy_stacked["Date"] = amy_stacked["Day of Year"] + " " + amy_stacked["Hour"]
    amy_stacked = amy_stacked.drop(columns=["Day of Year", "Hour"]).set_index("Date")

    # Set plot title, suptitle, and ylabel using original xr DataArray
    suptitle = "Average Meterological Year"
    title = ""
    if computation_method is not None:
        suptitle += ": " + computation_method
        if computation_method == "Warming Level Future":
            if warmlevel is not None:
                suptitle += " at " + str(warmlevel) + "°C "
        if computation_method == "Historical":
            suptitle += " Data"

    # Add months information
    try:  # Try leap year
        months = [
            datetime.datetime.strptime("2024." + idx_i, "%Y.%b-%d %I%p").strftime("%B")
            for idx_i in amy_stacked.index
        ]
    except:  # Try non leap year
        months = [
            datetime.datetime.strptime("2024." + idx_i, "%Y.%b-%d %I%p").strftime("%B")
            for idx_i in amy_stacked.index
        ]

    def check_if_all_identical(l):
        return all(i == l[0] for i in l)

    if check_if_all_identical(months):  # Add month to title
        title += "Month: " + months[0] + "\n"
    else:
        title += "Months: " + months[0] + "-" + months[-1] + "\n"

    if location_subset is not None:
        title += "Location Subset: " + location_subset

    # Make plot
    fig, axes = plt.subplots(1, 1, figsize=(7, 4))
    amy_lineplot = axes.plot(amy_stacked)
    axes.grid(alpha=0.25)
    plt.xticks(rotation=45)
    axes.xaxis.set_major_locator(MaxNLocator(10))
    axes.set_ylabel(variable)
    plt.suptitle(suptitle, fontsize=13, y=1.025)
    axes.set_title(title, fontsize=10, y=1)
    plt.close()
    return fig


# =========================== MAIN AVERAGE METEO YR OBJECT ==============================


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
            "objects": ["Warming Level Future", "Severe AMY"]
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
                " level and the historical baseline."),
            "Severe AMY": (
                "AMY computed by taking the difference between the 90th percentile of the 30-year future"
                " period centered around the selected warming level and the historical baseline."
            )
        },
    }

    # Define TMY params
    data_type = param.ObjectSelector(
        default="Absolute", objects=["Absolute", "Difference"]
    )

    # Define new advanced options param, that is dependent on the user selection in data_type
    computation_method = param.ObjectSelector(objects=dict())

    # Define new computation description param
    # This will provide a string description of the computation option selected
    # and will update dynamically depending on the user selections
    tmy_computation_description = param.ObjectSelector(objects=dict())

    # Colormap
    cmap = param.ObjectSelector(objects=dict())

    # Warming level selection
    warmlevel = param.ObjectSelector(default=1.5, objects=[1.5, 2, 3])

    # 30-yr ranges to use for AMY computation
    warming_year_average_range = {
        1.5: (2034, 2063),
        2: (2047, 2076),
        3: (2061, 2090),
    }

    def __init__(self, *args, **params):
        super().__init__(*args, **params)

        # Location defaults
        self.location.area_subset = "CA counties"
        self.location.cached_area = "Los Angeles County"

        # Initialze tmy_adanced_options param
        self.param["computation_method"].objects = self.tmy_advanced_options_dict[
            self.data_type
        ]["objects"]
        self.computation_method = self.tmy_advanced_options_dict[self.data_type][
            "default"
        ]

        # Initialize tmy_computation_description param
        self.tmy_computation_description = self.computatation_description_dict[
            self.data_type
        ][self.computation_method]

        # Postage data and anomalies defaults
        self.historical_tmy_data = retrieve_amy_data(
            selections=self.selections,
            location=self.location,
            _cat=self.cat,
            year_start=1981,
            year_end=2010,
        ).compute()
        self.future_tmy_data = retrieve_amy_data(
            _cat=self.cat,
            selections=self.selections,
            location=self.location,
            year_start=self.warming_year_average_range[self.warmlevel][0],
            year_end=self.warming_year_average_range[self.warmlevel][1],
        ).compute()

        # Colormap
        self.cmap = _read_ae_colormap(cmap="ae_orange", cmap_hex=False)

        # Selectors defaults
        self.selections.append_historical = False
        self.selections.area_average = "Yes"
        self.selections.resolution = "45 km"
        self.selections.scenario_historical = ["Historical Climate"]
        self.selections.scenario_ssp = []
        self.selections.time_slice = (1981, 2010)
        self.selections.timescale = "hourly"
        self.selections.variable = "Air Temperature at 2m"
        self.selections.simulation = ["ensmean"]

    # For reloading data and plots
    reload_data = param.Action(
        lambda x: x.param.trigger("reload_data"), label="Reload Data"
    )

    @param.depends("selections.variable", "data_type", watch=True)
    def _update_cmap(self):
        """Set colormap depending on variable"""
        cmap_name = var_catalog[
            (var_catalog["display_name"] == self.selections.variable)
            & (var_catalog["timescale"] == "hourly")
        ].colormap.item()

        # Set to diverging colormap if difference is selected
        if self.data_type == "Difference":
            cmap_name = "ae_diverging"

        # Read colormap hex
        self.cmap = _read_ae_colormap(cmap=cmap_name, cmap_hex=False)

    @param.depends("computation_method", "reload_data", "warmlevel", watch=True)
    def _update_data_to_be_returned(self):
        """Update self.selections so that the correct data is returned by app.retrieve()"""

        if self.computation_method == "Historical":
            self.selections.scenario_historical = ["Historical Climate"]
            self.selections.scenario_ssp = []
            self.selections.time_slice = (1981, 2010)

        elif self.computation_method == "Warming Level Future":
            self.selections.scenario_ssp = ["SSP 3-7.0 -- Business as Usual"]
            self.selections.scenario_historical = []
            self.selections.time_slice = self.warming_year_average_range[self.warmlevel]

        self.selections.simulation = ["ensmean"]
        self.selections.append_historical = False
        self.selections.area_average = "Yes"
        self.selections.timescale = "hourly"

    @param.depends("reload_data", watch=True)
    def _update_tmy_data(self):
        """If the button was clicked and the location or variable was changed,
        reload the tmy data from AWS"""

        self.historical_tmy_data = retrieve_amy_data(
            selections=self.selections,
            location=self.location,
            _cat=self.cat,
            year_start=1981,
            year_end=2010,
        ).compute()
        self.future_tmy_data = retrieve_amy_data(
            _cat=self.cat,
            selections=self.selections,
            location=self.location,
            year_start=self.warming_year_average_range[self.warmlevel][0],
            year_end=self.warming_year_average_range[self.warmlevel][1],
        ).compute()

    # Create a function that will update computation_method when data_type is modified
    @param.depends("data_type", watch=True)
    def _update_computation_method(self):
        self.param["computation_method"].objects = self.tmy_advanced_options_dict[
            self.data_type
        ]["objects"]
        self.computation_method = self.tmy_advanced_options_dict[self.data_type][
            "default"
        ]

    @param.depends("data_type", "computation_method", watch=True)
    def _update_tmy_computatation_description(self):
        self.tmy_computation_description = self.computatation_description_dict[
            self.data_type
        ][self.computation_method]

    @param.depends("reload_data", watch=False)
    def _tmy_hourly_heatmap(self):
        # update heatmap df and title with selections
        days_in_year = 366
        if self.data_type == "Absolute":
            if self.computation_method == "Historical":
                df = compute_amy(self.historical_tmy_data, days_in_year=days_in_year)
                title = "Average Meteorological Year: {}\nAbsolute {} Baseline".format(
                    self.location.cached_area, self.computation_method
                )
                clabel = (
                    self.selections.variable
                    + " ("
                    + self.historical_tmy_data.attrs["units"]
                    + ")"
                )
            else:
                df = compute_amy(self.future_tmy_data, days_in_year=days_in_year)
                title = "Average Meteorological Year: {}\nAbsolute {} at {}°C".format(
                    self.location.cached_area, self.computation_method, self.warmlevel
                )
                clabel = self.selections.variable + " (" + self.selections.units + ")"
        elif self.data_type == "Difference":
            cmap = _read_ae_colormap("ae_diverging", cmap_hex=False)
            if self.computation_method == "Warming Level Future":
                df = compute_amy(
                    self.future_tmy_data, days_in_year=days_in_year
                ) - compute_amy(self.historical_tmy_data, days_in_year=days_in_year)
                title = "Average Meteorological Year: {}\nDifference between {} at {}°C and Historical Baseline".format(
                    self.location.cached_area, self.computation_method, self.warmlevel
                )
                clabel = self.selections.variable + " (" + self.selections.units + ")"
            else:
                df = compute_severe_yr(
                    self.future_tmy_data, days_in_year=days_in_year
                ) - compute_amy(self.historical_tmy_data, days_in_year=days_in_year)
                title = "Severe Meteorological Year: {}\nDifference between {} at 90th percentile and Historical Baseline".format(
                    self.location.cached_area, self.computation_method
                )
                clabel = self.selections.variable + " (" + self.selections.units + ")"
        else:
            title = "Average Meteorological Year\n{}".format(self.location.cached_area)

        heatmap = _amy_heatmap(
            amy_df=df,
            title=title,
            cmap=self.cmap,
            cbar_label=self.selections.variable + "(" + self.selections.units + ")",
        )

        return heatmap


# =========================== OBJECT VISUALIZATION USING PARAM ==============================


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
                pn.widgets.RadioButtonGroup.from_param(tmy_ob.param.data_type),
                pn.widgets.Select.from_param(
                    tmy_ob.param.computation_method, name="Computation Options"
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
                pn.widgets.StaticText(name="", value="Model Resolution"),
                pn.widgets.RadioButtonGroup.from_param(selections.param.resolution),
                width=280,
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
        height=615,
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
