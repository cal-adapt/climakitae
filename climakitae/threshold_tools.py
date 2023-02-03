########################################
#                                      #
# THRESHOLD TOOLS                      #
#                                      #
########################################

import numpy as np
import pandas as pd
import xarray as xr
from scipy import stats
import lmoments3 as lm
from lmoments3 import distr as ldistr
from lmoments3 import stats as lstats
import matplotlib.pyplot as plt
from matplotlib import cm
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import geoviews as gv
import holoviews as hv
from holoviews import opts
import hvplot.pandas
import hvplot.xarray
import panel as pn
from .visualize import get_geospatial_plot


def get_ams(
    da, 
    extremes_type="max",
    duration=None,
    groupby=None,
    duration2=None,
    ):
    """
    Returns a data array of annual maximums.
    
    Optional arguments `duration`, `groupby`, and `duration2` define the type
    of event to find the annual maximums of. These correspond to the event
    types defined in the `get_exceedance_count` function. 

    `duration` must be specified as (X, 'hour')
    `groupby` must be specified as (Y, 'day')
    `duration2` must be specified as (Z, 'day')
    """

    extremes_types = ["max", "min"]
    if extremes_type not in extremes_types:
        raise ValueError(
            "invalid extremes type. expected one of the following: %s" % extremes_types
        )

    # To implement:
    # Need to check the duration and groupby arguments 
    #   make sure duration1 < groupby < duration2 (as is checked in the `get_exceedance_count` functions)

    # In the simplest case, we use the original data array to take annual 
    # extreme values from
    da_series = da

    if duration != None:
        # In this case, user is interested in extreme events lasting at least 
        # as long as the length of `duration`. 
        dur_len, dur_type = duration
        if dur_type != "hour" or da_series.frequency != "1hr":
            raise ValueError("Current specifications not yet implemented. `duration` options only implemented for `hour` specifications.")

        # First identify the min (max) value for each window of length `duration`
        if extremes_type == "max":
            da_series = da_series.rolling(time=dur_len, center=False).min("time")
        elif extremes_type == "min":
            da_series = da_series.rolling(time=dur_len, center=False).max("time")
    
    if groupby != None:
        # In this case, select the max (min) in each group. (This option is
        # really only meaningful when coupled with the `duration2` option.)
        group_len, group_type = groupby
        if group_type != 'day': 
            raise ValueError("`groupby` specifications only implemented for 'day' groupings.")
        
        # select the max (min) in each group
        if extremes_type == "max":
            da_series = da_series.resample(time=f"{group_len}D", label="left").max()
        elif extremes_type == "min":
            da_series = da_series.resample(time=f"{group_len}D", label="left").max()

    if duration2 != None:
        if groupby == None:
            raise ValueError("To use `duration2` option, must first use groupby.")
        # In this case, identify the min (max) value of the grouped values for
        # each window of length duration2. Must be in `days`.
        dur2_len, dur2_type = duration2
        if dur2_type != 'day':
            raise ValueError("`duration2` specification must be in days. example: `duration2 = (3, 'day')`.")

        # Now select the min (max) from the duration period
        if extremes_type == "max":
            da_series = da.rolling(time=dur2_len, center=False).min("time")
        elif extremes_type == "min":
            da_series = da.rolling(time=dur2_len, center=False).max("time")
    
    # Now select the most extreme value for each year in the series
    if extremes_type == "max":
        ams = da_series.resample(time="A").max(keep_attrs=True)
        ams.attrs["extremes type"] = "maxima"
    elif extremes_type == 'min':
        ams = da_series.resample(time='A').min(keep_attrs = True)
        ams.attrs['extremes type'] = 'minima'
    
    # Common attributes
    ams.attrs["extreme value extraction method"] = "block maxima"
    ams.attrs["block size"] = "1 year"
    ams.attrs["timeseries type"] = "annual max series"

    return ams


def get_lmom_distr(distr):
    """
    Returns corresponding l-moments distribution function from selected
    distribution name.
    """

    distrs = ["gev", "gumbel", "weibull", "pearson3", "genpareto"]

    if distr == "gev":
        lmom_distr = ldistr.gev
    elif distr == "gumbel":
        lmom_distr = ldistr.gum
    elif distr == "weibull":
        lmom_distr = ldistr.wei
    elif distr == "pearson3":
        lmom_distr = ldistr.pe3
    elif distr == "genpareto":
        lmom_distr = ldistr.gpa
    else:
        raise ValueError(
            "invalid distr type. expected one of the following: %s" % distrs
        )

    return lmom_distr


def get_fitted_distr(ams, distr, lmom_distr):
    """
    Returns fitted l-moments distribution function from l-moments.
    """

    lmoments = lmom_distr.lmom_fit(ams)

    if distr == "gev":
        fitted_distr = stats.genextreme(**lmoments)
    elif distr == "gumbel":
        fitted_distr = stats.gumbel_r(**lmoments)
    elif distr == "weibull":
        fitted_distr = stats.weibull_min(**lmoments)
    elif distr == "pearson3":
        fitted_distr = stats.pearson3(**lmoments)
    elif distr == "genpareto":
        fitted_distr = stats.genpareto(**lmoments)
    return lmoments, fitted_distr


def get_lmoments(ams, distr="gev", multiple_points=True):
    """
    Returns dataset of l-moments ratios from an inputed maximum series.
    """

    lmom_distr = get_lmom_distr(distr)
    ams_attributes = ams.attrs

    if multiple_points:
        ams = ams.stack(allpoints=["y", "x"]).squeeze().groupby("allpoints")

    lmoments = xr.apply_ufunc(
        lmom_distr.lmom_fit,
        ams,
        input_core_dims=[["time"]],
        exclude_dims=set(("time",)),
        output_core_dims=[[]],
    )

    lmoments = lmoments.rename("lmoments")
    new_ds = lmoments.to_dataset().to_array()

    if multiple_points:
        new_ds = new_ds.unstack("allpoints")

    new_ds.attrs = ams_attributes
    new_ds.attrs["distribution"] = "{}".format(str(distr))
    return new_ds


def get_ks_stat(ams, distr="gev", multiple_points=True):
    """
    Returns a dataset of ks test d-statistics and p-values from an inputed
    maximum series.
    """

    lmom_distr = get_lmom_distr(distr)
    ams_attributes = ams.attrs

    if multiple_points:
        ams = ams.stack(allpoints=["y", "x"]).squeeze().groupby("allpoints")

    def ks_stat(ams):
        lmoments, fitted_distr = get_fitted_distr(ams, distr, lmom_distr)

        if distr == "gev":
            cdf = "genextreme"
            args = (lmoments["c"], lmoments["loc"], lmoments["scale"])
        elif distr == "gumbel":
            cdf = "gumbel_r"
            args = (lmoments["loc"], lmoments["scale"])
        elif distr == "weibull":
            cdf = "weibull_min"
            args = (lmoments["c"], lmoments["loc"], lmoments["scale"])
        elif distr == "pearson3":
            cdf = "pearson3"
            args = (lmoments["skew"], lmoments["loc"], lmoments["scale"])
        elif distr == "genpareto":
            cdf = "genpareto"
            args = (lmoments["c"], lmoments["loc"], lmoments["scale"])

        try:
            ks = stats.kstest(ams, cdf, args=args)
            d_statistic = ks[0]
            p_value = ks[1]
        except (ValueError, ZeroDivisionError):
            d_statistic = np.nan
            p_value = np.nan

        return d_statistic, p_value

    d_statistic, p_value = xr.apply_ufunc(
        ks_stat,
        ams,
        input_core_dims=[["time"]],
        exclude_dims=set(("time",)),
        output_core_dims=[[], []],
    )

    d_statistic = d_statistic.rename("d_statistic")
    new_ds = d_statistic.to_dataset()
    new_ds["p_value"] = p_value

    if multiple_points:
        new_ds = new_ds.unstack("allpoints")

    new_ds["d_statistic"].attrs["stat test"] = "KS test"
    new_ds["p_value"].attrs["stat test"] = "KS test"
    new_ds.attrs = ams_attributes
    new_ds.attrs["distribution"] = "{}".format(str(distr))
    return new_ds


def calculate_return(fitted_distr, data_variable, arg_value):
    """
    Returns corresponding extreme value calculation for selected data variable.
    """

    try:
        if data_variable == "return_value":
            return_event = 1.0 - (1.0 / arg_value)
            return_value = fitted_distr.ppf(return_event)
            result = round(return_value, 5)

        elif data_variable == "return_prob":
            result = 1 - (fitted_distr.cdf(arg_value))

        elif data_variable == "return_period":
            return_prob = fitted_distr.cdf(arg_value)
            if return_prob == 1.0:
                result = np.nan
            else:
                return_period = -1.0 / (return_prob - 1.0)
                result = round(return_period, 3)

    except (ValueError, ZeroDivisionError, AttributeError):
        result = np.nan  

    return result


def bootstrap(ams, distr="gev", data_variable="return_value", arg_value=10):
    """
    Returns a bootstrap-calculated value for relevant parameters from an
    inputed maximum series.
    """

    data_variables = ["return_value", "return_prob", "return_period"]
    if data_variable not in data_variables:
        raise ValueError(
            "invalid data variable type. expected one of the following: %s"
            % data_variables
        )

    lmom_distr = get_lmom_distr(distr)

    sample_size = len(ams)
    new_ams = np.random.choice(ams, size=sample_size, replace=True)

    try:
        lmoments, fitted_distr = get_fitted_distr(new_ams, distr, lmom_distr)
        result = calculate_return(
            fitted_distr=fitted_distr,
            data_variable=data_variable,
            arg_value=arg_value,
        )
    except (ValueError, ZeroDivisionError):
        result = np.nan

    return result


def conf_int(
    ams,
    distr,
    data_variable,
    arg_value,
    bootstrap_runs,
    conf_int_lower_bound,
    conf_int_upper_bound,
):
    """
    Returns lower and upper limits of confidence interval given selected parameters.
    """

    bootstrap_values = []

    for _ in range(bootstrap_runs):
        result = bootstrap(
            ams,
            distr,
            data_variable,
            arg_value,
        )
        bootstrap_values.append(result)

    conf_int_array = np.percentile(
        bootstrap_values, [conf_int_lower_bound, conf_int_upper_bound]
    )

    conf_int_lower_limit = conf_int_array[0]
    conf_int_upper_limit = conf_int_array[1]
    return conf_int_lower_limit, conf_int_upper_limit

def get_return_variable(
    ams,
    data_variable,
    arg_value,
    distr="gev",
    bootstrap_runs=100,
    conf_int_lower_bound=2.5,
    conf_int_upper_bound=97.5,
    multiple_points=True,
):
    """
    Generic function used by `get_return_value`, `get_return_period`, and
    `get_return_prob`.

    Returns a dataset with the estimate of the requested data_variable, and
    confidence intervals.

    If data_variable == "return_value", then arg_value is the return period.
    If data_variable == "return_prob", then arg_value is the threshold value.
    If data_variable == "return_period", then arg_value is the return value.
    """

    data_variables = ['return_value', 'return_period', 'return_prob']
    if data_variable not in data_variables:
        raise ValueError(
            "invalid data_variable. "
        )

    lmom_distr = get_lmom_distr(distr)
    ams_attributes = ams.attrs

    if multiple_points:
        ams = ams.stack(allpoints=["y", "x"]).squeeze().groupby("allpoints")

    def return_variable(ams):
        try:
            lmoments, fitted_distr = get_fitted_distr(ams, distr, lmom_distr)
            return_variable = calculate_return(
                fitted_distr=fitted_distr,
                data_variable=data_variable,
                arg_value=arg_value,
            )
        except (ValueError, ZeroDivisionError):
            return_variable = np.nan
            
        conf_int_lower_limit, conf_int_upper_limit = conf_int(
            ams=ams,
            distr=distr,
            data_variable=data_variable,
            arg_value=arg_value,
            bootstrap_runs=bootstrap_runs,
            conf_int_lower_bound=conf_int_lower_bound,
            conf_int_upper_bound=conf_int_upper_bound,
        )

        return return_variable, conf_int_lower_limit, conf_int_upper_limit

    return_variable, conf_int_lower_limit, conf_int_upper_limit = xr.apply_ufunc(
        return_variable,
        ams,
        input_core_dims=[["time"]],
        exclude_dims=set(("time",)),
        output_core_dims=[[], [], []],
    )

    return_variable = return_variable.rename(data_variable)
    new_ds = return_variable.to_dataset()
    new_ds["conf_int_lower_limit"] = conf_int_lower_limit
    new_ds["conf_int_upper_limit"] = conf_int_upper_limit

    if multiple_points:
        new_ds = new_ds.unstack("allpoints")

    if data_variable == "return_value":
        new_ds[data_variable].attrs["return period"] = f"1 in {arg_value} year event"
    elif data_variable == "return_prob":
        new_ds[data_variable].attrs["threshold"] = f"exceedance of {arg_value} value event"
    elif data_variable == "return_period":
        new_ds[data_variable].attrs["return value"] = f"occurrence of a {arg_value} value event"

    new_ds["conf_int_lower_limit"].attrs[
        "confidence interval lower bound"
    ] = "{}th percentile".format(str(conf_int_lower_bound))
    new_ds["conf_int_upper_limit"].attrs[
        "confidence interval upper bound"
    ] = "{}th percentile".format(str(conf_int_upper_bound))

    new_ds.attrs = ams_attributes
    new_ds.attrs["distribution"] = "{}".format(str(distr))
    return new_ds

def get_return_value(
    ams,
    return_period=10,
    distr="gev",
    bootstrap_runs=100,
    conf_int_lower_bound=2.5,
    conf_int_upper_bound=97.5,
    multiple_points=True,
):
    """
    Returns dataset with return values and confidence intervals from maximum series.
    """ 
    return get_return_variable(
        ams, "return_value", return_period, distr, bootstrap_runs, 
        conf_int_lower_bound, conf_int_upper_bound, multiple_points
    )

def get_return_prob(
    ams,
    threshold,
    distr="gev",
    bootstrap_runs=100,
    conf_int_lower_bound=2.5,
    conf_int_upper_bound=97.5,
    multiple_points=True,
):
    """
    Returns dataset with return probabilities and confidence intervals from maximum series.
    """
    return get_return_variable(
        ams, "return_prob", threshold, distr, bootstrap_runs, 
        conf_int_lower_bound, conf_int_upper_bound, multiple_points
    )

def get_return_period(
    ams,
    return_value,
    distr="gev",
    bootstrap_runs=100,
    conf_int_lower_bound=2.5,
    conf_int_upper_bound=97.5,
    multiple_points=True,
):
    """
    Returns dataset with return periods and confidence intervals from maximum series.
    """
    return get_return_variable(
        ams, "return_period", return_value, distr, bootstrap_runs, 
        conf_int_lower_bound, conf_int_upper_bound, multiple_points
    )


# ===================== Functions for exceedance count =========================


def get_exceedance_count(
    da,
    threshold_value,
    duration1=None,
    period=(1, "year"),
    threshold_direction="above",
    duration2=None,
    groupby=None,
    smoothing=None,
):
    """
    Calculate the number of occurances of exceeding the specified threshold
    within each period.

    Returns an xarray with the same coordinates as the input data except for
    the time dimension, which will be collapsed to one value per period (equal
    to the number of event occurances in each period).

    Arguments:
    da -- an xarray.DataArray of some climate variable. Can have multiple
        scenarios, simulations, or x and y coordinates.
    threshold_value -- value against which to test exceedance

    Optional Keyword Arguments:
    period -- amount of time across which to sum the number of occurances,
        default is (1, "year"). Specified as a tuple: (x, time) where x is an
        integer, and time is one of: ["day", "month", "year"]
    threshold_direction -- string either "above" or "below", default is above.
    duration1 -- length of exceedance in order to qualify as an event (before grouping)
    groupby -- see examples for explanation. Typical grouping could be (1, "day")
    duration2 -- length of exceedance in order to qualify as an event (after grouping)
    smoothing -- option to average the result across multiple periods with a
        rolling average; value is either None or the number of timesteps to use
        as the window size
    """

    # --------- Type check arguments -------------------------------------------

    # Check compatibility of periods, durations, and groupbys
    if _is_greater(duration1, groupby):
        raise ValueError(
            "Incompatible `group` and `duration1` specification. Duration1 must be shorter than group."
        )
    if _is_greater(groupby, duration2):
        raise ValueError(
            "Incompatible `group` and `duration2` specification. Duration2 must be longer than group."
        )
    if _is_greater(groupby, period):
        raise ValueError(
            "Incompatible `group` and `period` specification. Group must be longer than period."
        )
    if _is_greater(duration2, period):
        raise ValueError(
            "Incompatible `duration` and `period` specification. Period must be longer than duration."
        )

    # Check compatibility of specifications with the data frequency (hourly, daily, or monthly)
    freq = (
        (1, "hour")
        if da.frequency == "hourly"
        else ((1, "day") if da.frequency == "daily" else (1, "month"))
    )
    if _is_greater(freq, groupby):
        raise ValueError(
            "Incompatible `group` specification: cannot be less than data frequency."
        )
    if _is_greater(freq, duration2):
        raise ValueError(
            "Incompatible `duration` specification: cannot be less than data frequency."
        )
    if _is_greater(freq, period):
        raise ValueError(
            "Incompatible `period` specification: cannot be less than data frequency."
        )

    # --------- Calculate occurances -------------------------------------------

    events_da = get_exceedance_events(
        da, threshold_value, threshold_direction, duration1, groupby
    )

    # --------- Apply specified duration requirement ---------------------------

    if duration2 is not None:
        dur_len, dur_type = duration2

        if (
            groupby is not None
            and groupby[1] == dur_type
            or groupby is None
            and freq[1] == dur_type
        ):
            window_size = dur_len
        else:
            raise ValueError(
                "Duration options for time types (i.e. hour, day) that are different than group or frequency not yet implemented"
            )

        # The "min" operation will return 0 if any time in the window is not an
        # event, which is the behavior we want. It will only return 1 for True
        # if all values in the duration window are 1.
        events_da = events_da.rolling(time=window_size, center=False).min("time")

    # --------- Sum occurances across each period ------------------------------

    period_len, period_type = period
    period_indexer = str.capitalize(
        period_type[0]
    )  # capitalize first letter to use as indexer in resample
    exceedance_count = events_da.resample(
        time=f"{period_len}{period_indexer}", label="left"
    ).sum()

    # Optional smoothing
    if smoothing is not None:
        exceedance_count = exceedance_count.rolling(time=smoothing, center=True).mean(
            "time"
        )

    # --------- Set new attributes for the counts DataArray --------------------
    exceedance_count.attrs["variable_name"] = da.name
    exceedance_count.attrs["variable_units"] = exceedance_count.units
    exceedance_count.attrs["period"] = period
    exceedance_count.attrs["duration1"] = duration1
    exceedance_count.attrs["group"] = groupby
    exceedance_count.attrs["duration2"] = duration2
    exceedance_count.attrs["threshold_value"] = threshold_value
    exceedance_count.attrs["threshold_direction"] = threshold_direction
    exceedance_count.attrs["units"] = _exceedance_count_name(exceedance_count)

    # Set name (for plotting, this will be the y-axis label)
    exceedance_count.name = "Count"

    return exceedance_count


def _is_greater(time1, time2):
    """
    Helper function for comparing user specifications of period, duration, and groupby.
    Examples:
        (1, "day"), (1, "year") --> False
        (3, "month"), (1, "month") --> True
    """
    order = ["hour", "day", "month", "year"]
    if time1 is None or time2 is None:
        return False
    elif time1[1] == time2[1]:
        return time1[0] > time2[0]
    else:
        return order.index(time1[1]) > order.index(time2[1])


def get_exceedance_events(
    da, threshold_value, threshold_direction="above", duration1=None, groupby=None
):
    """
    Returns an xarray that specifies whether each entry of `da` is a qualifying
    threshold event. Values are 0 for False, 1 for True, or NaN for NaNs.
    """

    # Identify occurances (and preserve NaNs)
    if threshold_direction == "above":
        events_da = (da > threshold_value).where(da.isnull() == False)
    elif threshold_direction == "below":
        events_da = (da < threshold_value).where(da.isnull() == False)
    else:
        raise ValueError(
            f"Unknown value for `threshold_direction` parameter: {threshold_direction}. Available options are 'above' or 'below'."
        )

    if duration1 is not None:
        dur_len, dur_type = duration1
        if dur_type != "hour" or da.frequency != "hourly":
            raise ValueError("Current specifications not yet implemented.")
        window_size = dur_len

        # The "min" operation will return 0 if any time in the window is not an
        # event, which is the behavior we want. It will only return 1 for True
        # if all values in the duration window are 1.
        events_da = events_da.rolling(time=window_size, center=False).min("time")

    # Groupby
    if groupby is not None:
        if (
            (groupby == (1, "hour") and da.frequency == "hourly")
            or (groupby == (1, "day") and da.frequency == "daily")
            or (groupby == (1, "month") and da.frequency == "monthly")
            or groupby == duration1
        ):
            # groupby specification is the same as data frequency, do nothing
            pass
        else:
            group_len, group_type = groupby
            indexer_type = str.capitalize(
                group_type[0]
            )  # capitalize the first letter to use as the indexer (i.e. H, D, M, or Y)
            group_totals = events_da.resample(
                time=f"{group_len}{indexer_type}", label="left"
            ).sum()  # sum occurences within each group
            events_da = (group_totals > 0).where(
                group_totals.isnull() == False
            )  # turn back into a boolean with preserved NaNs (0 or 1 for whether there is any occurance in the group)
    return events_da


def _exceedance_count_name(exceedance_count):
    """
    Helper function to build the appropriate name for the queried exceedance count.
    Examples:
        'Number of hours'
        'Number of days'
        'Number of 3-day events'
    """
    # If duration is used, this determines the event name
    dur = exceedance_count.duration2
    if dur is not None:
        d_num, d_type = dur
        if d_num != 1:
            event = f"{d_num}-{d_type} events"
        else:
            event = f"{d_type}s"  # ex: day --> days
    else:
        # otherwise use "groupby" if not None
        grp = exceedance_count.group
        if grp is not None:
            g_num, g_type = grp
            if g_num != 1:
                event = f"{g_num}-{g_type} events"
            else:
                event = f"{g_type}s"  # ex: day --> days
        else:
            # otherwise use data frequency info as the default event type
            if exceedance_count.frequency == "hourly":
                event = "hours"
            elif exceedance_count.frequency == "daily":
                event = "days"
            elif exceedance_count.frequency == "monthly":
                event = "months"
    return f"Number of {event}"


def plot_exceedance_count(exceedance_count):
    """
    Plots each simulation as a different color line.
    Drop down option to select different scenario.
    Currently can only plot for one location, so is expecting input to already be subsetted or an area average.
    """
    plot_obj = exceedance_count.hvplot.line(
        x="time",
        widget_location="bottom",
        by="simulation",
        groupby=["scenario"],
        title="",
        fontsize={"ylabel": "10pt"},
        legend="right",
    )
    return pn.Column(plot_obj)


def _exceedance_plot_title(exceedance_count):
    """
    Helper function for making the title for exceedance plots.
    Examples:
        'Air Temperatue at 2m: events above 35C'
        'Preciptation (total): events below 10mm'
    """
    return f"{exceedance_count.variable_name}: events {exceedance_count.threshold_direction} {exceedance_count.threshold_value}{exceedance_count.variable_units}"


def _exceedance_plot_subtitle(exceedance_count):
    """
    Examples:
        'Number of hours per year'
        'Number of 4-hour events per 3-months'
        'Number of days per year with conditions lasting at least 4-hours'
    """

    if exceedance_count.duration2 != exceedance_count.duration1:
        dur_len, dur_type = exceedance_count.duration1
        _s = "" if dur_len == 1 else "s"
        dur_str = f" with conditions lasting at least {dur_len} {dur_type}{_s}"
    else:
        dur_str = ""

    if exceedance_count.duration2 != exceedance_count.group:
        grp_len, grp_type = exceedance_count.group
        if grp_len == 1:
            grp_str = f" each {grp_type}"
        else:
            grp_str = f" every {grp_len} {grp_type}s"
    else:
        grp_str = ""

    per_len, per_type = exceedance_count.period
    if per_len == 1:
        period_str = f" each {per_type}"
    else:
        period_str = f" per {per_len}-{per_type} period"

    _subtitle = (
        _exceedance_count_name(exceedance_count) + period_str + dur_str + grp_str
    )
    return _subtitle
