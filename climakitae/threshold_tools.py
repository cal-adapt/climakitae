"""Helper functions for performing analyses related to thresholds"""

import numpy as np
import pandas as pd
import xarray as xr
from scipy import stats
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
import statsmodels as sm
from .utils import _read_ae_colormap
from warnings import warn

def calculate_ess(
    data,
    nlags=None
    ):
    """ 
    Function for calculating the effective sample size (ESS) of the provided data.

    Input array is assumed to be timeseries data with potential autocorrelation.
    """
    n=len(data)
    if nlags is None:
        nlags = n
    acf = sm.tsa.stattools.acf(data, nlags = nlags, fft = True)
    sums = 0
    for k in range(1, len(acf)):
        sums = sums + (n-k)*acf[k]/n

    return n/(1+2*sums)

def get_block_maxima(
    da, 
    extremes_type="max",
    duration=None,
    groupby=None,
    grouped_duration=None,
    check_ess=True,
    block_size=1
    ):
    """
    Function that converts data into block maximums, defaulting to annual maximums (default block size = 1 year).

    Takes input array and resamples by taking the maximum value over the specified block size.
    
    Optional arguments `duration`, `groupby`, and `grouped_duration` define the type
    of event to find the annual maximums of. These correspond to the event
    types defined in the `get_exceedance_count` function. 

    Parameters
    ----------
    da: xarray.DataArray
        DataArray from app.retrieve
    extremes_type: str
        option for max or min
        Defaults to max
    duration: tuple
        length of extreme event, specified as (4, 'hour')
    groupby: tuple
        group over which to look for max occurance, specified as (1, 'day')
    grouped_duration: tuple
        length of event after grouping, specified as (5, 'day')
    check_ess: boolean
        optional flag specifying whether to check the effective sample size (ESS)
        within the blocks of data, and throw a warning if the average ESS is too small.
        can be silenced with check_ess=False.
    block_size: int
        block size in years. default is 1 year.

    Returns
    -------
    xarray.DataArray
    """

    extremes_types = ["max", "min"]
    if extremes_type not in extremes_types:
        raise ValueError(
            "invalid extremes type. expected one of the following: %s" % extremes_types
        )

    # To implement:
    # Need to check the duration and groupby arguments 
    #   make sure duration1 < groupby < grouped_duration (as is checked in the `get_exceedance_count` functions)

    # In the simplest case, we use the original data array to take annual 
    # extreme values from
    da_series = da

    if duration != None:
        # In this case, user is interested in extreme events lasting at least 
        # as long as the length of `duration`. 
        dur_len, dur_type = duration
        if dur_type != "hour" or da_series.frequency not in ["1hr", "hourly"]:
            raise ValueError("Current specifications not yet implemented. `duration` options only implemented for `hour` specifications.")

        # First identify the min (max) value for each window of length `duration`
        if extremes_type == "max":
            da_series = da_series.rolling(time=dur_len, center=False).min("time")
        elif extremes_type == "min":
            da_series = da_series.rolling(time=dur_len, center=False).max("time")
    
    if groupby != None:
        # In this case, select the max (min) in each group. (This option is
        # really only meaningful when coupled with the `grouped_duration` option.)
        group_len, group_type = groupby
        if group_type != 'day': 
            raise ValueError("`groupby` specifications only implemented for 'day' groupings.")
        
        # select the max (min) in each group
        if extremes_type == "max":
            da_series = da_series.resample(time=f"{group_len}D", label="left").max()
        elif extremes_type == "min":
            da_series = da_series.resample(time=f"{group_len}D", label="left").min()

    if grouped_duration != None:
        if groupby == None:
            raise ValueError("To use `grouped_duration` option, must first use groupby.")
        # In this case, identify the min (max) value of the grouped values for
        # each window of length `grouped_duration``. Must be in `days`.
        dur2_len, dur2_type = grouped_duration
        if dur2_type != 'day':
            raise ValueError("`grouped_duration` specification must be in days. example: `grouped_duration = (3, 'day')`.")

        # Now select the min (max) from the duration period
        if extremes_type == "max":
            da_series = da_series.rolling(time=dur2_len, center=False).min("time")
        elif extremes_type == "min":
            da_series = da_series.rolling(time=dur2_len, center=False).max("time")
    
    # Now select the most extreme value for each year in the series
    if extremes_type == "max":
        bms = da_series.resample(time=f"{block_size}A").max(keep_attrs=True)
        bms.attrs["extremes type"] = "maxima"
    elif extremes_type == 'min':
        bms = da_series.resample(time=f"{block_size}A").min(keep_attrs = True)
        bms.attrs["extremes type"] = "minima"

    # Calculate the effective sample size of the computed event type in all blocks, check the average value
    if check_ess:
        # all_ess = da_series.groupby("time.year").apply(calculate_ess) # how to set this up correctly? this currently errors so using a for loop below instead
        all_ess = []
        for yr in set(da_series.time.dt.year.values):
            ess = calculate_ess(da_series.sel(time = f"{yr}"))
            all_ess.append(ess)
        average_ess = np.nanmean(all_ess)
        if average_ess < 25:
            warn(f"The average effective sample size in your data is {average_ess} per block, which is low. This may result in biased estimates of extreme value distributions when calculating return values, periods, and probabilities from this data.")
    
    # Common attributes
    bms.attrs["duration"] = duration
    bms.attrs["groupby"] = groupby
    bms.attrs["grouped_duration"] = grouped_duration
    bms.attrs["extreme value extraction method"] = f"block maxima"
    bms.attrs["block size"] = f"{block_size} year"
    bms.attrs["timeseries type"] = f"block {extremes_type} series"

    return bms


def _get_distr_func(distr):
    """Function that sets the scipy distribution object

    Sets corresponding distribution function from selected
    distribution name.

    Parameters
    ----------
    distr: str
        name of distribution to use

    Returns
    -------
    scipy.stats
    """

    distrs = ["gev", "gumbel", "weibull", "pearson3", "genpareto"]

    if distr == "gev":
        distr_func = stats.genextreme
    elif distr == "gumbel":
        distr_func = stats.gumbel_r
    elif distr == "weibull":
        distr_func = stats.weibull_min
    elif distr == "pearson3":
        distr_func = stats.pearson3
    elif distr == "genpareto":
        distr_func = stats.genpareto
    else:
        raise ValueError(
            "invalid distribution type. expected one of the following: %s" % distrs
        )

    return distr_func


def _get_fitted_distr(bms, distr, distr_func):
    """Function for fitting data to distribution function

    Takes data array and fits it to distribution function.

    Parameters
    ----------
    bms: xarray.DataArray
        Block maximum series, can be output from the function get_block_maxima()
    distr: str
    distr_func: scipy.stats

    Returns
    -------
    parameters: dict
        dictionary of distribution function parameters
    fitted_distr: scipy.rv_frozen
        frozen fitted distribution
    """

    def get_param_dict(p_names, p_values):
        """Function for building the dictionary of parameters used as argument
        to scipy.stats distribution functions.
        """
        return dict(zip(p_names, p_values))

    parameters = None
    fitted_distr = None

    p_values = distr_func.fit(bms)

    if distr == "gev":
        p_names = ("c", "loc", "scale")
        parameters = get_param_dict(p_names, p_values)
        fitted_distr = stats.genextreme(**parameters)
    elif distr == "gumbel":
        p_names = ("loc", "scale")
        parameters = get_param_dict(p_names, p_values)
        fitted_distr = stats.gumbel_r(**parameters)
    elif distr == "weibull":
        p_names = ("c", "loc", "scale")
        parameters = get_param_dict(p_names, p_values)
        fitted_distr = stats.weibull_min(**parameters)
    elif distr == "pearson3":
        p_names = ("skew", "loc", "scale")
        parameters = get_param_dict(p_names, p_values)
        fitted_distr = stats.pearson3(**parameters)
    elif distr == "genpareto":
        p_names = ("c", "loc", "scale")
        parameters = get_param_dict(p_names, p_values)
        fitted_distr = stats.genpareto(**parameters)
    else:
        raise ValueError("invalid distribution type.")
    return parameters, fitted_distr


def get_ks_stat(bms, distr="gev", multiple_points=True):
    """Function to perform kstest on input DataArray

    Creates a dataset of ks test d-statistics and p-values from an inputed
    maximum series.

    Parameters
    ----------
    bms: xarray.DataArray
        Block maximum series, can be output from the function get_block_maxima()
    distr: str
    multiple_points: boolean

    Returns
    -------
    xarray.Dataset
    """

    distr_func = _get_distr_func(distr)
    bms_attributes = bms.attrs

    if multiple_points:
        bms = (
            bms.stack(allpoints=["y", "x"])
            .dropna(dim="allpoints")
            .squeeze()
            .groupby("allpoints")
        )

    def ks_stat(bms):
        parameters, fitted_distr = _get_fitted_distr(bms, distr, distr_func)

        if distr == "gev":
            cdf = "genextreme"
            args = (parameters["c"], parameters["loc"], parameters["scale"])
        elif distr == "gumbel":
            cdf = "gumbel_r"
            args = (parameters["loc"], parameters["scale"])
        elif distr == "weibull":
            cdf = "weibull_min"
            args = (parameters["c"], parameters["loc"], parameters["scale"])
        elif distr == "pearson3":
            cdf = "pearson3"
            args = (parameters["skew"], parameters["loc"], parameters["scale"])
        elif distr == "genpareto":
            cdf = "genpareto"
            args = (parameters["c"], parameters["loc"], parameters["scale"])

        try:
            ks = stats.kstest(bms, cdf, args=args)
            d_statistic = ks[0]
            p_value = ks[1]
        except (ValueError, ZeroDivisionError):
            d_statistic = np.nan
            p_value = np.nan

        return d_statistic, p_value

    d_statistic, p_value = xr.apply_ufunc(
        ks_stat,
        bms,
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
    new_ds.attrs = bms_attributes
    new_ds.attrs["distribution"] = "{}".format(str(distr))
    new_ds["p_value"].attrs["units"] = None
    new_ds["d_statistic"].attrs["units"] = None
    return new_ds


def _calculate_return(fitted_distr, data_variable, arg_value):
    """Function to perform extreme value calculation on fitted distribution

    Runs corresponding extreme value calculation for selected data variable.
    Can be the return value, probability, or period.

    Parameters
    ----------
    fitted_distr: scipy.rv_frozen
        frozen fitted distribution
    data_variable: str
        can be return_value, return_prob, return_period
    arg_value: float
        value to do the calucation to

    Returns
    -------
    float
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


def _bootstrap(bms, distr="gev", data_variable="return_value", arg_value=10):
    """Function for making a bootstrap-calculated value from input array

    Determines a bootstrap-calculated value for relevant parameters from an
    inputed maximum series.

    Parameters
    ----------
    bms: xarray.DataArray
        Block maximum series, can be output from the function get_block_maxima()
    distr: str
    data_variable: str
        can be return_value, return_prob, return_period
    arg_value: float
        value to do the calculation to

    Returns
    -------
    float
    """

    data_variables = ["return_value", "return_prob", "return_period"]
    if data_variable not in data_variables:
        raise ValueError(
            "invalid data variable type. expected one of the following: %s"
            % data_variables
        )

    distr_func = _get_distr_func(distr)

    sample_size = len(bms)
    new_bms = np.random.choice(bms, size=sample_size, replace=True)

    try:
        parameters, fitted_distr = _get_fitted_distr(new_bms, distr, distr_func)
        result = _calculate_return(
            fitted_distr=fitted_distr,
            data_variable=data_variable,
            arg_value=arg_value,
        )
    except (ValueError, ZeroDivisionError):
        result = np.nan

    return result


def _conf_int(
    bms,
    distr,
    data_variable,
    arg_value,
    bootstrap_runs,
    conf_int_lower_bound,
    conf_int_upper_bound,
):
    """Function for genearating lower and upper limits of confidence interval

    Returns lower and upper limits of confidence interval given selected parameters.

    Parameters
    ----------
    bms: xarray.DataArray
        Block maximum series, can be output from the function get_block_maxima()
    distr: str
    data_variable: str
        can be return_value, return_prob, return_period
    arg_value: float
        value to do the calucation to
    conf_int_lower_bound: float
    conf_int_upper_bound: float

    Returns
    -------
    float, float
    """

    bootstrap_values = []

    for _ in range(bootstrap_runs):
        result = _bootstrap(
            bms,
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
    bms,
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
    bms_attributes = bms.attrs

    if multiple_points:
        bms = bms.stack(allpoints=["y", "x"]).squeeze().groupby("allpoints")

    def return_variable(bms):
        try:
            lmoments, fitted_distr = get_fitted_distr(bms, distr, lmom_distr)
            return_variable = calculate_return(
                fitted_distr=fitted_distr,
                data_variable=data_variable,
                arg_value=arg_value,
            )
        except (ValueError, ZeroDivisionError):
            return_variable = np.nan
            
        conf_int_lower_limit, conf_int_upper_limit = conf_int(
            bms=bms,
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
        bms,
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
        new_ds[data_variable].attrs["return period"] = f"1-in-{arg_value}-year event"
    elif data_variable == "return_prob":
        unit_threshold = bms_attributes["units"]
        new_ds[data_variable].attrs["threshold"] = f"exceedance of {arg_value} {unit_threshold} value event"
        new_ds[data_variable].attrs["units"] = None
    elif data_variable == "return_period":
        unit_return_value = bms_attributes["units"]
        new_ds[data_variable].attrs["return value"] = f"{arg_value} {unit_return_value} event"
        new_ds[data_variable].attrs["units"] = "years"

    new_ds["conf_int_lower_limit"].attrs[
        "confidence interval lower bound"
    ] = "{}th percentile".format(str(conf_int_lower_bound))
    new_ds["conf_int_upper_limit"].attrs[
        "confidence interval upper bound"
    ] = "{}th percentile".format(str(conf_int_upper_bound))

    new_ds.attrs = bms_attributes
    new_ds.attrs["distribution"] = "{}".format(str(distr))
    return new_ds

def _get_return_variable(
    bms,
    data_variable,
    arg_value,
    distr="gev",
    bootstrap_runs=100,
    conf_int_lower_bound=2.5,
    conf_int_upper_bound=97.5,
    multiple_points=True,
):
    """Generic function used by `get_return_value`, `get_return_period`, and
    `get_return_prob`.

    Returns a dataset with the estimate of the requested data_variable and
    confidence intervals.

    If data_variable == "return_value", then arg_value is the return period.
    If data_variable == "return_prob", then arg_value is the threshold value.
    If data_variable == "return_period", then arg_value is the return value.

    Parameters
    ----------
    bms: xarray.DataArray
        Block maximum series, can be output from the function get_block_maxima()
    data_variable: str
    arg_value: float
    distr: str
    bootstrap_runs: int
    conf_int_lower_bound: float
    conf_int_upper_bound: float
    multiple_points: boolean

    Returns
    -------
    xarray.Dataset
    """

    data_variables = ["return_value", "return_period", "return_prob"]
    if data_variable not in data_variables:
        raise ValueError(f"Invalid `data_variable`. Must be one of: {data_variables}")

    distr_func = _get_distr_func(distr)
    bms_attributes = bms.attrs

    if multiple_points:
        bms = (
            bms.stack(allpoints=["y", "x"])
            .dropna(dim="allpoints")
            .squeeze()
            .groupby("allpoints")
        )

    def _return_variable(bms):
        try:
            parameters, fitted_distr = _get_fitted_distr(bms, distr, distr_func)
            return_variable = _calculate_return(
                fitted_distr=fitted_distr,
                data_variable=data_variable,
                arg_value=arg_value,
            )
        except (ValueError, ZeroDivisionError):
            return_variable = np.nan

        conf_int_lower_limit, conf_int_upper_limit = _conf_int(
            bms=bms,
            distr=distr,
            data_variable=data_variable,
            arg_value=arg_value,
            bootstrap_runs=bootstrap_runs,
            conf_int_lower_bound=conf_int_lower_bound,
            conf_int_upper_bound=conf_int_upper_bound,
        )

        return return_variable, conf_int_lower_limit, conf_int_upper_limit

    return_variable, conf_int_lower_limit, conf_int_upper_limit = xr.apply_ufunc(
        _return_variable,
        bms,
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

    new_ds.attrs = bms_attributes

    if data_variable == "return_value":
        new_ds["return_value"].attrs["return period"] = f"1-in-{arg_value}-year event"
    elif data_variable == "return_prob":
        threshold_unit = bms_attributes["units"]
        new_ds["return_prob"].attrs[
            "threshold"
        ] = f"exceedance of {arg_value} {threshold_unit} event"
        new_ds["return_prob"].attrs["units"] = None
    elif data_variable == "return_period":
        return_value_unit = bms_attributes["units"]
        new_ds["return_period"].attrs[
            "return value"
        ] = f"{arg_value} {return_value_unit} event"
        new_ds["return_period"].attrs["units"] = "years"

    new_ds["conf_int_lower_limit"].attrs[
        "confidence interval lower bound"
    ] = "{}th percentile".format(str(conf_int_lower_bound))
    new_ds["conf_int_upper_limit"].attrs[
        "confidence interval upper bound"
    ] = "{}th percentile".format(str(conf_int_upper_bound))

    new_ds.attrs["distribution"] = f"{distr}"
    return new_ds


def get_return_value(
    bms,
    return_period=10,
    distr="gev",
    bootstrap_runs=100,
    conf_int_lower_bound=2.5,
    conf_int_upper_bound=97.5,
    multiple_points=True,
):
    """Creates xarray Dataset with return values and confidence intervals from maximum series.

    Parameters
    ----------
    bms: xarray.DataArray
        Block maximum series, can be output from the function get_block_maxima()
    return_period: float
        The recurrence interval (in years) for which to calculate the return value
    distr: str
        The type of extreme value distribution to fit
    bootstrap_runs: int
        Number of bootstrap samples
    conf_int_lower_bound: float
        Confidence interval lower bound
    conf_int_upper_bound: float
        Confidence interval upper bound
    multiple_points: boolean
        Whether or not the data contains multiple points (has x, y dimensions)

    Returns
    -------
    xarray.Dataset
        Dataset with return values and confidence intervals
    """
    return _get_return_variable(
        bms,
        "return_value",
        return_period,
        distr,
        bootstrap_runs,
        conf_int_lower_bound,
        conf_int_upper_bound,
        multiple_points,
    )


def get_return_prob(
    bms,
    threshold,
    distr="gev",
    bootstrap_runs=100,
    conf_int_lower_bound=2.5,
    conf_int_upper_bound=97.5,
    multiple_points=True,
):
    """Creates xarray Dataset with return probabilities and confidence intervals from maximum series.

    Parameters
    ----------
    bms: xarray.DataArray
        Block maximum series, can be output from the function get_block_maxima()
    threshold: float
        The threshold value for which to calculate the probability of exceedance
    distr: str
        The type of extreme value distribution to fit
    bootstrap_runs: int
        Number of bootstrap samples
    conf_int_lower_bound: float
        Confidence interval lower bound
    conf_int_upper_bound: float
        Confidence interval upper bound
    multiple_points: boolean
        Whether or not the data contains multiple points (has x, y dimensions)

    Returns
    -------
    xarray.Dataset
        Dataset with return probabilities and confidence intervals
    """
    return _get_return_variable(
        bms,
        "return_prob",
        threshold,
        distr,
        bootstrap_runs,
        conf_int_lower_bound,
        conf_int_upper_bound,
        multiple_points,
    )


def get_return_period(
    bms,
    return_value,
    distr="gev",
    bootstrap_runs=100,
    conf_int_lower_bound=2.5,
    conf_int_upper_bound=97.5,
    multiple_points=True,
):
    """Creates xarray Dataset with return periods and confidence intervals from maximum series.

    Parameters
    ----------
    bms: xarray.DataArray
        Block maximum series, can be output from the function get_block_maxima()
    return_value: float
        The threshold value for which to calculate the return period of occurance
    distr: str
        The type of extreme value distribution to fit
    bootstrap_runs: int
        Number of bootstrap samples
    conf_int_lower_bound: float
        Confidence interval lower bound
    conf_int_upper_bound: float
        Confidence interval upper bound
    multiple_points: boolean
        Whether or not the data contains multiple points (has x, y dimensions)

    Returns
    -------
    xarray.Dataset
        Dataset with return periods and confidence intervals
    """
    return _get_return_variable(
        bms,
        "return_period",
        return_value,
        distr,
        bootstrap_runs,
        conf_int_lower_bound,
        conf_int_upper_bound,
        multiple_points,
    )


# ===================== Functions for exceedance count =========================


def _get_exceedance_count(
    da,
    threshold_value,
    duration1=None,
    period=(1, "year"),
    threshold_direction="above",
    duration2=None,
    groupby=None,
    smoothing=None,
):
    """Calculate the number of occurances of exceeding the specified threshold
    within each period.

    Returns an xarray.DataArray with the same coordinates as the input data except for
    the time dimension, which will be collapsed to one value per period (equal
    to the number of event occurances in each period).

    Parameters
    ----------
    da: xarray.DataArray
        array of some climate variable. Can have multiple
        scenarios, simulations, or x and y coordinates.
    threshold_value: float
        value against which to test exceedance
    period: int
        amount of time across which to sum the number of occurances,
        default is (1, "year"). Specified as a tuple: (x, time) where x is an
        integer, and time is one of: ["day", "month", "year"]
    threshold_direction: str
        either "above" or "below", default is above.
    duration1: tuple
        length of exceedance in order to qualify as an event (before grouping)
    groupby: tuple
        see examples for explanation. Typical grouping could be (1, "day")
    duration2: tuple
        length of exceedance in order to qualify as an event (after grouping)
    smoothing: int
        option to average the result across multiple periods with a
        rolling average; value is either None or the number of timesteps to use
        as the window size

    Returns
    -------
    xarray.DataArray
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

    events_da = _get_exceedance_events(
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
    """Function that compares period, duration.

    Helper function for comparing user specifications of period, duration.

    Parameters
    ----------
    time1: tuple
        tuple of period (int), duration (str)
    time2: tuple
        tuple of period (int), duration (str)

    Returns
    -------
    boolean

    Examples
    --------
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


def _get_exceedance_events(
    da, threshold_value, threshold_direction="above", duration1=None, groupby=None
):
    """Function for generating logical array of threshold event occurance

    Returns an xarray that specifies whether each entry of `da` is a qualifying
    threshold event. Values are 0 for False, 1 for True, or NaN for NaNs.

    Parameters
    ----------
    da: xarray.DataArray
    threshold_value: float
        value against which to test exceedance
    threshold_direction: str
        either "above" or "below", default is above.
    duration1: tuple
        length of exceedance in order to qualify as an event (before grouping)
    groupby: tuple
        see examples for explanation. Typical grouping could be (1, "day")

    Returns
    -------
    xarray.DataArray
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
    """Function to generate exceedance count name

    Helper function to build the appropriate name for the queried exceedance count.

    Parameters
    ----------
    exceedance_count: xarray.DataArray

    Returns
    -------
    string

    Examples
    --------
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


def _plot_exceedance_count(exceedance_count):
    """Create panel column object with embedded plots

    Plots each simulation as a different color line.
    Drop down option to select different scenario.
    Currently can only plot for one location, so is expecting input to already be subsetted or an area average.

    Parameters
    ----------
    exceedance_count: xarray.DataArray

    Returns
    -------
    panel.Column
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
    """Function to build title for exceedance plots

    Helper function for making the title for exceedance plots.

    Parameters
    ----------
    exceedance_count: xarray.DataArray

    Returns
    -------
    string

    Examples
    --------
        'Air Temperatue at 2m: events above 35C'
        'Preciptation (total): events below 10mm'
    """
    return f"{exceedance_count.variable_name}: events {exceedance_count.threshold_direction} {exceedance_count.threshold_value}{exceedance_count.variable_units}"


def _exceedance_plot_subtitle(exceedance_count):
    """Function of build exceedance plot subtitle

    Helper function for making the subtile for exceedance plots.

    Parameters
    ----------
    exceedance_count: xarray.DataArray

    Returns
    -------
    string

    Examples
    --------
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


##### Visualize the data
def _rename_distr_abbrev(distr):
    """Makes abbreviated distribution name human-readable"""
    distr_abbrev = ["gev", "gumbel", "weibull", "pearson3", "genpareto"]
    distr_readable = [
        "GEV",
        "Gumbel",
        "Weibull",
        "Pearson Type III",
        "Generalized Pareto",
    ]
    return distr_readable[distr_abbrev.index(distr)]


def get_geospatial_plot(
    ds,
    data_variable,
    bar_min=None,
    bar_max=None,
    border_color="black",
    line_width=0.5,
    cmap="ae_orange",
    hover_fill_color="blue",
):
    """Returns an interactive map from inputed dataset and selected data variable.

    Parameters
    ----------
    ds: xr.Dataset
        Data to plot
    data_variable: str
        Valid variable option in input dataset
        Valid options: "d_statistic","p_value","return_value","return_prob","return_period"
    bar_min: float, optional
        Colorbar minimum value
    bar_max: float, optional
        Colorbar maximum value
    border_color: str, optional
        Color for state lines and international borders
        Default to black
    cmap: matplotlib colormap name or AE colormap names, optional
        Colormap to apply to data
        Default to "ae_orange" for mapped data or color-blind friendly "categorical_cb" for timeseries data.
    hover_fill_color: str, optional
        Default to "blue"

    Returns
    -------
    holoviews.core.overlay.Overlay
        Map of input data
    """

    if cmap in [
        "categorical_cb",
        "ae_orange",
        "ae_diverging",
        "ae_blue",
        "ae_diverging_r",
    ]:
        cmap = _read_ae_colormap(cmap=cmap, cmap_hex=True)

    data_variables = [
        "d_statistic",
        "p_value",
        "return_value",
        "return_prob",
        "return_period",
    ]
    if data_variable not in data_variables:
        raise ValueError(
            "invalid data variable type. expected one of the following: %s"
            % data_variables
        )

    if data_variable == "p_value":
        variable_name = "p-value"
    else:
        variable_name = data_variable.replace("_", " ").replace("'", "")

    distr_name = _rename_distr_abbrev(ds.attrs["distribution"])

    borders = gv.Path(gv.feature.states.geoms(scale="50m", as_element=False)).opts(
        color=border_color, line_width=line_width
    ) * gv.feature.coastline.geoms(scale="50m").opts(
        color=border_color, line_width=line_width
    )

    if data_variable in ["d_statistic", "p_value"]:
        attribute_name = (
            (ds[data_variable].attrs["stat test"])
            .replace("{", "")
            .replace("}", "")
            .replace("'", "")
        )

    if data_variable in ["return_value"]:
        attribute_name = (
            (ds[data_variable].attrs["return period"])
            .replace("{", "")
            .replace("}", "")
            .replace("'", "")
        )

    if data_variable in ["return_prob"]:
        attribute_name = (
            (ds[data_variable].attrs["threshold"])
            .replace("{", "")
            .replace("}", "")
            .replace("'", "")
        )

    if data_variable in ["return_period"]:
        attribute_name = (
            (ds[data_variable].attrs["return value"])
            .replace("{", "")
            .replace("}", "")
            .replace("'", "")
        )

    cmap_label = variable_name
    variable_unit = ds[data_variable].attrs["units"]
    if variable_unit:
        cmap_label = " ".join([cmap_label, "({})".format(variable_unit)])

    geospatial_plot = (
        ds.hvplot.quadmesh(
            "lon",
            "lat",
            data_variable,
            clim=(bar_min, bar_max),
            projection=ccrs.PlateCarree(),
            ylim=(30, 50),
            xlim=(-130, -100),
            title="{} for a {}\n({} distribution)".format(
                variable_name, attribute_name, distr_name
            ),
            cmap=cmap,
            clabel=cmap_label,
            hover_fill_color=hover_fill_color,
        )
        * borders
    )
    return geospatial_plot
