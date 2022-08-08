########################################
#                                      #
# THRESHOLD TOOLS                      #
#                                      #
########################################

from multiprocessing.sharedctypes import Value
import numpy as np
import pandas as pd
import xarray as xr
from scipy import stats
import panel as pn
import param
import math

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

from .visualize import get_geospatial_plot

#####################################################################


def get_ams(da, extremes_type="max"):
    """
    Returns a data array of annual maximums.
    """

    extremes_types = ["max"]
    if extremes_type not in extremes_types:
        raise ValueError(
            "invalid extremes type. expected one of the following: %s" % extremes_types
        )

    if extremes_type == "max":
        ams = da.resample(time="A").max(keep_attrs=True)
        ams.attrs["extreme value extraction method"] = "block maxima"
        ams.attrs["extremes type"] = "maxima"
        ams.attrs["block size"] = "1 year"
        ams.attrs["timeseries type"] = "annual max series"

    # if extremes_type == 'min':
    #     ams = da.resample(time='A').min(keep_attrs=True)
    #     ams.attrs['extreme value extraction method'] = 'block maxima'
    #     ams.attrs['extremes type'] = 'minima'
    #     ams.attrs['block size'] = '1 year'
    #     ams.attrs['timeseries type'] = 'annual min series'

    return ams


#####################################################################


def get_lmom_distr(distr):
    """
    Returns corresponding l-moments distribution function from selected distribution name.
    """

    distrs = ["gev", "gumbel", "weibull", "pearson3", "genpareto"]
    if distr not in distrs:
        raise ValueError(
            "invalid distr type. expected one of the following: %s" % distrs
        )

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

    return lmom_distr


#####################################################################


def get_fitted_distr(ams, distr, lmom_distr):
    """
    Returns fitted l-moments distribution function from l-moments.
    """

    if distr == "gev":
        lmoments = lmom_distr.lmom_fit(ams)
        fitted_distr = stats.genextreme(**lmoments)

    elif distr == "gumbel":
        lmoments = lmom_distr.lmom_fit(ams)
        fitted_distr = stats.gumbel_r(**lmoments)

    elif distr == "weibull":
        lmoments = lmom_distr.lmom_fit(ams)
        fitted_distr = stats.weibull_min(**lmoments)

    elif distr == "pearson3":
        lmoments = lmom_distr.lmom_fit(ams)
        fitted_distr = stats.pearson3(**lmoments)

    elif distr == "genpareto":
        lmoments = lmom_distr.lmom_fit(ams)
        fitted_distr = stats.genpareto(**lmoments)

    return lmoments, fitted_distr


#####################################################################


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


#####################################################################


def get_ks_stat(ams, distr="gev", multiple_points=True):
    """
    Returns a dataset of ks test d-statistics and p-values from an inputed maximum series.
    """

    lmom_distr = get_lmom_distr(distr)
    ams_attributes = ams.attrs

    if multiple_points:
        ams = ams.stack(allpoints=["y", "x"]).squeeze().groupby("allpoints")

    def ks_stat(ams):

        if distr == "gev":
            try:
                lmoments, fitted_distr = get_fitted_distr(ams, distr, lmom_distr)
                ks = stats.kstest(
                    ams,
                    "genextreme",
                    args=(lmoments["c"], lmoments["loc"], lmoments["scale"]),
                )
                d_statistic = ks[0]
                p_value = ks[1]
            except (ValueError, ZeroDivisionError):
                d_statistic = np.nan
                p_value = np.nan

        elif distr == "gumbel":
            try:
                lmoments, fitted_distr = get_fitted_distr(ams, distr, lmom_distr)
                ks = stats.kstest(
                    ams, "gumbel_r", args=(lmoments["loc"], lmoments["scale"])
                )
                d_statistic = ks[0]
                p_value = ks[1]
            except (ValueError, ZeroDivisionError):
                d_statistic = np.nan
                p_value = np.nan

        elif distr == "weibull":
            try:
                lmoments, fitted_distr = get_fitted_distr(ams, distr, lmom_distr)
                ks = stats.kstest(
                    ams,
                    "weibull_min",
                    args=(lmoments["c"], lmoments["loc"], lmoments["scale"]),
                )
                d_statistic = ks[0]
                p_value = ks[1]
            except (ValueError, ZeroDivisionError):
                d_statistic = np.nan
                p_value = np.nan

        elif distr == "pearson3":
            try:
                lmoments, fitted_distr = get_fitted_distr(ams, distr, lmom_distr)
                ks = stats.kstest(
                    ams,
                    "pearson3",
                    args=(lmoments["skew"], lmoments["loc"], lmoments["scale"]),
                )
                d_statistic = ks[0]
                p_value = ks[1]
            except (ValueError, ZeroDivisionError):
                d_statistic = np.nan
                p_value = np.nan

        elif distr == "genpareto":
            try:
                lmoments, fitted_distr = get_fitted_distr(ams, distr, lmom_distr)
                ks = stats.kstest(
                    ams,
                    "genpareto",
                    args=(lmoments["c"], lmoments["loc"], lmoments["scale"]),
                )
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


#####################################################################


def calculate_return(fitted_distr, data_variable, arg_value):
    """
    Returns corresponding extreme value calculation for selected data variable.
    """

    if data_variable == "return_value":
        try:
            return_event = 1.0 - (1.0 / arg_value)
            return_value = fitted_distr.ppf(return_event)
            result = round(return_value, 5)
        except (ValueError, ZeroDivisionError, AttributeError):
            result = np.nan

    elif data_variable == "return_prob":
        try:
            result = 1 - (fitted_distr.cdf(arg_value))
        except (ValueError, ZeroDivisionError, AttributeError):
            result = np.nan

    elif data_variable == "return_period":
        try:
            return_prob = fitted_distr.cdf(arg_value)
            if return_prob == 1.0:
                result = np.nan
            else:
                return_period = -1.0 / (return_prob - 1.0)
                result = round(return_period, 3)
        except (ValueError, ZeroDivisionError, AttributeError):
            result = np.nan

    return result


#####################################################################


def bootstrap(ams, distr="gev", data_variable="return_value", arg_value=10):
    """
    Returns a bootstrap-calculated value for relevant parameters from a inputed maximum series.
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

    if distr == "gev":
        try:
            lmoments, fitted_distr = get_fitted_distr(new_ams, distr, lmom_distr)
            result = calculate_return(
                fitted_distr=fitted_distr,
                data_variable=data_variable,
                arg_value=arg_value,
            )
        except (ValueError, ZeroDivisionError):
            result = np.nan

    elif distr == "gumbel":
        try:
            lmoments, fitted_distr = get_fitted_distr(new_ams, distr, lmom_distr)
            result = calculate_return(
                fitted_distr=fitted_distr,
                data_variable=data_variable,
                arg_value=arg_value,
            )
        except (ValueError, ZeroDivisionError):
            result = np.nan

    elif distr == "weibull":
        try:
            lmoments, fitted_distr = get_fitted_distr(new_ams, distr, lmom_distr)
            result = calculate_return(
                fitted_distr=fitted_distr,
                data_variable=data_variable,
                arg_value=arg_value,
            )
        except (ValueError, ZeroDivisionError):
            result = np.nan

    elif distr == "pearson3":
        try:
            lmoments, fitted_distr = get_fitted_distr(new_ams, distr, lmom_distr)
            result = calculate_return(
                fitted_distr=fitted_distr,
                data_variable=data_variable,
                arg_value=arg_value,
            )
        except (ValueError, ZeroDivisionError):
            result = np.nan

    elif distr == "genpareto":
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


#####################################################################


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
        result = bootstrap(ams, distr, data_variable, arg_value,)
        bootstrap_values.append(result)

    conf_int_array = np.percentile(
        bootstrap_values, [conf_int_lower_bound, conf_int_upper_bound]
    )

    conf_int_lower_limit = conf_int_array[0]
    conf_int_upper_limit = conf_int_array[1]

    return conf_int_lower_limit, conf_int_upper_limit


#####################################################################


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

    data_variable = "return_value"
    lmom_distr = get_lmom_distr(distr)
    ams_attributes = ams.attrs

    if multiple_points:
        ams = ams.stack(allpoints=["y", "x"]).squeeze().groupby("allpoints")

    def return_value(ams):

        if distr == "gev":
            try:
                lmoments, fitted_distr = get_fitted_distr(ams, distr, lmom_distr)
                return_value = calculate_return(
                    fitted_distr=fitted_distr,
                    data_variable=data_variable,
                    arg_value=return_period,
                )
            except (ValueError, ZeroDivisionError):
                return_value = np.nan
            conf_int_lower_limit, conf_int_upper_limit = conf_int(
                ams=ams,
                distr=distr,
                data_variable=data_variable,
                arg_value=return_period,
                bootstrap_runs=bootstrap_runs,
                conf_int_lower_bound=conf_int_lower_bound,
                conf_int_upper_bound=conf_int_upper_bound,
            )

        elif distr == "gumbel":
            try:
                lmoments, fitted_distr = get_fitted_distr(ams, distr, lmom_distr)
                return_value = calculate_return(
                    fitted_distr=fitted_distr,
                    data_variable=data_variable,
                    arg_value=return_period,
                )
            except (ValueError, ZeroDivisionError):
                return_value = np.nan
            conf_int_lower_limit, conf_int_upper_limit = conf_int(
                ams=ams,
                distr=distr,
                data_variable=data_variable,
                arg_value=return_period,
                bootstrap_runs=bootstrap_runs,
                conf_int_lower_bound=conf_int_lower_bound,
                conf_int_upper_bound=conf_int_upper_bound,
            )

        elif distr == "weibull":
            try:
                lmoments, fitted_distr = get_fitted_distr(ams, distr, lmom_distr)
                return_value = calculate_return(
                    fitted_distr=fitted_distr,
                    data_variable=data_variable,
                    arg_value=return_period,
                )
            except (ValueError, ZeroDivisionError):
                return_value = np.nan
            conf_int_lower_limit, conf_int_upper_limit = conf_int(
                ams=ams,
                distr=distr,
                data_variable=data_variable,
                arg_value=return_period,
                bootstrap_runs=bootstrap_runs,
                conf_int_lower_bound=conf_int_lower_bound,
                conf_int_upper_bound=conf_int_upper_bound,
            )

        elif distr == "pearson3":
            try:
                lmoments, fitted_distr = get_fitted_distr(ams, distr, lmom_distr)
                return_value = calculate_return(
                    fitted_distr=fitted_distr,
                    data_variable=data_variable,
                    arg_value=return_period,
                )
            except (ValueError, ZeroDivisionError):
                return_value = np.nan
            conf_int_lower_limit, conf_int_upper_limit = conf_int(
                ams=ams,
                distr=distr,
                data_variable=data_variable,
                arg_value=return_period,
                bootstrap_runs=bootstrap_runs,
                conf_int_lower_bound=conf_int_lower_bound,
                conf_int_upper_bound=conf_int_upper_bound,
            )

        elif distr == "genpareto":
            try:
                lmoments, fitted_distr = get_fitted_distr(ams, distr, lmom_distr)
                return_value = calculate_return(
                    fitted_distr=fitted_distr,
                    data_variable=data_variable,
                    arg_value=return_period,
                )
            except (ValueError, ZeroDivisionError):
                return_value = np.nan
            conf_int_lower_limit, conf_int_upper_limit = conf_int(
                ams=ams,
                distr=distr,
                data_variable=data_variable,
                arg_value=return_period,
                bootstrap_runs=bootstrap_runs,
                conf_int_lower_bound=conf_int_lower_bound,
                conf_int_upper_bound=conf_int_upper_bound,
            )

        return return_value, conf_int_lower_limit, conf_int_upper_limit

    return_value, conf_int_lower_limit, conf_int_upper_limit = xr.apply_ufunc(
        return_value,
        ams,
        input_core_dims=[["time"]],
        exclude_dims=set(("time",)),
        output_core_dims=[[], [], []],
    )

    return_value = return_value.rename("return_value")
    new_ds = return_value.to_dataset()
    new_ds["conf_int_lower_limit"] = conf_int_lower_limit
    new_ds["conf_int_upper_limit"] = conf_int_upper_limit

    if multiple_points:
        new_ds = new_ds.unstack("allpoints")

    new_ds["return_value"].attrs["return period"] = "1 in {} year event".format(
        str(return_period)
    )
    new_ds["conf_int_lower_limit"].attrs[
        "confidence interval lower bound"
    ] = "{}th percentile".format(str(conf_int_lower_bound))
    new_ds["conf_int_upper_limit"].attrs[
        "confidence interval upper bound"
    ] = "{}th percentile".format(str(conf_int_upper_bound))

    new_ds.attrs = ams_attributes
    new_ds.attrs["distribution"] = "{}".format(str(distr))

    return new_ds


#####################################################################


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

    data_variable = "return_prob"
    lmom_distr = get_lmom_distr(distr)
    ams_attributes = ams.attrs

    if multiple_points:
        ams = ams.stack(allpoints=["y", "x"]).squeeze().groupby("allpoints")

    def return_prob(ams):

        if distr == "gev":
            try:
                lmoments, fitted_distr = get_fitted_distr(ams, distr, lmom_distr)
                return_prob = calculate_return(
                    fitted_distr=fitted_distr,
                    data_variable=data_variable,
                    arg_value=threshold,
                )
            except (ValueError, ZeroDivisionError):
                return_prob = np.nan
            conf_int_lower_limit, conf_int_upper_limit = conf_int(
                ams=ams,
                distr=distr,
                data_variable=data_variable,
                arg_value=threshold,
                bootstrap_runs=bootstrap_runs,
                conf_int_lower_bound=conf_int_lower_bound,
                conf_int_upper_bound=conf_int_upper_bound,
            )

        elif distr == "gumbel":
            try:
                lmoments, fitted_distr = get_fitted_distr(ams, distr, lmom_distr)
                return_prob = calculate_return(
                    fitted_distr=fitted_distr,
                    data_variable=data_variable,
                    arg_value=threshold,
                )
            except (ValueError, ZeroDivisionError):
                return_prob = np.nan
            conf_int_lower_limit, conf_int_upper_limit = conf_int(
                ams=ams,
                distr=distr,
                data_variable=data_variable,
                arg_value=threshold,
                bootstrap_runs=bootstrap_runs,
                conf_int_lower_bound=conf_int_lower_bound,
                conf_int_upper_bound=conf_int_upper_bound,
            )

        elif distr == "weibull":
            try:
                lmoments, fitted_distr = get_fitted_distr(ams, distr, lmom_distr)
                return_prob = calculate_return(
                    fitted_distr=fitted_distr,
                    data_variable=data_variable,
                    arg_value=threshold,
                )
            except (ValueError, ZeroDivisionError):
                return_prob = np.nan
            conf_int_lower_limit, conf_int_upper_limit = conf_int(
                ams=ams,
                distr=distr,
                data_variable=data_variable,
                arg_value=threshold,
                bootstrap_runs=bootstrap_runs,
                conf_int_lower_bound=conf_int_lower_bound,
                conf_int_upper_bound=conf_int_upper_bound,
            )

        elif distr == "pearson3":
            try:
                lmoments, fitted_distr = get_fitted_distr(ams, distr, lmom_distr)
                return_prob = calculate_return(
                    fitted_distr=fitted_distr,
                    data_variable=data_variable,
                    arg_value=threshold,
                )
            except (ValueError, ZeroDivisionError):
                return_prob = np.nan
            conf_int_lower_limit, conf_int_upper_limit = conf_int(
                ams=ams,
                distr=distr,
                data_variable=data_variable,
                arg_value=threshold,
                bootstrap_runs=bootstrap_runs,
                conf_int_lower_bound=conf_int_lower_bound,
                conf_int_upper_bound=conf_int_upper_bound,
            )

        elif distr == "genpareto":
            try:
                lmoments, fitted_distr = get_fitted_distr(ams, distr, lmom_distr)
                return_prob = calculate_return(
                    fitted_distr=fitted_distr,
                    data_variable=data_variable,
                    arg_value=threshold,
                )
            except (ValueError, ZeroDivisionError):
                return_prob = np.nan
            conf_int_lower_limit, conf_int_upper_limit = conf_int(
                ams=ams,
                distr=distr,
                data_variable=data_variable,
                arg_value=threshold,
                bootstrap_runs=bootstrap_runs,
                conf_int_lower_bound=conf_int_lower_bound,
                conf_int_upper_bound=conf_int_upper_bound,
            )

        return return_prob, conf_int_lower_limit, conf_int_upper_limit

    return_prob, conf_int_lower_limit, conf_int_upper_limit = xr.apply_ufunc(
        return_prob,
        ams,
        input_core_dims=[["time"]],
        exclude_dims=set(("time",)),
        output_core_dims=[[], [], []],
    )

    return_prob = return_prob.rename("return_prob")
    new_ds = return_prob.to_dataset()
    new_ds["conf_int_lower_limit"] = conf_int_lower_limit
    new_ds["conf_int_upper_limit"] = conf_int_upper_limit

    if multiple_points:
        new_ds = new_ds.unstack("allpoints")

    new_ds["return_prob"].attrs["threshold"] = "exceedance of {} value event".format(
        str(threshold)
    )
    new_ds["conf_int_lower_limit"].attrs[
        "confidence interval lower bound"
    ] = "{}th percentile".format(str(conf_int_lower_bound))
    new_ds["conf_int_upper_limit"].attrs[
        "confidence interval upper bound"
    ] = "{}th percentile".format(str(conf_int_upper_bound))

    new_ds.attrs = ams_attributes
    new_ds.attrs["distribution"] = "{}".format(str(distr))

    return new_ds


#####################################################################


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

    data_variable = "return_period"
    lmom_distr = get_lmom_distr(distr)
    ams_attributes = ams.attrs

    if multiple_points:
        ams = ams.stack(allpoints=["y", "x"]).squeeze().groupby("allpoints")

    def return_period(ams):

        if distr == "gev":
            try:
                lmoments, fitted_distr = get_fitted_distr(ams, distr, lmom_distr)
                return_period = calculate_return(
                    fitted_distr=fitted_distr,
                    data_variable=data_variable,
                    arg_value=return_value,
                )
            except (ValueError, ZeroDivisionError):
                return_period = np.nan
            conf_int_lower_limit, conf_int_upper_limit = conf_int(
                ams=ams,
                distr=distr,
                data_variable=data_variable,
                arg_value=return_value,
                bootstrap_runs=bootstrap_runs,
                conf_int_lower_bound=conf_int_lower_bound,
                conf_int_upper_bound=conf_int_upper_bound,
            )

        elif distr == "gumbel":
            try:
                lmoments, fitted_distr = get_fitted_distr(ams, distr, lmom_distr)
                return_period = calculate_return(
                    fitted_distr=fitted_distr,
                    data_variable=data_variable,
                    arg_value=return_value,
                )
            except (ValueError, ZeroDivisionError):
                return_period = np.nan
            conf_int_lower_limit, conf_int_upper_limit = conf_int(
                ams=ams,
                distr=distr,
                data_variable=data_variable,
                arg_value=return_value,
                bootstrap_runs=bootstrap_runs,
                conf_int_lower_bound=conf_int_lower_bound,
                conf_int_upper_bound=conf_int_upper_bound,
            )

        elif distr == "weibull":
            try:
                lmoments, fitted_distr = get_fitted_distr(ams, distr, lmom_distr)
                return_period = calculate_return(
                    fitted_distr=fitted_distr,
                    data_variable=data_variable,
                    arg_value=return_value,
                )
            except (ValueError, ZeroDivisionError):
                return_period = np.nan
            conf_int_lower_limit, conf_int_upper_limit = conf_int(
                ams=ams,
                distr=distr,
                data_variable=data_variable,
                arg_value=return_value,
                bootstrap_runs=bootstrap_runs,
                conf_int_lower_bound=conf_int_lower_bound,
                conf_int_upper_bound=conf_int_upper_bound,
            )

        elif distr == "pearson3":
            try:
                lmoments, fitted_distr = get_fitted_distr(ams, distr, lmom_distr)
                return_period = calculate_return(
                    fitted_distr=fitted_distr,
                    data_variable=data_variable,
                    arg_value=return_value,
                )
            except (ValueError, ZeroDivisionError):
                return_period = np.nan
            conf_int_lower_limit, conf_int_upper_limit = conf_int(
                ams=ams,
                distr=distr,
                data_variable=data_variable,
                arg_value=return_value,
                bootstrap_runs=bootstrap_runs,
                conf_int_lower_bound=conf_int_lower_bound,
                conf_int_upper_bound=conf_int_upper_bound,
            )

        elif distr == "genpareto":
            try:
                lmoments, fitted_distr = get_fitted_distr(ams, distr, lmom_distr)
                return_period = calculate_return(
                    fitted_distr=fitted_distr,
                    data_variable=data_variable,
                    arg_value=return_value,
                )
            except (ValueError, ZeroDivisionError):
                return_period = np.nan
            conf_int_lower_limit, conf_int_upper_limit = conf_int(
                ams=ams,
                distr=distr,
                data_variable=data_variable,
                arg_value=return_value,
                bootstrap_runs=bootstrap_runs,
                conf_int_lower_bound=conf_int_lower_bound,
                conf_int_upper_bound=conf_int_upper_bound,
            )

        return return_period, conf_int_lower_limit, conf_int_upper_limit

    return_period, conf_int_lower_limit, conf_int_upper_limit = xr.apply_ufunc(
        return_period,
        ams,
        input_core_dims=[["time"]],
        exclude_dims=set(("time",)),
        output_core_dims=[[], [], []],
    )

    return_period = return_period.rename("return_period")
    new_ds = return_period.to_dataset()
    new_ds["conf_int_lower_limit"] = conf_int_lower_limit
    new_ds["conf_int_upper_limit"] = conf_int_upper_limit

    if multiple_points:
        new_ds = new_ds.unstack("allpoints")

    new_ds["return_period"].attrs[
        "return value"
    ] = "occurrence of a {} value event".format(str(return_value))
    new_ds["conf_int_lower_limit"].attrs[
        "confidence interval lower bound"
    ] = "{}th percentile".format(str(conf_int_lower_bound))
    new_ds["conf_int_upper_limit"].attrs[
        "confidence interval upper bound"
    ] = "{}th percentile".format(str(conf_int_upper_bound))

    new_ds.attrs = ams_attributes
    new_ds.attrs["distribution"] = "{}".format(str(distr))

    return new_ds

#------------- Functions for exceedance count ---------------------------------

def get_exceedance_events(
    da,
    threshold_value,
    threshold_direction = "above", 
    groupby = None
):
    """
    Returns an xarray that specifies whether each entry of da is a qualifying event. 
    0 for False, 1 for True, and NaN for NaN values
    """

    # Count occurances (and preserve NaNs)
    if threshold_direction == "above":
        events_da = (da > threshold_value).where(da.isnull()==False)
    elif threshold_direction == "below":
        events_da = (da < threshold_value).where(da.isnull()==False)
    else:
        raise ValueError(f"Unknown value for `threshold_direction` parameter: {threshold_direction}. Available options are 'above' or 'below'.")

    # Groupby 
    if groupby is not None:
        group_totals = _group_and_sum(events_da, groupby)
        events_da = (group_totals > 0).where(group_totals.isnull()==False)

    return events_da

def _group_and_sum(da, group_spec):
    """
    Helper function to sum data across time periods (i.e. days, months years).
    The `group_spec` argument is a tuple of the form (3, 'day') or (1, 'year').

    Return value is an xarray DataArray, with all the same dimensions except 
    for `time`, which will be collapsed in length along the groups, and the 
    coordinate values will be the date of the start of the group, rather than a 
    datetime. (TBD if this is the desired behavior for the time dimension.)

    This function is implemented by manually creating group numbers (i.e. 
    [0,0,...1,1,...]), assigning them as a dimension, then summing across 
    those groups. 
    """
    group_len, group_type = group_spec
    if (group_spec == (1, "hour") and da.frequency == "1hr") \
        or (group_spec == (1, "day") and da.frequency == "1day") \
        or (group_spec == (1, "month") and da.frequency == "1month"):
        # group_spec same as data frequency, do nothing
        pass
    elif group_spec == (1, "day"):
        # special case where it is simpler to sum by day using "time.date"
        day_totals = events_da.groupby("time.date").sum()
        events_da["date"] = pd.to_datetime(events_da.date)
        events_da = events_da.rename({"date":"time"})
    elif group_type == "day":
        # general case for grouping by some number of days
        dates = events_da.time.dt.date.values # get the date for each value in the time dimension
        days = [(d - dates[0]).days for d in dates] # calculate the day number for each value (i.e. [0,0,...1,1,...])
        groups = [math.floor(d / group_len) for d in days] # group the day numbers based on user-specified group length
        date_ids = dates[[groups.index(i) for i in list(set(groups))]] # save one date value for each group to use as the time index after summing
        events_da["time"] = groups # set the group numbers as the time dimension for summing
        group_totals = events_da.groupby("time").sum() # sum across the groups
        events_da["time"] = pd.to_datetime(date_ids) # reset the time dimension to the saved date values
    else:
        raise ValueError("Groupby options other than 'day' not yet implmented.")

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

def get_exceedance_count(
    da,
    threshold_value,
    period = (1, "year"),
    threshold_direction = "above", 
    duration = None,
    groupby = None,
    smoothing = None
):
    """
    Calculate the number of occurances of exceeding the specified threshold 
    within each period length.

    Returns an xarray with the same coordinates as the input data except for 
    the time dimension, which will be collapsed to one value per period (equal
    to the number of event occurances in each period).

    Arguments:
    da -- an xarray.DataArray of some climate variable. Can have mutliple scenarios, simulations,
        or x and y coordinates. 
    threshold_value -- value against which to test exceedance
    
    Optional Keyword Arguments:
    period -- amount of time across which to sum the number of occurances, default is (1, "year"). 
        Specified as a tuple: (x, time) where x is an integer, and time is one of: ["day", "month", "year"]
    threshold_direction -- string either "above" or "below", default is above.
    duration -- length of exceedance in order to qualify as an event
    groupby -- see examples for explanation. Typical grouping could be (1, "day")
    smoothing -- option to average the result across multiple periods with a rolling average (not yet implemented)
    """

    #--------- Type check arguments -------------------------------------------

    if duration is not None:
        if duration == (1, "hour"):
            duration = None
        else:
            raise ValueError("Other duration options not yet implemented. Please use (1, 'hour').")
    if smoothing is not None: raise ValueError("Smoothing option not yet implemented.")

    # Check compatibility of periods, durations, and groupbys
    if _is_greater(groupby, duration): raise ValueError("Incompatible `group` and `duration` specification. Duration must be longer than group.")
    if _is_greater(groupby, period): raise ValueError("Incompatible `group` and `period` specification. Group must be longer than period.")
    if _is_greater(duration, period): raise ValueError("Incompatible `duration` and `period` specification. Period must be longer than duration.")
    freq = (1, "hour") if da.frequency == "1hr" else ((1, "day") if da.frequency == "1day" else (1, "month"))
    if _is_greater(freq, groupby): raise ValueError("Incompatible `groupby` specification: cannot be less than data frequency.")
    if _is_greater(freq, duration): raise ValueError("Incompatible `duration` specification: cannot be less than data frequency.")
    if _is_greater(freq, period): raise ValueError("Incompatible `period` specification: cannot be less than data frequency.")

    #--------- Calculate occurances -------------------------------------------

    events_da = get_exceedance_events(da, threshold_value, threshold_direction, groupby)

    #--------- Group by time period and count ---------------------------------
    
    if period == (1, "year"):
        exceedance_count = events_da.groupby("time.year").sum()
    else:
        raise ValueError("Other period options not yet implemented. Please use (1, 'year').")
        # eventually, run:
        # exceedance_count = _group_and_sum(events_da, period)

    #--------- Set attributes for the counts DataArray ------------------------
    exceedance_count.attrs["variable_name"] = da.name
    exceedance_count.attrs["variable_units"] = exceedance_count.units
    exceedance_count.attrs["period"] = period
    exceedance_count.attrs["group"] = groupby
    exceedance_count.attrs["duration"] = duration
    exceedance_count.attrs["threshold_value"] = threshold_value
    exceedance_count.attrs["threshold_direction"] = threshold_direction
    exceedance_count.attrs["units"] = _exceedance_count_name(exceedance_count)
    exceedance_count.attrs["time"] = period[1] # for plotting: x-axis

    # Set name (for plotting, this will be the y-axis label)
    exceedance_count.name =  "Count"

    return exceedance_count

def _exceedance_count_name(exceedance_count):
    """
    Helper function to build the appropriate name for the queried exceedance count.
    Examples:
        'Number of hours per 1 year'
        'Number of days per 1 year'
        'Number of 3-day events per 1 year'
    """
    # If duration is used, this determines the event name
    dur = exceedance_count.duration
    if dur is not None:
        d_num, d_type = dur 
        if d_num != 1:
            event = f"{d_num}-{d_type} events"
        else:
            event = f"{d_type}s" # ex: day --> days
    else:
        # otherwise use "groupby" 
        freq = exceedance_count.group
        if freq is not None:
            f_num, f_type = freq
            if f_num != 1:
                event = f"{f_num}-{f_type} events"
            else:
                event = f"{f_type}s" # ex: day --> days
        else:
            # otherwise use data frequency info
            if exceedance_count.frequency == "1hr":
                event = "hours"
            elif exceedance_count.frequency == "1day":
                event = "days"
            elif exceedance_count.frequency == "1month":
                event = "months"

    return f"Number of {event} per " + " ".join(map(str, exceedance_count.period))

def _exceedance_plot_title(exceedance_count):
    """
    Helper function for making the title for exceedance plots.
    Examples:
        'Air Temperatue at 2m: events above 35C'
        'Preciptation (total): events below 10cm'
    """
    return f"{exceedance_count.variable_name}: events {exceedance_count.threshold_direction} {exceedance_count.threshold_value}{exceedance_count.variable_units}"

def plot_exceedance_count(exceedance_count):
    """
    Plots each simulation as a different color line.
    Drop down option to select different scenario.
    Currently can only plot for one location, so is expecting input to already be subsetted or an area average.
    """
    plot_obj = exceedance_count.hvplot.line(
        x=exceedance_count.attrs["time"], 
        widget_location="bottom", 
        by="simulation", 
        groupby=["scenario"],
        title=_exceedance_plot_title(exceedance_count),
        fontsize = {'title': '14pt', 'ylabel': '10pt'},
        legend = 'right',
        xlabel = str.capitalize(exceedance_count.attrs["time"])
    )
    return pn.Column(plot_obj)


#------------- Class and methods for the explore_exceedance GUI ---------------

class ExceedanceParams(param.Parameterized):
    """
    An object to hold exceedance count parameters, which depends only on the 'param' library.
    """
    # Define the params
    threshold_direction = param.ObjectSelector(default = "above", objects = ["above", "below"], label = "")
    threshold_value = param.Number(default = 0, label = "")
    period_length = param.Number(default = 1, bounds = (0, None), label = "")
    period_type = param.ObjectSelector(default = "year", objects = ["year", "month", "day", "hour"], label = "")
    group_length = param.Number(default = 1, bounds = (0, None), label = "")
    group_type = param.ObjectSelector(default = "hour", objects = ["year", "month", "day", "hour"], label = "")
    duration_length = param.Number(default = 1, bounds = (0, None), label = "")
    duration_type = param.ObjectSelector(default = "hour", objects = ["year", "month", "day", "hour"], label = "")

    def __init__(self, dataarray, **params):
        super().__init__(**params)
        self.data = dataarray
        self.threshold_value = round(dataarray.mean().values.item()) # Have the starting display value be the average of the data

    def transform_data(self):
        return get_exceedance_count(self.data, 
            threshold_value = self.threshold_value, 
            threshold_direction = self.threshold_direction,
            period = (self.period_length, self.period_type),
            groupby = (self.group_length, self.group_type),
            duration = (self.duration_length, self.duration_type))

    @param.depends("threshold_value", "threshold_direction", "period_length", "period_type", 
        "group_length", "group_type", "duration_length", "duration_type", watch=False)
    def view(self):
        try:
            to_plot = self.transform_data()
            obj = plot_exceedance_count(to_plot)
            return obj
        except ValueError as ve:
            return ve

def _exceedance_visualize(choices, option=1):
    """
    Uses holoviz 'panel' library to display the parameters and view defined for exploring exceedance.
    """
    _left_column_width = 375
    exceedance_count_panel = pn.Column(pn.Spacer(width=15), pn.Row(
        pn.Column(
            pn.Card(
                "Specify the event threshold of interest.",
                pn.Row(
                    choices.param.threshold_direction,
                    choices.param.threshold_value,
                    width = _left_column_width
                ),
                title = "Threshold",
            ),
            pn.Card(
                "Amount of time across which to sum the occurances.",
                pn.Row(
                    choices.param.period_length,
                    choices.param.period_type,
                    width = _left_column_width
                ),
                title = "Period",
            ),
            pn.Card(
                "Group occurances into single events.",
                pn.Row(
                    choices.param.group_length,
                    choices.param.group_type,
                    width = _left_column_width
                ),
                title = "Group",
            ),
            pn.Card(
                "Amount of time threshold is exceeded to qualify as an event.",
                pn.Row(
                    choices.param.duration_length,
                    choices.param.duration_type,
                    width = _left_column_width
                ),
                title = "Duration",
            ),
            width = _left_column_width
        ),
        pn.Spacer(width=15),
        choices.view
    ))
    
    if option==1:
        return exceedance_count_panel
    elif option==2:
        return pn.Tabs(
            ("Event counts", exceedance_count_panel), 
            ("Return values", pn.Row())
        ) 
    else:
        raise ValueError("Unknown option")
    
    


def explore_exceedance(da, option=1):
    exc_choices = ExceedanceParams(da)
    return _exceedance_visualize(exc_choices, option)