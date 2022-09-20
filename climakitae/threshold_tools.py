########################################
#                                      #
# THRESHOLD TOOLS                      #
#                                      #
########################################

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

from .data_loaders import _read_from_catalog
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

def get_exceedance_count(
    da,
    threshold_value,
    duration1 = None,
    period = (1, "year"),
    threshold_direction = "above", 
    duration2 = None,
    groupby = None,
    smoothing = None
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

    #--------- Type check arguments -------------------------------------------

    # Check compatibility of periods, durations, and groupbys
    if _is_greater(duration1, groupby): raise ValueError("Incompatible `group` and `duration1` specification. Duration1 must be shorter than group.")
    if _is_greater(groupby, duration2): raise ValueError("Incompatible `group` and `duration2` specification. Duration2 must be longer than group.")
    if _is_greater(groupby, period): raise ValueError("Incompatible `group` and `period` specification. Group must be longer than period.")
    if _is_greater(duration2, period): raise ValueError("Incompatible `duration` and `period` specification. Period must be longer than duration.")
    
    # Check compatibility of specifications with the data frequency (hourly, daily, or monthly)
    freq = (1, "hour") if da.frequency == "1hr" else ((1, "day") if da.frequency == "1day" else (1, "month"))
    if _is_greater(freq, groupby): raise ValueError("Incompatible `group` specification: cannot be less than data frequency.")
    if _is_greater(freq, duration2): raise ValueError("Incompatible `duration` specification: cannot be less than data frequency.")
    if _is_greater(freq, period): raise ValueError("Incompatible `period` specification: cannot be less than data frequency.")

    #--------- Calculate occurances -------------------------------------------

    events_da = get_exceedance_events(da, threshold_value, threshold_direction, duration1, groupby)

    #--------- Apply specified duration requirement ---------------------------

    if duration2 is not None:
        dur_len, dur_type = duration2

        if (groupby is not None and groupby[1] == dur_type) \
            or (groupby is None and freq[1] == dur_type):
            window_size = dur_len 
        else:
            raise ValueError("Duration options for time types (i.e. hour, day) that are different than group or frequency not yet implemented")

        # The "min" operation will return 0 if any time in the window is not an
        # event, which is the behavior we want. It will only return 1 for True 
        # if all values in the duration window are 1.
        events_da = events_da.rolling(time = window_size, center=False).min("time")

    #--------- Sum occurances across each period ------------------------------
    
    period_len, period_type = period
    period_indexer = str.capitalize(period_type[0]) # capitalize first letter to use as indexer in resample
    exceedance_count = events_da.resample(time = f"{period_len}{period_indexer}", label="left").sum()

    # Optional smoothing
    if smoothing is not None:
        exceedance_count = exceedance_count.rolling(time=smoothing, center=True).mean("time")

    #--------- Set new attributes for the counts DataArray --------------------
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
    exceedance_count.name =  "Count"

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
    da,
    threshold_value,
    threshold_direction = "above", 
    duration1 = None,
    groupby = None
):
    """
    Returns an xarray that specifies whether each entry of `da` is a qualifying 
    threshold event. Values are 0 for False, 1 for True, or NaN for NaNs.
    """

    # Identify occurances (and preserve NaNs)
    if threshold_direction == "above":
        events_da = (da > threshold_value).where(da.isnull()==False)
    elif threshold_direction == "below":
        events_da = (da < threshold_value).where(da.isnull()==False)
    else:
        raise ValueError(f"Unknown value for `threshold_direction` parameter: {threshold_direction}. Available options are 'above' or 'below'.")

    if duration1 is not None:
        dur_len, dur_type = duration1
        if dur_type != "hour" or da.frequency != "1hr":
            raise ValueError("Current specifications not yet implemented.")
        window_size = dur_len 

        # The "min" operation will return 0 if any time in the window is not an
        # event, which is the behavior we want. It will only return 1 for True 
        # if all values in the duration window are 1.
        events_da = events_da.rolling(time = window_size, center=False).min("time")

    # Groupby 
    if groupby is not None:
        if (groupby == (1, "hour") and da.frequency == "1hr") \
            or (groupby == (1, "day") and da.frequency == "1day") \
            or (groupby == (1, "month") and da.frequency == "1month") \
            or groupby == duration1:
            # groupby specification is the same as data frequency, do nothing
            pass
        else:
            group_len, group_type = groupby
            indexer_type = str.capitalize(group_type[0]) # capitalize the first letter to use as the indexer (i.e. H, D, M, or Y)
            group_totals = events_da.resample(time=f"{group_len}{indexer_type}", label="left").sum() # sum occurences within each group
            events_da = (group_totals > 0).where(group_totals.isnull()==False) # turn back into a boolean with preserved NaNs (0 or 1 for whether there is any occurance in the group)

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
            event = f"{d_type}s" # ex: day --> days
    else:
        # otherwise use "groupby" if not None
        grp = exceedance_count.group
        if grp is not None:
            g_num, g_type = grp
            if g_num != 1:
                event = f"{g_num}-{g_type} events"
            else:
                event = f"{g_type}s" # ex: day --> days
        else:
            # otherwise use data frequency info as the default event type
            if exceedance_count.frequency == "1hr":
                event = "hours"
            elif exceedance_count.frequency == "1day":
                event = "days"
            elif exceedance_count.frequency == "1month":
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
        title = "",
        fontsize = {'ylabel': '10pt'},
        legend = 'right',
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

    _subtitle = _exceedance_count_name(exceedance_count)+ period_str + dur_str + grp_str
    
    return _subtitle

#------------- Class and methods for the explore_exceedance GUI ---------------

class ThresholdDataParams(param.Parameterized):
    """
    An object that holds the "Data Options" parameters for the 
    explore.thresholds panel. 
    """
    def __init__(self, *args, **params):
        super().__init__(*args, **params)
        
        # Selectors defaults
        self.selections.append_historical = False
        self.selections.area_average = True
        self.selections.resolution = "45 km"
        self.selections.scenario = ["SSP 3-7.0 -- Business as Usual"]
        # self.selections.time_slice = (1980,2100)
        self.selections.time_slice = (2050, 2051)
        self.selections.timescale = "hourly"
        self.selections.variable = "Air Temperature at 2m"

        # Location defaults
        self.location.area_subset = 'CA counties'
        self.location.cached_area = 'Santa Clara County'

        # Get the underlying dataarray
        self.da = _read_from_catalog(selections = self.selections, location = self.location, cat = self._cat).compute()

    variable2 = param.ObjectSelector(default="Air Temperature at 2m",
        objects=["Air Temperature at 2m"]
    )

    cached_area2 = param.ObjectSelector(default="CA",
        objects=["CA"]
    )

    area_subset2 = param.ObjectSelector(
        default="states",
        objects=["states", "CA counties"],
    )

    # For reloading data
    reload_data = param.Action(lambda x: x.param.trigger('reload_data'), label='Reload Data')
    changed_loc_and_var = param.Boolean(default=True)

    @param.depends("area_subset2","cached_area2","variable2", watch=True)
    def _updated_bool_loc_and_var(self):
        """Update boolean if any changes were made to the location or variable"""
        self.changed_loc_and_var = True

    @param.depends("reload_data", watch=True)
    def _update_data(self):
        """If the button was clicked and the location or variable was changed,
        reload the postage stamp data from AWS"""
        if self.changed_loc_and_var == True:
            self.da = _read_from_catalog(selections = self.selections, location = self.location, cat = self._cat).compute()
            self.changed_loc_and_var = False

    @param.depends("variable2", watch=True)
    def _update_variable(self):
        """Update variable in selections object to reflect variable chosen in panel"""
        self.selections.variable = self.variable2

    @param.depends("area_subset2", watch=True)
    def _update_cached_area(self):
        """
        Makes the dropdown options for 'cached area' reflect the type of area subsetting
        selected in 'area_subset' (currently state, county, or watershed boundaries).
        """
        if self.area_subset2 == "CA counties":
            # setting this to the dict works for initializing, but not updating an objects list:
            self.param["cached_area2"].objects = ["Santa Clara County", "Los Angeles County"]
            self.cached_area2 = "Santa Clara County"
        elif self.area_subset2 == "states":
            self.param["cached_area2"].objects = ["CA"]
            self.cached_area2 = "CA"

    @param.depends("area_subset2","cached_area2",watch=True)
    def _updated_location(self):
        """Update locations object to reflect location chosen in panel"""
        self.location.area_subset = self.area_subset2
        self.location.cached_area = self.cached_area2

class ExceedanceParams(param.Parameterized):
    """
    An object to hold exceedance count parameters, which depends on the 'param' library.
    """
    # Define the params (before __init__ so that we can access them during __init__)
    threshold_direction = param.ObjectSelector(default = "above", objects = ["above", "below"], label = "Direction")
    threshold_value = param.Number(default = 0, label = "")
    duration1_length = param.Integer(default = 1, bounds = (0, None), label = "")
    duration1_type = param.ObjectSelector(default = "hour", objects = ["year", "month", "day", "hour"], label = "")
    period_length = param.Integer(default = 1, bounds = (0, None), label = "")
    period_type = param.ObjectSelector(default = "year", objects = ["year", "month", "day"], label = "")
    group_length = param.Integer(default = 1, bounds = (0, None), label = "")
    group_type = param.ObjectSelector(default = "hour", objects = ["year", "month", "day", "hour"], label = "")
    duration2_length = param.Integer(default = 1, bounds = (0, None), label = "")
    duration2_type = param.ObjectSelector(default = "hour", objects = ["year", "month", "day", "hour"], label = "")
    smoothing = param.ObjectSelector(default="None", objects=["None", "Running mean"], label = "Smoothing")
    num_timesteps = param.Integer(default=10, bounds=(0, None), label = "Number of timesteps")

    def __init__(self, dataarray, **params):
        super().__init__(**params)
        self.data = dataarray
        # Set the starting display value to be the average of the data 
        #   (TBD: do we want "rounding" to be different number of sig figs 
        #   depending on variable type?)
        self.threshold_value = round(dataarray.mean().values.item())
        self.param.threshold_value.label = f"Value (units: {dataarray.units})"

    def transform_data(self):
        return get_exceedance_count(self.data, 
            threshold_value = self.threshold_value, 
            threshold_direction = self.threshold_direction,
            duration1 = (self.duration1_length, self.duration1_type),
            period = (self.period_length, self.period_type),
            groupby = (self.group_length, self.group_type),
            duration2 = (self.duration2_length, self.duration2_type),
            smoothing = self.num_timesteps if self.smoothing == "Running mean" else None)

    @param.depends("threshold_value", "threshold_direction", 
        "duration1_length", "duration1_type", "period_length", 
        "period_type", "group_length", "group_type", "duration2_length", 
        "duration2_type", "smoothing", "num_timesteps", watch=False)
    def view(self):
        try:
            to_plot = self.transform_data()
            obj = plot_exceedance_count(to_plot)
        except Exception as e:
            # Display any raised Errors (instead of plotting) if any of the 
            # user specifications are incompatible or not yet implemented.
            return e
        return pn.Column(
            _exceedance_plot_title(to_plot),
            _exceedance_plot_subtitle(to_plot),
            obj
        )
        return obj


    @param.depends("smoothing")
    def smoothing_card(self):
        """A reactive panel card used by _exceedance_visualize that only 
        displays the num_timesteps option if smoothing is selected."""
        if self.smoothing != "None":
            smooth_row = pn.Row(
                self.param.smoothing,
                self.param.num_timesteps, 
                width=375
            )
        else:
            smooth_row = pn.Row(
                self.param.smoothing,
                width=375
            )
        return pn.Card(smooth_row, title = "Smoothing", collapsible = False)

    @param.depends("duration1_length", "duration1_type", watch=False)
    def group_row(self):
        """A reactive row for duration2 options that updates if group is updated"""
        self.group_length = self.duration1_length
        self.group_type = self.duration1_type
        return pn.Row(
            self.param.group_length, self.param.group_type, 
            width=375
        )

    @param.depends("group_length", "group_type", watch=False)
    def duration2_row(self):
        """A reactive row for duration2 options that updates if group is updated"""
        self.duration2_length = self.group_length
        self.duration2_type = self.group_type
        return pn.Row(
            self.param.duration2_length, self.param.duration2_type, 
            width=375
        )

def explore_exceedance(da, option=1):
    """
    Main function for displaying the threshold exceedance count GUI for a 
    provided DataArray `da`.
    """
    exc_choices = ExceedanceParams(da) # initialize an instance of the Param class for this dataarray
    return _exceedance_visualize(exc_choices, option) # display the holoviz panel

def _exceedance_visualize(choices, option=1):
    """
    Uses holoviz 'panel' library to display the parameters and view defined for exploring exceedance.
    """
    _left_column_width = 375

    if option==1:
        plot_card = choices.view
    elif option==2:
        # For show: potential option to display multiple tabs if we want to 
        # build this out as a broader GUI app for all threshold tools
        plot_card = pn.Tabs(
            ("Event counts", choices.view), 
            ("Return values", pn.Row()),
            ("Return periods", pn.Row())
        ) 
    else:
        raise ValueError("Unknown option")

    options_card = pn.Card(
        # Threshold value and direction
        pn.Row(
            choices.param.threshold_direction,
            choices.param.threshold_value,
            width = _left_column_width
        ),

        # DURATION 1
        "I'm interested in extreme conditions that last for . . .",
        pn.Row(
            choices.param.duration1_length,
            choices.param.duration1_type, width=375
        ),
        pn.layout.Divider(margin = (-10,0,-10,0)),

        # PERIOD
        "Show me a timeseries of the number of occurences every . . .",
        pn.Row(
            choices.param.period_length,
            choices.param.period_type,
            width = _left_column_width
        ),
        "Examples: for an annual timeseries, select '1-year'. For a seasonal timeseries, select '3-month'.",
        pn.layout.Divider(margin = (-10,0,-10,0)),

        # GROUP
        "Optional aggregation: I'm interested in the number of ___ that contain at least one occurance.",
        choices.group_row,

        # DURATION 2
        "After aggregation, I'm interested in occurances that last for . . .",
        choices.duration2_row,
        
        title = "Threshold event options",
        collapsible = False
    )

    exceedance_count_panel = pn.Column(pn.Spacer(width=15), pn.Row(
        pn.Column(
            options_card,
            choices.smoothing_card,
            width = _left_column_width
        ),
        pn.Spacer(width=15),
        pn.Column(
            plot_card
        )
    ))

    return exceedance_count_panel