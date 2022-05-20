########################################
#                                      #
# THRESHOLD TOOLS                      #
#                                      #
########################################

#####################################################################
# import packages

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

from .visualize import get_geospatial_plot

#####################################################################
# GET AMS
# input: data array
# export: data array of maximums


def get_ams(da, extremes_type="max"):

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
# GET L-MOMENTS DISTR
# input: distribution name
# export: corresponding l-moments distribution function


def get_lmom_distr(distr):

    distrs = ["gev", "gumbel", "weibull", "pearson3", "genpareto"]
    if distr not in distrs:
        raise ValueError(
            "invalid distr type. expected one of the following: %s" % distrs
        )

    if distr == "gev":
        lmom_distr = ldistr.gev

    if distr == "gumbel":
        lmom_distr = ldistr.gum

    if distr == "weibull":
        lmom_distr = ldistr.wei

    if distr == "pearson3":
        lmom_distr = ldistr.pe3

    if distr == "genpareto":
        lmom_distr = ldistr.gpa

    return lmom_distr


#####################################################################
# GET L-MOMENTS
# input: annual maximum series
# export: xarray dataset of l-moment ratios


def get_lmoments(ams, distr="gev", multiple_points=True):

    lmom_distr = get_lmom_distr(distr)
    ams_attributes = ams.attrs

    if multiple_points:
        ams = ams.stack(allpoints=["x", "y"]).squeeze().groupby("allpoints")

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
# GET KS STAT
# input: annual maximum series
# export: xarray dataset of ks test d-statistics and p-values


def get_ks_stat(ams, distr="gev", multiple_points=True):

    lmom_distr = get_lmom_distr(distr)
    ams_attributes = ams.attrs

    if multiple_points:
        ams = ams.stack(allpoints=["x", "y"]).squeeze().groupby("allpoints")

    def ks_stat(ams):

        if distr == "gev":
            try:
                lmoments = lmom_distr.lmom_fit(ams)
                fitted_distr = stats.genextreme(**lmoments)
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

        if distr == "gumbel":
            try:
                lmoments = lmom_distr.lmom_fit(ams)
                fitted_distr = stats.gumbel_r(**lmoments)
                ks = stats.kstest(
                    ams, "gumbel_r", args=(lmoments["loc"], lmoments["scale"])
                )
                d_statistic = ks[0]
                p_value = ks[1]
            except (ValueError, ZeroDivisionError):
                d_statistic = np.nan
                p_value = np.nan

        if distr == "weibull":
            try:
                lmoments = lmom_distr.lmom_fit(ams)
                fitted_distr = stats.weibull_min(**lmoments)
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

        if distr == "pearson3":
            try:
                lmoments = lmom_distr.lmom_fit(ams)
                fitted_distr = stats.pearson3(**lmoments)
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

        if distr == "genpareto":
            try:
                lmoments = lmom_distr.lmom_fit(ams)
                fitted_distr = stats.genpareto(**lmoments)
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
# BOOTSTRAP
# input: annual maximum series and relevant parameters
# export: boostrap-calculated value for relevant parameters


def bootstrap(ams, distr="gev", data_variable="return_value", arg_value=10):

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
            lmoments = lmom_distr.lmom_fit(new_ams)
            fitted_distr = stats.genextreme(**lmoments)

            if data_variable == "return_value":
                try:
                    return_event = 1.0 - (1.0 / arg_value)
                    return_value = fitted_distr.ppf(return_event)
                    result = round(return_value, 5)
                except (ValueError, ZeroDivisionError):
                    result = np.nan

            if data_variable == "return_prob":
                try:
                    result = 1 - (fitted_distr.cdf(arg_value))
                except (ValueError, ZeroDivisionError):
                    result = np.nan

            if data_variable == "return_period":
                try:
                    return_prob = fitted_distr.cdf(arg_value)
                    if return_prob == 1.0:
                        result = np.nan
                    else:
                        return_period = -1.0 / (return_prob - 1.0)
                        result = round(return_period, 3)
                except (ValueError, ZeroDivisionError):
                    result = np.nan

        except (ValueError, ZeroDivisionError):
            result = np.nan

    if distr == "gumbel":

        try:
            lmoments = lmom_distr.lmom_fit(new_ams)
            fitted_distr = stats.gumbel_r(**lmoments)

            if data_variable == "return_value":
                try:
                    return_event = 1.0 - (1.0 / arg_value)
                    return_value = fitted_distr.ppf(return_event)
                    result = round(return_value, 5)
                except (ValueError, ZeroDivisionError):
                    result = np.nan

            if data_variable == "return_prob":
                try:
                    result = 1 - (fitted_distr.cdf(arg_value))
                except (ValueError, ZeroDivisionError):
                    result = np.nan

            if data_variable == "return_period":
                try:
                    return_prob = fitted_distr.cdf(arg_value)
                    if return_prob == 1.0:
                        result = np.nan
                    else:
                        return_period = -1.0 / (return_prob - 1.0)
                        result = round(return_period, 3)
                except (ValueError, ZeroDivisionError):
                    result = np.nan

        except (ValueError, ZeroDivisionError):
            result = np.nan

    if distr == "weibull":

        try:
            lmoments = lmom_distr.lmom_fit(new_ams)
            fitted_distr = stats.weibull_min(**lmoments)

            if data_variable == "return_value":
                try:
                    return_event = 1.0 - (1.0 / arg_value)
                    return_value = fitted_distr.ppf(return_event)
                    result = round(return_value, 5)
                except (ValueError, ZeroDivisionError):
                    result = np.nan

            if data_variable == "return_prob":
                try:
                    result = 1 - (fitted_distr.cdf(arg_value))
                except (ValueError, ZeroDivisionError):
                    result = np.nan

            if data_variable == "return_period":
                try:
                    return_prob = fitted_distr.cdf(arg_value)
                    if return_prob == 1.0:
                        result = np.nan
                    else:
                        return_period = -1.0 / (return_prob - 1.0)
                        result = round(return_period, 3)
                except (ValueError, ZeroDivisionError):
                    result = np.nan

        except (ValueError, ZeroDivisionError):
            result = np.nan

    if distr == "pearson3":

        try:
            lmoments = lmom_distr.lmom_fit(new_ams)
            fitted_distr = stats.pearson3(**lmoments)

            if data_variable == "return_value":
                try:
                    return_event = 1.0 - (1.0 / arg_value)
                    return_value = fitted_distr.ppf(return_event)
                    result = round(return_value, 5)
                except (ValueError, ZeroDivisionError):
                    result = np.nan

            if data_variable == "return_prob":
                try:
                    result = 1 - (fitted_distr.cdf(arg_value))
                except (ValueError, ZeroDivisionError):
                    result = np.nan

            if data_variable == "return_period":
                try:
                    return_prob = fitted_distr.cdf(arg_value)
                    if return_prob == 1.0:
                        result = np.nan
                    else:
                        return_period = -1.0 / (return_prob - 1.0)
                        result = round(return_period, 3)
                except (ValueError, ZeroDivisionError):
                    result = np.nan

        except (ValueError, ZeroDivisionError):
            result = np.nan

    if distr == "genpareto":

        try:
            lmoments = lmom_distr.lmom_fit(new_ams)
            fitted_distr = stats.genpareto(**lmoments)

            if data_variable == "return_value":
                try:
                    return_event = 1.0 - (1.0 / arg_value)
                    return_value = fitted_distr.ppf(return_event)
                    result = round(return_value, 5)
                except (ValueError, ZeroDivisionError):
                    result = np.nan

            if data_variable == "return_prob":
                try:
                    result = 1 - (fitted_distr.cdf(arg_value))
                except (ValueError, ZeroDivisionError):
                    result = np.nan

            if data_variable == "return_period":
                try:
                    return_prob = fitted_distr.cdf(arg_value)
                    if return_prob == 1.0:
                        result = np.nan
                    else:
                        return_period = -1.0 / (return_prob - 1.0)
                        result = round(return_period, 3)
                except (ValueError, ZeroDivisionError):
                    result = np.nan

        except (ValueError, ZeroDivisionError):
            result = np.nan

    return result


#####################################################################
# GET RETURN VALUE
# input: annual maximum series and relevant parameters
# export: xarray dataset with return values and confidence intervals


def get_return_value(
    ams,
    return_period=10,
    distr="gev",
    bootstrap_runs=100,
    conf_int_lower_bound=2.5,
    conf_int_upper_bound=97.5,
    multiple_points=True,
):

    data_variable = "return_value"
    lmom_distr = get_lmom_distr(distr)
    ams_attributes = ams.attrs

    if multiple_points:
        ams = ams.stack(allpoints=["x", "y"]).squeeze().groupby("allpoints")

    def return_value(ams):

        if distr == "gev":
            try:
                lmoments = lmom_distr.lmom_fit(ams)
                fitted_distr = stats.genextreme(**lmoments)
                return_event = 1.0 - (1.0 / return_period)
                return_value = fitted_distr.ppf(return_event)
                return_value = round(return_value, 5)
            except (ValueError, ZeroDivisionError):
                return_value = np.nan

            bootstrap_values = []
            for _ in range(bootstrap_runs):
                result = bootstrap(
                    ams,
                    distr=distr,
                    data_variable=data_variable,
                    arg_value=return_period,
                )
                bootstrap_values.append(result)

            conf_int_array = np.percentile(
                bootstrap_values, [conf_int_lower_bound, conf_int_upper_bound]
            )
            conf_int_lower_limit = conf_int_array[0]
            conf_int_upper_limit = conf_int_array[1]

        if distr == "gumbel":
            try:
                lmoments = lmom_distr.lmom_fit(ams)
                fitted_distr = stats.gumbel_r(**lmoments)
                return_event = 1.0 - (1.0 / return_period)
                return_value = fitted_distr.ppf(return_event)
                return_value = round(return_value, 5)
            except (ValueError, ZeroDivisionError):
                return_value = np.nan

            bootstrap_values = []
            for _ in range(bootstrap_runs):
                result = bootstrap(
                    ams,
                    distr=distr,
                    data_variable=data_variable,
                    arg_value=return_period,
                )
                bootstrap_values.append(result)

            conf_int_array = np.percentile(
                bootstrap_values, [conf_int_lower_bound, conf_int_upper_bound]
            )
            conf_int_lower_limit = conf_int_array[0]
            conf_int_upper_limit = conf_int_array[1]

        if distr == "weibull":
            try:
                lmoments = lmom_distr.lmom_fit(ams)
                fitted_distr = stats.weibull_min(**lmoments)
                return_event = 1.0 - (1.0 / return_period)
                return_value = fitted_distr.ppf(return_event)
                return_value = round(return_value, 5)
            except (ValueError, ZeroDivisionError):
                return_value = np.nan

            bootstrap_values = []
            for _ in range(bootstrap_runs):
                result = bootstrap(
                    ams,
                    distr=distr,
                    data_variable=data_variable,
                    arg_value=return_period,
                )
                bootstrap_values.append(result)

            conf_int_array = np.percentile(
                bootstrap_values, [conf_int_lower_bound, conf_int_upper_bound]
            )
            conf_int_lower_limit = conf_int_array[0]
            conf_int_upper_limit = conf_int_array[1]

        if distr == "pearson3":
            try:
                lmoments = lmom_distr.lmom_fit(ams)
                fitted_distr = stats.pearson3(**lmoments)
                return_event = 1.0 - (1.0 / return_period)
                return_value = fitted_distr.ppf(return_event)
                return_value = round(return_value, 5)
            except (ValueError, ZeroDivisionError):
                return_value = np.nan

            bootstrap_values = []
            for _ in range(bootstrap_runs):
                result = bootstrap(
                    ams,
                    distr=distr,
                    data_variable=data_variable,
                    arg_value=return_period,
                )
                bootstrap_values.append(result)

            conf_int_array = np.percentile(
                bootstrap_values, [conf_int_lower_bound, conf_int_upper_bound]
            )
            conf_int_lower_limit = conf_int_array[0]
            conf_int_upper_limit = conf_int_array[1]

        if distr == "genpareto":
            try:
                lmoments = lmom_distr.lmom_fit(ams)
                fitted_distr = stats.genpareto(**lmoments)
                return_event = 1.0 - (1.0 / return_period)
                return_value = fitted_distr.ppf(return_event)
                return_value = round(return_value, 5)
            except (ValueError, ZeroDivisionError):
                return_value = np.nan

            bootstrap_values = []
            for _ in range(bootstrap_runs):
                result = bootstrap(
                    ams,
                    distr=distr,
                    data_variable=data_variable,
                    arg_value=return_period,
                )
                bootstrap_values.append(result)

            conf_int_array = np.percentile(
                bootstrap_values, [conf_int_lower_bound, conf_int_upper_bound]
            )
            conf_int_lower_limit = conf_int_array[0]
            conf_int_upper_limit = conf_int_array[1]

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
# GET RETURN PROBABILITY
# input: annual maximum series and relevant parameters
# export: xarray dataset with return probabilities and confidence intervals


def get_return_prob(
    ams,
    threshold,
    distr="gev",
    bootstrap_runs=100,
    conf_int_lower_bound=2.5,
    conf_int_upper_bound=97.5,
    multiple_points=True,
):

    data_variable = "return_prob"
    lmom_distr = get_lmom_distr(distr)
    ams_attributes = ams.attrs

    if multiple_points:
        ams = ams.stack(allpoints=["x", "y"]).squeeze().groupby("allpoints")

    def return_prob(ams):

        if distr == "gev":
            try:
                lmoments = lmom_distr.lmom_fit(ams)
                fitted_distr = stats.genextreme(**lmoments)
                return_prob = 1 - (fitted_distr.cdf(threshold))
            except (ValueError, ZeroDivisionError):
                return_prob = np.nan

            bootstrap_values = []
            for _ in range(bootstrap_runs):
                result = bootstrap(
                    ams, distr=distr, data_variable=data_variable, arg_value=threshold
                )
                bootstrap_values.append(result)

            conf_int_array = np.percentile(
                bootstrap_values, [conf_int_lower_bound, conf_int_upper_bound]
            )
            conf_int_lower_limit = conf_int_array[0]
            conf_int_upper_limit = conf_int_array[1]

        if distr == "gumbel":
            try:
                lmoments = lmom_distr.lmom_fit(ams)
                fitted_distr = stats.gumbel_r(**lmoments)
                return_prob = 1 - (fitted_distr.cdf(threshold))
            except (ValueError, ZeroDivisionError):
                return_prob = np.nan

            bootstrap_values = []
            for _ in range(bootstrap_runs):
                result = bootstrap(
                    ams, distr=distr, data_variable=data_variable, arg_value=threshold
                )
                bootstrap_values.append(result)

            conf_int_array = np.percentile(
                bootstrap_values, [conf_int_lower_bound, conf_int_upper_bound]
            )
            conf_int_lower_limit = conf_int_array[0]
            conf_int_upper_limit = conf_int_array[1]

        if distr == "weibull":
            try:
                lmoments = lmom_distr.lmom_fit(ams)
                fitted_distr = stats.weibull_min(**lmoments)
                return_prob = 1 - (fitted_distr.cdf(threshold))
            except (ValueError, ZeroDivisionError):
                return_prob = np.nan

            bootstrap_values = []
            for _ in range(bootstrap_runs):
                result = bootstrap(
                    ams, distr=distr, data_variable=data_variable, arg_value=threshold
                )
                bootstrap_values.append(result)

            conf_int_array = np.percentile(
                bootstrap_values, [conf_int_lower_bound, conf_int_upper_bound]
            )
            conf_int_lower_limit = conf_int_array[0]
            conf_int_upper_limit = conf_int_array[1]

        if distr == "pearson3":
            try:
                lmoments = lmom_distr.lmom_fit(ams)
                fitted_distr = stats.pearson3(**lmoments)
                return_prob = 1 - (fitted_distr.cdf(threshold))
            except (ValueError, ZeroDivisionError):
                return_prob = np.nan

            bootstrap_values = []
            for _ in range(bootstrap_runs):
                result = bootstrap(
                    ams, distr=distr, data_variable=data_variable, arg_value=threshold
                )
                bootstrap_values.append(result)

            conf_int_array = np.percentile(
                bootstrap_values, [conf_int_lower_bound, conf_int_upper_bound]
            )
            conf_int_lower_limit = conf_int_array[0]
            conf_int_upper_limit = conf_int_array[1]

        if distr == "genpareto":
            try:
                lmoments = lmom_distr.lmom_fit(ams)
                fitted_distr = stats.genpareto(**lmoments)
                return_prob = 1 - (fitted_distr.cdf(threshold))
            except (ValueError, ZeroDivisionError):
                return_prob = np.nan

            bootstrap_values = []
            for _ in range(bootstrap_runs):
                result = bootstrap(
                    ams, distr=distr, data_variable=data_variable, arg_value=threshold
                )
                bootstrap_values.append(result)

            conf_int_array = np.percentile(
                bootstrap_values, [conf_int_lower_bound, conf_int_upper_bound]
            )
            conf_int_lower_limit = conf_int_array[0]
            conf_int_upper_limit = conf_int_array[1]

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
# GET RETURN PERIOD
# input: annual maximum series and relevant parameters
# export: xarray dataset with return periods and confidence intervals


def get_return_period(
    ams,
    return_value,
    distr="gev",
    bootstrap_runs=100,
    conf_int_lower_bound=2.5,
    conf_int_upper_bound=97.5,
    multiple_points=True,
):

    data_variable = "return_period"
    lmom_distr = get_lmom_distr(distr)
    ams_attributes = ams.attrs

    if multiple_points:
        ams = ams.stack(allpoints=["x", "y"]).squeeze().groupby("allpoints")

    def return_period(ams):

        if distr == "gev":
            try:
                lmoments = lmom_distr.lmom_fit(ams)
                fitted_distr = stats.genextreme(**lmoments)
                return_prob = fitted_distr.cdf(return_value)
                if return_prob == 1.0:
                    return_period = np.nan
                else:
                    return_period = -1.0 / (return_prob - 1.0)
                    return_period = round(return_period, 3)
            except (ValueError, ZeroDivisionError):
                return_period = np.nan

            bootstrap_values = []
            for _ in range(bootstrap_runs):
                result = bootstrap(
                    ams,
                    distr=distr,
                    data_variable=data_variable,
                    arg_value=return_value,
                )
                bootstrap_values.append(result)

            conf_int_array = np.percentile(
                bootstrap_values, [conf_int_lower_bound, conf_int_upper_bound]
            )
            conf_int_lower_limit = conf_int_array[0]
            conf_int_upper_limit = conf_int_array[1]

        if distr == "gumbel":
            try:
                lmoments = lmom_distr.lmom_fit(ams)
                fitted_distr = stats.gumbel_r(**lmoments)
                return_prob = fitted_distr.cdf(return_value)
                if return_prob == 1.0:
                    return_period = np.nan
                else:
                    return_period = -1.0 / (return_prob - 1.0)
                    return_period = round(return_period, 3)
            except (ValueError, ZeroDivisionError):
                return_period = np.nan

            bootstrap_values = []
            for _ in range(bootstrap_runs):
                result = bootstrap(
                    ams,
                    distr=distr,
                    data_variable=data_variable,
                    arg_value=return_value,
                )
                bootstrap_values.append(result)

            conf_int_array = np.percentile(
                bootstrap_values, [conf_int_lower_bound, conf_int_upper_bound]
            )
            conf_int_lower_limit = conf_int_array[0]
            conf_int_upper_limit = conf_int_array[1]

        if distr == "weibull":
            try:
                lmoments = lmom_distr.lmom_fit(ams)
                fitted_distr = stats.weibull_min(**lmoments)
                return_prob = fitted_distr.cdf(return_value)
                if return_prob == 1.0:
                    return_period = np.nan
                else:
                    return_period = -1.0 / (return_prob - 1.0)
                    return_period = round(return_period, 3)
            except (ValueError, ZeroDivisionError):
                return_period = np.nan

            bootstrap_values = []
            for _ in range(bootstrap_runs):
                result = bootstrap(
                    ams,
                    distr=distr,
                    data_variable=data_variable,
                    arg_value=return_value,
                )
                bootstrap_values.append(result)

            conf_int_array = np.percentile(
                bootstrap_values, [conf_int_lower_bound, conf_int_upper_bound]
            )
            conf_int_lower_limit = conf_int_array[0]
            conf_int_upper_limit = conf_int_array[1]

        if distr == "pearson3":
            try:
                lmoments = lmom_distr.lmom_fit(ams)
                fitted_distr = stats.pearson3(**lmoments)
                return_prob = fitted_distr.cdf(threshold)
                if return_prob == 1.0:
                    return_period = np.nan
                else:
                    return_period = -1.0 / (return_prob - 1.0)
                    return_period = round(return_period, 3)
            except (ValueError, ZeroDivisionError):
                return_period = np.nan

            bootstrap_values = []
            for _ in range(bootstrap_runs):
                result = bootstrap(
                    ams,
                    distr=distr,
                    data_variable=data_variable,
                    arg_value=return_value,
                )
                bootstrap_values.append(result)

            conf_int_array = np.percentile(
                bootstrap_values, [conf_int_lower_bound, conf_int_upper_bound]
            )
            conf_int_lower_limit = conf_int_array[0]
            conf_int_upper_limit = conf_int_array[1]

        if distr == "genpareto":
            try:
                lmoments = lmom_distr.lmom_fit(ams)
                fitted_distr = stats.genpareto(**lmoments)
                return_prob = fitted_distr.cdf(return_value)
                if return_prob == 1.0:
                    return_period = np.nan
                else:
                    return_period = -1.0 / (return_prob - 1.0)
                    return_period = round(return_period, 3)
            except (ValueError, ZeroDivisionError):
                return_period = np.nan

            bootstrap_values = []
            for _ in range(bootstrap_runs):
                result = bootstrap(
                    ams,
                    distr=distr,
                    data_variable=data_variable,
                    arg_value=return_value,
                )
                bootstrap_values.append(result)

            conf_int_array = np.percentile(
                bootstrap_values, [conf_int_lower_bound, conf_int_upper_bound]
            )
            conf_int_lower_limit = conf_int_array[0]
            conf_int_upper_limit = conf_int_array[1]

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


#####################################################################
# ARCHIVE


# def get_aicc_stat(ams, multiple_points=True):

#     ams_attributes = ams.attrs

#     if multiple_points:
#         ams = ams.stack(allpoints=["x", "y"]).squeeze().groupby("allpoints")

#     def aicc_stat(ams):

#         try:
#             lmoments_gev = ldistr.gev.lmom_fit(ams)
#             aicc_gev = ["gev", lstats.AICc(ams, "gev", lmoments_gev)]
#         except (ValueError, ZeroDivisionError):
#             aicc_gev = ["gev", np.nan]

#         try:
#             lmoments_gum = ldistr.gum.lmom_fit(ams)
#             aicc_gum = ["gumbel", lstats.AICc(ams, "gum", lmoments_gum)]
#         except (ValueError, ZeroDivisionError):
#             aicc_gum = ["gumbel", np.nan]

#         try:
#             lmoments_wei = ldistr.wei.lmom_fit(ams)
#             aicc_wei = ["weibull", lstats.AICc(ams, "wei", lmoments_wei)]
#         except (ValueError, ZeroDivisionError):
#             aicc_wei = ["weibull", np.nan]

#         try:
#             lmoments_pe3 = ldistr.pe3.lmom_fit(ams)
#             aicc_pe3 = ["pearson3", lstats.AICc(ams, "pe3", lmoments_pe3)]
#         except (ValueError, ZeroDivisionError):
#             aicc_pe3 = ["pearson3", np.nan]

#         try:
#             lmoments_gpa = ldistr.gpa.lmom_fit(ams)
#             aicc_gpa = ["genpareto", lstats.AICc(ams, "gpa", lmoments_gpa)]
#         except (ValueError, ZeroDivisionError):
#             aicc_gpa = ["genpareto", np.nan]

#         try:
#             lmoments_gpa = ldistr.gpa.lmom_fit(ams)
#             aicc_gpa = ["genpareto", lstats.AICc(ams, "gpa", lmoments_gpa)]
#         except (ValueError, ZeroDivisionError):
#             aicc_gpa = ["genpareto", np.nan]

#         all_aicc_results = (
#             str(aicc_gev)
#             + str(aicc_gum)
#             + str(aicc_wei)
#             + str(aicc_pe3)
#             + str(aicc_gpa)
#         )
#         all_aicc_results_string = str(all_aicc_results)

#         lowest_aicc_value = min(
#             aicc_gev[1], aicc_gum[1], aicc_wei[1], aicc_pe3[1], aicc_gpa[1]
#         )

#         if lowest_aicc_value == aicc_gev[1]:
#             lowest_aicc_distr = aicc_gev[0]
#         elif lowest_aicc_value == aicc_gum[1]:
#             lowest_aicc_distr = aicc_gum[0]
#         elif lowest_aicc_value == aicc_wei[1]:
#             lowest_aicc_distr = aicc_wei[0]
#         elif lowest_aicc_value == aicc_pe3[1]:
#             lowest_aicc_distr = aicc_pe3[0]
#         elif lowest_aicc_value == aicc_gpa[1]:
#             lowest_aicc_distr = aicc_gpa[0]

#         return all_aicc_results, lowest_aicc_distr, lowest_aicc_value

#     all_aicc_results, lowest_aicc_distr, lowest_aicc_value = xr.apply_ufunc(
#         aicc_stat,
#         ams,
#         input_core_dims=[["time"]],
#         exclude_dims=set(("time",)),
#         output_core_dims=[[], [], []]

#         #dask_gufunc_kwargs=dict({"allow_rechunk"==True, "output_dtypes"==[ams.dtype]})
#     )

#     all_aicc_results = all_aicc_results.rename("all_aicc_results")
#     new_ds = all_aicc_results.to_dataset()
#     new_ds["lowest_aicc_distr"] = lowest_aicc_distr
#     new_ds["lowest_aicc_value"] = lowest_aicc_value

#     if multiple_points:
#         new_ds = new_ds.unstack("allpoints")

#     new_ds.attrs = ams_attributes
#     new_ds.attrs["model selection technique"] = "AICc"

#     return new_ds
