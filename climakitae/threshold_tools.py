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

from .visualize import get_frequency_plot, get_geospatial_plot

#####################################################################


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


def get_lmoments(ams, distr="gev", multiple_points=True):

    distrs = ["gev", "gumbel", "weibull", "pearson3", "genpareto"]
    if distr not in distrs:
        raise ValueError(
            "invalid distr type. expected one of the following: %s" % distrs
        )

    ams_attributes = ams.attrs

    if multiple_points == True:
        ams = ams.stack(allpoints=["x", "y"]).squeeze().groupby("allpoints")

    def apply_function(ams):
        def function(ams):

            if distr == "gev":
                lmoments = ldistr.gev.lmom_fit(ams)

            if distr == "gumbel":
                lmoments = ldistr.gum.lmom_fit(ams)

            if distr == "weibull":
                lmoments = ldistr.wei.lmom_fit(ams)

            if distr == "pearson3":
                lmoments = ldistr.pe3.lmom_fit(ams)

            if distr == "genpareto":
                lmoments = ldistr.gpa.lmom_fit(ams)

            return lmoments

        return xr.apply_ufunc(
            function,
            ams,
            input_core_dims=[["time"]],
            exclude_dims=set(("time",)),
            output_core_dims=[[]],
        )

    lmoments = apply_function(ams)

    new_ds = lmoments.to_dataset()
    new_ds = new_ds.rename({"T2": "lmoments"})

    if multiple_points == True:
        new_ds = new_ds.unstack("allpoints")

    new_ds.attrs = ams_attributes
    new_ds.attrs["distribution"] = "{}".format(str(distr))

    return new_ds


#####################################################################


def get_aicc_stat(ams, multiple_points=True):

    ams_attributes = ams.attrs

    if multiple_points == True:
        ams = ams.stack(allpoints=["x", "y"]).squeeze().groupby("allpoints")

    def apply_function(ams):
        def function(ams):

            try:
                lmoments_gev = ldistr.gev.lmom_fit(ams)
                aicc_gev = ["gev", lstats.AICc(ams, "gev", lmoments_gev)]
            except ValueError:
                aicc_gev = ["gev", np.nan]

            try:
                lmoments_gum = ldistr.gum.lmom_fit(ams)
                aicc_gum = ["gumbel", lstats.AICc(ams, "gum", lmoments_gum)]
            except ValueError:
                aicc_gum = ["gumbel", np.nan]

            try:
                lmoments_wei = ldistr.wei.lmom_fit(ams)
                aicc_wei = ["weibull", lstats.AICc(ams, "wei", lmoments_wei)]
            except ValueError:
                aicc_wei = ["weibull", np.nan]

            try:
                lmoments_pe3 = ldistr.pe3.lmom_fit(ams)
                aicc_pe3 = ["pearson3", lstats.AICc(ams, "pe3", lmoments_pe3)]
            except ValueError:
                aicc_pe3 = ["pearson3", np.nan]

            try:
                lmoments_gpa = ldistr.gpa.lmom_fit(ams)
                aicc_gpa = ["genpareto", lstats.AICc(ams, "gpa", lmoments_gpa)]
            except ValueError:
                aicc_gpa = ["genpareto", np.nan]

            all_aicc_results = (
                str(aicc_gev)
                + str(aicc_gum)
                + str(aicc_wei)
                + str(aicc_pe3)
                + str(aicc_gpa)
            )
            all_aicc_results_string = str(all_aicc_results)

            lowest_aicc_value = min(
                aicc_gev[1], aicc_gum[1], aicc_wei[1], aicc_pe3[1], aicc_gpa[1]
            )

            if lowest_aicc_value == aicc_gev[1]:
                lowest_aicc_distr = aicc_gev[0]
            elif lowest_aicc_value == aicc_gum[1]:
                lowest_aicc_distr = aicc_gum[0]
            elif lowest_aicc_value == aicc_wei[1]:
                lowest_aicc_distr = aicc_wei[0]
            elif lowest_aicc_value == aicc_pe3[1]:
                lowest_aicc_distr = aicc_pe3[0]
            elif lowest_aicc_value == aicc_gpa[1]:
                lowest_aicc_distr = aicc_gpa[0]

            return all_aicc_results, lowest_aicc_distr, lowest_aicc_value

        return xr.apply_ufunc(
            function,
            ams,
            input_core_dims=[["time"]],
            exclude_dims=set(("time",)),
            output_core_dims=[[], [], []],
            dask = 'allowed', vectorize = True
        )

    all_aicc_results, lowest_aicc_distr, lowest_aicc_value = apply_function(ams)

    all_aicc_results = all_aicc_results.rename("all_aicc_results")
    new_ds = all_aicc_results.to_dataset()
    new_ds["lowest_aicc_distr"] = lowest_aicc_distr
    new_ds["lowest_aicc_value"] = lowest_aicc_value

    if multiple_points == True:
        new_ds = new_ds.unstack("allpoints")

    new_ds.attrs = ams_attributes
    new_ds.attrs["model selection technique"] = "AICc"

    return new_ds


#####################################################################


def get_ks_stat(ams, distr="gev", multiple_points=True):

    distrs = ["gev", "gumbel", "weibull", "pearson3", "genpareto"]
    if distr not in distrs:
        raise ValueError(
            "invalid distr type. expected one of the following: %s" % distrs
        )

    ams_attributes = ams.attrs

    if multiple_points == True:
        ams = ams.stack(allpoints=["x", "y"]).squeeze().groupby("allpoints")

    def apply_function(ams):
        def function(ams):

            if distr == "gev":
                try:
                    lmoments = ldistr.gev.lmom_fit(ams)
                    fitted_distr = stats.genextreme(**lmoments)
                    ks = stats.kstest(
                        ams,
                        "genextreme",
                        args=(lmoments["c"], lmoments["loc"], lmoments["scale"]),
                    )
                    d_statistic = ks[0]
                    p_value = ks[1]
                except ValueError:
                    d_statistic = np.nan
                    p_value = np.nan

            if distr == "gumbel":
                try:
                    lmoments = ldistr.gum.lmom_fit(ams)
                    fitted_distr = stats.gumbel_r(**lmoments)
                    ks = stats.kstest(
                        ams, "gumbel_r", args=(lmoments["loc"], lmoments["scale"])
                    )
                    d_statistic = ks[0]
                    p_value = ks[1]
                except ValueError:
                    d_statistic = np.nan
                    p_value = np.nan

            if distr == "weibull":
                try:
                    lmoments = ldistr.wei.lmom_fit(ams)
                    fitted_distr = stats.weibull_min(**lmoments)
                    ks = stats.kstest(
                        ams,
                        "weibull_min",
                        args=(lmoments["c"], lmoments["loc"], lmoments["scale"]),
                    )
                    d_statistic = ks[0]
                    p_value = ks[1]
                except ValueError:
                    d_statistic = np.nan
                    p_value = np.nan

            if distr == "pearson3":
                try:
                    lmoments = ldistr.pe3.lmom_fit(ams)
                    fitted_distr = stats.pearson3(**lmoments)
                    ks = stats.kstest(
                        ams,
                        "pearson3",
                        args=(lmoments["skew"], lmoments["loc"], lmoments["scale"]),
                    )
                    d_statistic = ks[0]
                    p_value = ks[1]
                except ValueError:
                    d_statistic = np.nan
                    p_value = np.nan

            if distr == "genpareto":
                try:
                    lmoments = ldistr.gpa.lmom_fit(ams)
                    fitted_distr = stats.genpareto(**lmoments)
                    ks = stats.kstest(
                        ams,
                        "genpareto",
                        args=(lmoments["c"], lmoments["loc"], lmoments["scale"]),
                    )
                    d_statistic = ks[0]
                    p_value = ks[1]
                except ValueError:
                    d_statistic = np.nan
                    p_value = np.nan

            return d_statistic, p_value

        return xr.apply_ufunc(
            function,
            ams,
            input_core_dims=[["time"]],
            exclude_dims=set(("time",)),
            output_core_dims=[[], []],
            dask = 'allowed', vectorize = True
        )

    d_statistic, p_value = apply_function(ams)

    d_statistic = d_statistic.rename("d_statistic")
    new_ds = d_statistic.to_dataset()
    new_ds["p_value"] = p_value

    if multiple_points == True:
        new_ds = new_ds.unstack("allpoints")

    new_ds["d_statistic"].attrs["stat test"] = "KS test"
    new_ds["p_value"].attrs["stat test"] = "KS test"
    new_ds.attrs = ams_attributes
    new_ds.attrs["distribution"] = "{}".format(str(distr))

    return new_ds


#####################################################################


def get_return_value(ams, return_period=10, distr="gev", multiple_points=True):

    distrs = ["gev", "gumbel", "weibull", "pearson3", "genpareto"]
    if distr not in distrs:
        raise ValueError(
            "invalid distr type. expected one of the following: %s" % distrs
        )

    ams_attributes = ams.attrs

    if multiple_points == True:
        ams = ams.stack(allpoints=["x", "y"]).squeeze().groupby("allpoints")

    def apply_function(ams):
        def function(ams):

            if distr == "gev":
                try:
                    lmoments = ldistr.gev.lmom_fit(ams)
                    fitted_distr = stats.genextreme(**lmoments)
                    return_event = 1.0 - (1.0 / return_period)
                    return_value = fitted_distr.ppf(return_event)
                    return_value = round(return_value, 5)
                except ValueError:
                    return_value = np.nan

            if distr == "gumbel":
                try:
                    lmoments = ldistr.gum.lmom_fit(ams)
                    fitted_distr = stats.gumbel_r(**lmoments)
                    return_event = 1.0 - (1.0 / return_period)
                    return_value = fitted_distr.ppf(return_event)
                    return_value = round(return_value, 5)
                except ValueError:
                    return_value = np.nan

            if distr == "weibull":
                try:
                    lmoments = ldistr.wei.lmom_fit(ams)
                    fitted_distr = stats.weibull_min(**lmoments)
                    return_event = 1.0 - (1.0 / return_period)
                    return_value = fitted_distr.ppf(return_event)
                    return_value = round(return_value, 5)
                except ValueError:
                    return_value = np.nan

            if distr == "pearson3":
                try:
                    lmoments = ldistr.pe3.lmom_fit(ams)
                    fitted_distr = stats.pearson3(**lmoments)
                    return_event = 1.0 - (1.0 / return_period)
                    return_value = fitted_distr.ppf(return_event)
                    return_value = round(return_value, 5)
                except ValueError:
                    return_value = np.nan

            if distr == "genpareto":
                try:
                    lmoments = ldistr.gpa.lmom_fit(ams)
                    fitted_distr = stats.genpareto(**lmoments)
                    return_event = 1.0 - (1.0 / return_period)
                    return_value = fitted_distr.ppf(return_event)
                    return_value = round(return_value, 5)
                except ValueError:
                    return_value = np.nan

            return return_value

        return xr.apply_ufunc(
            function,
            ams,
            input_core_dims=[["time"]],
            exclude_dims=set(("time",)),
            output_core_dims=[[]],
            dask = 'allowed', vectorize = True
        )

    return_value = apply_function(ams)

    return_value = return_value.rename("return_value")
    new_ds = return_value.to_dataset()

    if multiple_points == True:
        new_ds = new_ds.unstack("allpoints")

    new_ds["return_value"].attrs["return period"] = "1 in {} year event".format(
        str(return_period)
    )
    new_ds.attrs = ams_attributes
    new_ds.attrs["distribution"] = "{}".format(str(distr))

    return new_ds


#####################################################################


def get_return_prob(ams, threshold, distr="gev", multiple_points=True):

    distrs = ["gev", "gumbel", "weibull", "pearson3", "genpareto"]
    if distr not in distrs:
        raise ValueError(
            "invalid distr type. expected one of the following: %s" % distrs
        )

    ams_attributes = ams.attrs

    if multiple_points == True:
        ams = ams.stack(allpoints=["x", "y"]).squeeze().groupby("allpoints")

    def apply_function(ams):
        def function(ams):

            if distr == "gev":
                try:
                    lmoments = ldistr.gev.lmom_fit(ams)
                    fitted_distr = stats.genextreme(**lmoments)
                    return_prob = 1 - (fitted_distr.cdf(threshold))
                except ValueError:
                    return_prob = np.nan

            if distr == "gumbel":
                try:
                    lmoments = ldistr.gum.lmom_fit(ams)
                    fitted_distr = stats.gumbel_r(**lmoments)
                    return_prob = 1 - (fitted_distr.cdf(threshold))
                except ValueError:
                    return_prob = np.nan

            if distr == "weibull":
                try:
                    lmoments = ldistr.wei.lmom_fit(ams)
                    fitted_distr = stats.weibull_min(**lmoments)
                    return_prob = 1 - (fitted_distr.cdf(threshold))
                except ValueError:
                    return_prob = np.nan

            if distr == "pearson3":
                try:
                    lmoments = ldistr.pe3.lmom_fit(ams)
                    fitted_distr = stats.pearson3(**lmoments)
                    return_prob = 1 - (fitted_distr.cdf(threshold))
                except ValueError:
                    return_prob = np.nan

            if distr == "genpareto":
                try:
                    lmoments = ldistr.gpa.lmom_fit(ams)
                    fitted_distr = stats.genpareto(**lmoments)
                    return_prob = 1 - (fitted_distr.cdf(threshold))
                except ValueError:
                    return_prob = np.nan

            return return_prob

        return xr.apply_ufunc(
            function,
            ams,
            input_core_dims=[["time"]],
            exclude_dims=set(("time",)),
            output_core_dims=[[]],
            dask = 'allowed', vectorize = True
        )

    return_prob = apply_function(ams)

    return_prob = return_prob.rename("return_prob")
    new_ds = return_prob.to_dataset()

    if multiple_points == True:
        new_ds = new_ds.unstack("allpoints")

    new_ds["return_prob"].attrs["threshold"] = "exceedance of {} value event".format(
        str(threshold)
    )
    new_ds.attrs = ams_attributes
    new_ds.attrs["distribution"] = "{}".format(str(distr))

    return new_ds


#####################################################################


def get_return_period(ams, return_value, distr="gev", multiple_points=True):

    distrs = ["gev", "gumbel", "weibull", "pearson3", "genpareto"]
    if distr not in distrs:
        raise ValueError(
            "invalid distr type. expected one of the following: %s" % distrs
        )

    ams_attributes = ams.attrs

    if multiple_points == True:
        ams = ams.stack(allpoints=["x", "y"]).squeeze().groupby("allpoints")

    def apply_function(ams):
        def function(ams):
            if distr == "gev":
                try:
                    lmoments = ldistr.gev.lmom_fit(ams)
                    fitted_distr = stats.genextreme(**lmoments)
                    return_prob = fitted_distr.cdf(return_value)
                    if return_prob == 1.0:
                        return_period = np.nan
                    else:
                        return_period = -1.0 / (return_prob - 1.0)
                        return_period = round(return_period, 3)
                except ValueError:
                    return_period = np.nan

            if distr == "gumbel":
                try:
                    lmoments = ldistr.gum.lmom_fit(ams)
                    fitted_distr = stats.gumbel_r(**lmoments)
                    return_prob = fitted_distr.cdf(return_value)
                    if return_prob == 1.0:
                        return_period = np.nan
                    else:
                        return_period = -1.0 / (return_prob - 1.0)
                        return_period = round(return_period, 3)
                except ValueError:
                    return_period = np.nan

            if distr == "weibull":
                try:
                    lmoments = ldistr.wei.lmom_fit(ams)
                    fitted_distr = stats.weibull_min(**lmoments)
                    return_prob = fitted_distr.cdf(return_value)
                    if return_prob == 1.0:
                        return_period = np.nan
                    else:
                        return_period = -1.0 / (return_prob - 1.0)
                        return_period = round(return_period, 3)
                except ValueError:
                    return_period = np.nan

            if distr == "pearson3":
                try:
                    lmoments = ldistr.pe3.lmom_fit(ams)
                    fitted_distr = stats.pearson3(**lmoments)
                    return_prob = fitted_distr.cdf(threshold)
                    if return_prob == 1.0:
                        return_period = np.nan
                    else:
                        return_period = -1.0 / (return_prob - 1.0)
                        return_period = round(return_period, 3)
                except ValueError:
                    return_period = np.nan

            if distr == "genpareto":
                try:
                    lmoments = ldistr.gpa.lmom_fit(ams)
                    fitted_distr = stats.genpareto(**lmoments)
                    return_prob = fitted_distr.cdf(return_value)
                    if return_prob == 1.0:
                        return_period = np.nan
                    else:
                        return_period = -1.0 / (return_prob - 1.0)
                        return_period = round(return_period, 3)
                except ValueError:
                    return_period = np.nan

            return return_period

        return xr.apply_ufunc(
            function,
            ams,
            input_core_dims=[["time"]],
            exclude_dims=set(("time",)),
            output_core_dims=[[]],
            dask = 'allowed', vectorize = True
        )

    return_period = apply_function(ams)

    return_period = return_period.rename("return_period")
    new_ds = return_period.to_dataset()

    if multiple_points == True:
        new_ds = new_ds.unstack("allpoints")

    new_ds["return_period"].attrs[
        "return value"
    ] = "occurrence of a {} value event".format(str(return_value))
    new_ds.attrs = ams_attributes
    new_ds.attrs["distribution"] = "{}".format(str(distr))

    return new_ds


#####################################################################
