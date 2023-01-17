########################################
#                                      #
# VISUALIZE                            #
#                                      #
########################################

import numpy as np
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt
from matplotlib import cm
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import geoviews as gv
import holoviews as hv
from holoviews import opts
import hvplot.pandas
import hvplot.xarray


def rename_distr_abbrev(distr):
    '''Makes abbreviated distribution name human-readable''' 
    distr_abbrev = ["gev", "gumbel", "weibull", "pearson3", "genpareto"]
    distr_readable = \
        ['GEV', 'Gumbel', 'Weibull', 'Pearson Type III', 'Generalized Pareto']
    return distr_readable[distr_abbrev.index(distr)]


def get_geospatial_plot(
    ds,
    data_variable,
    bar_min=None,
    bar_max=None,
    border_color="black",
    line_width=0.5,
    cmap="Wistia",
    hover_fill_color="blue",
):
    """
    Returns a geospatial plot (hvplot) from inputed dataset and selected data variable.
    """

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

    if data_variable == 'p_value':
        variable_name = 'p-value'
    else:
        variable_name = data_variable.replace("_", " ").replace("'", "")

    distr_name = rename_distr_abbrev(ds.attrs["distribution"])

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
    variable_unit = ds[data_variable].attrs['units']
    if variable_unit:
        cmap_label = ' '.join([cmap_label, '({})'.format(variable_unit)])

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
