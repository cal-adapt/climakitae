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

# ==============================================================================

def get_geospatial_plot(
    ds,
    data_variable,
    bar_min = None,
    bar_max = None,
    border_color = "black",
    line_width = 0.5,
    cmap = "Wistia",
    hover_fill_color = "blue",
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

    variable_name = data_variable.replace("_", " ").replace("'", "").title()

    distr_name = ds.attrs["distribution"].replace("'", "").title()

    borders = gv.Path(gv.feature.states.geoms(scale = "50m", as_element = False)).opts(
        color = border_color, line_width = line_width
    ) * gv.feature.coastline.geoms(scale = "50m").opts(
        color = border_color, line_width = line_width
    )

    if data_variable in ["d_statistic", "p_value"]:
        attribute_name = (
            (ds[data_variable].attrs["stat test"])
            .replace("{", "")
            .replace("}", "")
            .replace("'", "")
            .title()
        )

    if data_variable in ["return_value"]:
        attribute_name = (
            (ds[data_variable].attrs["return period"])
            .replace("{", "")
            .replace("}", "")
            .replace("'", "")
            .title()
        )

    if data_variable in ["return_prob"]:
        attribute_name = (
            (ds[data_variable].attrs["threshold"])
            .replace("{", "")
            .replace("}", "")
            .replace("'", "")
            .title()
        )

    if data_variable in ["return_period"]:
        attribute_name = (
            (ds[data_variable].attrs["return value"])
            .replace("{", "")
            .replace("}", "")
            .replace("'", "")
            .title()
        )

    geospatial_plot = (
        ds.hvplot.quadmesh(
            "lon",
            "lat",
            data_variable,
            clim = (bar_min, bar_max),
            projection = ccrs.PlateCarree(),
            ylim = (30, 50),
            xlim = (-130, -100),
            title = "{} For A {} ({} Distribution)".format(
                variable_name, attribute_name, distr_name
            ),
            cmap = cmap,
            hover_fill_color = hover_fill_color,
        )
        * borders
    )

    return geospatial_plot
