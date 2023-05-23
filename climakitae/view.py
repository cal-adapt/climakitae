"""Backend function for creating generic visualizations of xarray DataArray."""

import warnings
import xarray as xr
import numpy as np
import pandas as pd
import hvplot.xarray
import matplotlib.pyplot as plt
import pkg_resources
from .utils import _reproject_data, _read_ae_colormap

# Import package data
var_catalog_resource = pkg_resources.resource_filename(
    "climakitae", "data/variable_descriptions.csv"
)
var_catalog = pd.read_csv(var_catalog_resource, index_col=None)


def _compute_vmin_vmax(da_min, da_max):
    """Compute min, max, and center for plotting"""
    vmin = np.nanpercentile(da_min, 1)
    vmax = np.nanpercentile(da_max, 99)
    # define center for diverging symmetric data
    if (vmin < 0) and (vmax > 0):
        # dabs = abs(vmax) - abs(vmin)
        sopt = True
    else:
        sopt = None
    return vmin, vmax, sopt


def _visualize(data, lat_lon=True, width=None, height=None, cmap=None):
    """Create a generic visualization of the data

    Visualization will depend on the shape of the input data.
    Works much faster if the data has already been loaded into memory.

    Parameters
    ----------
    data: xr.DataArray
        Input data
    lat_lon: bool, optional
        Reproject to lat/lon coords?
        Default to True.
    width: int, optional
        Width of plot
        Default to hvplot default
    height: int, optional
        Height of plot
        Default to hvplot.image default
    cmap: matplotlib colormap name or AE colormap names
        Colormap to apply to data
        Default to "ae_orange" for mapped data or color-blind friendly "categorical_cb" for timeseries data.

    Returns
    -------
    holoviews.core.spaces.DynamicMap
        Interactive map or lineplot
    matplotlib.figure.Figure
        xarray default map
        Only produced if gridded data doesn't have sufficient cells for hvplot

    Raises
    ------
    UserWarning
        Warn user that the function will be slow if data has not been loaded into memory
    """

    # Warn user about speed if passing a zarr to the function
    if data.chunks is None or str(data.chunks) == "Frozen({})":
        pass
    else:
        warnings.warn(
            "This function may be quite slow unless you call .compute() on your data before passing it to app.view()"
        )

    # Workflow if data contains spatial coordinates
    if set(["x", "y"]).issubset(set(data.dims)) or set(["lon", "lat"]).issubset(
        set(data.dims)
    ):
        # If simulation is a dimension, make it so the colorbar plots the min and max across the simulations
        # Such that the colorbar is standardized
        vmin = None
        vmax = None
        sopt = None
        if "simulation" in data.dims:
            # But, only do this if the data is already read into memory
            # Or else the computation of min and max will take forever
            if data.chunks is None or str(data.chunks) == "Frozen({})":
                min_data = data.min(dim="simulation")
                max_data = data.max(dim="simulation")
                vmin, vmax, sopt = _compute_vmin_vmax(min_data, max_data)

        # Set default cmap if no user input
        if cmap is None:
            try:
                if data.frequency in ["monthly", "daily"]:
                    timescale = "daily/monthly"
                else:
                    timescale = data.frequency
                cmap = var_catalog[
                    (var_catalog["display_name"] == data.name)
                    # & (var_catalog["timescale"] == timescale)
                ].colormap.values[0]
            except:  # If variable not found, set to ae_orange without raising error
                cmap = "ae_orange"

        # Must have more than one grid cell to generate a map
        if (len(data["lon"]) <= 1) or (len(data["lat"]) <= 1):
            print(
                "Your data contains only one grid cell in height and/or width. A plot will be created using a default method that may or may not have spatial coordinates as the x and y axes."
            )  # Warn user that plot may be weird

            # Set default cmap if no user input
            # Different if using matplotlib (no "hex")
            if cmap in [
                "categorical_cb",
                "ae_orange",
                "ae_diverging",
                "ae_blue",
                "ae_diverging_r",
            ]:
                cmap = _read_ae_colormap(cmap=cmap, cmap_hex=False)

            with warnings.catch_warnings():
                # Silence annoying matplotlib deprecation error
                warnings.simplefilter("ignore")

                # Use generic static xarray plot
                try:
                    _matplotlib_plot = data.isel(time=0).plot(cmap=cmap)
                except:
                    _matplotlib_plot = data.isel(
                        time=0
                    ).plot()  # Make histogram for data the plotting function doesn't know how to handle
                _plot = plt.gcf()  # Add plot to figure
                plt.close()  # Close to prevent annoying matplotlib collections object line from showing in notebook
        # If there's more than one grid cell, generate a pretty map
        else:
            # Define colorbar label using variable and units
            try:
                clabel = data.name + " (" + data.attrs["units"] + ")"
            except:  # Try except just in case units attribute is missing from data
                clabel = data.name

            # Set default cmap if no user input
            # Different if using hvplot (we need "hex")
            if cmap in [
                "categorical_cb",
                "ae_orange",
                "ae_diverging",
                "ae_blue",
                "ae_diverging_r",
            ]:
                cmap = _read_ae_colormap(cmap=cmap, cmap_hex=True)

            # Set default width & height
            if width is None:
                width = 550
            if height is None:
                height = 450

            if set(["x", "y"]).issubset(set(data.dims)):
                # Reproject data to lat/lon
                if lat_lon == True:
                    try:
                        data = _reproject_data(
                            xr_da=data, proj="EPSG:4326", fill_value=np.nan
                        )
                    except:  # Reprojection can fail if the data doesn't have a crs element. If that happens, just carry on without projection (i.e. don't raise an error)
                        pass

                # Create map
                _plot = data.hvplot.image(
                    x="x",
                    y="y",
                    grid=True,
                    clabel=clabel,
                    cmap=cmap,
                    width=width,
                    height=height,
                    clim=(vmin, vmax),
                    sopt=sopt,
                )
            else:
                # Create map
                _plot = data.hvplot.image(
                    x="lon",
                    y="lat",
                    grid=True,
                    clabel=clabel,
                    cmap=cmap,
                    width=width,
                    height=height,
                    clim=(vmin, vmax),
                    sopt=sopt,
                )

    # Workflow if data contains only time dimension
    elif "time" in data.dims:
        # Default colormap for timeseries data
        if cmap is None:
            cmap = "categorical_cb"
            cmap = _read_ae_colormap(cmap=cmap, cmap_hex=True)

        # Set default width & height
        if width is None:
            width = 600
        if height is None:
            height = 300

        # Create lineplot
        _plot = data.hvplot.line(x="time", width=width, height=height, color=cmap)

    # Error raised if data does not contain [x,y] or time dimensions
    else:
        raise ValueError(
            "Input data must contain valid spatial dimensions (x,y and/or lat,lon) and/or time dimensions"
        )
    return _plot
