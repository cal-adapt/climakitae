"""Backend function for creating generic visualizations of xarray DataArray."""

import warnings
import numpy as np
import hvplot.xarray
import matplotlib.pyplot as plt
import panel as pn
from climakitae.util.utils import reproject_data, read_ae_colormap
from climakitae.core.data_interface import DataInterface


def compute_vmin_vmax(da_min, da_max):
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


def view(data, lat_lon=True, width=None, height=None, cmap=None):
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

    variable_descriptions = DataInterface().variable_descriptions

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
                vmin, vmax, sopt = compute_vmin_vmax(min_data, max_data)

        # Set default cmap if no user input
        if cmap is None:
            try:
                if data.frequency in ["monthly", "daily"]:
                    timescale = "daily, monthly"
                else:
                    timescale = data.frequency
                cmap = variable_descriptions[
                    (variable_descriptions["display_name"] == data.name)
                    # & (variable_descriptions["timescale"] == timescale)
                ].colormap.values[0]
            except:  # If variable not found, set to ae_orange without raising error
                cmap = "ae_orange"

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
            cmap = read_ae_colormap(cmap=cmap, cmap_hex=True)

        # Set default width & height
        if width is None:
            width = 550
        if height is None:
            height = 450

        if set(["x", "y"]).issubset(set(data.dims)):
            x = "x"
            y = "y"
            # Reproject data to lat/lon
            if lat_lon == True:
                try:
                    data = reproject_data(
                        xr_da=data, proj="EPSG:4326", fill_value=np.nan
                    )
                except:  # Reprojection can fail if the data doesn't have a crs element. If that happens, just carry on without projection (i.e. don't raise an error)
                    pass
        if set(["lat", "lon"]).issubset(set(data.dims)):
            x = "lon"
            y = "lat"

        # Create map
        try:
            if len(data[x]) > 1 and len(data[y]) > 1:
                # If data has more than one grid cell, make a pretty map
                _plot = data.hvplot.image(
                    x=x,
                    y=y,
                    grid=True,
                    clabel=clabel,
                    cmap=cmap,
                    width=width,
                    height=height,
                    clim=(vmin, vmax),
                    sopt=sopt,
                )
            else:
                # Make a scatter plot if it's just one grid cell
                print(
                    "Warning: your input data has 2 or less gridcells. Due to plotting limitations for small areas, a scatter plot will be generated."
                )
                _plot = data.hvplot.scatter(
                    x=x,
                    y=y,
                    hover_cols=data.name,  # Add variable name as hover column
                    grid=True,
                    clabel=clabel,
                    cmap=cmap,
                    width=width,
                    height=height,
                    s=150,  # Size of marker
                    clim=(vmin, vmax),
                    sopt=sopt,
                )
        except:
            # Print message instead of raising error
            print("Unknown error: default map could not be generated for input data.")
            _plot = None

    # Workflow if data contains only time dimension
    elif "time" in data.dims:
        # Default colormap for timeseries data
        if cmap is None:
            cmap = "categorical_cb"
            cmap = read_ae_colormap(cmap=cmap, cmap_hex=True)

        # Set default width & height
        if width is None:
            width = 600
        if height is None:
            height = 300

        # Create lineplot
        _plot = data.hvplot.line(x="time", width=width, height=height, color=cmap)

    # Error raised if data does not contain [x,y] or time dimensions
    else:
        print(
            "Default plot could not be generated: input data must contain valid spatial dimensions (x,y and/or lat,lon) and/or time dimensions"
        )
        _plot = None

    # Put plot object into a panel Pane object
    # _plot_as_pane = pn.Pane(_plot)

    return _plot
