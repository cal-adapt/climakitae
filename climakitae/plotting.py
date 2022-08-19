import cartopy.crs as ccrs
import hvplot.xarray
import hvplot.pandas

def plot(to_plot, overlay=None, overlay_dims=None):
    """
    Function for plotting climate data, with optional overlay point data.

    Arguments:
    to_plot -- xarray DataArray
    overlay -- optional point data, pandas dataframe 
    overlay_dims -- tuple indicating the names of the longitude and lattitude
        variables in the overlay data. May be ("x", "y"), for example. Will 
        default to ("lon", "lat") if None provided.
    """

    _groupby = [d for d in ['time','scenario','simulation'] if d in to_plot.dims]

    # Make the gridded variable map
    quadmesh_plot = to_plot.hvplot.quadmesh('lon', 'lat', 
        groupby=_groupby, 
        crs=ccrs.PlateCarree(),
        projection=ccrs.Orthographic(-118, 40),
        project=True,
        rasterize=True,
        coastline=True, 
        features=['borders'])

    # Add overlay points if provided
    if overlay is not None:
        overlay_dims = ("lon", "lat") if overlay_dims is None else overlay_dims
        overlay_plot = overlay.hvplot.scatter(*overlay_dims, color="black")
        return quadmesh_plot * overlay_plot
    else:
        return quadmesh_plot