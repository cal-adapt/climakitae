from climakitae.util.utils import get_closest_gridcell, stack_sims_across_locs
from climakitae.core.data_load import load
import time
import xarray as xr
import pyproj


def batch_select(approach, selections, points, load_data=False, progress_bar=True):
    """
    Conducts batch mode analysis on a series of points for a given metric.

    Parameters
    ----------
    selections: `Select` object
        Selections object that describes the area of interest. The `area_subset` and `cached_area` attributes are automatically overwritten.
    points: np.array
        An array at lat/lon points to gather the specified data at.
    load_data: Boolean
        A boolean that tells the function whether or not to load the data into memory.

    Returns
    -------
    cells_of_interest: xr.DataArray of the gridcells that the points lie within, aggregated together into one DataArray. It can or cannot be loaded into memory, depending on `load_data`.
    """
    # Add selections attributes to cover the entire domain since we don't know exactly where the selected points lie.
    selections.area_subset = "none"
    selections.cached_area = ["entire domain"]
    data = selections.retrieve()

    if approach == "Time":
        # Remove leap days, if applicable
        data = data.sel(time=~((data.time.dt.month == 2) & (data.time.dt.day == 29)))

    # Find the closest gridcells for each of the passed in points and concatenate them on a new 'points' dimension to go from 2D grid to 1D series of points
    # da_points = get_closest_gridcell(data, points[:, 0], points[:, 1])
    lat_lon_to_model_projection = pyproj.Transformer.from_crs(
        crs_from="epsg:4326",  # Lat/lon
        crs_to=data.rio.crs,  # Model projection
        always_xy=True,
    )
    xs, ys = lat_lon_to_model_projection.transform(points[:,1], points[:,0])
    x_idx = data.x.to_index().get_indexer(xs, method='nearest')
    y_idx = data.x.to_index().get_indexer(ys, method='nearest')

    da_points = data.isel(x=xr.DataArray(x_idx, dims='points'), y=xr.DataArray(y_idx, dims='points'))
    
    # Load in the cells of interest into memory, if desired.
    if load_data:
        t3 = time.time()
        cells_of_interest = load(da_points, progress_bar=progress_bar)
        t4 = time.time()
        print(f"Total time to load: {t4 - t3} seconds")

    return cells_of_interest
