from climakitae.util.utils import get_closest_gridcell, stack_sims_across_locs
from climakitae.core.data_load import load
import xarray as xr


def batch_select(selections, points, load_data=False, progress_bar=True):
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

    def _retrieve_pts(data, points):
        """Retrieving all individual points within the entire domain of data pulled."""
        data_pts = []
        for point in points:
            lat, lon = point
            closest_cell = get_closest_gridcell(data, lat, lon, print_coords=False)
            stacked_data = stack_sims_across_locs(closest_cell)
            data_pts.append(closest_cell)
        return data_pts

    print(f"Batch retrieving all {len(points)} points passed in...\n")

    # Add selections attributes to cover the entire domain since we don't know exactly where the selected points lie.
    selections.area_subset = "none"
    selections.cached_area = ["entire domain"]

    print("we are here")

    data = selections.retrieve()
    print("this is done")

    data_pts = _retrieve_pts(data, points)

    # Combine data points into a single xr.Dataset
    cells_of_interest = xr.concat(data_pts, dim="simulation").chunk(chunks="auto")

    # Load in the cells of interest into memory, if desired.
    if load_data:
        cells_of_interest = load(cells_of_interest, progress_bar=progress_bar)

    return cells_of_interest


def batch_analysis(sims, metric):
    """
    Runs an analysis against a loaded set of simulations.
    """
    return metric(sims)
