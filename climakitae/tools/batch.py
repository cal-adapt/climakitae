from climakitae.util.utils import get_closest_gridcell, stack_sims_across_locs
from climakitae.core.data_load import load
import xarray as xr


def batch_select(selection_params, points, approach, load_data=True, progress_bar=True):
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

    def _retrieve_pts(data, sim_dim_name, points):
        """Retrieving all individual points within the entire domain of data pulled."""
        data_pts = []
        for point in points:
            lat, lon = point
            closest_cell = get_closest_gridcell(data, lat, lon, print_coords=False)
            stacked_data = stack_sims_across_locs(closest_cell, sim_dim_name)
            data_pts.append(stacked_data)
        return data_pts

    print(f"Batch retrieving all {len(points)} points passed in...\n")

    dim_name = "simulation" if approach == "time" else "all_sims"

    # Add selections attributes to cover the entire domain since we don't know exactly where the selected points lie.
    selection_params.area_subset = "none"
    selection_params.cached_area = ["entire domain"]

    if approach == "time":
        data = selection_params.retrieve()
        data_pts = _retrieve_pts(data, dim_name, points)

    elif approach == "warming_level":
        selection_params.calculate()

        # This will only retrieve points for 1 warming level at a time.
        data = selection_params.sliced_data[
            selection_params.wl_params.warming_levels[0]
        ]
        data_pts = _retrieve_pts(data, dim_name, points)

    # Combine data points into a single xr.Dataset
    cells_of_interest = xr.concat(data_pts, dim=dim_name).chunk(chunks="auto")

    # Load in the cells of interest into memory, if desired.
    if load_data:
        cells_of_interest = load(cells_of_interest, progress_bar=progress_bar)

    return cells_of_interest


def batch_analysis(sims, metric):
    """
    Runs an analysis against a loaded set of simulations.
    """
    return metric(sims)
