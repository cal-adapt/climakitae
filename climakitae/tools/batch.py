from climakitae.util.utils import get_closest_gridcell
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

    def _retrieve_pts(data, sim_dim_name):
        """Retrieving all individual points within the entire domain of data pulled."""
        data_pts = []
        for point in points:
            lat, lon = point
            closest_cell = get_closest_gridcell(
                data, lat, lon, print_coords=False
            ).squeeze()
            closest_cell[sim_dim_name] = [
                "{}_{}_{}".format(
                    sim_name, closest_cell.lat.item(), closest_cell.lon.item()
                )
                for sim_name in closest_cell[sim_dim_name]
            ]
            data_pts.append(closest_cell)
        return data_pts

    # Add selections attributes to cover the entire domain since we don't know exactly where the selected points lie.
    selection_params.area_subset = "none"
    selection_params.cached_area = ["entire domain"]

    if approach == "time":
        data = selection_params.retrieve()
        data_pts = _retrieve_pts(data, "simulation")
        time_length = data_pts[0].time.size

    elif approach == "warming_level":
        data = selection_params.calculate()
        for wl in data.keys():
            data_pts = _retrieve_pts(data[wl], "all_sims")

        # Find the WL time dim name
        wl_time_dim = [dim for dim in data_pts[0].dims if "from_center" in dim][0]
        time_length = data_pts[0][wl_time_dim].size

    # Combine data points into a single xr.Dataset
    cells_of_interest = xr.concat(data_pts, dim="simulation").chunk((1, time_length))

    # Load in the cells of interest into memory, if desired.
    if load_data:
        cells_of_interest = load(cells_of_interest, progress_bar=progress_bar)

    return cells_of_interest


def batch_analysis(sims, metric):
    """
    Runs an analysis against a loaded set of simulations.
    """
    return metric(sims)
