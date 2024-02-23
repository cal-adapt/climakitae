from climakitae.core.data_interface import Select
from climakitae.util.utils import get_closest_gridcell
import xarray as xr


def batch_select(points):
    """
    Conducts batch mode analysis on a series of points for a given metric.
    """
    # Retrieve data for the entire service area
    selections = Select()
    selections.data_type = "Gridded"
    selections.area_subset = "none"
    selections.cached_area = ["entire domain"]
    selections.timescale = "hourly"
    selections.variable_type = "Derived Index"
    selections.variable = "NOAA Heat Index"
    selections.resolution = "9 km"
    selections.time_slice = (1981, 2010)
    data = selections.retrieve()

    # Find the closest gridcell for each lat, lon pair in a series of points
    data_pts = []
    for point in points:
        lat, lon = point
        closest_cell = get_closest_gridcell(
            data, lat, lon, print_coords=False
        ).squeeze()
        closest_cell["simulation"] = [
            "{}_{}_{}".format(
                sim_name, closest_cell.lat.item(), closest_cell.lon.item()
            )
            for sim_name in closest_cell.simulation
        ]
        data_pts.append(closest_cell)

    # Combine data points into a single xr.Dataset
    cells_of_interest = xr.concat(data_pts, dim="simulation").chunk(
        (8, len(closest_cell.time))
    )

    # Load in the cells of interest
    loaded_vals = cells_of_interest.compute()

    return loaded_vals


def batch_analysis(sims, metric):
    """
    Runs an analysis against a loaded set of simulations.
    """
    return metric(sims)
