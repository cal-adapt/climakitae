import numpy as np
import xarray as xr

from climakitae.core.data_interface import DataParameters
from climakitae.core.data_load import load
from climakitae.util.utils import get_closest_gridcells


def batch_select(
    approach: str,
    selections: DataParameters,
    points: np.ndarray,
    load_data: bool = False,
    progress_bar: bool = True,
) -> xr.DataArray:
    """Conducts batch mode analysis on a series of points for a given metric.

    Parameters
    ----------
    approach : str
    selections : DataParameters
        Selections object that describes the area of interest. The `area_subset` and `cached_area` attributes are automatically overwritten.
    points : np.ndarray
        An array at lat/lon points to gather the specified data at.
    load_data : boolean
        A boolean that tells the function whether or not to load the data into memory.
    progress_bar : boolean
        A boolean that determines whether progress bar is displayed.

    Returns
    -------
    cells_of_interest: xr.DataArray
        Gridcells that the points lie within, aggregated together into one DataArray. It can or cannot be loaded into memory, depending on `load_data`.

    """
    print(f"Batch retrieving all {len(points)} points passed in...\n")

    # Add selections attributes to cover the entire domain since we don't know exactly where the selected points lie.
    selections.area_subset = "none"
    selections.cached_area = ["entire domain"]

    data = selections.retrieve()

    if approach == "Time":
        # Remove leap days, if applicable
        data = data.sel(time=~((data.time.dt.month == 2) & (data.time.dt.day == 29)))

    # Find the closest gridcells for each of the passed in points and concatenate them on a new 'points' dimension to go from 2D grid to 1D series of points
    cells_of_interest = get_closest_gridcells(data, points[:, 0], points[:, 1])

    # Load in the cells of interest into memory, if desired.
    if load_data:
        cells_of_interest = load(cells_of_interest, progress_bar=progress_bar)

    return cells_of_interest
