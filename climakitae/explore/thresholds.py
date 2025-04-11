import xarray as xr
from climakitae.core.data_interface import DataParameters
from climakitae.core.data_load import read_catalog_from_select


def get_threshold_data(selections: DataParameters) -> xr.DataArray:
    """
    This function pulls data from the catalog and reads it into memory

    Arguments
    ---------
    selections: DataParameters
        object holding user's selections

    Returns
    -------
    data: xr.DataArray
        data to use for creating postage stamp data
    """
    # Read data from catalog
    data = read_catalog_from_select(selections)
    data = data.compute()  # Read into memory
    return data
