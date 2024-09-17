import pandas as pd
from climakitae.core.data_load import read_catalog_from_select


def get_threshold_data(self):
    """
    This function pulls data from the catalog and reads it into memory

    Arguments
    ---------
    selections: DataParameters
        object holding user's selections
    cat: intake_esm.core.esm_datastore
        data catalog

    Returns
    -------
    data: xr.DataArray
        data to use for creating postage stamp data
    """
    # Read data from catalog
    data = read_catalog_from_select(self)
    data = data.compute()  # Read into memory
    return data
