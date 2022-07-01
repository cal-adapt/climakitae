import xarray as xr
import numpy as np
import pytest
import os
from climakitae.derive_variables import _compute_total_precip


@pytest.fixture
def test_data(rootdir): 
    """ Read in test dataset using xarray. """
    filename = "test_data/test_dataset_2022_2022_monthly_45km.nc"
    filepath = os.path.join(rootdir, filename)
    ds = xr.open_dataset(filepath)  
    return ds

@pytest.fixture
def xr_negative(test_data): 
    """Create an xr.Dataset with the same shape as test_data, but replace all values with negative fill value. """
    fill_value = -100
    xr_ds = xr.full_like(test_data, fill_value=fill_value, dtype=float)
    return xr_ds

def test_precip_return_type(test_data): 
    """Ensure function returns an xr.DataArray object. """
    total_precip = _compute_total_precip(cumulus_precip=test_data["RAINC"], gridcell_precip=test_data["RAINNC"])
    assert type(total_precip) == xr.core.dataarray.DataArray
    
#def test_precip_check_for_negatives(test_data, xr_negative): 
#    """Ensure function raises ValueError if output is negative. """
#    with pytest.raises(ValueError) as e_info:
#        tot_precip = _compute_total_precip(cumulus_precip=test_data["RAINC"], gridcell_precip=xr_negative["RAINNC"])
    
def test_precip_check_expected_value(test_data): 
    """Ensure function returns expected value. """
    test_data_subset = test_data.sel(simulation="cesm2", scenario="SSP 2-4.5 -- Middle of the Road", time="2022-01-01") # Get small subset of testing data 
    total_precip = _compute_total_precip(cumulus_precip=test_data_subset["RAINC"], gridcell_precip=test_data_subset["RAINNC"])
    expected_value = test_data_subset["RAINC"] + test_data_subset["RAINNC"] # Value you expect the function to return      
    assert total_precip.equals(expected_value)
    
    
