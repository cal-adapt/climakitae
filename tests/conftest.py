import os
import pytest
import xarray as xr

@pytest.fixture
def rootdir():
    """Add path to test data as fixture. """
    return os.path.dirname(os.path.abspath("tests/test_data"))

@pytest.fixture
def test_data(rootdir): 
    """ Read in test dataset using xarray. """
    filename = "test_data/test_dataset_2022_2022_monthly_45km.nc"
    filepath = os.path.join(rootdir, filename)
    ds = xr.open_dataset(filepath)  
    return ds