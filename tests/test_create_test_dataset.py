"""
This script contains a test to check that the `_read_data_for_var` function 
from the `create_test_dataset.py` script runs without error.
"""
import intake 

from climakitae.selectors import DataSelector, LocSelectorArea
from climakitae.core import _get_catalog_contents
from create_test_dataset import _read_data_for_var

def test_read_data_for_var():

    _cat = intake.open_catalog("https://cadcat.s3.amazonaws.com/cae.yaml")
    _selections = DataSelector(choices=_get_catalog_contents(_cat))
    _location = LocSelectorArea()

    # Make sure call to _read_data_for_var runs without error
    xr_da = _read_data_for_var(cat=_cat,
                            selections=_selections,
                            location=_location,
                            variable="Air Temperature at 2m", 
                            year_start=2030, 
                            year_end=2031, 
                            timescale="monthly", 
                            append_historical=False, 
                            resolution="45 km", 
                            scenarios=['SSP 2-4.5 -- Middle of the Road'])