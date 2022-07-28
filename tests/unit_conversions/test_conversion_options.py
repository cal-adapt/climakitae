""" This script runs tests that unit conversions module contains conversion options that match those in the csv file. 
This test is designed to catch a specific error: If someone adds an alternative unit option to the variable_descriptions.csv file, but a conversion for that unit does not exist in the _convert_units function. """

import pytest 
import pandas as pd
import numpy as np 
import xarray as xr
from climakitae.unit_conversions import _convert_units
from climakitae.utils import _read_var_csv
import pkg_resources
CSV_FILE = pkg_resources.resource_filename('climakitae', 'data/variable_descriptions.csv')


def unit_combos(): 
    """Get combination of native units and alternate unit options in the proper format for a pytest. 
    The output looks something like this: 
    [('K', 'K'),
     ('K', 'degC,'),
     ('K', 'degF,'),
     ('K', 'degR'),
     ('mm', 'mm'),
     ('mm', 'inches'),
     ('m/s', 'm/s'),
     ('m/s', 'knots'),
     ('Pa', 'Pa'),
     ('Pa', 'mb,'),
     ('Pa', 'hPa,'),
     ('Pa', 'inHg'),
     ('kg/kg', 'kg/kg'),
     ('kg/kg', 'g/kg')]
    """
    pd_csv = pd.read_csv(CSV_FILE, usecols=["native_unit","alt_unit_options"])
    pd_conversions = pd_csv.drop_duplicates().dropna()
    native_units = pd_conversions["native_unit"].values
    alt_units = pd_conversions["alt_unit_options"].values
    alt_units = [x.split(', ') for x in alt_units]
    l_iter = [(native_units[i], alt_units[i][x]) for i in range(len(native_units)) for x in range(len(alt_units[i]))]
    for native_unit, selected_unit in l_iter: 
        yield native_unit, selected_unit
        
@pytest.mark.parametrize('native_unit, selected_unit', unit_combos())
def test_unit_conversion_exists(native_unit, selected_unit):
    """Test that a conversion exists for each (native_unit, selected_unit) conversion pair. """
    dummy_da = xr.DataArray(np.arange(1,10,1)) 
    da_converted = _convert_units(da=dummy_da, native_units=native_unit, selected_units=selected_unit)
    assert da_converted.attrs["units"] == selected_unit, "This is not a valid unit conversion for the native unit"