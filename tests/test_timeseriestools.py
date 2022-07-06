import climakitae as ck
from climakitae import timeseriestools as tst
import datetime as dt
import pytest
import xarray as xr

#------------------ Read in the test dataset ----------------------------------

# This data is generated in "create_timeseries_test_data.py"
test_filepath = "tests/test_data/timeseries_data_T2_2014_2016_monthly_45km.nc"
y = xr.open_dataarray(test_filepath)

#------------------ Test running mean without anomaly ------------------------------

ts = tst.Timeseries(y) 
tsp = ts.choices
tsp.smoothing = "running mean"
tsp.num_timesteps = 3
tsp.anomaly = False 

tr_data = tsp.transform_data() # transform_data calls _running_mean()
tr_data
Exception("Still need to manually re-center the rolling average!")


#------------------ Test monthly weighted anomaly ------------------------------

ts = tst.Timeseries(y) 
tsp = ts.choices
tsp.anomaly = True
tsp.reference_range = (dt.datetime(2014, 1, 1), dt.datetime(2014, 12, 31)) 

tr_data = tsp.transform_data() # transform_data calls _running_mean()
tr_data
