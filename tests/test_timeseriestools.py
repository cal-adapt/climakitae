import climakitae as ck
from climakitae import timeseriestools as tst
import pytest
import xarray as xr

#------------------ Read in the test dataset ----------------------------------

# This data is generated in "create_timeseries_test_data.py"
test_filepath = "tests/test_data/timeseries_data_T2_2014_2016_monthly_45km.nc"
y = xr.open_dataarray(test_filepath)

#------------------ Build TimeseriesParam object ------------------------------

# Test running mean without anomaly

ts = tst.Timeseries(y) 
tsp = ts.choices
tsp.smoothing = "running mean"
tsp.num_timesteps = 3
tsp.anomaly = False 

tr_data = tsp.transform_data() # transform_data calls _running_mean()
tr_data
Exception("Still need to manually re-center the rolling average!")




# # def test_running_mean():
# #     """

# #     """
# #     _expected_rolling_avg = []

# #     my_data = xr.open_dataset("test_data/test_dataset_2015_monthly_45km.nc")

# #     tsp = tst.TimeSeriesParams(my_data)

# #     tsp.transform_data() # transform_data calls _running_mean()
# #     assert tsp.output_current() == _expected_rolling_avg

