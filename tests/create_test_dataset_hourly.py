"""Create test dataset for hourly data"""

import xarray as xr

# Import DataParameters from climakitae and initialize
from climakitae.core.data_interface import DataParameters

selections = DataParameters()

# Set selections
selections.timescale = "hourly"
selections.resolution = "45 km"
selections.scenario_historical = []
selections.scenario_ssp = ["SSP 3-7.0"]
selections.data_type = "Gridded"
selections.downscaling_method = "Dynamical"
selections.time_slice = (2015, 2015)
selections.area_subset = "CA counties"
selections.cached_area = ["Los Angeles County"]

# Get air temp in K
selections.variable = "Air Temperature at 2m"
units = "K"
T_da = selections.retrieve()

# Load mixing ratio data
selections.variable = "Water Vapor Mixing Ratio at 2m"
selections.units = "kg kg-1"
q2_da = selections.retrieve()

# Load air pressure data
selections.variable = "Surface Pressure"
selections.units = "Pa"
pressure_da = selections.retrieve()

# Load u10 data
selections.variable = "West-East component of Wind at 10m"
selections.units = "m s-1"
u10_da = selections.retrieve()

# Load v10 data
selections.variable = "North-South component of Wind at 10m"
selections.units = "m s-1"
v10_da = selections.retrieve()

# Merge to form one dataset
ds = xr.merge([T_da, q2_da, pressure_da, u10_da, v10_da])

# Subset time
ds_jan01 = ds.sel(time="Jan 01 2015")

# Output to netcdf
ds_jan01.to_netcdf("test_data/test_dataset_01Jan2015_LAcounty_45km_hourly.nc")
