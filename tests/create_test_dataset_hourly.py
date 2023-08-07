"""Create test dataset for hourly data"""

# Import climakitae and initialize Application object
import climakitae as ck
import xarray as xr

app = ck.Application()

# Set app.selections
app.selections.timescale = "hourly"
app.selections.resolution = "45 km"
app.selections.scenario_historical = []
app.selections.scenario_ssp = ["SSP 3-7.0 -- Business as Usual"]
app.selections.data_type = "Gridded"
app.selections.downscaling_method = ["Dynamical"]
app.selections.time_slice = (2015, 2015)
app.selections.area_subset = "CA counties"
app.selections.cached_area = "Los Angeles County"

# Get air temp in K
app.selections.variable = "Air Temperature at 2m"
app.units = "K"
T_da = app.retrieve()

# Load mixing ratio data
app.selections.variable = "Water Vapor Mixing Ratio at 2m"
app.selections.units = "kg kg-1"
q2_da = app.retrieve()

# Load air pressure data
app.selections.variable = "Surface Pressure"
app.selections.units = "Pa"
pressure_da = app.retrieve()

# Load u10 data
app.selections.variable = "West-East component of Wind at 10m"
app.selections.units = "m s-1"
u10_da = app.retrieve()

# Load v10 data
app.selections.variable = "North-South component of Wind at 10m"
app.selections.units = "m s-1"
v10_da = app.retrieve()

# Merge to form one dataset
ds = xr.merge([T_da, q2_da, pressure_da, u10_da, v10_da])

# Subset time
ds_jan01 = ds.sel(time="Jan 01 2015")

# Output to netcdf
ds_jan01.to_netcdf("test_data/test_dataset_01Jan2015_LAcounty_45km_hourly.nc")
