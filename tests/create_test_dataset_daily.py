"""Create test dataset for daily data"""

# Import climakitae and initialize Application object
import climakitae as ck

app = ck.Application()

# Set app.selections
app.selections.timescale = "daily"
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
T = app.retrieve()

# Subset time
T_jan = T.sel(time="Jan 2015")

# Output to netcdf
T_jan.to_netcdf("test_data/test_dataset_Jan2015_LAcounty_45km_daily.nc")
