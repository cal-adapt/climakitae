"""Create test dataset for daily data"""

# Import DataParameters from climakitae and initialize
from climakitae.core.data_interface import DataParameters

selections = DataParameters()

# Set selections
selections.timescale = "daily"
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
T = selections.retrieve()

# Subset time
T_jan = T.sel(time="Jan 2015")

# Output to netcdf
T_jan.to_netcdf("test_data/test_dataset_Jan2015_LAcounty_45km_daily.nc")
