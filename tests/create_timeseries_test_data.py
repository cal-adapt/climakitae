from climakitae.data_loaders import _read_from_catalog
from climakitae.selectors import DataSelector, LocSelectorArea


#------------- Retrieve small test dataset ------------------------------------
# For Timeseries tools, dataset must have appended_historical=True

year_start=2014
year_end=2016
timescale="monthly"
resolution="45 km"
variable="T2"

selections = DataSelector(
    append_historical=True, 
    area_average=False, 
    name='DataSelector00101', 
    resolution=resolution, 
    scenario=['SSP 2-4.5 -- Middle of the Road', 'Historical Climate'], # 'Historical Climate' cannnot be listed first in the scenario list, causes an error in data_loaders
    time_slice=(year_start, year_end), 
    timescale=timescale, 
    variable=variable)
location = LocSelectorArea(
    area_subset='none',
    cached_area='CA',
    latitude=(32.5, 42), 
    longitude=(-125.5, -114), 
    name='LocSelectorArea00102')

xr_da = _read_from_catalog(selections=selections, location=location)
 
#------------- Manual location subset -----------------------------------------

# Coordinates of San Diego to Joshua Tree area
lon_0 = min_lon = -117.1611 # San Diego 
lat_0 = min_lat = 32.7157
lon_1 = max_lon = -115.9010 # Joshua Tree  
lat_1 = max_lat = 33.8734

# Crop the data 
mask_lon = (xr_da.lon >= min_lon) & (xr_da.lon <= max_lon)
mask_lat = (xr_da.lat >= min_lat) & (xr_da.lat <= max_lat)
test_data_precompute = xr_da.where(mask_lon & mask_lat, drop=True)

# Compute
test_data = test_data_precompute.compute()

# Save the file
filename = "timeseries_data_"+variable+"_"+str(year_start)+"_"+str(year_end)+"_"+timescale+"_"+resolution # Leave off .nc !
filename = filename.replace(" ", "")
filepath = "tests/test_data/"+filename+".nc" # Path to file 
test_data.to_netcdf(path=filepath, mode='w') # Output 