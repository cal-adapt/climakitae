"""create_test_dataset.py

This script uses functions from the data_loaders, core, and selectors modules to construct small datasets for testing.
Depending on the number of variables you use to construct your dataset and the geographic subset you want, this script can take a while to run.

"""

# Install dependencies
import sys
import xarray as xr
from climakitae.core.data_load import read_catalog_from_select
from climakitae.core.data_interface import DataParameters
from climakitae.core.constants import SSPS


# ----------------- CHOOSE SETTINGS FOR TEST DATASET -----------------

# These settings will indicate what data you want to read in from the AWS catalog.
# The variables will be read in individually using the read_from_catalog function, which returns an xarray DataArray. At the end, we will combine the individual DataArrays to form a single xarray Dataset object.

# ---- Settings to generate testing file test_dataset_2022_2022_monthly_45km
# variable_list = ['Air Temperature at 2m', 'Precipitation (total)', 'Relative Humidity', 'Wind Magnitude at 10m', 'West-East component of Wind at 10m', 'North-South component of Wind at 10m', 'Surface Pressure', '2m Water Vapor Mixing Ratio', 'Surface skin temperature', 'Shortwave surface downward diffuse irradiance', 'Instantaneous downwelling longwave flux at bottom', 'Instantaneous downwelling clear sky longwave flux at bottom', 'Instantaneous upwelling longwave flux at bottom', 'Instantaneous upwelling clear sky longwave flux at bottom', 'Instantaneous downwelling shortwave flux at bottom', 'Instantaneous downwelling clear sky shortwave flux at bottom', 'Instantaneous upwelling shortwave flux at bottom', 'Instantaneous upwelling clear sky shortwave flux at bottom', 'Snowfall (snow and ice)', 'Precipitation (cumulus portion only)', 'Precipitation (grid-scale portion only)']
# year_start = 2022 # Starting year for time slice (integer)
# year_end = 2022 # Ending year for time slice (integer)
# timescale = "monthly" # Timescale (string): hourly, daily, or monthly
# resolution = "45 km" # Resolution (string): 3 km, 9 km , or 45 km
# append_historical = False # Append historical data? (boolean) year_start must be < 2015
# area_average = False
# scenarios = ['SSP 2-4.5', 'SSP 3-7.0', 'SSP 5-8.5']
# filename = None

# ---- Settings to generate testing file timeseries_data_T2_2014_2016_monthly_45km.nc
# variable_list = ["Air Temperature at 2m"]
# year_start = 2014
# year_end = 2016
# timescale = "monthly"
# resolution = "45 km"
# append_historical = True
# area_average = False
# scenarios = ['SSP 2-4.5', 'Historical Climate']
# filename = "timeseries_data_T2_2014_2016_monthly_45km"

# ---- Settings to generate testing file threshold_data_T2_2050_2051_hourly_45km.nc
variable_list = ["Air Temperature at 2m"]
year_start = 2050
year_end = 2051
timescale = "hourly"
resolution = "45 km"
append_historical = False
area_average = "Yes"
scenarios = ["SSP 2-4.5"]
filename = "threshold_data_T2_2050_2051_hourly_45km"


# -----------------  READ IN DATA FROM AWS CATALOG -----------------


def _read_data_for_var(
    selections,
    variable="Air Temperature at 2m",
    year_start=2013,
    year_end=2016,
    append_historical=True,
    timescale="monthly",
    resolution="45 km",
    scenarios=SSPS,
):
    """Read data from catalog for a given variable."""

    selections.append_historical = append_historical
    selections.area_average = "No"
    selections.resolutions = resolution
    selections.scenario = scenarios
    selections.time_slice = (year_start, year_end)
    selections.timescale = timescale
    selections.variable = variable

    xr_da = read_catalog_from_select(selections=selections)
    return xr_da


def progressBar(i, tot):
    """Display a progress bar inside a for loop
    Based on Stack Overflow answer from Mark Rushakoff: https://stackoverflow.com/questions/3002085/how-to-print-out-status-bar-and-percentage

    Args:
        i (int): iteration number
        tot (int): total number of iterations
    """
    j = (i + 1) / tot
    sys.stdout.write("\r[%-20s] %d%% complete" % ("=" * int(20 * j), 100 * j))
    sys.stdout.flush()


# Print settings
print(
    "Reading in data from AWS for {0} variables: {1}".format(
        len(variable_list), ", ".join(map(str, variable_list))
    )
)
print("Year start: {0}".format(year_start))
print("Year end: {0}".format(year_end))
print("Timescale: {0}".format(timescale))
print("Append historical data: {0}".format(append_historical))
print("Resolution: {0}".format(resolution))
print("Scenarios: {0}".format(", ".join(map(str, scenarios))))

# Get catalog, DataSelector, and LocSelectorArea
_selections = DataParameters()


# Read in each variable individually into an xr.DataArray
xr_da_list = []
for i in range(len(variable_list)):
    variable = variable_list[i]
    xr_da = _read_data_for_var(
        selections=_selections,
        variable=variable,
        year_start=year_start,
        year_end=year_end,
        timescale=timescale,
        append_historical=append_historical,
        resolution=resolution,
        scenarios=scenarios,
    )
    xr_da_list.append(xr_da)
    progressBar(i, len(variable_list))  # Display progress bar


# Merge to form a single xr.Dataset with several data variables
xr_ds = xr.merge(xr_da_list)


# ----------------- SUBSET DATA TO A SMALLER GEOGRAPHIC REGION -----------------

# To avoid having a huge dataset, we just want to use a geographic subset.
# At the time of writing this script (June 20, 2022), the subsetting function built into climakitae was broken. # So, instead we just subset the data using xarray. To do this, we construct a simple lat/lon bounding box of any region we're interested, and filter the data from the dataset using .where()
# This section constructs a box between San Diego, CA and Joshua Tree, CA

# San Diego coordinates
lon_0 = min_lon = -117.1611
lat_0 = min_lat = 32.7157

# Joshua Tree coordinates
lon_1 = max_lon = -115.9010
lat_1 = max_lat = 33.8734

# Crop the data
print("\nCropping data...", end="")
mask_lon = (xr_ds.lon >= min_lon) & (xr_ds.lon <= max_lon)
mask_lat = (xr_ds.lat >= min_lat) & (xr_ds.lat <= max_lat)
test_dataset = xr_ds.where(mask_lon & mask_lat, drop=True)
print("COMPLETE.")

# Area average?
if area_average == "Yes":
    test_dataset = test_dataset.mean(["x", "y"])

# Load lazy dask data
print("Loading data...", end="")
test_dataset = test_dataset.compute()
print("COMPLETE.")


# ----------------- SAVE FILE AS NETCDF -----------------

# Do you want to download the data?
download_data = True  # Boolean True/False

# Define filename and output folder
output_folder = "test_data"  # Set to "" if you want to store in your current directory (where the notebook is running)
if filename is None:
    filename = (
        "test_dataset_"
        + str(year_start)
        + "_"
        + str(year_end)
        + "_"
        + timescale
        + "_"
        + resolution
    )  # Leave off .nc !
filename = filename.replace(" ", "")  # Remove any spaces from the string
filepath = output_folder + "/" + filename + ".nc"  # Path to file
print("Filename: {0}".format(filename))

if download_data == True:
    test_dataset.to_netcdf(path=filepath, mode="w")  # Output
    print("File saved to: {0}".format(filepath))
else:
    print("Data not downloaded. Set download_data = True to download data")
