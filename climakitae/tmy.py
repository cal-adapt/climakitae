"""
Calculates the Typical Meterological Year (TMY) for the Cal-Adapt: Analytics Engine using the full historical period available (1980-2015).
Working Group 4 (Aug 31, 2022) version focuses on air temperature and precipitation, with all variables being available at a later date.
"""

## Import libraries
import numpy as np
# import pandas as pd
import metpy.calc
import metpy
from netCDF4 import Dataset


## PROCESS
# for each month, the average variable over the whole measurement period is determined + average variable value in each month during the measurement period
# data for the month that has the average value most closely equal to the monthly average over the whole measurement period is chosen as the TMY data for that month
# process is repeated for each month in the year
# months are added together to provide a full year of hourly samples

## TESTING WITH JUST AIR TEMPERATURE AT PRESENT + HARD-CODING
## Steps 1: Read in data
fn = "data/dummy_dataset_1980_2100_SSP3.7.0_historical_appended.nc"     # Nicole dummy dataset focusing on Joshua Tree region for testing
data = Dataset(fn)
tas = data.variables['Air Temperature at 2m'][:,:,5:35*12,:,:]  # subsetting to grab just the historcial period full years (data starts at 9/1980)
tas_area = np.average(tas, axis=(0,1,3,4))

## Step 2: Remove missing data for statistics?
## there are no missing data in the dummy dataset, so skipping for now -- TO DO FOR FULL DATA

## Step 3: Select most typical year of data for given month/day -- Sandia method (hold off on for now)
# data should be area averaged
## hard coding in to use dummy dataset


def find_nearest(array, value=0.0):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return idx

m = np.arange(0,12,1) # months
monthly_index_list = [] # which year for a specific month (ex: all Januarys) best matches the mean
for i in m:
    mon_list = tas_area[i::12]
    mon_mean = np.average(mon_list, axis=0)
    ideal_mon = find_nearest(mon_list - mon_mean)
    monthly_index_list.append(ideal_mon)


## Step 5: Calculate TMY
full_index_list = [] # corresponding indices in the full data range to the monthly_index_list
for i in m:
    tmy_idx = i+(12*monthly_index_list[i])
    full_index_list.append(tmy_idx)

## Step 6: Merge selected months of TMY together -- THIS IS A TIME SERIES
tmy_data = [tas_area[i] for i in full_index_list]


## Step 7: Export to csv for easy reading in?
## organize as time, temperature, precipitation


# ## -----------------------------------------------------------------------------------------------------------------------------------------------
# ## Focus on just available variables for now?  and not full Sandia method...?
# ## Calculate dewpoint temperature from air temperature and relative humidity
# def _compute_dewpoint_temperature(temperature, relative_humidity, variable_name="DEWPOINT_TEMPERATURE"):
#     """Computes dewpoint temperature using air temperature and relative humidity
#
#     Args:
#         temperature (xr.DataArray): Temperature in Kelvin (unit?)
#         relative humidity (xr.DataArray): Relative Humidity in %
#         variable_name (string): Name to assign DataArray object (default to "DEWPOINT_TEMPERATURE")
#
#     Returns:
#         dewpt_temp (xr.DataArray): Dewpoint temperature
#
#     """
#     dewpt_temp = metpy.calc.dewpoint_from_relative_humidity(temperature=temperature, relative_humidity=relative_humidity)
#     dewpt_temp = dewpt_temp.metpy.dequantify()
#     # metpy function returns a pint.Quantity object, which can cause issues with dask. This can be undone using the dequantify function. For more info: https://unidata.github.io/MetPy/latest/tutorials/xarray_tutorial.html
#
#     # Assign descriptive name and attributes
#     dewpt_temp.name = variable_name
#     dewpt_temp.attrs["description"] = "Dewpoint temperature"
#
#     return dewpt_temp
#
#
# ##relative_humidity (pint.Quantity) – Relative humidity expressed as a ratio in the range 0 < relative_humidity <= 1
# ## check how relative humidity outputs -- is it 0 to 100 or 0 to 1
