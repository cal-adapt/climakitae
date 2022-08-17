"""
Calculates the Typical Meterological Year (TMY) for the Cal-Adapt: Analytics Engine using the full historical period available (1980-2015).
Working Group 4 (Aug 31, 2022) version focuses on air temperature and precipitation, with all variables being available at a later date.
"""

## Import libraries
import numpy as np
import pandas as pd
import metpy.calc
import metpy

## Steps 1: Read in data
fn = "data/dummy_dataset_1980_2100_SSP3.7.0_historical_appended.nc"     # Nicole dummy dataset focusing on Joshua Tree region for testing


## Step 2: Remove missing data for statistics?

## Step 3: Calculate cumulative distribution function for given property

## Step 4: Select most typical year of data for given month/day -- Sandia method (hold off on for now)

## Step 5: Calculate TMY given Sandia method

## Step 6: Merge selected months of TMY together

## Step 7: Export to csv for easy reading in?


## -----------------------------------------------------------------------------------------------------------------------------------------------
## Focus on just available variables for now?  and not full Sandia method...?
## Calculate dewpoint temperature from air temperature and relative humidity
def _compute_dewpoint_temperature(temperature, relative_humidity, variable_name="DEWPOINT_TEMPERATURE"):
    """Computes dewpoint temperature using air temperature and relative humidity

    Args:
        temperature (xr.DataArray): Temperature in Kelvin (unit?)
        relative humidity (xr.DataArray): Relative Humidity in %
        variable_name (string): Name to assign DataArray object (default to "DEWPOINT_TEMPERATURE")

    Returns:
        dewpt_temp (xr.DataArray): Dewpoint temperature

    """
    dewpt_temp = metpy.calc.dewpoint_from_relative_humidity(temperature=temperature, relative_humidity=relative_humidity)
    dewpt_temp = dewpt_temp.metpy.dequantify()
    # metpy function returns a pint.Quantity object, which can cause issues with dask. This can be undone using the dequantify function. For more info: https://unidata.github.io/MetPy/latest/tutorials/xarray_tutorial.html

    # Assign descriptive name and attributes
    dewpt_temp.name = variable_name
    dewpt_temp.attrs["description"] = "Dewpoint temperature"

    return dewpt_temp


##relative_humidity (pint.Quantity) – Relative humidity expressed as a ratio in the range 0 < relative_humidity <= 1
## check how relative humidity outputs -- is it 0 to 100 or 0 to 1
