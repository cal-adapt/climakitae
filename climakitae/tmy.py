"""
Calculates the Typical Meterological Year (TMY) for the Cal-Adapt: Analytics Engine using the full historical period available (1980-2015).
Working Group 4 (Aug 31, 2022) version focuses on air temperature and precipitation, with all variables being available at a later date.
"""


##### --------------- COPY/PASTE FROM EXPLORE.PY --------------------

# TMY files in climakitae/data 
tmy_filenames = ['tmy_future-minus-hist_rh_45km_CA_15degC.csv',
 'tmy_future-minus-hist_rh_45km_CA_2degC.csv',
 'tmy_future-minus-hist_rh_45km_CA_3degC.csv',
 'tmy_future-minus-hist_rh_45km_losangeles_15degC.csv',
 'tmy_future-minus-hist_rh_45km_losangeles_2degC.csv',
 'tmy_future-minus-hist_rh_45km_losangeles_3degC.csv',
 'tmy_future-minus-hist_rh_45km_santaclara_15degC.csv',
 'tmy_future-minus-hist_rh_45km_santaclara_2degC.csv',
 'tmy_future-minus-hist_rh_45km_santaclara_3degC.csv',
 'tmy_future-minus-hist_temp_45km_CA_15degC.csv',
 'tmy_future-minus-hist_temp_45km_CA_2degC.csv',
 'tmy_future-minus-hist_temp_45km_CA_3degC.csv',
 'tmy_future-minus-hist_temp_45km_losangeles_15degC.csv',
 'tmy_future-minus-hist_temp_45km_losangeles_2degC.csv',
 'tmy_future-minus-hist_temp_45km_losangeles_3degC.csv',
 'tmy_future-minus-hist_temp_45km_santaclara_15degC.csv',
 'tmy_future-minus-hist_temp_45km_santaclara_2degC.csv',
 'tmy_future-minus-hist_temp_45km_santaclara_3degC.csv']
cached_tmy_files = [pkg_resources.resource_filename('climakitae', 'data/cached_tmy/'+file) for file in tmy_filenames]


def _read_cached_tmy_df(cached_tmy_files, variable, warmlevel, cached_area):
    """Read in cached tmy file corresponding to a given variable, warmlevel, and cached area.
    Returns a dataframe"""

    # Subset list by variable
    if variable == "Relative Humidity":
        cached_tmy_var = [file for file in cached_tmy_files if "rh" in file]
    elif variable == "Air Temperature at 2m":
        cached_tmy_var = [file for file in cached_tmy_files if "temp" in file]

    # Subset list by warming level
    if warmlevel == 1.5:
        cached_tmy_warmlevel = [file for file in cached_tmy_var if "15degC" in file]
    elif warmlevel == 2:
        cached_tmy_warmlevel = [file for file in cached_tmy_var if "2degC" in file]
    elif warmlevel == 3:
        cached_tmy_warmlevel = [file for file in cached_tmy_var if "3degC" in file]

    # Subset list by location
    if cached_area == "CA":
        tmy = [file for file in cached_tmy_warmlevel if "CA" in file]
    if cached_area == "Santa Clara County":
        tmy = [file for file in cached_tmy_warmlevel if "santaclara" in file]
    if cached_area == "Los Angeles County" :
        tmy = [file for file in cached_tmy_warmlevel if "losangeles" in file]

    # Read in file as pandas dataframe
    df = pd.read_csv(tmy[0], index_col=0)
    
    # Name columns and index 
    df.columns.name = "Hour of Day"
    df.index.name = "Day of Year"
    
    return df

@param.depends("reload_data2", watch=False)
def _TMY_hourly_heatmap(self):
    """Generate a TMY hourly heatmap using hourly data"""

    # hard-coding in for now
    warming_year_average_range = {
        1.5 : (2034,2063),
        2 : (2047,2076),
        3 : (2061,2090),
        4 : (2076, 2100)
    }


    def remove_repeats(xr_data):
        """
        Remove hours that have repeats.
        This occurs if two hours have the same absolute difference from the mean.
        Returns numpy array
        """
        unq, unq_idx, unq_cnt = np.unique(xr_data.time.dt.hour.values, return_inverse=True, return_counts=True)
        cnt_mask = unq_cnt > 1
        cnt_idx, = np.nonzero(cnt_mask)
        idx_mask = np.in1d(unq_idx, cnt_idx)
        idx_idx, = np.nonzero(idx_mask)
        srt_idx = np.argsort(unq_idx[idx_mask])
        dup_idx = np.split(idx_idx[srt_idx], np.cumsum(unq_cnt[cnt_mask])[:-1])
        if len(dup_idx[0]) > 0:
            dup_idx_keep_first_val = np.concatenate([dup_idx[x][1:] for x in range(len(dup_idx))], axis=0)
            cleaned_np = np.delete(xr_data.values, dup_idx_keep_first_val)
            return cleaned_np
        else:
            return xr_data.values

    df = _read_cached_tmy_df(
        cached_tmy_files=cached_tmy_files,
        variable=self.variable2,
        warmlevel=self.warmlevel,
        cached_area=self.cached_area2
    )

    # Set to PST time -- hardcoded
    df = df[['8','9','10','11','12','13','14','15','16','17','18','19','20','21','22','23','24','1','2','3','4','5','6','7']]
    col_h=[]
    for i in np.arange(1,25,1):
        col_h.append(str(i))
    df.columns = col_h

    if self.variable2 == "Air Temperature at 2m":
        cm = "YlOrRd"
        cl = (0,6)  # hardcoding this in, full range of warming level response for 2m air temp
    elif self.variable2 == "Relative Humidity":
        cm = "PuOr"
        cl = (-7,7) # hardcoding this in, full range of warming level response for relhumid

    heatmap = df.hvplot.heatmap(
        x='columns',
        y='index',
        title='Typical Meteorological Year\nDifference between a {}°C future and historical baseline'.format(self.warmlevel),
        cmap=cm,
        xaxis='bottom',
        xlabel="Hour of Day (PST)",
        ylabel="Day of Year",clabel=self.postage_data.name + " ("+self.postage_data.attrs["units"]+")",
        width=800, height=350).opts(
        fontsize={'title': 15, 'xlabel':12, 'ylabel':12},
        clim=cl
    )
    return heatmap

TMY = pn.Column(
        pn.widgets.StaticText(
           value="A typical meteorological year is calculated by selecting the 24 hours for every day that best represent multi-model mean conditions during a 30-year period – 1981-2010 for the historical baseline or centered on the year the warming level is reached.",
           width = 700
        ),
        warming_levels._TMY_hourly_heatmap
    )

##### --------------- END OF COPY/PASTE FROM EXPLORE.PY --------------------


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
