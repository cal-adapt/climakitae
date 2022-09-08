"""
Calculates the Typical Meterological Year (TMY) for the Cal-Adapt: Analytics Engine using a standard climatological period (1981-2010) for the historical baseline,
and uses a 30-year window around when a designated warming level is exceeded for the SSP3-7.0 future scenario for 1.5°C, 2°C, and 3°C. Additional functionality for 4°C is forthcoming.
Working Group 4 (Aug 31, 2022) version focuses on air temperature and relative humidity, with all variables being available at a later date.
"""

## PROCESS: typical meteorological year
# for each hour, the average variable over the whole climatological period is determined
# data for that hour that has the value most closely equal (smalleset absolute difference) to the hourly average over the whole measurement period is chosen as the TMY data for that hour
# process is repeated for each hour in the year
# hours are added together to provide a full year of hourly samples

# To be developed:
# app.explore.tmy()
# absolute tmy of raw data: "preliminary unbiased data corrected data"
# difference tmys
# extreme/severe tmys: can think about percentiles above extreme values to represent
# can also select diurnal cycle
# download/export functionality
# watch cluster worker numbers for a target time to complete computations
# potential recent data transformation data cacheing

## Import libraries
import numpy as np
# import pandas as pd
import metpy.calc
import metpy
from netCDF4 import Dataset

##### --------------- COPY/PASTE FROM EXPLORE.PY --------------------

# TMY files in climakitae/data
# tmy_filenames = ['tmy_future-minus-hist_rh_45km_CA_15degC.csv',
#  'tmy_future-minus-hist_rh_45km_CA_2degC.csv',
#  'tmy_future-minus-hist_rh_45km_CA_3degC.csv',
#  'tmy_future-minus-hist_rh_45km_losangeles_15degC.csv',
#  'tmy_future-minus-hist_rh_45km_losangeles_2degC.csv',
#  'tmy_future-minus-hist_rh_45km_losangeles_3degC.csv',
#  'tmy_future-minus-hist_rh_45km_santaclara_15degC.csv',
#  'tmy_future-minus-hist_rh_45km_santaclara_2degC.csv',
#  'tmy_future-minus-hist_rh_45km_santaclara_3degC.csv',
#  'tmy_future-minus-hist_temp_45km_CA_15degC.csv',
#  'tmy_future-minus-hist_temp_45km_CA_2degC.csv',
#  'tmy_future-minus-hist_temp_45km_CA_3degC.csv',
#  'tmy_future-minus-hist_temp_45km_losangeles_15degC.csv',
#  'tmy_future-minus-hist_temp_45km_losangeles_2degC.csv',
#  'tmy_future-minus-hist_temp_45km_losangeles_3degC.csv',
#  'tmy_future-minus-hist_temp_45km_santaclara_15degC.csv',
#  'tmy_future-minus-hist_temp_45km_santaclara_2degC.csv',
#  'tmy_future-minus-hist_temp_45km_santaclara_3degC.csv']
# cached_tmy_files = [pkg_resources.resource_filename('climakitae', 'data/cached_tmy/'+file) for file in tmy_filenames]


# def _read_cached_tmy_df(cached_tmy_files, variable, warmlevel, cached_area):
#     """Read in cached tmy file corresponding to a given variable, warmlevel, and cached area.
#     Returns a dataframe"""
#
#     # Subset list by variable
#     if variable == "Relative Humidity":
#         cached_tmy_var = [file for file in cached_tmy_files if "rh" in file]
#     elif variable == "Air Temperature at 2m":
#         cached_tmy_var = [file for file in cached_tmy_files if "temp" in file]
#
#     # Subset list by warming level
#     if warmlevel == 1.5:
#         cached_tmy_warmlevel = [file for file in cached_tmy_var if "15degC" in file]
#     elif warmlevel == 2:
#         cached_tmy_warmlevel = [file for file in cached_tmy_var if "2degC" in file]
#     elif warmlevel == 3:
#         cached_tmy_warmlevel = [file for file in cached_tmy_var if "3degC" in file]
#
#     # Subset list by location
#     if cached_area == "CA":
#         tmy = [file for file in cached_tmy_warmlevel if "CA" in file]
#     if cached_area == "Santa Clara County":
#         tmy = [file for file in cached_tmy_warmlevel if "santaclara" in file]
#     if cached_area == "Los Angeles County" :
#         tmy = [file for file in cached_tmy_warmlevel if "losangeles" in file]
#
#     # Read in file as pandas dataframe
#     df = pd.read_csv(tmy[0], index_col=0)
#
#     # Name columns and index
#     df.columns.name = "Hour of Day"
#     df.index.name = "Day of Year"
#
#     return df

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

    ## NEED TO SET BACK TO AWS CATALOG READING IN FOR ALL VARIABLES
    # df = _read_cached_tmy_df(
    #     cached_tmy_files=cached_tmy_files,
    #     variable=self.variable2,
    #     warmlevel=self.warmlevel,
    #     cached_area=self.cached_area2
    # )

    # Set to PST time -- hardcoded
    df = df[['8','9','10','11','12','13','14','15','16','17','18','19','20','21','22','23','24','1','2','3','4','5','6','7']]
    col_h=[]
    for i in np.arange(1,25,1):
        col_h.append(str(i))
    df.columns = col_h

    # think about best practices data presentation here
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



## ----------------------------------------------- old explore ^ ---------

# think about options here for display
# absolute/unbias corrected
# difference for warming levels 

def _display_tmy(selections, location, _cat):
    """
    Creates a new TMY focus panel object to display user selections
    """
    user_options = pn.Card(
            pn.Row(
                pn.Column(
                    pn.widgets.StaticText(name="", value='Warming level (°C)'),
                    pn.widgets.RadioButtonGroup.from_param(warming_levels.param.warmlevel, name=""),
                    pn.widgets.Select.from_param(warming_levels.param.variable2, name="Data variable"),
                    pn.widgets.StaticText.from_param(selections.param.variable_description),
                    pn.widgets.Button.from_param(warming_levels.param.reload_data2, button_type="primary", width=150, height=30),
                    width = 230),
                pn.Column(
                    pn.widgets.Select.from_param(warming_levels.param.area_subset2, name="Area subset"),
                    pn.widgets.Select.from_param(warming_levels.param.cached_area2, name="Cached area"),
                    location.view,
                    width = 230)
                )
        , title="Data Options", collapsible=False, width=460, height=500
    )

    TMY = pn.Column(
        pn.widgets.StaticText(
           value="A typical meteorological year is calculated by selecting the 24 hours for every day that best represent multi-model mean conditions during a 30-year period – 1981-2010 for the historical baseline or centered on the year the warming level is reached.",
           width = 700
        ),
        warming_levels._TMY_hourly_heatmap
    )
