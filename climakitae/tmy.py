"""
Calculates the Typical Meterological Year (TMY) for the Cal-Adapt: Analytics Engine using a standard climatological period (1981-2010) for the historical baseline,
and uses a 30-year window around when a designated warming level is exceeded for the SSP3-7.0 future scenario for 1.5°C, 2°C, and 3°C. Additional functionality for 4°C is forthcoming.
Working Group 4 (Aug 31, 2022) version focuses on air temperature and relative humidity, with all variables being available for Analytics Engine Beta Launch.
"""

## PROCESS: typical meteorological year
# for each hour, the average variable over the whole climatological period is determined
# data for that hour that has the value most closely equal (smalleset absolute difference) to the hourly average over the whole measurement period is chosen as the TMY data for that hour
# process is repeated for each hour in the year
# hours are added together to provide a full year of hourly samples

# To be developed: will delete these notes once finalized
# app.explore.tmy -- COMPLETED
# absolute tmy of raw data: "preliminary unbiased data corrected data" -- IN PROGRESS
# difference tmys -- IN PROGRESS (FROM OLD EXPLORE )
# extreme/severe tmys: can think about percentiles above extreme values to represent
# can also select diurnal cycle
# download/export functionality
# watch cluster worker numbers for a target time to complete computations
# potential recent data transformation data cacheing


import cartopy.crs as ccrs
import hvplot.xarray
import hvplot.pandas
import xarray as xr
import holoviews as hv
from holoviews import opts
from matplotlib.figure import Figure
import numpy as np
import pandas as pd
import param
import panel as pn
import intake
import warnings
from .data_loaders import _read_from_catalog
from .selectors import DataSelector, LocSelectorArea
# from .utils import _read_var_csv, _read_ae_colormap
import intake
import pkg_resources


## Needs to produce 3 different kinds of TMY
## 1: Absolute/unbias corrected raw tmy
## 2: Future-minus-historical warming level tmy (see warming_levels)
## 3: Severe/extremes tmy based upon historical baseline and a designated threshold/percentile

class TypicalMeteorologicalYear(param.Parameterized):
    """
    An object that holds the "Data Options" paramters for the
    explore.tmy panel.
    """
    def __init__(self, *args, **params):
        super().__init__(*args, **params)

        # Selectors defaults
        self.selections.append_historical = False
        self.selections.area_average = True
        self.selections.resolution = "45 km"
        self.selections.scenario = ["Historical"]       # setting for historical
        self.selections.time_slice = (1981,2010)
        self.selections.timescale = "hourly"
        self.selections.variable = "Air Temperature at 2m"

        # Location defaults
        self.location.area_subset = 'states'
        self.location.cached_area = 'CA'

    # For the difference TMY maps
    warmlevel = param.ObjectSelector(default=1.5,
        objects=[1.5, 2, 3, 4]
    )

    # For reloading data and plots
    reload_data2 = param.Action(lambda x: x.param.trigger('reload_data2'), label='Reload Data')
    changed_loc_and_var = param.Boolean(default=True)

    variable2 = param.ObjectSelector(default="Air Temperature at 2m",
        objects=["Air Temperature at 2m"]
    )

    cached_area2 = param.ObjectSelector(default="CA",
        objects=["CA"]
    )

    area_subset2 = param.ObjectSelector(
        default="states",
        objects=["states", "CA counties"],
    )

    @param.depends("area_subset2","cached_area2","variable2", watch=True)
    def _updated_bool_loc_and_var(self):
        """Update boolean if any changes were made to the location or variable"""
        self.changed_loc_and_var = True

    @param.depends("variable2", watch=True)
    def _update_variable(self):
        """Update variable in selections object to reflect variable chosen in panel"""
        self.selections.variable = self.variable2

    @param.depends("area_subset2", watch=True)
    def _update_cached_area(self):
        """
        Makes the dropdown options for 'cached area' reflect the type of area subsetting
        selected in 'area_subset' (currently state, county, or watershed boundaries).
        """
        if self.area_subset2 == "CA counties":
            # setting this to the dict works for initializing, but not updating an objects list:
            self.param["cached_area2"].objects = ["Santa Clara County", "Los Angeles County"]
            self.cached_area2 = "Santa Clara County"
        elif self.area_subset2 == "states":
            self.param["cached_area2"].objects = ["CA"]
            self.cached_area2 = "CA"

    @param.depends("area_subset2","cached_area2",watch=True)
    def _updated_location(self):
        """Update locations object to reflect location chosen in panel"""
        self.location.area_subset = self.area_subset2
        self.location.cached_area = self.cached_area2


    # reload_data = param.Action(lambda x: x.param.trigger('reload_data'), label="Reload Data")
    @param.depends("reload_data", watch=False)
    def _tmy_hourly_heatmap(self):
        def _get_hist_heatmap_data():
            """Get historical data from AWS catalog"""
            heatmap_selections = self.selections
            heatmap_selections.append_historical = False
            heatmap_selections.area_average = True
            heatmap_selections.resolution = "45 km" ## KEEPING FOR NOW
            heatmap_selections.scenario = ["Historical Climate"]
            heatmap_selections.time_slice = (1981,2010) # to match historical 30-year average
            heatmap_selections.timescale = "hourly"
            xr_da = _read_from_catalog(
                selections=heatmap_selections,
                location=self.location,
                cat=self.catalog
            )
            return xr_da

        ## hard-coding in for now
        warming_year_average_range = {
            1.5 : (2034,2063),
            2 : (2047,2076),
            3 : (2061,2090),
            4 : (2076,2100) ## Need to consider this further
        }

        def _get_future_heatmap_data():
            """Gets data from AWS catalog based upon desired warming level"""
            heatmap_selections = self.selections
            heatmap_selections.append_historical = False
            heatmap_selections.area_average = True
            heatmap_selections.resolution = "45 km"
            heatmap_selections.scenario = ["SSP 3-7.0 -- Business as Usual"]
            heatmap_selections.time_slice = warming_year_average_range[self.warmlevel]
            heatmap_selections.timescale = "hourly"
            xr_da2 = _read_from_catalog(
                selections=heatmap_selections,
                location=self.location,
                cat=self.catalog
            )
            return xr_da2

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

        ## Grab data from AWS
        data_hist = _get_hist_heatmap_data()
        data_hist = data_hist.mean(dim="simulation").isel(scenario=0).compute()
        # data_future = _get_future_heatmap_data()
        # data_future = data_future.mean(dim="simulation").isel(scenario=0).compute()

        ## Compute hourly TMY for each hour of the year
        days_in_year = 366
        def tmy_calc(data, days_in_year=366):
            """
            Calculates the typical meteorological year based on a designated period of time.
            Applicable for both the historical and future periods.
            Returns: list of tmy
            """
            hourly_list = []
            for x in np.arange(1, days_in_year+1, 1):
                data_on_day_x = data.where(data.time.dt.dayofyear == x, drop = True)
                data_grouped = data_on_day_x.groupby("time.hour")
                mean_by_hour = data_grouped.mean()
                min_diff = abs(data_grouped - mean_by_hour).groupby("time.hour").min()
                typical_hourly_data_on_day_x = data_on_day_x.where(abs(data_grouped - mean_by_hour).groupby("time.hour") == min_diff, drop = True).sortby("time.hour")
                np_typical_hourly_data_on_day_x = remove_repeats(typical_hourly_data_on_day_x)
                hourly_list.append(np_typical_hourly_data_on_day_x)
            return hourly_list

        tmy_hist = tmy_calc(data_hist)
        # tmy_future = tmy_calc(data_future)

        ## Funnel data into pandas DataFrame object
        df_hist = pd.DataFrame(tmy_hourly, columns = np.arange(1,25,1), index=np.arange(1,days_in_year+1,1))
        df_hist = df_hist.iloc[::-1] # Reverse index
        # df_future = pd.DataFrame(tmy_future, columns = np.arange(1,25,1), index=np.arange(1,days_in_year+1,1))
        # df_future = df_future.iloc[::-1]

        ## Create difference heatmaps based on selected warming level
        df = df_hist # absolute unbias corrected version
        # df = df_future - df_hist # future difference version
        # df = df_extreme - df_hist # extreme version

        ## Set to PST time -- hardcoded
        df = df[['8','9','10','11','12','13','14','15','16','17','18','19','20','21','22','23','24','1','2','3','4','5','6','7']]
        col_h=[]
        for i in np.arange(1,25,1):
            col_h.append(str(i))
        df.columns = col_h

        # # think about best practices data presentation here
        # if self.variable2 == "Air Temperature at 2m":
        #     cm = "YlOrRd"
        #     cl = (0,6)  # hardcoding this in, full range of warming level response for 2m air temp
        # elif self.variable2 == "Relative Humidity":
        #     cm = "PuOr"
        #     cl = (-7,7) # hardcoding this in, full range of warming level response for relhumid

        heatmap = df.hvplot.heatmap(
            x='columns',
            y='index',
            # title='Typical Meteorological Year\nDifference between a {}°C-warming level future and the historical baseline'.format(self.warmlevel),
            title='Typical Meteorological Year\nAbsolute value for Historical Baseline\n{}'.format(self.location_subset2),
            cmap="YlOrRd",
            xaxis='bottom',
            xlabel="Hour of Day (UTC)",
            ylabel="Day of Year",clabel="Air Temperature at 2m " + " (°C)",
            width=800, height=350).opts(
            clim=(0,6), fontsize={'title': 15, 'xlabel':12, 'ylabel':12} # clim=(0,6) is for air temperature; clim=(-1,1) for relative humidity?
        )
        return heatmap


#--------------------------------------------------------------------------------------------
# def _tmy_visualize(data, cmap=None):
def _tmy_visualize(self):
    """
    Creates a new TMY focus panel object to display user selections
    """
    tmy_ob = TypicalMeteorologicalYear(selections=self.selections, location=self.location)

    user_options = pn.Card(
            pn.Row(
                pn.Column(
                    pn.widgets.StaticText(name="", value='Typical Meteorological Year Options'),
                    pn.widgets.Select.from_param(tmy_ob.param.variable2, name="Data variable"),
                    pn.widgets.StaticText.from_param(self.selections.param.variable_description),
                    pn.widgets.Button.from_param(tmy_ob.param.reload_data, button_type="primary", width=150, height=30),
                    width = 230),
                pn.Column(
                    pn.widgets.Select.from_param(tmy_ob.param.area_subset2, name="Location"),
                    self.location.view,
                    width = 230)
                )
        , title="Data Options", collapsible=False, width=460, height=500
    )

    TMY = pn.Card(
        pn.widgets.StaticText(
           value=" ",
           width = 700
           ),
        tmy_ob._tmy_hourly_heatmap
    )

    mthd_bx = pn.Card(
        "A typical meteorological year is calculated by selecting the 24 hours for every day that best represent multi-model mean conditions during a 30-year period – 1981-2010 for the historical baseline or centered on the year the warming level is reached. Alternatively, can put what data was selected here as a visual reminder. ",
        title="Methodology", collapsible=True, width=1160
    )

    tmy_panel = pn.Column(
        pn.Row(user_options, TMY),
        mthd_bx
    )
    return tmy_panel
