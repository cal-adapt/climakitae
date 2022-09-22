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
# absolute tmy of raw data: "preliminary unbiased data corrected data" -- COMPLETED
# difference tmys -- IN PROGRESS (FROM OLD EXPLORE )
# extreme/severe tmys: can think about percentiles above extreme values to represent, stress years?
# can also select diurnal cycle
# download/export functionality
# watch cluster worker numbers for a target time to complete computations

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
# from .utils import _read_var_csv, _read_ae_colormap
import intake
import pkg_resources

import logging  # Silence warnings
logging.getLogger("param").setLevel(logging.CRITICAL)

xr.set_options(keep_attrs=True) # Keep attributes when mutating xr objects


## Needs to produce 3 different kinds of TMY
## 1: Absolute/unbias corrected raw tmy
## 2: Future-minus-historical warming level tmy (see warming_levels)
## 3: Severe/extremes tmy based upon historical baseline and a designated threshold/percentile

def _get_historical_tmy_data(selections, location, catalog):
    """Get historical data from AWS catalog"""
    selections.append_historical = False
    selections.area_average = True
    selections.resolution = "45 km" ## KEEPING FOR NOW
    selections.scenario = ["Historical Climate"]
    selections.time_slice = (1981,2010) # to match historical 30-year average
    selections.timescale = "hourly"
    historical_da = _read_from_catalog(
        selections=selections,
        location=location,
        cat=catalog
    )
    
    # Compute mean of all simulations 
    # Compute zarr data 
    historical_da_mean = historical_da.mean(dim="simulation").isel(scenario=0).compute() 
    return historical_da_mean


def _get_future_heatmap_data(selections, location, catalog, warmlevel):
    """Gets data from AWS catalog based upon desired warming level"""
    
    warming_year_average_range = {
        1.5 : (2034,2063),
        2 : (2047,2076),
        3 : (2061,2090),
        4 : (2076,2100) ## Need to consider this further
    }
    
    selections.append_historical = False
    selections.area_average = True
    selections.resolution = "45 km"
    selections.scenario = ["SSP 3-7.0 -- Business as Usual"]
    selections.time_slice = warming_year_average_range[warmlevel]
    selections.timescale = "hourly"
    future_da = _read_from_catalog(
        selections=selections,
        location=location,
        cat=catalog
    )
    
    # Compute mean of all simulations 
    # Compute zarr data 
    future_da_mean = future_da.mean(dim="simulation").isel(scenario=0).compute() 
    return future_da_mean


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
        
        # Postage data and anomalies defaults
        self.historical_tmy_data = _get_historical_tmy_data(
            selections=self.selections, 
            location=self.location, 
            catalog=self.catalog
        )
        self.future_tmy_data = _get_future_heatmap_data(
            selections=self.selections, 
            location=self.location, 
            catalog=self.catalog, 
            warmlevel=1.5
        )

    # TMY options to display
    tmy_options = param.ObjectSelector(default='Absolute',
        objects=['Absolute', 'Difference'])

    abs_tmy_options = param.ObjectSelector(default='Historical',
        objects=['Historical', 'Warming Level Future'])

    diff_tmy_options = param.ObjectSelector(default='Warming Level Future',
        objects=['Warming Level Future', 'Severe TMY'])

    # For the difference TMY maps
    warmlevel = param.ObjectSelector(default=1.5,
        objects=[1.5, 2, 3])     # removing 4°C option for TMY

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
    
    # For reloading data and plots
    reload_data = param.Action(lambda x: x.param.trigger('reload_data'), label='Reload Data')
        

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

    @param.depends("area_subset2","cached_area2", watch=True)
    def _updated_location(self):
        """Update locations object to reflect location chosen in panel"""
        self.location.area_subset = self.area_subset2
        self.location.cached_area = self.cached_area2
        
    @param.depends("reload_data", watch=True)
    def _update_tmy_data(self):
        """If the button was clicked and the location or variable was changed,
        reload the tmy data from AWS"""
        self.historical_tmy_data = _get_historical_tmy_data(
            selections=self.selections, 
            location=self.location, 
            catalog=self.catalog
        )
        self.future_tmy_data = _get_future_heatmap_data(
            selections=self.selections, 
            location=self.location, 
            catalog=self.catalog, 
            warmlevel=self.warmlevel
        )
        
    @param.depends("reload_data", watch=False)
    def _tmy_hourly_heatmap(self):
        
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

        tmy_hist = tmy_calc(self.historical_tmy_data, days_in_year=days_in_year)
        tmy_future = tmy_calc(self.future_tmy_data, days_in_year=days_in_year)

        ## Funnel data into pandas DataFrame object
        df_hist = pd.DataFrame(tmy_hist, columns=np.arange(1,25,1), index=np.arange(1,days_in_year+1,1))
        df_hist = df_hist.iloc[::-1]
        df_future = pd.DataFrame(tmy_future, columns = np.arange(1,25,1), index=np.arange(1,days_in_year+1,1))
        df_future = df_future.iloc[::-1]

        df = df_future - df_hist # difference 

        clabel = self.variable2 #+ " ("+self.variable2.attrs["units"]+")"
        title = "Typical Meteorological Year\nAbsolute Value for Historical Baseline\n{}".format(self.cached_area2)

        df.columns = ['12am','1am','2am','3am','4am','5am','6am','7am','8am','9am','10am','11am','12pm','1pm','2pm','3pm','4pm','5pm','6pm','7pm','8pm','9pm','10pm','11pm']

        dy_labs = []
        for x in np.arange(1,367,1):
            dy_labs.append(x)

        heatmap = df.hvplot.heatmap(
            x='columns',
            y='index',
            title=title,
            cmap="YlOrRd",
            xaxis='bottom',
            xlabel="Hour of Day (PST)",
            ylabel="Day of Year", clabel=clabel, rot=60,
            width=800, height=350).opts(
            fontsize={'title': 15, 'xlabel':12, 'ylabel':12} # clim=(0,6) is for air temperature; clim=(-1,1) for relative humidity?
        )

        return heatmap
    


#--------------------------------------------------------------------------------------------
def _tmy_visualize(tmy_ob, selections, location):
    """
    Creates a new TMY focus panel object to display user selections
    """
    user_options = pn.Card(
            pn.Row(
                pn.Column(
                    pn.widgets.RadioButtonGroup.from_param(tmy_ob.param.tmy_options, name=" "),
                    pn.widgets.Select.from_param(tmy_ob.param.abs_tmy_options, name="Absolute AMY Options"),
                    pn.widgets.Select.from_param(tmy_ob.param.diff_tmy_options, name="Difference AMY Options"),
                    pn.widgets.StaticText(name="", value="Warming level (°C)"),
                    pn.widgets.RadioButtonGroup.from_param(tmy_ob.param.warmlevel),
                    pn.widgets.Select.from_param(tmy_ob.param.variable2, name="Data variable"),
                    pn.widgets.StaticText.from_param(selections.param.variable_description),
                    width=230),
                pn.Column(
                    pn.widgets.Select.from_param(tmy_ob.param.area_subset2, name="Location"),
                    location.view,
                    pn.widgets.Button.from_param(tmy_ob.param.reload_data, button_type="primary", width=150, height=30),
                    width=230)
                )
        , title="How do you want to investigate AMY?", collapsible=False, width=460, height=500
    )

    mthd_bx = pn.Column(
        pn.widgets.StaticText(
            value="A average meteorological year is calculated by selecting the 24 hours for every day that best represent multi-model mean conditions during a 30-year period – 1981-2010 for the historical baseline or centered on the year the warming level is reached. Alternatively, can put what data was selected here as a visual reminder.",
            width=400
        ),
    )

    TMY = pn.Card(
        pn.widgets.StaticText(
           value="Absolute TMY",
           width = 650, height=500
           ),
        tmy_ob._tmy_hourly_heatmap
    )

    tmy_tabs = pn.Card(
        pn.Tabs(
            ("TMY Heatmap", tmy_ob._tmy_hourly_heatmap),
            ("Methodology", mthd_bx)
        ),
    title="Average Meteorological Year", width = 850, height=500, collapsible=False,
    )

    tmy_panel = pn.Column(
        pn.Row(user_options, tmy_tabs)
    )

    return tmy_panel
