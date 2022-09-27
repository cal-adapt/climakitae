"""
Calculates the Average Meterological Year (TMY) for the Cal-Adapt: Analytics Engine using a standard climatological period (1981-2010) for the historical baseline,
and uses a 30-year window around when a designated warming level is exceeded for the SSP3-7.0 future scenario for 1.5°C, 2°C, and 3°C. Additional functionality for 4°C is forthcoming.
Working Group 4 (Aug 31, 2022) version focuses on air temperature and relative humidity, with all variables being available for Analytics Engine Beta Launch.
The AMY is comparable to a typical meteorological year, but not quite the same full methodology.
"""

## PROCESS: average meteorological year
# for each hour, the average variable over the whole climatological period is determined
# data for that hour that has the value most closely equal (smalleset absolute difference) to the hourly average over the whole measurement period is chosen as the AMY data for that hour
# process is repeated for each hour in the year
# repeat values (where multiple years have the same smallest abs value) are removed, earliest occurence selected for AMY
# hours are added together to provide a full year of hourly samples

## Produces 3 different kinds of AMY
## 1: Absolute/unbias corrected raw AMY, either historical or warming level-centered future
## 2: Future-minus-historical warming level AMY (see warming_levels)
## 3: Severe AMY based upon historical baseline and a designated threshold/percentile

import cartopy.crs as ccrs
import hvplot.xarray
import hvplot.pandas
import xarray as xr
# import climtas
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
from .utils import _read_ae_colormap, _read_var_csv
import intake
import pkg_resources

import logging  # Silence warnings
logging.getLogger("param").setLevel(logging.CRITICAL)

var_descrip_pkg = pkg_resources.resource_filename('climakitae', 'data/variable_descriptions.csv')
var_descrip = _read_var_csv(var_descrip_pkg, index_col="description")

xr.set_options(keep_attrs=True) # Keep attributes when mutating xr objects

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


class AverageMeteorologicalYear(param.Parameterized):
    """
    An object that holds the "Data Options" paramters for the
    explore.tmy panel.
    """

    units2 = param.ObjectSelector(objects=dict())
    variable2 = param.ObjectSelector(default="Air Temperature at 2m", objects=dict())
    cached_area2 = param.ObjectSelector(default="CA", objects=dict())
    area_subset2 = param.ObjectSelector(default="states", objects=["CA counties", "states"])

    # Create dictionary of TMY advanced options depending on TMY type
    tmy_advanced_options_dict = {
        "Absolute": {
            "default":"Historical",
            "objects":["Historical","Warming Level Future"]
        },
        "Difference": {
            "default":"Warming Level Future",
            "objects":["Warming Level Future"] #,"Severe AMY"]
        }
    }

    # Create a dictionary that briefly explains the computation being perfomed
    computatation_description_dict = {
        "Absolute": {
            "Historical": "AMY computed using the historical baseline for 1981-2010.",
            "Warming Level Future": "AMY computed using the 30-year future period centered around when the selected warming level is reached."
        },
        "Difference": {
            "Warming Level Future": "AMY computed by taking the difference between the 30-year future period centered around the selected warming level and the historical baseline."
            # "Severe AMY": "AMY computed by taking the difference between the 90th percentile of the 30-year future period centered around the selected warming level and the historical baseline."
        }
    }

    # Define TMY params
    tmy_options = param.ObjectSelector(
        default='Absolute',
        objects=['Absolute', 'Difference']
    )

    # Define new advanced options param, that is dependent on the user selection in tmy_options
    tmy_advanced_options = param.ObjectSelector(objects=dict())

    # Define new computation description param
    # This will provide a string description of the computation option selected and will update dynamically depending on the user selections
    tmy_computation_description = param.ObjectSelector(objects=dict())

    def __init__(self, *args, **params):
        super().__init__(*args, **params)

        # Selectors defaults
        self.selections.append_historical = False
        self.selections.area_average = True
        self.selections.resolution = "45 km"
        self.selections.scenario = ["Historical"]  # setting for historical
        self.selections.time_slice = (1981,2010)
        self.selections.timescale = "hourly"
        self.selections.variable = "Air Temperature at 2m"

        self.units2 = self.selections.descrip_dict[self.selections.variable]["native_unit"]

        # Location defaults
        self.location.area_subset = 'states'
        self.location.cached_area = 'CA'

        self.param["variable2"].objects = self.selections.param.variable.objects
        self.param["cached_area2"].objects = self.location.param.cached_area.objects

        # Initialze tmy_adanced_options param
        self.param["tmy_advanced_options"].objects = self.tmy_advanced_options_dict[self.tmy_options]["objects"]
        self.tmy_advanced_options = self.tmy_advanced_options_dict[self.tmy_options]["default"]

        # Initialize tmy_computation_description param
        self.tmy_computation_description = self.computatation_description_dict[self.tmy_options][self.tmy_advanced_options]

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

    warmlevel = param.ObjectSelector(default=1.5,
        objects=[1.5, 2, 3])

    variable2 = param.ObjectSelector(default="Air Temperature at 2m", objects=dict())
    cached_area2 = param.ObjectSelector(default="CA", objects=dict())
    area_subset2 = param.ObjectSelector(default="states", objects=["states", "CA counties"])

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
            self.param["cached_area2"].objects = list(
                self.location._geography_choose[self.area_subset2].keys()
            )
            self.cached_area2 = "Sacramento County"
        elif self.area_subset2 == "states":
            self.param["cached_area2"].objects = ["CA"]
            self.cached_area2 = "CA"

    @param.depends("area_subset2","cached_area2",watch=True)
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

    # Create a function that will update tmy_advanced_options when tmy_options is modified
    @param.depends("tmy_options", watch=True)
    def _update_tmy_advanced_options(self):
        self.param["tmy_advanced_options"].objects = self.tmy_advanced_options_dict[self.tmy_options]["objects"]
        self.tmy_advanced_options = self.tmy_advanced_options_dict[self.tmy_options]["default"]

    @param.depends("tmy_options", "tmy_advanced_options", watch=True)
    def _update_tmy_computatation_description(self):
        self.tmy_computation_description = self.computatation_description_dict[self.tmy_options][self.tmy_advanced_options]

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
            Calculates the average meteorological year based on a designated period of time.
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

        # update heatmap df and title with selections
        if self.tmy_options == "Absolute":
            if self.tmy_advanced_options == "Historical":
                df = df_hist
                title = "Average Meteorological Year: {}\nAbsolute {} Baseline".format(self.cached_area2, self.tmy_advanced_options)
                clabel = self.variable2 + " (" +self.historical_tmy_data.attrs["units"]+")"
            else:
                df = df_future
                title = "Average Meteorological Year: {}\nAbsolute {} at {}°C".format(self.cached_area2, self.tmy_advanced_options, self.warmlevel)
                clabel = self.variable2 + " (" +self.future_tmy_data.attrs["units"]+")"
        elif self.tmy_options == "Difference":
            cmap = _read_ae_colormap("ae_diverging", cmap_hex = True)
            if self.tmy_advanced_options == "Warming Level Future":
                df = df_future - df_hist
                title = "Average Meteorological Year: {}\nDifference between {} at {}°C and Historical Baseline".format(self.cached_area2, self.tmy_advanced_options, self.warmlevel)
                clabel = self.variable2 + " (" +self.historical_tmy_data.attrs["units"]+")"
            else:
                df = df_future - df_hist # placeholder for now for severe amy
                title = "Average Meteorological Year: {}\nDifference between {} at 90th percentile and Historical Baseline".format(self.cached_area2, self.tmy_advanced_options)
                clabel = self.variable2 + " (" +self.historical_tmy_data.attrs["units"]+")"
        else:
            title = "Average Meteorological Year\n{}".format(self.cached_area2)

        # Manual re-ordering for PST time from UTC and easy-to-understand labels
        df = df[[8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,1,2,3,4,5,6,7]]
        df.columns = ['12am','1am','2am','3am','4am','5am','6am','7am','8am','9am','10am','11am','12pm','1pm','2pm','3pm','4pm','5pm','6pm','7pm','8pm','9pm','10pm','11pm']

        cmap_name = var_descrip[self.variable2]["default_cmap"]
        cmap = _read_ae_colormap(cmap=cmap_name, cmap_hex=True)
        # if df.min() < 0:
        #     cmap = _read_ae_colormap(cmap="ae_diverging", cmap_hex = True)

        heatmap = df.hvplot.heatmap(
            x='columns',
            y='index',
            title=title,
            cmap=cmap,
            xaxis='bottom',
            xlabel="Hour of Day (PST)",
            ylabel="Day of Year", clabel=clabel, rot=60,
            width=800, height=350).opts(
            fontsize={'title': 15, 'xlabel':12, 'ylabel':12}
        )
        heatmap.opts(toolbar="below")

        return heatmap

#--------------------------------------------------------------------------------------------
def _amy_visualize(tmy_ob, selections, location):
    """
    Creates a new AMY focus panel object to display user selections
    """
    user_options = pn.Card(
            pn.Row(
                pn.Column(
                    pn.widgets.StaticText(name="", value="Average Meteorological Year Type"),
                    pn.widgets.RadioButtonGroup.from_param(tmy_ob.param.tmy_options),
                    pn.widgets.Select.from_param(tmy_ob.param.tmy_advanced_options, name="Computation Options"),
                    pn.widgets.StaticText.from_param(tmy_ob.param.tmy_computation_description, name=""),
                    pn.widgets.StaticText(name="", value="Warming level (°C)"),
                    pn.widgets.RadioButtonGroup.from_param(tmy_ob.param.warmlevel),
                    pn.widgets.Select.from_param(tmy_ob.param.variable2, name="Data variable"),
                    pn.widgets.StaticText.from_param(selections.param.variable_description),
                    width=230),
                pn.Column(
                    pn.widgets.Select.from_param(tmy_ob.param.area_subset2, name="Location"),
                    pn.widgets.Select.from_param(tmy_ob.param.cached_area2, name="Cached area"),
                    location.view,
                    pn.widgets.Button.from_param(tmy_ob.param.reload_data, button_type="primary", width=150, height=30),
                    width=230)
                )
        , title=" How do you want to investigate AMY?", collapsible=False, width=460, height=500
    )

    mthd_bx = pn.Column(
        pn.widgets.StaticText(
            value="An average meteorological year is calculated by selecting the 24 hours for every day that best represent multi-model mean conditions during a 30-year period – 1981-2010 \
            for the historical baseline or centered on the year the warming level is reached. Absolute average meteorolgoical year profiles represent data that is not bias corrected,  \
            please exercise caution when analyzing. The 'severe' AMY is calculated using the 90th percentile of future warming level data at the selected warming level, and is compared to the historical baseline.",
            width=400
        ),
    )

    TMY = pn.Card(
        pn.widgets.StaticText(
           value="Absolute AMY",
           width = 600, height=500
           ),
        tmy_ob._tmy_hourly_heatmap
    )

    tmy_tabs = pn.Card(
        pn.Tabs(
            ("AMY Heatmap", tmy_ob._tmy_hourly_heatmap),
            ("Methodology", mthd_bx)
        ),
    title=" Average Meteorological Year", width = 850, height=500, collapsible=False,
    )

    tmy_panel = pn.Column(
        pn.Row(user_options, tmy_tabs)
    )

    return tmy_panel
