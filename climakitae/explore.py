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
import intake

import pkg_resources

# Import package data 
ssp119 = pkg_resources.resource_filename('climakitae', 'data/tas_global_SSP1_1_9.csv')
ssp126 = pkg_resources.resource_filename('climakitae', 'data/tas_global_SSP1_2_6.csv')
ssp245 = pkg_resources.resource_filename('climakitae', 'data/tas_global_SSP2_4_5.csv')
ssp370 = pkg_resources.resource_filename('climakitae', 'data/tas_global_SSP3_7_0.csv')
ssp585 = pkg_resources.resource_filename('climakitae', 'data/tas_global_SSP5_8_5.csv')
hist = pkg_resources.resource_filename('climakitae', 'data/tas_global_Historical.csv')
dummy_data = pkg_resources.resource_filename('climakitae', 'data/dummy_dataset_1980_2100_SSP3.7.0_historical_appended.nc')



class WarmingLevels(param.Parameterized):
    
    ## ---------- Params used for GMT context plot ----------
    
    warmlevel = param.ObjectSelector(default=1.5, 
        objects=[1.5, 2, 3, 4]
    )
    ssp = param.ObjectSelector(default="SSP 3-7.0 -- Business as Usual",
        objects=["SSP 2-4.5 -- Middle of the Road","SSP 3-7.0 -- Business as Usual","SSP 5-8.5 -- Burn it All"]
    ) 
    
    ## ---------- Reset certain DataSelector and LocSelectorArea options ----------
    def __init__(self, *args, **params):
        super().__init__(*args, **params)
        
        self.selections.append_historical = True
        self.selections.area_average = False
        self.selections.resolution = "9 km"
        self.selections.scenario = ["SSP 3-7.0 -- Business as Usual"]
        self.selections.time_slice = (1980,2100)
        self.selections.timescale = "monthly" 
        self.selections.variable = "Air Temperature at 2m"

        self.location.area_subset = 'states'
        self.location.cached_area = 'CA'
    
    ## ---------- Modify options in selectors.py DataSelectors object ----------
    
    variable2 = param.ObjectSelector(default="Air Temperature at 2m", 
        objects=["Air Temperature at 2m","Precipitation (total)"]
        )    
    location_subset2 = param.ObjectSelector(default="California", 
        objects=["California","Entire domain"]
        )
        
    @param.depends("variable2", watch=True)
    def _update_variable(self): 
        self.selections.variable = self.variable2
    
    @param.depends("location_subset2", watch=True)
    def _update_location(self): 
        if self.location_subset2 == "California":
            self.location.area_subset = "states"
            self.location.cached_area = "CA" 
        elif self.location_subset2 == "Entire domain": 
            self.location.area_subset = "none"
        else: 
            raise ValueError("You've encountered a bug in the code. See the ModifiedSelections class in explore.py")
            

    reload_data = param.Action(lambda x: x.param.trigger('reload_data'), label='Reload Data')
    @param.depends("reload_data", watch=False)
    def _TMY_hourly_heatmap(self):
        def _get_heatmap_data():    
            """Get data from AWS catalog"""
            heatmap_selections = self.selections
            heatmap_selections.append_historical = False
            heatmap_selections.area_average = True
            heatmap_selections.resolution = "45 km"
            heatmap_selections.scenario = ["Historical Climate"]
            heatmap_selections.time_slice = (1980,2000)
            heatmap_selections.timescale = "hourly" 
            xr_da = _read_from_catalog(
                selections=heatmap_selections, 
                location=self.location, 
                cat=self.catalog
            )
            return xr_da
        
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
        
        # Grab data from AWS 
        data = _get_heatmap_data()
        data = data.mean(dim="simulation").isel(scenario=0).compute()
        
        # Compute hourly TMY for each day of the year
        days_in_year = 366
        tmy_hourly = []
        for x in np.arange(1,days_in_year+1,1): 
            data_on_day_x = data.where(data.time.dt.dayofyear == x, drop=True)
            data_grouped = data_on_day_x.groupby("time.hour")
            mean_by_hour = data_grouped.mean()
            min_diff = abs(data_grouped - mean_by_hour).groupby("time.hour").min()
            typical_hourly_data_on_day_x = data_on_day_x.where(abs(data_grouped - mean_by_hour).groupby("time.hour") == min_diff, drop=True).sortby("time.hour")
            np_typical_hourly_data_on_day_x = remove_repeats(typical_hourly_data_on_day_x)
            tmy_hourly.append(np_typical_hourly_data_on_day_x)

        # Funnel data into pandas DataFrame object 
        df = pd.DataFrame(tmy_hourly, columns = np.arange(1,25,1), index=np.arange(1,days_in_year+1,1))
        df = df.iloc[::-1] # Reverse index 
        
        # Create heatmap 
        heatmap = df.hvplot.heatmap(
            x='columns', 
            y='index', 
            title='Typical Meteorological Year Heatmap', 
            cmap="coolwarm", 
            xaxis='bottom', 
            xlabel="Hour of Day (UTC)",
            ylabel="Day of Year",clabel=data.name + " ("+data.units+")",
            width=800, height=350).opts( 
            fontsize={'title': 15, 'xlabel':12, 'ylabel':12}
        )
        return heatmap

    @param.depends("variable2", watch=False)
    def _GCM_PostageStamps(self): 
        def _get_data(): 
            pkg_data = xr.open_dataset(dummy_data)
            return pkg_data[self.variable2]
            
        data = _get_data()
        
        fig = Figure(figsize=(7, 5), tight_layout=True)

        # Placeholder indices 
        for ax_index, time_index in zip(np.arange(1,6),np.arange(1,6)):
            # Ideally these should all have the same colorbar
            ax = fig.add_subplot(2,4,ax_index,projection=ccrs.LambertConformal())
            xr_pl = data.isel(time=time_index,simulation=0,scenario=0).plot(
                ax=ax, shading='auto', cmap="coolwarm",add_colorbar=False
                )
            ax.set_title("plot {0}".format(ax_index))
            ax.coastlines(linewidth=1, color = 'black', zorder = 10) # Coastlines
            ax.gridlines(linewidth=0.25, color='gray', alpha=0.9, crs=ccrs.PlateCarree(), linestyle = '--',draw_labels=False)

        fig.suptitle("Variable: "+data.name+"; Units: "+data.attrs["units"], fontsize=16)
        mpl_pane = pn.pane.Matplotlib(fig, dpi=144)
        return mpl_pane

    
    
    @param.depends("warmlevel","ssp", watch=False)
    def _GMT_context_plot(self): 
        """ Display static GMT plot using package data. """
        ## Plot dimensions 
        width=575
        height=300

        ## Read in data
        ssp119_data = pd.read_csv(ssp119, index_col='Year')
        ssp126_data = pd.read_csv(ssp126, index_col='Year')
        ssp245_data = pd.read_csv(ssp245, index_col='Year')
        ssp370_data = pd.read_csv(ssp370, index_col='Year')
        ssp585_data = pd.read_csv(ssp585, index_col='Year')
        hist_data = pd.read_csv(hist, index_col='Year')

        ## Plot figure
        hist_t = np.arange(1950,2015,1)
        cmip_t = np.arange(2015,2100,1)

        ## https://pyam-iamc.readthedocs.io/en/stable/tutorials/ipcc_colors.html
        c119 = "#00a9cf"
        c126 = "#003466"
        c245 = "#f69320"
        c370 = "#df0000"
        c585 = "#980002"

        ipcc_data = (hist_data.hvplot(y="Mean", color="k", label="Historical", width=width, height=height) *
                hist_data.hvplot.area(x="Year", y="5%", y2="95%", alpha=0.1, color="k", ylabel="°C", xlabel="", ylim=[-1,5], xlim=[1950,2100]) * # very likely range
                 ssp119_data.hvplot(y="Mean", color=c119, label="SSP1-1.9") *
                 # ssp126_data.hvplot.area(x="Year", y="5%", y2="95%", alpha=0.1, color=c126) * # very likely range
                 ssp126_data.hvplot(y="Mean", color=c126, label="SSP1-2.6") *
                 ssp245_data.hvplot(y="Mean", color=c245, label="SSP2-4.5") *
                 ssp370_data.hvplot(y="Mean", color=c370, label="SSP3-7.0") *
                 ssp585_data.hvplot(y="Mean", color=c585, label="SSP5-8.5")
                )


        # SSP intersection lines
        cmip_t = np.arange(2015,2101,1)

        # warming level connection lines & additional labeling
        warmlevel_line = hv.HLine(self.warmlevel).opts(color="black", line_width=1.0) * hv.Text(x=1964, y=self.warmlevel+0.25, text=".    " + str(self.warmlevel) + "°C warming level").opts(style=dict(text_font_size='8pt'))

        ssp_dict = {
            "SSP 2-4.5 -- Middle of the Road":(ssp245_data,c245),
            "SSP 3-7.0 -- Business as Usual":(ssp370_data,c370), 
            "SSP 5-8.5 -- Burn it All":(ssp585_data,c585)
        }
        
        ssp_selected = ssp_dict[self.ssp][0] # data selected 
        ssp_color = ssp_dict[self.ssp][1] # color corresponding to ssp selected 
        
        # Shading around selected SSP 
        ssp_shading = ssp_selected.hvplot.area(x="Year", y="5%", y2="95%", alpha=0.1, color=ssp_color) # very likely range
        
        # If the mean/upperbound/lowerbound does not cross threshold, set to 2100 (not visible)
        if (np.argmax(ssp_selected["Mean"] > self.warmlevel)) > 0:
                ssp_int = hv.VLine(cmip_t[0] + np.argmax(ssp_selected["Mean"] > self.warmlevel)).opts(color=ssp_color, line_dash="dashed", line_width=1)
        else:
            ssp_int = hv.VLine(cmip_t[0] + 2100).opts(color=ssp_color, line_dash="dashed", line_width=1)

        if (np.argmax(ssp_selected["95%"] > self.warmlevel)) > 0:
            ssp_firstdate = hv.VLine(cmip_t[0] + np.argmax(ssp_selected["95%"] > self.warmlevel)).opts(color=ssp_color,  line_width=1)
        else:
            ssp_firstdate = hv.VLine(cmip_t[0] + 2100).opts(color=ssp_color,  line_width=1)

        if (np.argmax(ssp_selected["5%"] > self.warmlevel)) > 0:
            ssp_lastdate = hv.VLine(cmip_t[0] + np.argmax(ssp_selected["5%"] > self.warmlevel)).opts(color=ssp_color,  line_width=1)
        else:
            ssp_lastdate = hv.VLine(cmip_t[0] + 2100).opts(color=ssp_color, line_width=1)


        ## Bar to connect firstdate and lastdate of threshold cross
        bar_y = -0.5
        yr_len = [(cmip_t[0] + np.argmax(ssp_selected["95%"] > self.warmlevel), bar_y), (cmip_t[0] + np.argmax(ssp_selected["5%"] > self.warmlevel), bar_y)]
        yr_rng = (np.argmax(ssp_selected["5%"] > self.warmlevel) - np.argmax(ssp_selected["95%"] > self.warmlevel))
        if yr_rng > 0:
            interval = hv.Path(yr_len).opts(color=ssp_color, line_width=1) * hv.Text(x=cmip_t[0] + np.argmax(ssp_selected["95%"] > self.warmlevel)+5,
                                                                            y=bar_y+0.25,
                                                                            text = str(yr_rng) + " yrs").opts(style=dict(text_font_size='8pt'))
        else: # Removes "bar" in case the upperbound is beyond 2100
            interval = hv.Path([(0,0), (0,0)]) # hardcoding for now, likely a better way to handle

        to_plot = ipcc_data * warmlevel_line * ssp_int * ssp_shading * ssp_lastdate * ssp_firstdate * interval
        to_plot.opts(opts.Overlay(title='Global surface temperature change relative to 1850-1900', fontsize=12))
        to_plot.opts(legend_position='bottom', fontsize=10)

        return to_plot
    

def _display_warming_levels(selections, location, _cat):

    # Warming levels object 
    warming_levels = WarmingLevels(selections=selections, location=location, catalog=_cat)
    
    # Create panel doodad!
    user_options = pn.Card(
            pn.Row(
                pn.Column(
                    pn.widgets.StaticText(name="", value='Warming Level (°C)'),
                    pn.widgets.RadioButtonGroup.from_param(warming_levels.param.warmlevel, name=""),
                    pn.widgets.Select.from_param(warming_levels.param.variable2, name="Data variable"),
                    pn.widgets.StaticText.from_param(selections.param.variable_description),
                    width = 230), 
                pn.Column(
                    pn.widgets.Select.from_param(warming_levels.param.location_subset2, name="Location"),
                    location.view,
                    width = 230)
                )
        , title="Data Options", collapsible=False, width=460, height=420
    )         
    
    GMT_plot = pn.Card(
            pn.widgets.Select.from_param(warming_levels.param.ssp, name="Scenario", width=250),
            warming_levels._GMT_context_plot,
            title="When is the warming level reached?", 
            collapsible=False, width=600, height=420
        ) 
    
    TMY = pn.Column(
        pn.widgets.Button.from_param(warming_levels.param.reload_data, button_type="primary", width=150, height=30),
        warming_levels._TMY_hourly_heatmap
    )
    
    map_tabs = pn.Card(
        pn.Tabs(
            ("Maps of individual simulations", warming_levels._GCM_PostageStamps),
            ("Maps of cross-model statistics: mean/median/max/min", pn.Row()), 
            ("Typical meteorological year", TMY), 
        ), 
    title="Regional response at selected warming level", 
    width = 800, height=500, collapsible=False,
    )
        
    
    panel_doodad = pn.Column(
        pn.Row(user_options, GMT_plot), 
        map_tabs
    )
    
    return panel_doodad


def postage_stamps(data,var,warming_threshold):    
    
    '''
    outputs postage stamp type figures
    showing anomaly maps for a given variable and threshold.
    
    
    ds = xarray dataset
    var = variable (string), currently tuned for T at 2m 
    (this is because the colorbar and cmap are hardcoded).
    warming_threshold = desired warming level (string),
    currently must be either:
    1.5C, 2.0C, 3.0C, 4.0C exactly.
    '''   
    
    var = data[var_str]
    var.attrs['reference_range'] = '1981','2010'# hist threshold


    # year ranges for each warming threshold for each model
    # this is a placeholder for a function which reads in these years
    # from the CSV file
    warming_15 = [(str(y-15),str(y+14)) for y in [2044,2050,2045,2053,2053]]
    warming_2 = [(str(y-15),str(y+14)) for y in [2055,2061,2056,2072,2069]]
    warming_3 = [(str(y-15),str(y+14)) for y in [2078,2077,2073,np.nan,np.nan]]
    warming_4 = [(str(y-15),str(y+14)) for y in [np.nan,np.nan,2091,np.nan,np.nan]]

    my_simulations = ['cesm2', 'cnrm-esm2-1', 'ec-earth3-veg', 'fgoals-g3', 'mpi-esm1-2-lr']

    # make dicts for simulation
    # and the year each threshold is reached
    # hardcoded for now
    # note: not all sims reach all thresholds
    # might want to warn users
    dict_15 = dict(zip(my_simulations, warming_15))
    dict_20 = dict(zip(my_simulations, warming_2))
    dict_30 = dict(zip(my_simulations[0:3], warming_3[0:3]))
    dict_40 = dict(zip(my_simulations[2:3], warming_4[2:3]))

    thresh_dict = {'1.5C' : dict_15,
             '2.0C' : dict_20,
             '3.0C' : dict_30,
             '4.0C' : dict_40}

    ### generate weights based off days per month
    month_length = data.time.dt.days_in_month
    ### first monthly weights, ie, days in month / days in year
    mon_wgts = month_length.groupby("time.year") / month_length.groupby("time.year").sum()
    ### then annual weights, days in year / total days

    ### want to ensure nans do not impact the weighting
    # code from https://ncar.github.io/esds/posts/2021/yearly-averages-xarray/
    # Setup our masking for nan values
    cond = var.isnull()
    ones = xr.where(cond, 0.0, 1.0)
    # Calculate the numerator
    var_sum = (var * mon_wgts).resample(time="AS").sum(dim="time")
    # Calculate the denominator
    ones_out = (ones * mon_wgts).resample(time="AS").sum(dim="time")
    # weighted mean
    wgt_ann_mean = var_sum / ones_out

    var_hist = var.sel(time=slice(*wgt_ann_mean.reference_range))

    ### now for annual weights, days in year / total days
    ### need to do this for each time period
    ### note: each time period differs by the year the  
    ### warming threshold is reached.
    year_length = month_length.groupby("time.year").sum() 
    year_length.name = "days_in_year"
    year_length = year_length.assign_coords(year=wgt_ann_mean.time.values)

    hist_time_slice = year_length.sel(year=slice(*var.reference_range))
    hist_ann_wgts = hist_time_slice / hist_time_slice.sum()
    hist_ann_wgts = hist_ann_wgts.rename({'year' : 'time'})

    # # get the weighted 30-year historical statistics
    ann_hist_mean = wgt_ann_mean.sel(time=slice(*wgt_ann_mean.reference_range))
    hist_wgtd = ann_hist_mean.weighted(hist_ann_wgts)
    hist_mean = hist_wgtd.mean("time")
    hist_mean.name = 'Historical mean'

    fig = Figure(figsize=(8, 9))

    # dictionary of simulations and years they reach the given threshold
    warm_dict = thresh_dict[thresh_str]

    # make a dataset of means for given T threshold
    # and then for the anomalies

    for i,sim in enumerate(warm_dict.keys()):

        my_years = warm_dict[sim]
        #### compute the weihted means

        var_warm = var.sel(simulation=sim,time=slice(*my_years))
        warm_time_slice = year_length.sel(year=slice(*my_years))
        warm_ann_wgts = warm_time_slice / warm_time_slice.sum()
        warm_ann_wgts = warm_ann_wgts.rename({'year' : 'time'})

        # get the weighted 30-year threshold statistics
        ann_warm_mean = wgt_ann_mean.sel(simulation=sim,time=slice(*my_years))
        warm_wgtd = ann_warm_mean.weighted(warm_ann_wgts)

        # and find the anomaly
        warm_anom = warm_wgtd.mean("time") - hist_mean.sel(simulation=sim)

        ######## PANEL 1 CODE
        ax = fig.add_subplot(2,3,i+1,projection=ccrs.LambertConformal())
        cs = warm_anom.plot(
                ax=ax, shading='auto', cmap="coolwarm", transform = ccrs.LambertConformal(),
                add_colorbar = False, vmin=-3,vmax=3)

        ax.set_title(sim)
        ax.coastlines(linewidth=1, color = 'black', zorder = 10) # Coastlines
        gl = ax.gridlines(linewidth=0.25, color='gray', alpha=0.9, crs=ccrs.PlateCarree(), 
                                  linestyle = '--',draw_labels=False, x_inline=False)

        gl.bottom_labels = True
        if (i==0) or (i==3):
            gl.left_labels = True    

        # need to concatenate across simulations
        # ... there has to be a better way
        if (i==0):
            warm_all_anoms = warm_anom

        elif (i==1):
            warm_all_anoms = xr.concat([warm_all_anoms,warm_anom],dim='simulation')

        else:
            warm_all_anoms = xr.concat([warm_all_anoms,warm_anom],dim='simulation')      


    # Adjust the location of the subplots on the page to make room for the colorbar
    fig.subplots_adjust(bottom=0.2, top=0.9, left=0.1, right=0.9,
                        wspace=0.1, hspace=0.4)
    # Add a colorbar axis at the bottom of the graph
    cbar_ax = fig.add_axes([0.2, 0.1, 0.65, 0.02])
    # Draw the colorbar
    cbar=fig.colorbar(cs, cax=cbar_ax,orientation='horizontal',
                     label='$^\circ$C')   
    fig.suptitle(var_str+ ' Anomalies for '+thresh_str+' Warming',y=.98)
    mpl_pane = pn.pane.Matplotlib(fig, dpi=144)
    display(mpl_pane)   


    ########## PANEL 2 CODE
    # now compute min, median, max anomalies
    min_anom = warm_all_anoms.min(dim='simulation')
    min_anom.name = "Min"
    max_anom = warm_all_anoms.max(dim='simulation')
    max_anom.name = "Max"
    med_anom = warm_all_anoms.median(dim='simulation')
    med_anom.name = "Median"

    stat_anoms = xr.merge([min_anom,med_anom,max_anom])

    fig = Figure(figsize=(8, 4))

    for ax_index, stat_str in zip([1,2,3,4],
        ["Median","Min","Max"]): 

    # Ideally these should all have the same colorbar
        ax = fig.add_subplot(1,3,ax_index,projection=ccrs.LambertConformal())
        cs = stat_anoms[stat_str].isel(scenario=0).plot(
            ax=ax, shading='auto', cmap="coolwarm",add_colorbar = False, vmin=-3,vmax=3
            )

        ax.set_title(stat_str)
        ax.coastlines(linewidth=1, color = 'black', zorder = 10) # Coastlines
        gl = ax.gridlines(linewidth=0.25, color='gray', alpha=0.9, crs=ccrs.PlateCarree(), 
                                  linestyle = '--',draw_labels=False, x_inline=False)

        gl.bottom_labels = gl.left_labels = True   


    # Adjust the location of the subplots on the page to make room for the colorbar
    fig.subplots_adjust(bottom=0.1, top=.95, left=0.1, right=0.88,
                        wspace=0.35, hspace=.4)
    # Add a colorbar axis at the bottom of the graph
    cbar_ax = fig.add_axes([0.9, 0.28, 0.04, 0.5])
    # Draw the colorbar
    cbar=fig.colorbar(cs, cax=cbar_ax,orientation='vertical',
                     label='$^\circ$C')   
    fig.suptitle(var_str+ ' Anomalies for '+thresh_str+' Warming Across Models',y=1)
    mpl_pane = pn.pane.Matplotlib(fig, dpi=144)
    display(mpl_pane)    