import cartopy.crs as ccrs
import hvplot.xarray
import hvplot.pandas
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
    def _GCM_PostageStamps(self): 
        def _get_data():    
            """Get data from AWS catalog"""
            xr_da = _read_from_catalog(
                selections=self.selections, 
                location=self.location, 
                cat=self.catalog
            )
            return xr_da
        
        data = _get_data()
        
        # Crop data to improve speed during testing
        data_cropped = data.isel(x=np.arange(10,30), y=np.arange(10,30))

        fig = Figure(figsize=(6, 4), tight_layout=True)

        # Placeholder indices 
        for ax_index, time_index in zip([1,2],[0,1]): 
            # Ideally these should all have the same colorbar
            ax = fig.add_subplot(1,2,ax_index,projection=ccrs.LambertConformal())
            xr_pl = data_cropped.isel(time=time_index,simulation=0,scenario=0).plot(
                ax=ax, shading='auto', cmap="coolwarm"
                )
            ax.set_title("plot {0}".format(ax_index))
            ax.coastlines(linewidth=1, color = 'black', zorder = 10) # Coastlines
            ax.gridlines(linewidth=0.25, color='gray', alpha=0.9, crs=ccrs.PlateCarree(), linestyle = '--',draw_labels=False)

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
                 ssp370_data.hvplot.area(x="Year", y="5%", y2="95%", alpha=0.1, color=c370) * # very likely range
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

        to_plot = ipcc_data * warmlevel_line * ssp_int * ssp_lastdate * ssp_firstdate * interval
        to_plot.opts(opts.Overlay(title='Global surface temperature change relative to 1850-1900', fontsize=12))
        to_plot.opts(legend_position='bottom', fontsize=10)

        return to_plot
        
    
    

def _display_warming_levels(selections, location, _cat):

    # Warming levels object 
    warming_levels = WarmingLevels(selections=selections, location=location, catalog=_cat)
    
    # Create panel doodad!
    user_options = pn.Card(
        pn.Column(
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
                ),
            pn.widgets.Button.from_param(warming_levels.param.reload_data, button_type="primary", width=200, height=50)
        )
        , title="Data Options", collapsible=False, width=460, height=420
    )         
    
    GMT_plot = pn.Card(
            pn.widgets.Select.from_param(warming_levels.param.ssp, name="Scenario", width=250),
            warming_levels._GMT_context_plot,
            title="Global Mean Temperature Context Plot", 
            collapsible=False, width=600, height=420
        ) 
    
    
    map_tabs = pn.Card(
        pn.Tabs(
            ("Model means", warming_levels._GCM_PostageStamps),
            ("Model difference from historical", pn.Row()), 
            ("Typical meteorological year", pn.Row()), 
        ), 
    title="Global Circulation Model Maps", width = 800, height=500, collapsible=False
    )
        
    
    panel_doodad = pn.Column(
        pn.Row(user_options, GMT_plot), 
        map_tabs
    )
    
    return panel_doodad