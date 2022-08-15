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


def GMTContextPlot(width=500,height=300): 
    """ Display static GMT plot using package data. """
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
                 ssp126_data.hvplot.area(x="Year", y="5%", y2="95%", alpha=0.1, color=c126) * # very likely range
                 ssp126_data.hvplot(y="Mean", color=c126, label="SSP1-2.6") *
                 ssp245_data.hvplot(y="Mean", color=c245, label="SSP2-4.5") *
                 ssp370_data.hvplot.area(x="Year", y="5%", y2="95%", alpha=0.1, color=c370) * # very likely range
                 ssp370_data.hvplot(y="Mean", color=c370, label="SSP3-7.0") *
                 ssp585_data.hvplot(y="Mean", color=c585, label="SSP5-8.5")
                )

    # 3.0°C connection lines
    warmlevel = 3.0
    warmlevel_line =  hv.HLine(warmlevel).opts(color="black", line_width=1.0)

    # SSP intersection lines
    ssp370_int = hv.VLine(cmip_t[0] + np.argmax(ssp370_data["Mean"] > warmlevel)).opts(color=c370, line_dash="dashed", line_width=1)
    ssp585_int = hv.VLine(cmip_t[0] + np.argmax(ssp585_data["Mean"] > warmlevel)).opts(color=c585, line_dash="dashed", line_width=1)

    to_plot = ipcc_data * warmlevel_line * ssp370_int * ssp585_int
    to_plot.opts(opts.Overlay(title='Global surface temperature change relative to 1850-1900', fontsize=12))
    to_plot.opts(legend_position='bottom')

    return to_plot


class ScenarioSSP(param.Parameterized): 
    """Create new parameter object that is an object selector for the scenarios object"""
    scenario2 = param.ObjectSelector(default='SSP 3-7.0 -- Business as Usual', 
        objects=['SSP 2-4.5 -- Middle of the Road','SSP 3-7.0 -- Business as Usual','SSP 5-8.5 -- Burn it All']
        )    
    def __init__(self, *args, **params):
        super().__init__(*args, **params)
        
    @param.depends("scenario2", watch=True)
    def _update_selections(self): 
        modified_scenario_list = [self.scenario2]
        self.selections.scenario = modified_scenario_list
    
    
def _load_default_data(selections, location, catalog, modified_scenario, area_average=False): 
    """Load default data to be displayed whenever app.explore() is called """
    
    modified_scenario_list = [modified_scenario.scenario2]

    selections.append_historical = False
    selections.area_average = area_average
    selections.resolution = "45 km"
    selections.scenario = modified_scenario_list
    selections.time_slice = (2015,2100)
    selections.timescale = "monthly" 
    selections.variable = "Air Temperature at 2m"
    
    default_data = _read_from_catalog(selections=selections, 
                                   location=location, 
                                   cat=catalog)
    return default_data


def GCM_PostageStamps(data): 
    
    # Crop data to improve speed during testing
    data_cropped = data.isel(x=np.arange(50,100), y=np.arange(30,80))
    
    fig = Figure(figsize=(6, 4), tight_layout=True)

    # Placeholder indices 
    for ax_index, time_index, plot_title in zip([1,2,3,4],[0,1,2,3],
        ["mean (placeholder)","median (placeholder)","min (placeholder)","max (placeholder)"]): 
        # Ideally these should all have the same colorbar
        ax = fig.add_subplot(2,2,ax_index,projection=ccrs.LambertConformal())
        xr_pl = data_cropped.isel(time=time_index,simulation=0,scenario=0).plot(
            ax=ax, shading='auto', cmap="coolwarm"
            )
        ax.set_title(plot_title)
        ax.coastlines(linewidth=1, color = 'black', zorder = 10) # Coastlines
        ax.gridlines(linewidth=0.25, color='gray', alpha=0.9, crs=ccrs.PlateCarree(), linestyle = '--',draw_labels=False)

    mpl_pane = pn.pane.Matplotlib(fig, dpi=144)
    return mpl_pane

def AreaAverageLinePlot(data): 
    data_subset = data.isel(time=np.arange(0,30), scenario=0)
    lineplot = data_subset.hvplot.line(
       x="time", y="Air Temperature at 2m", by="simulation"
    )
    return lineplot

def _display_warming_levels(selections, location, _cat):
    # Load default data 
    modified_scenario = ScenarioSSP(selections=selections)
    #default_data_area_average = _load_default_data(area_average=True, selections=selections, location=location, catalog=_cat, modified_scenario=modified_scenario)
    default_data = _load_default_data(area_average=False, selections=selections, location=location, catalog=_cat, modified_scenario=modified_scenario)
    
    # Create panel doodad!
    user_options = pn.Card(
        pn.Row(
            pn.Column(
                pn.widgets.Select.from_param(selections.param.resolution, name="Model resolution"),
                pn.widgets.Select.from_param(modified_scenario.param.scenario2, name="SSP"),
                location.param.area_subset,
                location.param.cached_area,
                width = 210
                ), 
            location.view
            )
        , title="Data Options", collapsible=False, width = 440, height=290
    )         
    
    GMT_plot = pn.Card(
            GMTContextPlot(width=525),
            title="Global Mean Temperature Context Plot", 
            collapsible=False, width=550, height=360
        ) 
    
    area_average_line_plot = pn.Card(
            GMTContextPlot(width=525), # Replace with area average plot
            title="Area Average Line Plot", 
            collapsible=False, width=550, height=360
        ) 
    
    postage_stamps = pn.Card(
        GCM_PostageStamps(data=default_data),
        collapsible=False,
        width = 440, height=340,
        title="Global Circulation Model Maps"
    )
            
    left_column = pn.Column(
        user_options, 
        postage_stamps
    )
    
    right_column = pn.Column(GMT_plot, area_average_line_plot)
    
    panel_doodad = pn.Row(left_column, right_column)
    
    return panel_doodad