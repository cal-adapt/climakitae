import cartopy.crs as ccrs
import hvplot.xarray
import hvplot.pandas
import xarray as xr
import holoviews as hv
from holoviews import opts
from matplotlib.figure import Figure
import numpy as np
import pandas as pd
import geopandas as gpd
import param
import panel as pn
import intake
import intake
import s3fs
import pyproj
from shapely.geometry import box
from shapely.ops import transform
import regionmask
from .data_loaders import _read_from_catalog
from .selectors import DataSelector, LocSelectorArea
import pkg_resources
import matplotlib.colors as mcolors
import logging
logging.getLogger("param").setLevel(logging.CRITICAL)

# Import package data
ssp119 = pkg_resources.resource_filename('climakitae', 'data/tas_global_SSP1_1_9.csv')
ssp126 = pkg_resources.resource_filename('climakitae', 'data/tas_global_SSP1_2_6.csv')
ssp245 = pkg_resources.resource_filename('climakitae', 'data/tas_global_SSP2_4_5.csv')
ssp370 = pkg_resources.resource_filename('climakitae', 'data/tas_global_SSP3_7_0.csv')
ssp585 = pkg_resources.resource_filename('climakitae', 'data/tas_global_SSP5_8_5.csv')
hist = pkg_resources.resource_filename('climakitae', 'data/tas_global_Historical.csv')

# Global warming levels file (years when warming level is reached)
gwl_file = pkg_resources.resource_filename('climakitae', 'data/gwl_1981-2010ref.csv')
gwl_times = pd.read_csv(gwl_file, index_col=[0,1])

## Read in data
ssp119_data = pd.read_csv(ssp119, index_col='Year')
ssp126_data = pd.read_csv(ssp126, index_col='Year')
ssp245_data = pd.read_csv(ssp245, index_col='Year')
ssp370_data = pd.read_csv(ssp370, index_col='Year')
ssp585_data = pd.read_csv(ssp585, index_col='Year')
hist_data = pd.read_csv(hist, index_col='Year')


def _get_postage_data(area_subset2, cached_area2, variable2, location):

    """
    This function pulls pre-compiled data from AWS and then subsets it using recylced code from the data_loaders module

    Args:
        area_subset2 (str): area subset
        cached_area2 (str): cached area
        variable2 (str): variable
        location (LocSelectorArea object from selectors.py): location object containing boundary information

    Returns:
        postage_data (xr.DataArray): data to use for creating postage stamp data

    """

    # Get data from AWS
    fs = s3fs.S3FileSystem(anon=True)
    fp = fs.open('s3://cadcat/tmp/t2m_and_rh_9km_ssp370_monthly_CA.nc')
    pkg_data = xr.open_dataset(fp)

    # Select variable & scenario from dataset
    da = pkg_data[variable2]
    postage_data = da.where(da.scenario == "Historical + SSP 3-7.0 -- Business as Usual", drop=True)

    # Perform area subset based on user selections
    if area_subset2 == "states":
        ds_region = None # Data is already subsetted to CA
    elif area_subset2 in ["CA watersheds","CA counties"]:
        shape_index = int(
            location._geography_choose[area_subset2][cached_area2]
        )
        if area_subset2 == "CA watersheds":
            shape = location._geographies._ca_watersheds
            shape = shape[shape["OBJECTID"] == shape_index].iloc[0].geometry
            wgs84 = pyproj.CRS('EPSG:4326')
            psdo_merc = pyproj.CRS('EPSG:3857')
            project = pyproj.Transformer.from_crs(psdo_merc, wgs84, always_xy=True).transform
            shape = transform(project, shape)
        elif area_subset2 == "CA counties":
            shape = location._geographies._ca_counties
            shape = shape[shape.index == shape_index].iloc[0].geometry
        ds_region = regionmask.Regions(
            [shape], abbrevs=["geographic area"], name="area mask"
        )

    if ds_region:
        # Attributes are arrays, must be items to call pyproj.CRS.from_cf
        for attr_np in ["earth_radius","latitude_of_projection_origin","longitude_of_central_meridian"]:
            postage_data['Lambert_Conformal'].attrs[attr_np] = postage_data['Lambert_Conformal'].attrs[attr_np].item()
        data_crs = ccrs.CRS(pyproj.CRS.from_cf(postage_data['Lambert_Conformal'].attrs))
        output = data_crs.transform_points(ccrs.PlateCarree(),
                                               x=ds_region.coords[0][:,0],
                                               y=ds_region.coords[0][:,1])

        postage_data = postage_data.sel(x=slice(np.nanmin(output[:,0]), np.nanmax(output[:,0])),
            y=slice(np.nanmin(output[:,1]), np.nanmax(output[:,1])))

        mask = ds_region.mask(postage_data.lon, postage_data.lat, wrap_lon=False)
        assert (
            False in mask.isnull()
        ), "Insufficient gridcells are contained within the bounds."
        postage_data = postage_data.where(np.isnan(mask) == False)

    return postage_data


def get_anomaly_data(data, warmlevel=3.0):
    """
    Helper function for calculating warming level anomalies.

    Args:
        data (xr.DataArray)
        warmlevel (float): warming level

    Returns:
        warm_all_anoms (xr.DatArray): warming level anomalies computed from input data
    """
    model_case = {'cesm2':'CESM2', 'cnrm-esm2-1':'CNRM-ESM2-1',
              'ec-earth3-veg':'EC-Earth3-Veg', 'fgoals-g3':'FGOALS-g3',
              'mpi-esm1-2-lr':'MPI-ESM1-2-LR'}

    all_sims = xr.Dataset()
    for simulation in data.simulation.values:
        for scenario in ['ssp370']:
            one_ts = data.sel(simulation=simulation).squeeze() #,scenario=scenario) #scenario names are longer strings
            centered_time = pd.to_datetime(gwl_times[str(float(warmlevel))][model_case[simulation]][scenario]).year
            if not np.isnan(centered_time):
                anom = one_ts.sel(time=slice(str(centered_time-15),str(centered_time+14))).mean('time') - one_ts.sel(time=slice('1981','2010')).mean('time')
                all_sims[simulation] = anom
    return all_sims.to_array('simulation')


def _compute_vmin_vmax(da_min,da_max):
    """Compute min, max, and center for plotting"""
    vmin = np.nanpercentile(da_min, 1)
    vmax = np.nanpercentile(da_max, 99)
    # define center for diverging symmetric data
    if (vmin < 0) and (vmax > 0):
        # dabs = abs(vmax) - abs(vmin)
        sopt = True
    else:
        sopt = None
    return vmin, vmax, sopt

def _make_hvplot(data, clabel, clim, cmap, sopt, title, width=200, height=225): 
    """Make single map"""
    _plot = data.hvplot.image(
        x="x", y="y", 
        grid=True, width=width, height=height, xaxis=None, yaxis=None,
        clabel=clabel, clim=clim, cmap=cmap, # Colorbar 
        symmetric=sopt, title=title
    )
    return _plot


class WarmingLevels(param.Parameterized):

    ## ---------- Reset certain DataSelector and LocSelectorArea options ----------
    def __init__(self, *args, **params):
        super().__init__(*args, **params)
        
        # Selectors defaults
        self.selections.append_historical = True
        self.selections.area_average = False
        self.selections.resolution = "9 km"
        self.selections.scenario = ["SSP 3-7.0 -- Business as Usual"]
        self.selections.time_slice = (1980,2100)
        self.selections.timescale = "monthly"
        self.selections.variable = "Air Temperature at 2m"

        # Location defaults
        self.location.area_subset = 'states'
        self.location.cached_area = 'CA'

        # Postage data and anomalies defaults
        self.postage_data = _get_postage_data(
            area_subset2=self.location.area_subset, cached_area2=self.location.cached_area, variable2=self.selections.variable, location=self.location
        )
        self._warm_all_anoms = get_anomaly_data(data=self.postage_data, warmlevel=1.5)

    ## ---------- Params & global variables ----------

    warmlevel = param.ObjectSelector(default=1.5,
        objects=[1.5, 2, 3]
    )
    ssp = param.ObjectSelector(default="All",
        objects=[
            "All",
            "SSP 1-1.9 -- Very Low Emissions Scenario", 
            "SSP 1-2.6 -- Low Emissions Scenario", 
            "SSP 2-4.5 -- Middle of the Road",
            "SSP 3-7.0 -- Business as Usual",
            "SSP 5-8.5 -- Burn it All"
        ]
    )


    # For reloading postage stamp data and plots
    reload_data2 = param.Action(lambda x: x.param.trigger('reload_data2'), label='Reload Data')
    changed_loc_and_var = param.Boolean(default=True)

    variable2 = param.ObjectSelector(default="Air Temperature at 2m",
        objects=["Air Temperature at 2m","Relative Humidity"]
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

    @param.depends("reload_data2", watch=True)
    def _update_postage_data(self):
        """If the button was clicked and the location or variable was changed,
        reload the postage stamp data from AWS"""
        if self.changed_loc_and_var == True:
            self.postage_data = _get_postage_data(area_subset2=self.area_subset2, cached_area2=self.cached_area2, variable2=self.variable2, location=self.location)
            self.changed_loc_and_var = False
        self._warm_all_anoms = get_anomaly_data(data=self.postage_data, warmlevel=self.warmlevel)


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

    @param.depends("reload_data2", watch=False)
    def _GCM_PostageStamps_MAIN(self):
        
        # Get plot data 
        all_plot_data = self._warm_all_anoms
        if self.variable2 == "Relative Humidity": 
            all_plot_data = all_plot_data*100
            
        # Get int number of simulations
        num_simulations = len(all_plot_data.simulation.values)
            
        # Set up plotting arguments 
        clabel = self.variable2 + " ("+self.postage_data.attrs["units"]+")"
        if self.variable2 == "Air Temperature at 2m":
            cmap = "YlOrRd"
        elif self.variable2 == "Relative Humidity":
            cmap = "PuOr"
        else: 
            cmap = "viridis"
            
        if (num_simulations <= 3):
            width = 200 
            height = 225
        else: # Make plots a little smaller if there are more than 3 
            width = 190
            height = 210
            

        # Compute 1% min and 99% max of all simulations
        vmin_l, vmax_l = [],[]
        for sim in range(num_simulations):
            data = all_plot_data.isel(simulation=sim)
            vmin_i, vmax_i, sopt = _compute_vmin_vmax(data, data)
            vmin_l.append(vmin_i)
            vmax_l.append(vmax_i)
        vmin = min(vmin_l)
        vmax = max(vmax_l)
    
        
        # Make each plot 
        all_plots = _make_hvplot( # Need to make the first plot separate from the loop
            data=all_plot_data.isel(simulation=0), 
            clabel=clabel, clim=(vmin,vmax), cmap=cmap, sopt=sopt,
            title=all_plot_data.isel(simulation=0).simulation.item(), 
            width=width, height=height
        )
        for sim_i in range(1,num_simulations): 
            pl_i = _make_hvplot(
                data=all_plot_data.isel(simulation=sim_i), 
                clabel=clabel, clim=(vmin,vmax), cmap=cmap, sopt=sopt,
                title=all_plot_data.isel(simulation=sim_i).simulation.item(), 
                width=width, height=height
            )
            all_plots += pl_i
        
        all_plots.cols(3) # Organize into 3 columns 
        all_plots.opts(title=self.variable2+ ': Anomalies for '+str(self.warmlevel)+'°C Warming by Simulation') # Add title
        all_plots.opts(toolbar="right") # Set toolbar location
        all_plots.opts(hv.opts.Layout(merge_tools=True)) # Merge toolbar 
        return all_plots
        
        
    @param.depends("reload_data2", watch=False)
    def _GCM_PostageStamps_STATS(self):
        
        # Get plot data 
        all_plot_data = self._warm_all_anoms
        if self.variable2 == "Relative Humidity": 
            all_plot_data = all_plot_data*100
        
        # Compute stats
        min_data = all_plot_data.min(dim='simulation')
        max_data = all_plot_data.max(dim='simulation')
        med_data = all_plot_data.median(dim='simulation')
        mean_data = all_plot_data.mean(dim='simulation')
        
        # Set up plotting arguments 
        clabel = self.variable2 + " ("+self.postage_data.attrs["units"]+")"
        vmin, vmax, sopt = _compute_vmin_vmax(min_data,max_data)
        if self.variable2 == "Air Temperature at 2m":
            cmap = "YlOrRd"
        elif self.variable2 == "Relative Humidity":
            cmap = "PuOr"
        else: 
            cmap = "viridis"
        
        # Make plots
        min_plot = _make_hvplot(data=min_data, clabel=clabel, cmap=cmap, clim=(vmin,vmax), sopt=sopt, title="Minimum")
        max_plot = _make_hvplot(data=max_data, clabel=clabel, cmap=cmap,  clim=(vmin,vmax), sopt=sopt, title="Maximum")
        med_plot = _make_hvplot(data=med_data, clabel=clabel, cmap=cmap,  clim=(vmin,vmax), sopt=sopt, title="Median")
        mean_plot = _make_hvplot(data=mean_data, clabel=clabel, cmap=cmap, clim=(vmin,vmax), sopt=sopt, title="Mean")

        all_plots = (mean_plot+med_plot+min_plot+max_plot)
        all_plots.opts(title=self.variable2+ ': Anomalies for '+str(self.warmlevel)+'°C Warming Across Models') # Add title
        all_plots.opts(toolbar="below") # Set toolbar location
        all_plots.opts(hv.opts.Layout(merge_tools=True)) # Merge toolbar 
        return all_plots



    @param.depends("warmlevel","ssp", watch=False)
    def _GMT_context_plot(self):
        """ Display GMT plot using package data that updates whenever the warming level or SSP is changed by the user. """
        ## Plot dimensions
        width=575
        height=300

        ## Plot figure
        hist_t = np.arange(1950,2015,1)
        cmip_t = np.arange(2015,2100,1)

        ## https://pyam-iamc.readthedocs.io/en/stable/tutorials/ipcc_colors.html
        c119 = "#00a9cf"
        c126 = "#003466"
        c245 = "#f69320"
        c370 = "#df0000"
        c585 = "#980002"

        ipcc_data = (
            hist_data.hvplot(y="Mean", color="k", label="Historical", width=width, height=height) *
            hist_data.hvplot.area(x="Year", y="5%", y2="95%", alpha=0.1, color="k", ylabel="°C", xlabel="", ylim=[-1,5], xlim=[1950,2100])
        ) 
        if self.ssp == "All":
            ipcc_data = (
                ipcc_data *
                ssp119_data.hvplot(y="Mean", color=c119, label="SSP1-1.9") *
                ssp126_data.hvplot(y="Mean", color=c126, label="SSP1-2.6") *
                ssp245_data.hvplot(y="Mean", color=c245, label="SSP2-4.5") *
                ssp370_data.hvplot(y="Mean", color=c370, label="SSP3-7.0") *
                ssp585_data.hvplot(y="Mean", color=c585, label="SSP5-8.5")
            )
        elif self.ssp == "SSP 1-1.9 -- Very Low Emissions Scenario": 
            ipcc_data = (ipcc_data * ssp119_data.hvplot(y="Mean", color=c119, label="SSP1-1.9"))
        elif self.ssp == "SSP 1-2.6 -- Low Emissions Scenario": 
            ipcc_data = (ipcc_data * ssp126_data.hvplot(y="Mean", color=c126, label="SSP1-2.6"))
        elif self.ssp == "SSP 2-4.5 -- Middle of the Road": 
            ipcc_data = (ipcc_data * ssp245_data.hvplot(y="Mean", color=c245, label="SSP2-4.5"))
        elif self.ssp == "SSP 3-7.0 -- Business as Usual": 
            ipcc_data = (ipcc_data * ssp370_data.hvplot(y="Mean", color=c370, label="SSP3-7.0"))
        elif self.ssp == "SSP 5-8.5 -- Burn it All": 
            ipcc_data = (ipcc_data * ssp585_data.hvplot(y="Mean", color=c585, label="SSP5-8.5"))


        # SSP intersection lines
        cmip_t = np.arange(2015,2101,1)

        # Warming level connection lines & additional labeling
        warmlevel_line = hv.HLine(self.warmlevel).opts(color="black", line_width=1.0) * hv.Text(x=1964, y=self.warmlevel+0.25, text=".    " + str(self.warmlevel) + "°C warming level").opts(style=dict(text_font_size='8pt'))

        # Create plot 
        to_plot = ipcc_data * warmlevel_line

        if self.ssp != "All": 
            # Label to give addional plot info 
            info_label = "Intersection information"
            
            # Add interval line and shading around selected SSP 
            ssp_dict = {
                "SSP 1-1.9 -- Very Low Emissions Scenario":(ssp119_data, c119), 
                "SSP 1-2.6 -- Low Emissions Scenario":(ssp126_data, c126), 
                "SSP 2-4.5 -- Middle of the Road":(ssp245_data,c245),
                "SSP 3-7.0 -- Business as Usual":(ssp370_data,c370),
                "SSP 5-8.5 -- Burn it All":(ssp585_data,c585)
            }
            
            ssp_selected = ssp_dict[self.ssp][0] # data selected
            ssp_color = ssp_dict[self.ssp][1] # color corresponding to ssp selected

            # Shading around selected SSP
            ci_label = "90% interval"
            ssp_shading = ssp_selected.hvplot.area(
                x="Year", 
                y="5%", 
                y2="95%", 
                alpha=0.28, 
                color=ssp_color, 
                label=ci_label
            ) 
            to_plot = to_plot*ssp_shading 

            # If the mean/upperbound/lowerbound does not cross threshold, set to 2100 (not visible)
            if (np.argmax(ssp_selected["Mean"] > self.warmlevel)) > 0:
                
                # Add dashed line 
                label1 = "Warming level reached"
                year_warmlevel_reached = ssp_selected.where(ssp_selected["Mean"] > self.warmlevel).dropna().index[0]
                ssp_int = hv.Curve(
                    [[year_warmlevel_reached,-2],[year_warmlevel_reached,10]], 
                    label=label1
                ).opts(color=ssp_color, line_dash="dashed", line_width=1) 
                ssp_int = ssp_int * hv.Text(
                    x=year_warmlevel_reached-2, y=4.5, 
                    text=str(int(year_warmlevel_reached)), 
                    rotation=90, 
                    label=label1
                ).opts(style=dict(text_font_size='8pt',color=ssp_color))
                to_plot *= ssp_int # Add to plot 
                
            if ((np.argmax(ssp_selected["95%"] > self.warmlevel)) > 0) and ((np.argmax(ssp_selected["5%"] > self.warmlevel)) > 0):
                # Make 95% CI line
                x_95 = cmip_t[0] + np.argmax(ssp_selected["95%"] > self.warmlevel)
                ssp_firstdate = hv.Curve(
                    [[x_95,-2],[x_95,10]], 
                    label=ci_label
                ).opts(color=ssp_color, line_width=1)
                to_plot *= ssp_firstdate
                
                # Make 5% CI line
                x_5 = cmip_t[0] + np.argmax(ssp_selected["5%"] > self.warmlevel)
                ssp_lastdate = hv.Curve(
                    [[x_5,-2],[x_5,10]], 
                    label=ci_label
                ).opts(color=ssp_color, line_width=1)
                to_plot *= ssp_lastdate 
                
                ## Bar to connect firstdate and lastdate of threshold cross
                bar_y = -0.5
                yr_len = [(x_95, bar_y), (x_5, bar_y)]
                yr_rng = (np.argmax(ssp_selected["5%"] > self.warmlevel) - np.argmax(ssp_selected["95%"] > self.warmlevel))
                if yr_rng > 0:
                    interval = hv.Curve(
                        [[x_95,bar_y],[x_5,bar_y]],
                        label=ci_label
                    ).opts(color=ssp_color, line_width=1) * hv.Text(
                        x=x_95+5, 
                        y=bar_y+0.25, 
                        text=str(yr_rng) + "yrs", 
                        label=ci_label
                    ).opts(style=dict(text_font_size='8pt', color=ssp_color))
                    
                    to_plot *= interval

        to_plot.opts(opts.Overlay(title='Global mean surface temperature change relative to 1850-1900', fontsize=12))
        to_plot.opts(legend_position='bottom', fontsize=10)

        return to_plot


def _display_warming_levels(selections, location, _cat):

    # Warming levels object
    warming_levels = WarmingLevels(selections=selections, location=location)

    # Create panel doodad!
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
        , title="Data Options", collapsible=False, width=460, height=515
    )

    GMT_plot = pn.Card(
            pn.Column(
                "Shading around selected global emissions scenario shows 90% interval across different simulations. Dotted line indicates when the multi-model ensemble reaches the selected warming level, while solid vertical lines indicate when the earliest and latest simulations of that scenario reach the warming level. Figure and data are reproduced from the [IPCC AR6 Summary for Policymakers Fig 8](https://www.ipcc.ch/report/ar6/wg1/figures/summary-for-policymakers/figure-spm-8/).",
                pn.widgets.Select.from_param(warming_levels.param.ssp, name="Scenario", width=250),
                warming_levels._GMT_context_plot,
            ),
            title="When do different scenarios reach the warming level?",
            collapsible=False, width=600, height=515
        )

    postage_stamps_MAIN = pn.Column(
        pn.widgets.StaticText(
            value="Panels show difference between 30-year average centered on the year each GCM (name of model titles each panel) reaches the specified warming level and average from 1981-2010.",
            width=800
        ),
        pn.Row(
            warming_levels._GCM_PostageStamps_MAIN,
            pn.Column(
                pn.widgets.StaticText(
                    value="<br><br><br>", 
                    width=200
                ),
                pn.widgets.StaticText(
                    value="<b>Tip</b>: There's a toolbar to the side of the maps. \
        Try clicking the magnifying glass to zoom in on a particular region. \
        You can also click the save button to save a copy of the figure to your computer.", 
                    width=200, 
                    style={"border":"1.2px red solid","padding":"5px","border-radius":"4px","font-size":"13px"})
            )
        )
    )

    postage_stamps_STATS = pn.Column(
        pn.widgets.StaticText(
            value="Panels show simulation that represents average, median, minimum, or maximum conditions across all models. Minimum and maximum values were calculated across simulations for each grid cell, so one map may contain grid cells from different simulations. Median and mean maps show those respective summaries across simulations at each grid cell.",
            width=800
        ),
        warming_levels._GCM_PostageStamps_STATS
    )

    map_tabs = pn.Card(
        pn.Tabs(
            ("Maps of individual simulations", postage_stamps_MAIN),
            ("Maps of cross-model statistics: mean/median/max/min", postage_stamps_STATS),
        ),
    title="Regional response at selected warming level",
    width = 850, height=600, collapsible=False,
    )

    panel_doodad = pn.Column(
        pn.Row(user_options, GMT_plot),
        map_tabs
    )

    return panel_doodad
