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

# Import package data
ssp119 = pkg_resources.resource_filename('climakitae', 'data/tas_global_SSP1_1_9.csv')
ssp126 = pkg_resources.resource_filename('climakitae', 'data/tas_global_SSP1_2_6.csv')
ssp245 = pkg_resources.resource_filename('climakitae', 'data/tas_global_SSP2_4_5.csv')
ssp370 = pkg_resources.resource_filename('climakitae', 'data/tas_global_SSP3_7_0.csv')
ssp585 = pkg_resources.resource_filename('climakitae', 'data/tas_global_SSP5_8_5.csv')
hist = pkg_resources.resource_filename('climakitae', 'data/tas_global_Historical.csv')
cached_tmy_files = [pkg_resources.resource_filename('climakitae', 'data/cached_tmy/'+file) for file in pkg_resources.resource_listdir('climakitae', 'data/cached_tmy')]

# Global warming levels file (years when warming level is reached)
gwl_file = pkg_resources.resource_filename('climakitae', 'data/gwl_1981-2010ref.csv')
gwl_times = pd.read_csv(gwl_file, index_col=[0,1])

# # URLs to shapefiles for example point data from CEC (power plants and substations available)
# CEC_shapefile_URLs = {
#     'power_plants' : "https://opendata.arcgis.com/api/v3/datasets/4a702cd67be24ae7ab8173423a768e1b_0/downloads/data?format=geojson&spatialRefId=4326&where=1%3D1",
#     'substations' : "https://cecgis-caenergy.opendata.arcgis.com/datasets/CAEnergy::california-electric-substations.geojson?outSR=%7B%22latestWkid%22%3A3857%2C%22wkid%22%3A102100%7D"
# }
# # Read in the values once (so that it does not re-download every time plots are updated)
# power_plants = gpd.read_file(CEC_shapefile_URLs['power_plants']).rename(columns = {'Lon_WGS84':'lon', 'Lat_WGS84':'lat'})
# substations = gpd.read_file(CEC_shapefile_URLs['substations']).rename(columns = {'Lon_WGS84':'lon', 'Lat_WGS84':'lat'})

## Read in data
ssp119_data = pd.read_csv(ssp119, index_col='Year')
ssp126_data = pd.read_csv(ssp126, index_col='Year')
ssp245_data = pd.read_csv(ssp245, index_col='Year')
ssp370_data = pd.read_csv(ssp370, index_col='Year')
ssp585_data = pd.read_csv(ssp585, index_col='Year')
hist_data = pd.read_csv(hist, index_col='Year')



def read_cached_tmy_df(cached_tmy_files, variable, warmlevel, cached_area):
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
    return df


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
    ssp = param.ObjectSelector(default="SSP 3-7.0 -- Business as Usual",
        objects=["SSP 2-4.5 -- Middle of the Road","SSP 3-7.0 -- Business as Usual","SSP 5-8.5 -- Burn it All"]
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

#     # Option to overlay CEC point data on the MAIN postage stamp plots
#     overlay_MAIN = param.ObjectSelector(default = "None", 
#         objects = ["None", "Power plants", "Substations"], 
#         label = "Infrastructure point data"
#     )

#     # Option to overlay CEC point data on the STATS postage stamp plots
#     overlay_STATS = param.ObjectSelector(default = "None", 
#         objects = ["None", "Power plants", "Substations"], 
#         label = "Infrastructure point data"
#     )


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
    def _TMY_hourly_heatmap(self):
        """Generate a TMY hourly heatmap using hourly data"""
#         def _get_hist_heatmap_data():
#             """Get historical data from AWS catalog"""
#             heatmap_selections = self.selections
#             heatmap_selections.append_historical = False
#             heatmap_selections.area_average = True
#             heatmap_selections.resolution = "45 km"
#             heatmap_selections.scenario = ["Historical Climate"]
#             heatmap_selections.time_slice = (1981,2010) # to match historical 30-year average
#             heatmap_selections.timescale = "hourly"
#             xr_da = _read_from_catalog(
#                 selections=heatmap_selections,
#                 location=self.location,
#                 cat=self.catalog
#             )

#             # if self.variable2 == ('Precipitation (total)'):   # need to include snowfall eventually
#             #     xr_da = deaccumulate_precip(xr_da)

#             return xr_da


        # hard-coding in for now
        warming_year_average_range = {
            1.5 : (2034,2063),
            2 : (2047,2076),
            3 : (2061,2090),
            4 : (2076, 2100)
        }
#         def _get_future_heatmap_data():
#             """Gets data from AWS catalog based upon desired warming level"""
#             heatmap_selections = self.selections
#             heatmap_selections.append_historical = False
#             heatmap_selections.area_average = True
#             heatmap_selections.resolution = "45 km"
#             heatmap_selections.scenario = ["SSP 3-7.0 -- Business as Usual"]
#             heatmap_selections.time_slice = warming_year_average_range[self.warmlevel]
#             heatmap_selections.timescale = "hourly"
#             xr_da2 = _read_from_catalog(
#                 selections=heatmap_selections,
#                 location=self.location,
#                 cat=self.catalog
#             )

#             return xr_da2

#         def deaccumulate_precip(xr_data):
#             """
#             Deaccumulates the precipitation (total) by taking the difference between subsequent timesteps.
#             Returns xr.DataArray.
#             """
#             da_deacc = np.ediff1d(xr_data, to_begin=0.0)
#             da_deacc = np.where(da_deacc<0, 0.0, da_deacc)

#             da = xr.DataArray(
#                 data = da_deacc,
#                 dims = ["time"],
#                 coords=dict(
#                     time=xr_data["time"],

#                 ),
#                 attrs=dict(
#                     description="De-accumulated precipitation (total)",
#                 ),
#             )
#             return da

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

#         # Grab data from AWS
#         data_hist = _get_hist_heatmap_data()
#         data_hist = data_hist.mean(dim="simulation").isel(scenario=0).compute()
#         data_future = _get_future_heatmap_data()
#         data_future = data_future.mean(dim="simulation").isel(scenario=0).compute()

#         # Compute hourly TMY for each day of the year
#         days_in_year = 366
#         def tmy_calc(data, days_in_year=366):
#             """Calculates the typical meteorological year based for both historical and future periods.
#             Returns two lists, one for the historical tmy and one for the future tmy.
#             """
#             hourly_list = []
#             for x in np.arange(1,days_in_year+1,1):
#                 data_on_day_x = data.where(data.time.dt.dayofyear == x, drop=True)
#                 data_grouped = data_on_day_x.groupby("time.hour")
#                 mean_by_hour = data_grouped.mean()
#                 min_diff = abs(data_grouped - mean_by_hour).groupby("time.hour").min()
#                 typical_hourly_data_on_day_x = data_on_day_x.where(abs(data_grouped - mean_by_hour).groupby("time.hour") == min_diff, drop=True).sortby("time.hour")
#                 np_typical_hourly_data_on_day_x = remove_repeats(typical_hourly_data_on_day_x)
#                 hourly_list.append(np_typical_hourly_data_on_day_x)

#             return hourly_list

#         tmy_hourly = tmy_calc(data_hist)
#         tmy_future = tmy_calc(data_future)

#         # Funnel data into pandas DataFrame object
#         df_hist = pd.DataFrame(tmy_hourly, columns = np.arange(1,25,1), index=np.arange(1,days_in_year+1,1))
#         df_hist = df_hist.iloc[::-1] # Reverse index

#         df_future = pd.DataFrame(tmy_future, columns = np.arange(1,25,1), index=np.arange(1,days_in_year+1,1))
#         df_future = df_future.iloc[::-1]

#         # Create difference heatamp between future and historical baseline
#         df = df_future - df_hist

        df = read_cached_tmy_df(
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
            df = df * 100
            cm = "PuOr"
            cl = (-15,15) # hardcoding this in, full range of warming level response for relhumid

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


    @param.depends("reload_data2", watch=False)
    def _GCM_PostageStamps_MAIN(self):

        all_plot_data = self._warm_all_anoms
        
        if self.variable2 == "Air Temperature at 2m": 
            cmap = "YlOrRd"
        elif self.variable2 == "Relative Humidity": 
            cmap = "PuOr"

        sim_plots = all_plot_data.hvplot.quadmesh('lon','lat',
            by='simulation',
            subplots = True,
            width = 250, height = 200,
            crs=ccrs.PlateCarree(),
            projection=ccrs.Orthographic(-118, 40),
            project=True, rasterize=False, dynamic=False,
            coastline=True, features=['borders'],
            cmap = cmap
            ).cols(3)

        # if self.overlay_MAIN == "Power plants":
        #     return sim_plots * power_plants.hvplot(color="black",s=2,geo=True,projection=ccrs.Orthographic(-118, 40))
        # elif self.overlay_MAIN == "Substations":
        #     return sim_plots * substations.hvplot(color="black",s=2,geo=True,projection=ccrs.Orthographic(-118, 40))
        # else:
        #     return sim_plots
        
        return sim_plots

    @param.depends("reload_data2", watch=False)
    def _GCM_PostageStamps_STATS(self):

        all_plot_data = self._warm_all_anoms

        min_data = all_plot_data.min(dim='simulation')
        max_data = all_plot_data.max(dim='simulation')
        med_data = all_plot_data.median(dim='simulation')
        mean_data = all_plot_data.mean(dim='simulation')

        def _make_plot(data, title, cmap="coolwarm"):
            _plot = data.hvplot.quadmesh('lon','lat',
                title = title,
                width = 300, height = 250,
                crs=ccrs.PlateCarree(),
                projection=ccrs.Orthographic(-118, 40),
                project=True, rasterize=False, dynamic=False,
                coastline=True, features=['borders'],
                cmap=cmap
                )
            # if self.overlay_STATS == "Power plants":
            #     return _plot * power_plants.hvplot(color="black",s=4,geo=True,projection=ccrs.Orthographic(-118, 40))
            # elif self.overlay_STATS == "Substations":
            #     return _plot * substations.hvplot(color="black",s=4,geo=True,projection=ccrs.Orthographic(-118, 40))
            # else:
            #     return _plot
        
            return _plot 
        
        if self.variable2 == "Air Temperature at 2m": 
            cmap = "YlOrRd"
        elif self.variable2 == "Relative Humidity": 
            cmap = "PuOr"
            
        mean_plot = _make_plot(mean_data, "Mean", cmap=cmap)
        med_plot = _make_plot(med_data, "Median", cmap=cmap)
        max_plot = _make_plot(max_data, "Maximum", cmap=cmap)
        min_plot = _make_plot(min_data, "Minimum", cmap=cmap)

        plot_grid = pn.Column(
            pn.Row(mean_plot, med_plot),
            pn.Row(max_plot, min_plot)
        )

        return plot_grid


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

        ipcc_data = (hist_data.hvplot(y="Mean", color="k", label="Historical", width=width, height=height) *
                hist_data.hvplot.area(x="Year", y="5%", y2="95%", alpha=0.1, color="k", ylabel="°C", xlabel="", ylim=[-1,5], xlim=[1950,2100]) * # very likely range
                 ssp119_data.hvplot(y="Mean", color=c119, label="SSP1-1.9") *
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
        to_plot.opts(opts.Overlay(title='Global mean surface temperature change relative to 1850-1900', fontsize=12))
        to_plot.opts(legend_position='bottom', fontsize=10)

        return to_plot


def _display_warming_levels(selections, location, _cat):

    # Warming levels object
    warming_levels = WarmingLevels(selections=selections, location=location, catalog=_cat)

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
        , title="Data Options", collapsible=False, width=460, height=500
    )

    GMT_plot = pn.Card(
            pn.Column(
                "Shading around selected scenario shows variation across different simulations. Dotted line indicates when the multi-model ensemble reaches the selected warming level, while solid vertical lines indicate when the earliest and latest simulations of that scenario reach the warming level.", 
                pn.widgets.Select.from_param(warming_levels.param.ssp, name="Scenario", width=250),
                warming_levels._GMT_context_plot,
            ),
            title="When do different scenarios reach the warming level?",
            collapsible=False, width=600, height=500
        )

    TMY = pn.Column(
        pn.widgets.StaticText(
           value="A typical meteorological year is calculated by selecting the 24 hours for every day that best represent multi-model mean conditions during a 30-year period – 1981-2010 for the historical baseline or centered on the year the warming level is reached.",
           width = 700
        ),
        warming_levels._TMY_hourly_heatmap
    )

    postage_stamps_MAIN = pn.Column(
        pn.widgets.StaticText(
            value="Panels show difference between 30-year average centered on the year each model reaches the specified warming level and average from 1981-2010.",
            width = 700
        ),
        warming_levels._GCM_PostageStamps_MAIN
    )

    postage_stamps_STATS = pn.Column(
        pn.widgets.StaticText(
            value="Panels show simulation that represents average, minimum, or maximum conditions across all models.",
            width = 700
        ),
        warming_levels._GCM_PostageStamps_STATS
    )

    map_tabs = pn.Card(
        pn.Tabs(
            ("Maps of individual simulations", postage_stamps_MAIN),
            ("Maps of cross-model statistics: mean/median/max/min", postage_stamps_STATS),
            ("Typical meteorological year", TMY),
        ),
    title="Regional response at selected warming level",
    width = 800, height=700, collapsible=False,
    )

    panel_doodad = pn.Column(
        pn.Row(user_options, GMT_plot),
        map_tabs
    )

    return panel_doodad
