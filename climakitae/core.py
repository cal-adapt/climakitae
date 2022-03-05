import numpy as np
import xarray as xr
import pandas as pd
import geopandas as gpd
import xesmf as xe
import param
import panel as pn
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from cycler import cycler
from itertools import cycle
import geoviews as gv
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from shapely.geometry import Point, box, Polygon
import holoviews as hv
from holoviews import opts
import hvplot.pandas
import hvplot.xarray
import intake

#constants: instead will be read from database of some kind:
_cached_stations = ['','BAKERSFIELD MEADOWS FIELD','BLYTHE ASOS','BURBANK-GLENDALE-PASADENA AIRPORT','LOS ANGELES DOWNTOWN USC CAMPUS','NEEDLES AIRPORT','FRESNO YOSEMITE INTERNATIONAL AIRPORT','IMPERIAL COUNTY AIRPORT','LAS VEGAS MCCARRAN INTERNATIONAL AP','LOS ANGELES INTERNATIONAL AIRPORT','LONG BEACH DAUGHERTY FIELD','MERCED MUNICIPAL AIRPORT','MODESTO CITY-COUNTY AIRPORT','SAN DIEGO MIRAMAR WSCMO','OAKLAND METRO INTERNATIONAL AIRPORT','OXNARD VENTURA COUNTY AIRPORT','PALM SPRINGS REGIONAL AIRPORT','RIVERSIDE MUNICIPAL AIRPORT','RED BLUFF MUNICIPAL AIRPORT','SACRAMENTO EXECUTIVE AIRPORT','SAN DIEGO LINDBERGH FIELD','SANTA BARBARA MUNICIPAL AIRPORT','SAN LUIS OBISPO AIRPORT','GILLESPIE FIELD','SAN FRANCISCO INTERNATIONAL AIRPORT','SAN JOSE INTERNATIONAL AIRPORT','SANTA ANA JOHN WAYNE AIRPORT','THERMAL / PALM SPRINGS','UKIAH MUNICIPAL AIRPORT','LANCASTER WILLIAM J FOX FIELD']

# get the list of data variables from one of the zarr files:
cat = intake.open_catalog('s3://cdcat/cae.yaml') 
_ds = cat[list(cat)[0]].to_dask()
_variable_choices_hourly_wrf = [_ds[i].attrs['description'] for i in _ds.data_vars] 
_variable_choices_hourly_wrf = _variable_choices_hourly_wrf+['precipitation (total)', 'wind 10m magnitude'] #which we'll derive from what's there
#expand this dictionary to also be dependent on LOCA vs WRF:
_variable_choices_daily_loca = ['Temperature','Maximum Relative Humidity','Minimum Relative Humidity','Solar Radiation','Wind Speed','Wind Direction']
_variable_choices_hourly_loca = ['Temperature','Maximum Relative Humidity','Minimum Relative Humidity','Solar Radiation','Wind Speed','Wind Direction']

_variable_choices_hourly = {'Dynamical': _variable_choices_hourly_wrf,
                           'Statistical': _variable_choices_hourly_loca}

_variable_choices = {'hourly': _variable_choices_hourly,
                     'daily': _variable_choices_hourly+['extra daily thing']} #this will be different based on whatever's in an example file

# hard-coded options:
_scenario_choices = ['SSP 2-4.5 -- Middle of the Road','SSP 3-7.0 -- Business as Usual', 'SSP 5-8.5 -- Burn it All']
_warming_level_choices = ['2','3','4'] #DEGREES
_export_formats = ['NetCDF (.nc)','.csv','geotiff']

#=== Select ===================================
location = Point() # empty shapely object

class LocSelectorArea(param.Parameterized):
    '''
    Used to produce a panel of widgets for entering one of the types of location information used to 
    define a timeseries from an average over an area. Will update 'location' with whatever reflects the 
    last change made to any one of the options. User can 
    1. update the latitude or longitude of a bounding box
    2. select a predefined area, from a set of shapefiles we pre-select
    [future: 3. upload their own shapefile with the outline of a natural or administrative geographic area]
    '''
    latitude = param.Range(default=(41, 42), bounds=(10, 67)) #would be nice to have a widget to draw a box on a map...
    longitude = param.Range(default=(-148,-120), bounds=(-156.82317,-84.18701)) 
    #shapefile = param.FileSelector(path='../../*/*.shp*', precedence=0.5) #not for March 2022
    cached_area = param.ObjectSelector(objects=['CA','Sierra','LA County']) #this is a placeholder list
    
    @param.depends('cached_area',watch=True)
    def _update_loc_cached(self):
        '''Updates the 'location' object to be the Polygon associated with the selected geographic area.'''
        location = _areas_database[self.cached_area] #need this database to exist...
    
    # not for soft launch:
    #@param.depends('shapefile',watch=True)
    #def _update_loc_shp(self):
    #    '''Updates the 'location' object to be the polygon in the uploaded shapefile.
    #    Dealing with user-uploaded data of any kind might not be in the soft launch.'''
    #    # probably need to do checking for valid input (also for security reasons)
    #    user_location = gpd.read_file("shapefile.shp")
    #    assert user_location.geom_type in ['Point','Polygon'], "Please upload a valid shapefile."
    #    # maybe also offer interactive selecting if there's more than one
    #    # polygon in the shapefile!
    #    location = user_location
        
    @param.depends('latitude',watch=True)
    def _update_loc_lat(self):
        '''Updates the 'location' object to be the box associated with the entered latitude range.'''
        location = box(self.latitude[0],self.latitude[1],self.longitude[0],self.longitude[1])
    
    @param.depends('longitude',watch=True)
    def _update_loc_long(self):
        '''Updates the 'location' object to be the box associated with the entered longitude range.'''
        location = box(self.latitude[0],self.latitude[1],self.longitude[0],self.longitude[1])
    
class LocSelectorPoint(param.Parameterized):
    '''
    If the user wants a timeseries that pertains to a point, they may choose from among a set of 
    pre-calculated station locations. Later this class can be extended to accomodate user-specified
    station data. Produces a panel from which to select from pre-calculated stations, and updates
    the 'location' object accordingly.
    '''
    cached_station = param.Selector(objects=_cached_stations) 
    
    @param.depends('cached_station',watch=True)
    def _update_location(self):
        '''Updates the 'location' object to be the point associated with the selected station.'''
        location = _stations_database[self.cached_station]

class DataSelector(param.Parameterized): 
    timescale = param.ObjectSelector(objects=['hourly','daily'])
    variable = param.ObjectSelector(objects=_variable_choices['hourly'])
    
    @param.depends('timescale', watch=True)
    def _update_variables(self):
        variables = _variable_choices[self.timescale][self.dyn_stat]
        self.param['variable'].objects = variables
        self.variable = variables[0]
        
    dyn_stat = param.ListSelector(objects=['Dynamical','Statistical'])
    
    @param.depends('dyn_stat', watch=True)
    def _update_variables(self):
        variables = _variable_choices[self.timescale][self.dyn_stat]
        self.param['variable'].objects = variables
        self.variable = variables[0]

    scenario = param.ListSelector(objects=_scenario_choices)
    append_historical = param.Boolean()

selections = DataSelector() #this is global

def select(location_type='area average'):
    '''
    Called at the beginning of the workflow, to capture user selections. Displays panel of widgets from which 
    to make selections. Location selection widgets depend on location_type argument, which can have only one 
    of two values: 'area average' or 'station'. Modifies 'selections' object, which is used by generate() to 
    build an appropriate xarray Dataset. 
    '''
    assert location_type in ['area average','station'], "Please enter either 'area average' or 'station'."
    
    _which_loc_input = {'area average': LocSelectorArea, 'station': LocSelectorPoint}
    location_chooser = pn.Row(_which_loc_input[location_type].param)
        
    #warming_levels = pn.widgets.CheckBoxGroup(name='Warming Levels',options=_warming_level_choices) 
    #scenario = pn.widgets.CheckBoxGroup(name='Scenarios',options=_scenario_choices)
    #dyn_stat = pn.widgets.CheckBoxGroup(name='Dynamical/Statistical',options=['Dynamical','Statistical'])
    #return pn.Column(pn.Row(obj.param,dyn_stat), pn.Row(warming_levels, scenario))
    first_col = pn.Column(selections.param.timescale, pn.widgets.RadioButtonGroup.from_param(selections.param.dyn_stat), selections.param.variable, pn.widgets.CheckBoxGroup.from_param(selections.param.scenario), selections.param.append_historical)
    return pn.Row(first_col,location_chooser)
    
#=== Generate ===================================                                                                                             
def generate():
    '''
    Currently a placeholder, which reads in data from a pre-processed file. Intended to use the information in 
    'selections' and 'locations' to generate an xarray Dataset as specified, and return that Dataset object.
    '''
    dataOneModel = pd.read_csv('archive_workshops/workshop#1example/timeSeries_tas_global.csv', header=[0,1], index_col=0)
    return dataOneModel


