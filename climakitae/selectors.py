import datetime as dt
import param
import panel as pn
import intake
from shapely.geometry import box, Polygon
from matplotlib.figure import Figure
import matplotlib.ticker as ticker
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import geopandas as gpd
import pandas as pd
import pkg_resources
import warnings
from .unit_conversions import _get_unit_conversion_options
from .catalog_convert import (
    _resolution_to_gridlabel,
    _timescale_to_table_id,
    _scenario_to_experiment_id,
)

# Import package data
var_catalog_resource = pkg_resources.resource_filename(
    "climakitae", "data/variable_descriptions.csv"
)
var_catalog = pd.read_csv(var_catalog_resource, index_col=None)
unit_options_dict = _get_unit_conversion_options()


# =========================== LOCATION SELECTIONS ==============================

# constants: instead will be read from database of some kind:
_cached_stations = [
    "",
    "BAKERSFIELD MEADOWS FIELD",
    "BLYTHE ASOS",
    "BURBANK-GLENDALE-PASADENA AIRPORT",
    "LOS ANGELES DOWNTOWN USC CAMPUS",
    "NEEDLES AIRPORT",
    "FRESNO YOSEMITE INTERNATIONAL AIRPORT",
    "IMPERIAL COUNTY AIRPORT",
    "LAS VEGAS MCCARRAN INTERNATIONAL AP",
    "LOS ANGELES INTERNATIONAL AIRPORT",
    "LONG BEACH DAUGHERTY FIELD",
    "MERCED MUNICIPAL AIRPORT",
    "MODESTO CITY-COUNTY AIRPORT",
    "SAN DIEGO MIRAMAR WSCMO",
    "OAKLAND METRO INTERNATIONAL AIRPORT",
    "OXNARD VENTURA COUNTY AIRPORT",
    "PALM SPRINGS REGIONAL AIRPORT",
    "RIVERSIDE MUNICIPAL AIRPORT",
    "RED BLUFF MUNICIPAL AIRPORT",
    "SACRAMENTO EXECUTIVE AIRPORT",
    "SAN DIEGO LINDBERGH FIELD",
    "SANTA BARBARA MUNICIPAL AIRPORT",
    "SAN LUIS OBISPO AIRPORT",
    "GILLESPIE FIELD",
    "SAN FRANCISCO INTERNATIONAL AIRPORT",
    "SAN JOSE INTERNATIONAL AIRPORT",
    "SANTA ANA JOHN WAYNE AIRPORT",
    "THERMAL / PALM SPRINGS",
    "UKIAH MUNICIPAL AIRPORT",
    "LANCASTER WILLIAM J FOX FIELD",
]


class Boundaries:
    def __init__(self):
        self._cat = intake.open_catalog(
            "https://cadcat.s3.amazonaws.com/parquet/catalog.yaml"
        )
        self._us_states = self._cat.states.read()
        self._ca_counties = self._cat.counties.read().sort_values("NAME")
        self._ca_watersheds = self._cat.huc8.read().sort_values("Name")
        self._ca_utilities = self._cat.utilities.read()
        self._ca_forecast_zones = self._cat.dfz.read()
        
        # For Forecast Zones named "Other", replace that with the name of the county
        self._ca_forecast_zones.loc[
            self._ca_forecast_zones["FZ_Name"] == "Other", "FZ_Name"
        ] = self._ca_forecast_zones["FZ_Def"]
        #self._ca_forecast_zones = self._ca_forecast_zones[["OBJECTID","FZ_Name","geometry"]]

    def get_us_states(self):
        """
        Returns a custom sorted dictionary of state abbreviations and indices.
        """
        _states_subset_list = [
            "CA",
            "NV",
            "OR",
            "WA",
            "UT",
            "MT",
            "ID",
            "AZ",
            "CO",
            "NM",
            "WY",
        ]
        _us_states_subset = self._us_states.query("abbrevs in @_states_subset_list")[
            ["abbrevs"]
        ]
        _us_states_subset["abbrevs"] = pd.Categorical(
            _us_states_subset["abbrevs"], categories=_states_subset_list
        )
        _us_states_subset.sort_values(by="abbrevs", inplace=True)
        return dict(zip(_us_states_subset.abbrevs, _us_states_subset.index))

    def get_ca_counties(self):
        """
        Returns a dictionary of California counties and their indices
        in the geoparquet file.
        """
        return pd.Series(
            self._ca_counties.index, index=self._ca_counties["NAME"]
        ).to_dict()

    def get_ca_watersheds(self):
        """
        Returns a lookup dictionary for CA watersheds that references
        the geoparquet file.
        """
        return pd.Series(
            self._ca_watersheds.index, index=self._ca_watersheds["Name"]
        ).to_dict()
    
    def get_forecast_zones(self):
        """
        Returns a lookup dictionary for CA watersheds that references
        the geoparquet file.
        """        
        return pd.Series(
            self._ca_forecast_zones.index, index = self._ca_forecast_zones["FZ_Name"]
        ).to_dict()
    
    def get_ious_pous(self): 
        """
        Returns a lookup dictionary for IOUs & POUs that references 
        the geoparquet file. 
        """
        put_at_top = [ # Put in the order you want it to appear in the dropdown
            "Pacific Gas & Electric Company",
            "San Diego Gas & Electric",
            "Southern California Edison",
            "Los Angeles Department of Water & Power",
            "Sacramento Municipal Utility District"
        ]
        other_IOUs_POUs_list = [
            ut for ut in self._ca_utilities["Utility"] if 
            ut not in put_at_top
        ]
        other_IOUs_POUs_list = sorted(other_IOUs_POUs_list) # Put in alphabetical order
        ordered_list = put_at_top + other_IOUs_POUs_list
        _subset = self._ca_utilities.query("Utility in @ordered_list")[["Utility"]]
        _subset["Utility"] = pd.Categorical(
            _subset["Utility"], 
            categories = ordered_list
        )
        _subset.sort_values(by = "Utility", inplace = True)
        return dict(zip(_subset["Utility"], _subset.index))

    def boundary_dict(self):
        """
        This returns a dictionary of lookup dictionaries for each set of
        geoparquet files that the user might be choosing from. It is used to
        populate the selector object dynamically as the category in
        'LocSelectorArea.area_subset' changes.
        """
        _all_options = {
            "none": {"entire domain": 0},
            "lat/lon": {"coordinate selection": 0},
            "states": self.get_us_states(),
            "CA counties": self.get_ca_counties(),
            "CA watersheds": self.get_ca_watersheds(),
            "CA Electric Load Serving Entities (IOU & POU)": self.get_ious_pous(), 
            "CA Electricity Demand Forecast Zones": self.get_forecast_zones()
        }
        return _all_options

class LocSelectorArea(param.Parameterized):
    """
    Used to produce a panel of widgets for entering one of the types of location
    information used to define a timeseries from an average over an area. Will
    update 'location' with whatever reflects the last change made to any one of
    the options. User can:
    1. update the latitude or longitude of a bounding box
    2. select a predefined area, from a set of geoparquet files
        in S3 we pre-select
    [future: 3. upload their own shapefile with the outline of a natural or
        administrative geographic area]
    """

    area_subset = param.ObjectSelector(objects = dict())
    cached_area = param.ObjectSelector(objects = dict())
    default_lat = (32.5, 42)
    default_lon = (-125.5, -114)
    latitude = param.Range(
        default = default_lat, 
        bounds=(10, 67)
    )
    longitude = param.Range(
        default = default_lon, 
        bounds = (-156.82317, -84.18701)
    )
    _lat_lon_warning = param.String(
        default = "", 
        doc = "Warning if user is messing with lat/lon slider, \
        but lat/lon is not selected for area subset."
    ) 
    

    def __init__(self, **params):
        super().__init__(**params)
        
        # Get geography boundaries and selection options 
        self._geographies = Boundaries()
        self._geography_choose = self._geographies.boundary_dict()
        
        # Set params 
        self.area_subset = "none"
        self.param["area_subset"].objects = list(self._geography_choose.keys())
        self.param["cached_area"].objects = list(self._geography_choose[self.area_subset].keys())

    _wrf_bb = {
        "45 km": Polygon(
            [
                (-123.52125549316406, 9.475631713867188),
                (-156.8231658935547, 35.449039459228516),
                (-102.43182373046875, 67.32866668701172),
                (-84.18701171875, 26.643436431884766),
            ]
        ),
        "9 km": Polygon(
            [
                (-116.69509887695312, 22.267112731933594),
                (-138.42117309570312, 43.23344802856445),
                (-110.90779113769531, 57.5806770324707),
                (-94.9368896484375, 31.627288818359375),
            ]
        ),
        "3 km": Polygon(
            [
                (-117.80029, 29.978943),
                (-127.95593, 40.654625),
                (-120.79376, 44.8999),
                (-111.23247, 33.452168),
            ]
        ),
    }
                    
    @param.depends("latitude","longitude", watch = True)
    def _update_area_subset_to_lat_lon(self):
        """
        Makes the dropdown options for 'area subset' reflect that the user is
        adjusting the latitude or longitude slider.
        """
        if self.area_subset != "lat/lon": 
            self.area_subset = "lat/lon" 
    
    @param.depends("area_subset", watch = True)
    def _update_cached_area(self):
        """
        Makes the dropdown options for 'cached area' reflect the type of area
        subsetting selected in 'area_subset' (currently state, county, or
        watershed boundaries).
        """
        self.param["cached_area"].objects = list(
            self._geography_choose[self.area_subset].keys()
        )
        self.cached_area = list(self._geography_choose[self.area_subset].keys())[0]

    @param.depends("latitude", "longitude", "area_subset", "cached_area", watch=False)
    def view(self):
        geometry = box(
            self.longitude[0], self.latitude[0], self.longitude[1], self.latitude[1]
        )

        fig0 = Figure(figsize=(4.25, 4.25))
        proj = ccrs.Orthographic(-118, 40)
        crs_proj4 = proj.proj4_init  # used below
        xy = ccrs.PlateCarree()
        ax = fig0.add_subplot(111, projection = proj)
        ax.set_extent([-150, -88, 8, 66], crs = xy)
        ax.set_facecolor("grey")

        # Plot the boundaries of the WRF domains on an existing set of axes for
        # a map. Hard-coding these numbers makes it faster.
        _colors = ["k", "dodgerblue", "darkorange"]
        for i, domain in enumerate(["45 km", "9 km", "3 km"]):
            # Plot domain:
            ax.add_geometries(
                [self._wrf_bb[domain]],
                crs=ccrs.PlateCarree(),
                edgecolor=_colors[i],
                facecolor="white",
            )

        ax.coastlines()
        ax.add_feature(cfeature.BORDERS)
        ax.add_feature(cfeature.STATES, linewidth=0.5)
        ax.annotate(
            "45-km grid",
            xy=(-154, 33.8),
            rotation=28,
            xycoords=xy._as_mpl_transform(ax),
        )
        ax.annotate(
            "9-km",
            xy = (-135, 42),
            rotation = 32,
            xycoords = xy._as_mpl_transform(ax),
            color = "k",
        )
        ax.annotate(
            "3-km",
            xy = (-127, 39),
            rotation = 32,
            xycoords = xy._as_mpl_transform(ax),
            color = "k",
        )
        mpl_pane = pn.pane.Matplotlib(fig0, dpi=144)

        def plot_subarea(boundary_dataset, extent):
            ax.set_extent(extent, crs = xy)
            subarea = boundary_dataset[boundary_dataset.index == shape_index]
            df_ae = subarea.to_crs(crs_proj4)
            df_ae.plot(ax = ax, color = "b", zorder = 2)
            mpl_pane.param.trigger("object")

        if self.area_subset == "lat/lon":
            ax.set_extent([-150, -88, 8, 66], crs = xy)
            ax.add_geometries(
                [geometry], crs = ccrs.PlateCarree(), edgecolor = "b", facecolor = "None"
            )
        elif self.area_subset != "none":
            shape_index = int(
                self._geography_choose[self.area_subset][self.cached_area]
            )
            if self.area_subset == "states":
                plot_subarea(self._geographies._us_states, [-130, -100, 25, 50])
            elif self.area_subset == "CA counties":
                plot_subarea(self._geographies._ca_counties, [-125, -114, 31, 43])
            elif self.area_subset == "CA watersheds":
                plot_subarea(self._geographies._ca_watersheds, [-125, -114, 31, 43])
            elif self.area_subset == "CA Electric Load Serving Entities (IOU & POU)":
                plot_subarea(self._geographies._ca_utilities, [-125, -114, 31, 43])
            elif self.area_subset == "CA Electricity Demand Forecast Zones":
                plot_subarea(self._geographies._ca_forecast_zones, [-125, -114, 31, 43])
                
        return mpl_pane

class LocSelectorPoint(param.Parameterized):
    """
    If the user wants a timeseries that pertains to a point, they may choose
    from among a set of pre-calculated station locations. Later this class can
    be extended to accomodate user-specified station data. Produces a panel from
    which to select from pre-calculated stations, and updates the 'location'
    object accordingly.
    """

    cached_station = param.Selector(objects=_cached_stations)

    @param.depends("cached_station", watch=True)
    def _update_location(self):
        """Updates the 'location' object to be the point associated with the selected station."""
        location = _stations_database[self.cached_station]


# ============================ DATA SELECTIONS =================================

def _get_simulation_options(cat, activity_id, table_id, grid_label, experiment_id): 
    """Get simulations for user selections. This function is not intended to provide user options, 
    but is rather used to provide information only. It also serves to remove ensmean as an option.
    
    Args: 
        cat (intake catalog): catalog
        activity_id (str): dataset name (i.e. "WRF")
        table_id (str): timescale
        grid_label (str): resolution
        experiment_id (list of str): selected scenario/s
    
    Returns: 
        simulation_options (list of strs): valid simulation (source_id) options for input 
    
    """
    
    # Get catalog subset from user inputs
    with warnings.catch_warnings(record = True):
        cat_subset = cat.search(
            activity_id = activity_id, 
            table_id = table_id, 
            grid_label = grid_label, 
            experiment_id = experiment_id
        )
    
    # Get all unique simulation options from catalog selection 
    try: 
        simulation_options = cat_subset.unique()["source_id"]["values"] 
        if "ensmean" in simulation_options: 
            simulation_options.remove("ensmean") # Remove ensemble means
    except: 
        simulation_options = []   
    return simulation_options 

def _get_variable_options_df(var_catalog, unique_variable_ids, timescale):
    """Get variable information for a subset of unique variable ids.
    Args:
        var_catalog (df): variable catalog information. read in from csv file
        unique_variable_ids (list of strs): list of unique variable ids from catalog. Used to subset var_catalog
        timescale (str): hourly, daily, or monthly
    Returns:
        variable_options_df (pd.DataFrame): var_catalog information subsetted by unique_variable_ids
    """
    if timescale in ["daily", "monthly"]:
        timescale = "daily/monthly"
    # Catalog options and derived options together 
    var_options_plus_derived = unique_variable_ids+["rh_derived", "wind_speed_derived","dew_point_derived"]
    variable_options_df = var_catalog[
        (var_catalog["show"] == True) & # Make sure it's a valid variable selection
        (var_catalog["variable_id"].isin(var_options_plus_derived) & # Make sure variable_id is part of the catalog options for user selections
        (var_catalog["timescale"] == timescale) # Make sure its the right timescale
        )
    ]
    return variable_options_df

def _get_data_selection_description(variable, units, timescale, resolution, 
                                    time_slice, scenario_historical, scenario_ssp, 
                                    _area_average_yes_no, location): 
    
    """
    Make a long string to output to the user to show all their current selections.
    Updates whenever any of the input values are changed. 
    """
    
    # Edit how the scenarios are printed in the description to make it reader-friendly 
    if (True in ["SSP" in one for one in scenario_ssp]):
        if "Historical Climate" in scenario_historical: 
            scenario_print = ["Historical + " + ssp[:9] for ssp in scenario_ssp]
        else: 
            scenario_print = [ssp[:9] for ssp in scenario_ssp]
    else: 
        scenario_print = scenario_ssp + scenario_historical
        
    # Show lat/lon selection only if area_subset == lat/lon 
    if location.area_subset == "lat/lon":
        #bbox = min Longitude , min Latitude , max Longitude , max Latitude 
        cached_area_print = "bounding box <br>\
            ({:.2f}".format(location.longitude[0]) + ", {:.2f}".format(location.latitude[0]) + "\
            , {:.2f}".format(location.longitude[1]) + ", {:.2f}".format(location.latitude[1]) + ")"
    elif location.area_subset == "none": 
        cached_area_print = "entire " + str(resolution) + " grid"
    else: 
        cached_area_print = str(location.cached_area)
    

    _data_selection_description = "<font size='+0.10'>Data selections: </font><br> \
        <ul> \
            <li><b>variable:</b> " + str(variable) +"</li> \
            <li><b>units:</b> "+ str(units) + "</li> \
            <li><b>temporal resolution: </b>" + str(timescale) + "</li> \
            <li><b>model resolution: </b>" + str(resolution) +"</li> \
            <li><b>timeslice: </b>" + str(time_slice[0]) + " - " + str(time_slice[1]) + "</li> \
            <li><b>datasets:</b> " + ", ".join(scenario_print) + "</li> \
        </ul>"
    _location_selection_description = "<font size='+0.10'>Location selections: </font><br> \
        <ul> \
            <li><b>location:</b> " + cached_area_print +"</li> \
            <li><b>compute area average?</b> " + str(_area_average_yes_no) + "</li> \
        </ul>"
    return _data_selection_description + _location_selection_description

class DataSelector(param.Parameterized):
    """
    An object to hold data parameters, which depends only on the 'param'
    library. Currently used in '_display_select', which uses 'panel' to draw the
    gui, but another UI could in principle be used to update these parameters
    instead.
    """

    # Defaults
    default_variable = "Air Temperature at 2m"
    time_slice = param.Range(
        default = (1980, 2015), 
        bounds = (1950, 2100)
    )
    resolution = param.ObjectSelector(
        default = "45 km", 
        objects = ["45 km", "9 km", "3 km"]
    )
    timescale = param.ObjectSelector(
        default = "monthly", 
        objects = ["hourly", "daily", "monthly"]
    )
    scenario_historical = param.ListSelector(
        default = ["Historical Climate"], 
        objects = ["Historical Reconstruction (ERA5-WRF)", "Historical Climate"]
    )
    _area_average_yes_no = param.ObjectSelector(
        default = "No",
        objects = ["Yes","No"], 
        doc = "Used to make the select panel more readable. \
        Set to Yes if area_average = True, and No if not."
    ) 

    # Empty params, initialized in __init__
    downscaling_method = param.ObjectSelector(objects = dict())
    scenario_ssp = param.ListSelector(objects = dict())
    simulation = param.ListSelector(objects = dict())
    variable = param.ObjectSelector(objects = dict())
    units = param.ObjectSelector(objects = dict())
    extended_description = param.ObjectSelector(objects = dict())
    variable_id = param.ObjectSelector(objects = dict())
    area_average = param.Boolean()
    _data_warning = param.String(
        default = "", 
        doc = "Warning if user has made a bad selection"
    )
    _data_selection_description = param.String(
        default = "", 
        doc = "Description of the user data selections."
    )
    
    # Temporal range of each dataset 
    historical_climate_range = (1980, 2015) 
    historical_reconstruction_range = (1950, 2022) 
    ssp_range = (2015, 2100) 

    def __init__(self, **params):
        # Set default values
        super().__init__(**params)

        # Downscaling method selection
        self.downscaling_method = "WRF"

        # Variable catalog info
        self.cat_subset = self.cat.search(
            activity_id = self.downscaling_method,
            table_id = _timescale_to_table_id(self.timescale),
            grid_label = _resolution_to_gridlabel(self.resolution)
        )
        self.unique_variable_ids = self.cat_subset.unique()["variable_id"]["values"]
        # Get more info about that subset of unique variable ids 
        self.variable_options_df = _get_variable_options_df( 
            var_catalog = var_catalog,
            unique_variable_ids = self.unique_variable_ids,
            timescale = self.timescale
        )

        # Set scenario param
        scenario_ssp_options = [
            _scenario_to_experiment_id(scen, reverse = True) for scen in 
            self.cat_subset.unique()["experiment_id"]["values"] if "ssp" in scen
        ]
        for scenario_i in [ 
            "SSP 2-4.5 -- Middle of the Road",
            "SSP 3-7.0 -- Business as Usual",
            "SSP 5-8.5 -- Burn it All"
        ]:
            if scenario_i in scenario_ssp_options: # Reorder list
                scenario_ssp_options.remove(scenario_i) # Remove item
                scenario_ssp_options.append(scenario_i) # Add to back of list
        self.param["scenario_ssp"].objects = scenario_ssp_options
        self.scenario_ssp = []

        # Set variable param
        self.param["variable"].objects = self.variable_options_df.display_name.values
        self.variable = self.default_variable
        
        # Set simulation param 
        self.simulation = _get_simulation_options(
            cat = self.cat,
            activity_id = self.downscaling_method,
            table_id = _timescale_to_table_id(self.timescale),
            grid_label = _resolution_to_gridlabel(self.resolution),
            experiment_id = [
                _scenario_to_experiment_id(scen) for scen in 
                self.scenario_ssp + self.scenario_historical
            ]
        )

        # Set colormap, units, & extended description
        var_info = self.variable_options_df[
            self.variable_options_df["display_name"] == self.variable
        ]  
        
        # Set params that are not selected by the user
        self.colormap = var_info.colormap.item()
        self.units = var_info.unit.item()
        self.extended_description = var_info.extended_description.item()
        self.variable_id = var_info.variable_id.item()
        self._data_warning = ""
        self._data_selection_description = _get_data_selection_description(
            variable = self.variable, 
            units = self.units, 
            timescale = self.timescale, 
            resolution = self.resolution, 
            time_slice = self.time_slice, 
            scenario_historical = self.scenario_historical, 
            scenario_ssp = self.scenario_ssp, 
            _area_average_yes_no = self._area_average_yes_no,
            location = self.location
        )
        
    @param.depends("area_average", "units", "variable", "scenario_historical", 
                   "scenario_ssp", "timescale", "resolution", "time_slice", 
                   "_area_average_yes_no", "location.area_subset", "location.cached_area",
                   "location.longitude", "location.latitude", watch = True) 
    def _update_data_selection_description(self): 
        self._data_selection_description = _get_data_selection_description(
            variable = self.variable, 
            units = self.units, 
            timescale = self.timescale, 
            resolution = self.resolution, 
            time_slice = self.time_slice, 
            scenario_historical = self.scenario_historical, 
            scenario_ssp = self.scenario_ssp, 
            _area_average_yes_no = self._area_average_yes_no,
            location = self.location
        )
    @param.depends("_area_average_yes_no", watch = True)
    def _update_area_average_yes_no(self): 
        self.area_average = True if self._area_average_yes_no == "Yes" else False

    @param.depends("timescale", "resolution", watch = True)
    def _update_var_options(self):
        """Update unique variable options"""
        self.cat_subset = self.cat.search(
            activity_id = self.downscaling_method,
            table_id = _timescale_to_table_id(self.timescale),
            grid_label = _resolution_to_gridlabel(self.resolution)
        )
        self.unique_variable_ids = self.cat_subset.unique()["variable_id"]["values"]

        # Get more info about that subset of unique variable ids
        self.variable_options_df = _get_variable_options_df(
            var_catalog = var_catalog,
            unique_variable_ids = self.unique_variable_ids,
            timescale = self.timescale,
        )

        # Reset variable dropdown
        var_options = self.variable_options_df.display_name.values
        self.param["variable"].objects = var_options
        if self.variable not in var_options:
            self.variable = var_options[0]

    @param.depends("resolution", "location.area_subset", watch = True)
    def _update_states_3km(self):
        if self.location.area_subset == "states":
            if self.resolution == "3 km":
                self.location.param["cached_area"].objects = [
                    "CA",
                    "NV",
                    "OR",
                    "UT",
                    "AZ",
                ]
                self.location.cached_area = "CA"
            else:
                self.location.param[
                    "cached_area"
                ].objects = self.location._geography_choose["states"].keys()

    @param.depends("variable", "timescale", watch = True)
    def _update_unit_options(self):
        """Update unit options and native units for selected variable."""
        var_info = self.variable_options_df[
            self.variable_options_df["display_name"] == self.variable
        ]  # Get info for just that variable
        native_unit = var_info.unit.item()
        if (
            native_unit in unit_options_dict.keys()
        ):  # See if there's unit conversion options for native variable
            self.param["units"].objects = unit_options_dict[native_unit]
        else:  # Just use native units if no conversion options available
            self.param["units"].objects = [native_unit]
        self.units = native_unit

    @param.depends("variable", "timescale", "resolution", watch = True)
    def _update_cmap_and_extended_description(self):
        var_info = self.variable_options_df[
            self.variable_options_df["display_name"] == self.variable
        ]  # Get info for just that variable
        self.colormap = var_info.colormap.item()
        self.extended_description = var_info.extended_description.item()
        self.variable_id = var_info.variable_id.item()

    @param.depends("resolution", "scenario_ssp", "scenario_historical", watch=True)
    def _update_scenarios(self):
        """
        Update scenario options. Raise data warning if a bad selection is made.
        """     
        
        # Get scenario options in catalog format
        scenario_ssp_options = [
            _scenario_to_experiment_id(scen, reverse = True) for scen in 
            self.cat_subset.unique()["experiment_id"]["values"] if "ssp" in scen
        ]
        for scenario_i in [ 
            "SSP 2-4.5 -- Middle of the Road",
            "SSP 3-7.0 -- Business as Usual",
            "SSP 5-8.5 -- Burn it All"
        ]:
            if scenario_i in scenario_ssp_options: # Reorder list
                scenario_ssp_options.remove(scenario_i) # Remove item
                scenario_ssp_options.append(scenario_i) # Add to back of list
        self.param["scenario_ssp"].objects = scenario_ssp_options
        self.scenario_ssp = [x for x in self.scenario_ssp if x in scenario_ssp_options]
         
    @param.depends("scenario_ssp", "scenario_historical", "time_slice", watch = True) 
    def _update_data_warning(self): 
        """Update warning raised to user based on their data selections."""
        data_warning = ""
        bad_time_slice_warning = "You've selected a time slice that is outside the temporal range of the \
            selected data." 
        # Warning based on data scenario selections
        if ( # Warn user that they cannot have SSP data and ERA5-WRF data 
            (True in ["SSP" in one for one in self.scenario_ssp]) and 
            ("Historical Reconstruction (ERA5-WRF)" in self.scenario_historical)
        ) :
            data_warning = "Historical Reconstruction (ERA5-WRF) data is not available with SSP data. \
            Try using the Historical Climate data instead."
            
        elif ( # Warn user if no data is selected
            (not True in ["SSP" in one for one in self.scenario_ssp]) and 
            (not True in ["Historical" in one for one in self.scenario_historical])
        ):
            data_warning = "Please select as least one dataset."
            
        elif ( # If both historical options are selected, warn user the data will be cut
            ("Historical Reconstruction (ERA5-WRF)" in self.scenario_historical) and 
            ("Historical Climate" in self.scenario_historical)
        ):  
            data_warning = "The timescale of Historical Reconstruction (ERA5-WRF) data will be cut \
            to match that of the Historical Climate data if both are retrieved."
       
        # Warnings based on time slice selections
        if ( 
            (not True in ["SSP" in one for one in self.scenario_ssp]) and 
            ("Historical Climate" in self.scenario_historical)
        ):
            if (
                (self.time_slice[0] < self.historical_climate_range[0]) or 
                (self.time_slice[1] > self.historical_climate_range[1])
            ): 
                data_warning = bad_time_slice_warning
        elif (True in ["SSP" in one for one in self.scenario_ssp]): 
            if (not True in ["Historical" in one for one in self.scenario_historical]): 
                if (
                    (self.time_slice[0] < self.ssp_range[0]) or 
                    (self.time_slice[1] > self.ssp_range[1])
                ): 
                    data_warning = bad_time_slice_warning
            else: 
                if (
                    (self.time_slice[0] < self.historical_climate_range[0]) or 
                    (self.time_slice[1] > self.ssp_range[1])
                ): 
                    data_warning = bad_time_slice_warning
        elif (self.scenario_historical == ["Historical Reconstruction (ERA5-WRF)"]): 
            if (
                (self.time_slice[0] < self.historical_reconstruction_range[0]) or 
                (self.time_slice[1] > self.historical_reconstruction_range[1])
            ):  
                data_warning = bad_time_slice_warning
        
        # Show warning
        self._data_warning = data_warning
            
    @param.depends("scenario_ssp", "scenario_historical", watch = True)
    def _update_time_slice_range(self):
        """
        Will discourage the user from selecting a time slice that does not exist
        for any of the selected scenarios, by updating the default range of years.
        """
        low_bound, upper_bound = self.time_slice
        
        if self.scenario_historical == ["Historical Climate"]: 
            low_bound, upper_bound = self.historical_climate_range
        elif self.scenario_historical == ["Historical Reconstruction (ERA5-WRF)"]: 
            low_bound, upper_bound = self.historical_reconstruction_range
        elif ( # If both historical options are selected, and no SSP is selected
            all([x in ['Historical Reconstruction (ERA5-WRF)', 'Historical Climate'] 
                 for x in self.scenario_historical]) and 
            (not True in ["SSP" in one for one in self.scenario_ssp])
        ): 
            low_bound, upper_bound = self.historical_climate_range
               
        if True in ["SSP" in one for one in self.scenario_ssp]: 
            if "Historical Climate" in self.scenario_historical: # If also append historical 
                low_bound = self.historical_climate_range[0]
            else: 
                low_bound = self.ssp_range[0]
            upper_bound = self.ssp_range[1]

        self.time_slice = (low_bound, upper_bound)
    
    @param.depends("scenario_ssp","scenario_historical", "timescale",watch = True)
    def _update_simulation(self): 
        """Simulation options will change if the scenario changes, 
        or if the timescale changes, due to the fact that the ensmean
        data is available (and needs to be removed) for hourly data."""
        self.simulation = _get_simulation_options(
            cat = self.cat,
            activity_id = self.downscaling_method,
            table_id = _timescale_to_table_id(self.timescale),
            grid_label = _resolution_to_gridlabel(self.resolution),
            experiment_id = [
                _scenario_to_experiment_id(scen) for scen in 
                self.scenario_ssp + self.scenario_historical
            ]
        )

    @param.depends("time_slice", "scenario_ssp", "scenario_historical", watch = False)
    def view(self):
        """
        Displays a timeline to help the user visualize the time ranges
        available, and the subset of time slice selected.
        """
        fig0 = Figure(figsize=(4.25,2))
        ax = fig0.add_subplot(111)
        ax.spines["right"].set_color("none")
        ax.spines["left"].set_color("none")
        ax.yaxis.set_major_locator(ticker.NullLocator())
        ax.spines["top"].set_color("none")
        ax.xaxis.set_ticks_position("bottom")
        ax.set_xlim(1950, 2100)
        ax.set_ylim(0, 1)
        ax.tick_params(labelsize=11)
        ax.xaxis.set_major_locator(ticker.AutoLocator())
        ax.xaxis.set_minor_locator(ticker.AutoMinorLocator())
        mpl_pane = pn.pane.Matplotlib(fig0, dpi=144)

        y_offset = 0.15
        if (self.scenario_ssp is not None) and (self.scenario_historical is not None):
            for scen in self.scenario_ssp + self.scenario_historical:
                
                if (["SSP" in one for one in self.scenario_ssp]): 
                    if scen in ["Historical Climate","Historical Reconstruction (ERA5-WRF)"]: 
                        continue 
                        
                if scen == "Historical Reconstruction (ERA5-WRF)":
                    color = "darkblue"
                    if "Historical Climate" in self.scenario_historical: 
                        center = 1997.5  # 1980-2014
                        x_width = 17.5
                        ax.annotate("Reconstruction", xy = (1967, y_offset + 0.06), fontsize = 12)
                    else: 
                        center = 1986  # 1950-2022
                        x_width = 36
                        ax.annotate("Reconstruction", xy = (1955, y_offset + 0.06), fontsize = 12)
                    
                elif scen == "Historical Climate":
                    color = "c"
                    center = 1997.5  # 1980-2014
                    x_width = 17.5
                    ax.annotate("Historical", xy = (1979, y_offset + 0.06), fontsize = 12) 
                    
                elif "SSP" in scen:
                    center = 2057.5  # 2015-2100
                    x_width = 42.5
                    scenario_label = scen[:10]
                    if "2-4.5" in scen:
                        color = "#f69320"
                    elif "3-7.0" in scen:
                        color = "#df0000"
                    elif "5-8.5" in scen:
                        color = "#980002"
                    if ("Historical Climate" in self.scenario_historical):
                        ax.errorbar(
                            x = 1997.5,
                            y = y_offset,
                            xerr = 17.5,
                            linewidth = 8,
                            color = "c"
                        ) 
                        ax.annotate("Historical", xy = (1979, y_offset + 0.06), fontsize = 12)
                    ax.annotate(scen[:10], xy = (2035, y_offset + 0.06), fontsize = 12)

                ax.errorbar(
                    x = center,
                    y = y_offset,
                    xerr = x_width,
                    linewidth = 8,
                    color = color
                )
                
                y_offset += 0.28
                
        ax.fill_betweenx(
            [0, 1], 
            self.time_slice[0], 
            self.time_slice[1], 
            alpha = 0.8, 
            facecolor = "lightgrey"
        )
        return mpl_pane
    
class SelectionDescription(param.Parameterized):
    """
    An object to hold a description of the user's data and location selections. 
    """ 
    def __init__(self, **params):
        super().__init__(**params)

    
# ================ DISPLAY LOCATION/DATA SELECTIONS IN PANEL ===================

def _display_select(selections, location):
    """
    Called by 'select' at the beginning of the workflow, to capture user
    selections. Displays panel of widgets from which to make selections.
    Modifies 'selections' object, which is used by generate() to build an 
    appropriate xarray Dataset. 
    """

    location_chooser = pn.Row(
        pn.Column(
            pn.widgets.Select.from_param(location.param.area_subset, name="Subset the data by..."),
            pn.widgets.Select.from_param(location.param.cached_area, name="Location selection"), 
            location.param.latitude, 
            location.param.longitude,
            pn.widgets.StaticText(
                    value = "<b>Compute an area average of your data over \
                        the selected region?</b>", 
                    name = ""
            ),
            pn.widgets.RadioButtonGroup.from_param(
                selections.param._area_average_yes_no, 
                inline = True
            ),
            width = 275
        ),
        location.view,
    )
    
    data_options = pn.Column(
        selections.param.variable,
        pn.widgets.StaticText.from_param(
            selections.param.extended_description, name=""
        ),
        pn.widgets.StaticText(name="", value="Variable Units"),
        pn.widgets.RadioButtonGroup.from_param(selections.param.units),
        selections.param.timescale,
        pn.widgets.StaticText(name="", value="Model Resolution"),
        pn.widgets.RadioButtonGroup.from_param(selections.param.resolution),
        width = 285
    )
    
    scenario_options = pn.Column(
        selections.view,
        selections.param.time_slice,
        pn.widgets.StaticText(
            value = "<br>Estimates of recent historical climatic conditions", 
            name = "Historical Data"
        ),
        pn.widgets.CheckBoxGroup.from_param(selections.param.scenario_historical),
        pn.widgets.StaticText(
            value = "<br>SSP options represent end-of-century range", 
            name = "Future Model Data"
        ),
        pn.widgets.CheckBoxGroup.from_param(selections.param.scenario_ssp),
        pn.widgets.StaticText.from_param(
            selections.param._data_warning, 
            name = "", 
            style = {"color":"red"}
        ), 
        width = 310
    )

    tabs = pn.Card(
        pn.Tabs(
            ("Make your data selections", pn.Row(scenario_options, data_options)),
            ("Subset data by location", location_chooser)
        ),
        title = "Select your data and region of interest",
        height = 550, 
        width = 595, 
        collapsible = False,
    )
    
    how_to_use = pn.Card(
        pn.widgets.StaticText(
            value = "In the first tab, select the data. In the second tab, subset your selected \
            data by location and choose whether or not to compute an area average over the \
            selected region. To retrieve the data, use the climakitae function app.retrieve().", 
            name = ""
        ),
        title = "How to use this panel", 
        width = 285,
        height = 165, 
        collapsible = False
    )
    
    your_selections = pn.Card(
        pn.widgets.StaticText.from_param(
            selections.param._data_selection_description, 
            name = ""
        ),
        title = "Current selections", 
        width = 285,
        height = 310,
        collapsible = False
    )
    
    return pn.Row(
        tabs, 
        pn.Column(
            how_to_use, 
            your_selections)
    ) 


# =============================== EXPORT DATA ==================================

class UserFileChoices:

    # reserved for later: text boxes for dataset to export
    # as well as a file name
    # data_var_name = param.String()
    # output_file_name = param.String()

    def __init__(self):
        self._export_format_choices = ["Pick a file format", "CSV", "GeoTIFF", "NetCDF"]


class FileTypeSelector(param.Parameterized):
    """
    If the user wants to export an xarray dataset, they can choose
    their preferred format here. Produces a panel from which to select a
    supported file type.
    """

    user_options = UserFileChoices()
    output_file_format = param.ObjectSelector(
        objects=user_options._export_format_choices
    )

    def _export_file_type(self):
        """Updates the 'user_export_format' object to be the format specified by the user."""
        user_export_format = self.output_file_format


def _user_export_select(user_export_format):
    """
    Called by 'export' at the end of the workflow. Displays panel
    from which to select the export file format. Modifies 'user_export_format'
    object, which is used by data_export() to export data to the user in their
    specified format.
    """

    data_to_export = pn.widgets.TextInput(
        name="Data to export", placeholder="Type name of dataset here"
    )

    # reserved for later: text boxes for dataset to export
    # as well as a file name
    # file_name = pn.widgets.TextInput(name='File name',
    #                                 placeholder='Type file name here')
    # file_input_col = pn.Column(user_export_format.param, data_to_export, file_name)
    return pn.Row(user_export_format.param)
