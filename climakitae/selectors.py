import param
import panel as pn
import intake
from shapely.geometry import box  # , Point, Polygon
from matplotlib.figure import Figure
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import regionmask
import numpy as np
import geopandas as gpd
import pandas as pd

# support methods for core.Application.select

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

# === Select ===================================
class Boundaries:
    def __init__(self):
        self._us_states = regionmask.defined_regions.natural_earth_v4_1_0.us_states_50
        self._ca_counties = gpd.read_file(
            "https://tigerweb.geo.census.gov/arcgis/rest/services/TIGERweb/State_County/MapServer/1/query?where=STATE=06&f=geojson"
        )
        self._ca_counties = self._ca_counties.sort_values("NAME")

        self._ca_watersheds_file = "https://gis.data.cnra.ca.gov/datasets/02ff4971b8084ca593309036fb72289c_0.zip?outSR=%7B%22latestWkid%22%3A3857%2C%22wkid%22%3A102100%7D"
        self._ca_watersheds = gpd.read_file(self._ca_watersheds_file)
        self._ca_watersheds = self._ca_watersheds.sort_values("Name")

    def get_us_states(self):
        """
        Opens regionmask to retrieve a dictionary of state abbreviations:index
        """
        _state_lookup = dict(
            [
                (
                    abbrev,
                    np.argwhere(np.asarray(self._us_states.abbrevs) == abbrev)[0][0],
                )
                for abbrev in [
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
            ]
        )
        return _state_lookup

    def get_ca_counties(self):
        """
        Returns a dictionary of California counties and their indices in the shapefile.
        """
        return pd.Series(
            self._ca_counties.index, index=self._ca_counties["NAME"]
        ).to_dict()

    def get_ca_watersheds(self):
        """
        Returns a lookup dictionary for CA watersheds that references the shapefile.
        """
        return pd.Series(
            self._ca_watersheds["OBJECTID"].values, index=self._ca_watersheds["Name"]
        ).to_dict()

    def boundary_dict(self):
        """
        This returns a dictionary of lookup dictionaries for each set of shapefiles that
        the user might be choosing from. It is used to populate the selector object dynamically
        as the category in 'LocSelectorArea.area_subset' changes.
        """
        _all_options = {
            "states": self.get_us_states(),
            "CA counties": self.get_ca_counties(),
            "CA watersheds": self.get_ca_watersheds(),
        }
        return _all_options


class LocSelectorArea(param.Parameterized):
    """
    Used to produce a panel of widgets for entering one of the types of location information used to
    define a timeseries from an average over an area. Will update 'location' with whatever reflects the
    last change made to any one of the options. User can
    1. update the latitude or longitude of a bounding box
    2. select a predefined area, from a set of shapefiles we pre-select
    [future: 3. upload their own shapefile with the outline of a natural or administrative geographic area]
    """

    area_subset = param.ObjectSelector(
        default="none",
        objects=["none", "lat/lon", "states", "CA counties", "CA watersheds"],
    )
    # would be nice if these lat/lon sliders were greyed-out when lat/lon subset option is not selected
    latitude = param.Range(default=(32.5, 42), bounds=(10, 67))
    longitude = param.Range(default=(-125.5, -114), bounds=(-156.82317, -84.18701))
    _geographies = Boundaries()
    _geography_choose = (
        _geographies.boundary_dict()
    ) 
    cached_area = param.ObjectSelector(
        default="CA", objects=list(_geography_choose["states"].keys())
    )

    @param.depends("area_subset", watch=True)
    def _update_cached_area(self):
        """
        Makes the dropdown options for 'cached area' reflect the type of area subsetting
        selected in 'area_subset' (currently state, county, or watershed boundaries).
        """
        if self.area_subset in ["states", "CA counties", "CA watersheds"]:
            # setting this to the dict works for initializing, but not updating an objects list:
            self.param["cached_area"].objects = list(
                self._geography_choose[self.area_subset].keys()
            )
            self.cached_area = list(self._geography_choose[self.area_subset].keys())[0]

    @param.depends("latitude", "longitude", "area_subset", "cached_area", watch=False)
    def view(self):
        geometry = box(
            self.longitude[0], self.latitude[0], self.longitude[1], self.latitude[1]
        )

        fig0 = Figure(figsize=(3, 3))
        proj = ccrs.Orthographic(-118, 40)
        crs_proj4 = proj.proj4_init  # used below
        ax = fig0.add_subplot(111, projection=proj)
        ax.set_extent([-160, -84, 8, 68], crs=ccrs.PlateCarree())
        ax.coastlines()
        ax.add_feature(cfeature.STATES, linewidth=0.5)

        mpl_pane = pn.pane.Matplotlib(fig0, dpi=144)
        if self.area_subset == "lat/lon":
            ax.set_extent([-160, -84, 8, 68], crs=ccrs.PlateCarree())
            ax.add_geometries([geometry], crs=ccrs.PlateCarree(),edgecolor='b',facecolor='None')
        elif self.area_subset == "states":
            ax.set_extent([-130, -100, 25, 50], crs=ccrs.PlateCarree())
            shape_index = int(
                self._geography_choose[self.area_subset][self.cached_area]
            )
            self._geographies._us_states[[shape_index]].plot(
                ax=ax, add_label=False, line_kws=dict(color="b")
            )
            mpl_pane.param.trigger("object")
        elif self.area_subset == "CA counties":
            ax.set_extent([-125, -114, 31, 43], crs=ccrs.PlateCarree())
            shapefile = self._geographies._ca_counties
            shape_index = int(
                self._geography_choose[self.area_subset][self.cached_area]
            )
            county = shapefile[shapefile.index == shape_index]
            df_ae = county.to_crs(crs_proj4)
            df_ae.plot(ax=ax, color="b")
            mpl_pane.param.trigger("object")
        elif self.area_subset == "CA watersheds":
            ax.set_extent([-125, -114, 31, 43], crs=ccrs.PlateCarree())
            shapefile = self._geographies._ca_watersheds
            shape_index = int(
                self._geography_choose[self.area_subset][self.cached_area]
            )
            basin = shapefile[shapefile["OBJECTID"] == shape_index]
            df_ae = basin.to_crs(crs_proj4)
            df_ae.plot(ax=ax, color="b")
            mpl_pane.param.trigger("object")

        return mpl_pane


class LocSelectorPoint(param.Parameterized):
    """
    If the user wants a timeseries that pertains to a point, they may choose from among a set of
    pre-calculated station locations. Later this class can be extended to accomodate user-specified
    station data. Produces a panel from which to select from pre-calculated stations, and updates
    the 'location' object accordingly.
    """

    cached_station = param.Selector(objects=_cached_stations)

    @param.depends("cached_station", watch=True)
    def _update_location(self):
        """Updates the 'location' object to be the point associated with the selected station."""
        location = _stations_database[self.cached_station]


class CatalogContents:
    def __init__(self):
        # get the list of data variables from one of the zarr files:
        self._cat = intake.open_catalog("https://cdcat.s3.amazonaws.com/cae.yaml")
        _ds = self._cat[list(self._cat)[0]].to_dask()
        _variable_choices_hourly_wrf = {
            v.attrs["description"].capitalize(): k for k, v in _ds.data_vars.items()
        }
        # _variable_choices_hourly_wrf = _variable_choices_hourly_wrf +['precipitation (total)', 'wind 10m magnitude'] #which we'll derive from what's there
        # expand this dictionary to also be dependent on LOCA vs WRF:
        _variable_choices_daily_loca = [
            "Temperature",
            "Maximum Relative Humidity",
            "Minimum Relative Humidity",
            "Solar Radiation",
            "Wind Speed",
            "Wind Direction",
            "Precipitation",
        ]
        _variable_choices_hourly_loca = ["Temperature", "Precipitation"]

        _variable_choices_hourly = {
            "Dynamical": _variable_choices_hourly_wrf,
            "Statistical": _variable_choices_hourly_loca,
        }
        _variable_choices_daily = {
            "Dynamical": _variable_choices_hourly_wrf,
            "Statistical": _variable_choices_daily_loca,
        }

        self._variable_choices = {
            "hourly": _variable_choices_hourly,
            "daily": _variable_choices_daily,
        }

        # hard-coded options:
        self._scenario_choices = {
            "historical": "Historical Climate",
            "": "Historical Reconstruction",
            "ssp245": "SSP 2-4.5 -- Middle of the Road",
            "ssp370": "SSP 3-7.0 -- Business as Usual",
            "ssp585": "SSP 5-8.5 -- Burn it All",
        }

        self._resolutions = list(
            set(e.metadata["nominal_resolution"] for e in self._cat.values())
        )

        _scenario_list = []
        for resolution in self._resolutions:
            _temp = list(
                set(
                    e.metadata["experiment_id"]
                    for e in self._cat.values()
                    if e.metadata["nominal_resolution"] == resolution
                )
            )
            _temp.sort() #consistent order
            _scenario_subset = [(self._scenario_choices[e], e) for e in _temp]
            _scenario_subset = dict(_scenario_subset)
            _scenario_list.append((resolution, _scenario_subset))
        self._scenarios = dict(_scenario_list)


class DataSelector(param.Parameterized):
    """
    An object to hold data parameters, which depends only on the 'param' library.
    Currently used in '_display_select', which uses 'panel' to draw the gui, but another
    UI could in principle be used to update these parameters instead.
    """

    _choices = CatalogContents()
    variable = param.ObjectSelector(
        default="T2", objects=_choices._variable_choices["hourly"]["Dynamical"]
    )
    timescale = param.ObjectSelector(
        default="hourly", objects=["hourly", "daily", "monthly"]
    )  # for WRF, will just coarsen data to start

    # not needed yet until we have LOCA data:
    # dyn_stat = param.ListSelector(objects=['Dynamical','Statistical'])

    # @param.depends('timescale','dyn_stat', watch=True)
    # def _update_variables(self):
    #    '''
    #    Updates variable choices, which are different between dynamical/statistical, and
    #    for statistical are also different for hourly/daily.
    #    '''
    #    variables = choices._variable_choices[self.timescale][self.dyn_stat]
    #    self.param['variable'].objects = variables
    #    self.variable = variables[0]
    scenario = param.ListSelector(objects=list(_choices._scenarios["45 km"].keys()),allow_None=True)
    resolution = param.ObjectSelector(default="45 km", objects=_choices._resolutions)
    append_historical = param.Boolean(default=False)
    
    @param.depends("resolution","append_historical", watch=True)
    def _update_scenarios(self):
        _list_of_scenarios = list(self._choices._scenarios[self.resolution].keys())
        if self.append_historical:
            if "Historical Climate" in _list_of_scenarios:
                _list_of_scenarios.remove("Historical Climate")
        self.param["scenario"].objects = _list_of_scenarios
        self.scenario = [None] #resetting this avoids indexing errors with prior values

    area_average = param.Boolean(default=False)


def _display_select(selections, location, location_type="area average"):
    """
    Called by 'select' at the beginning of the workflow, to capture user selections. Displays panel of widgets
    from which to make selections. Location selection widgets depend on location_type argument, which can
    have only one of two values: 'area average' or 'station'. Modifies 'selections' object, which is used
    by generate() to build an appropriate xarray Dataset.
    Currently, core.Application.select does not pass an argument for location_type -- 'area average' is
    the only choice, until station-based data are available.
    """
    assert location_type in [
        "area average",
        "station",
    ], "Please enter either 'area average' or 'station'."

    # _which_loc_input = {'area average': LocSelectorArea, 'station': LocSelectorPoint}
    location_chooser = pn.Row(location.param, location.view)

    # add in when we have LOCA data too:
    # pn.widgets.RadioButtonGroup.from_param(selections.param.dyn_stat),
    first_row = pn.Row(
        pn.Column(
            selections.param.timescale,
            selections.param.variable,
            pn.widgets.RadioButtonGroup.from_param(selections.param.resolution),
        ),
        pn.Column(
            pn.widgets.CheckBoxGroup.from_param(selections.param.scenario),
            selections.param.append_historical,
            selections.param.area_average,
        )
    )
    return pn.Column(first_row, location_chooser)
