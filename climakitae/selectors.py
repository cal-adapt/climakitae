import param
import panel as pn
import intake
from shapely.geometry import box, Polygon
from matplotlib.figure import Figure
import matplotlib.ticker as ticker
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import regionmask
import numpy as np
import geopandas as gpd
import pandas as pd
import datetime as dt

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
            "https://tigerweb.geo.census.gov/arcgis/rest/services/TIGERweb/State_County/MapServer/1/query?where=STATE='06'&f=geojson"
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
    _geography_choose = _geographies.boundary_dict()
    cached_area = param.ObjectSelector(
        default="CA", objects=list(_geography_choose["states"].keys())
    )

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
        '3 km':Polygon(
            [
                (-117.80029, 29.978943),
                (-127.95593, 40.654625),
                (-120.79376, 44.8999),
                (-111.23247, 33.452168)
            ]
        )
    }

    @param.depends("cached_area", watch=True)
    def _update_area_subset(self):
        """
        Makes the dropdown options for 'area subset' reflect the kind of subsetting
        that the user is adjusting.
        """
        _previous = self.cached_area
        if (self.area_subset == "none") or (self.area_subset == "lat/lon"):
            for option in ["states", "CA counties", "CA watersheds"]:
                if _previous in list(self._geography_choose[option].keys()):
                    self.area_subset = option
                    self.cached_area = _previous

    @param.depends("latitude", "longitude", watch=True)
    def _update_area_subset_to_lat_lon(self):
        """
        Makes the dropdown options for 'area subset' reflect that the user is
        adjusting the latitude or longitude slider.
        """
        if self.area_subset != "lat/lon":
            self.area_subset = "lat/lon"

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
        xy = ccrs.PlateCarree()
        ax = fig0.add_subplot(111, projection=proj)
        ax.set_extent([-150, -88, 8, 66], crs=xy)
        ax.set_facecolor("grey")

        # Plot the boundaries of the WRF domains on an existing set of axes for a map.
        # Hard-coding these numbers makes it faster.
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
            xy=(-135, 42),
            rotation=32,
            xycoords=xy._as_mpl_transform(ax),
            color="k",
        )
        ax.annotate(
            "3-km",
            xy=(-127, 39),
            rotation=32,
            xycoords=xy._as_mpl_transform(ax),
            color="k",
        )
        mpl_pane = pn.pane.Matplotlib(fig0, dpi=144)
        if self.area_subset == "lat/lon":
            ax.set_extent([-150, -88, 8, 66], crs=xy)
            ax.add_geometries(
                [geometry], crs=ccrs.PlateCarree(), edgecolor="b", facecolor="None"
            )
        elif self.area_subset == "states":
            ax.set_extent([-130, -100, 25, 50], crs=xy)
            shape_index = int(
                self._geography_choose[self.area_subset][self.cached_area]
            )
            self._geographies._us_states[[shape_index]].plot(
                ax=ax, add_label=False, line_kws=dict(color="b")
            )
            mpl_pane.param.trigger("object")
        elif self.area_subset == "CA counties":
            ax.set_extent([-125, -114, 31, 43], crs=xy)
            shapefile = self._geographies._ca_counties
            shape_index = int(
                self._geography_choose[self.area_subset][self.cached_area]
            )
            county = shapefile[shapefile.index == shape_index]
            df_ae = county.to_crs(crs_proj4)
            df_ae.plot(ax=ax, color="b", zorder=2)
            mpl_pane.param.trigger("object")
        elif self.area_subset == "CA watersheds":
            ax.set_extent([-125, -114, 31, 43], crs=xy)
            shapefile = self._geographies._ca_watersheds
            shape_index = int(
                self._geography_choose[self.area_subset][self.cached_area]
            )
            basin = shapefile[shapefile["OBJECTID"] == shape_index]
            df_ae = basin.to_crs(crs_proj4)
            df_ae.plot(ax=ax, color="b", zorder=2)
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
        self._cat = intake.open_catalog("https://cadcat.s3.amazonaws.com/cae.yaml")
        _ds = self._cat[list(self._cat)[0]].to_dask()
        _variable_choices_hourly_wrf = {
            v.attrs["description"].capitalize(): k for k, v in _ds.data_vars.items()
        }
        
        # Add derived variables 
        # Dictionary key (i.e. Precipitation (total)) will appear in list of variable options. 
        # Dictionary item (i.e. TOT_PRECIP) must match name of DataArray assigned in data_loaders module. 
        _variable_choices_hourly_wrf.update(
            {"Precipitation (total)": "TOT_PRECIP",  
             "Relative Humidity": "REL_HUMIDITY", 
             "Wind Magnitude at 10 m": "WIND_MAG"}
        ) 
        
        # remove some variables from the list, which will be superceded by higher quality hydrology
        _to_drop = ["Surface runoff", "Subsurface runoff", "Snow water equivalent"]
        [_variable_choices_hourly_wrf.pop(k) for k in _to_drop]
        # give better names to some descriptions, and reorder:
        _variable_choices_hourly_wrf["Surface Pressure"] = _variable_choices_hourly_wrf[
            "Sfc pressure"
        ]
        _variable_choices_hourly_wrf.pop("Sfc pressure")
        _variable_choices_hourly_wrf[
            "2m Air Temperature"
        ] = _variable_choices_hourly_wrf["Temp at 2 m"]
        _variable_choices_hourly_wrf.pop("Temp at 2 m")
        _variable_choices_hourly_wrf[
            "2m Water Vapor Mixing Ratio"
        ] = _variable_choices_hourly_wrf["Qv at 2 m"]
        _variable_choices_hourly_wrf.pop("Qv at 2 m")
        _variable_choices_hourly_wrf[
            "West-East component of Wind at 10m"
        ] = _variable_choices_hourly_wrf["U at 10 m"]
        _variable_choices_hourly_wrf.pop("U at 10 m")
        _variable_choices_hourly_wrf[
            "North-South component of Wind at 10m"
        ] = _variable_choices_hourly_wrf["V at 10 m"]
        _variable_choices_hourly_wrf.pop("V at 10 m")
        _variable_choices_hourly_wrf[
            "Snowfall (snow and ice)"
        ] = _variable_choices_hourly_wrf["Accumulated total grid scale snow and ice"]
        _variable_choices_hourly_wrf.pop("Accumulated total grid scale snow and ice")
        _move_to_end = [k for k in _variable_choices_hourly_wrf if "Instantaneous" in k]
        for k in _move_to_end:
            _variable_choices_hourly_wrf[k] = _variable_choices_hourly_wrf.pop(k)

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
            _temp.sort()  # consistent order
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
        default="monthly", objects=["hourly", "daily", "monthly"]
    )  # for WRF, will just coarsen data to start

    time_slice = param.Range(default=(1980, 2015), bounds=(1950, 2100))

    scenario = param.ListSelector(
        default=["Historical Climate"],
        objects=list(_choices._scenarios["45 km"].keys()),
        allow_None=True,
    )
    resolution = param.ObjectSelector(default="45 km", objects=_choices._resolutions)
    append_historical = param.Boolean(default=False)

    @param.depends("resolution", "append_historical", "scenario", watch=True)
    def _update_scenarios(self):
        """
        The scenarios available will depend on the resolution (more will be available for 9km
        than 3km for WRF eventually). Also ensures that "Historical Climate" is not
        redundantly displayed when "Append historical" is also selected.
        """
        _list_of_scenarios = list(self._choices._scenarios[self.resolution].keys())
        self.param["scenario"].objects = _list_of_scenarios
        if self.append_historical and self.scenario is not None:
            if "Historical Climate" in self.scenario:
                _scenarios = self.scenario
                _scenarios.remove("Historical Climate")
                self.scenario = _scenarios

    @param.depends("scenario", "append_historical", watch=True)
    def _update_time_slice_range(self):
        """
        Will discourage the user from selecting a time slice that does not exist for any
        of the selected scenarios, by updating the default range of years.
        """
        low_bound, upper_bound = self.time_slice
        if "Historical Reconstruction" not in self.scenario:
            low_bound = 1980
            if "Historical Climate" not in self.scenario and not self.append_historical:
                low_bound = 2015
            else:
                low_bound = 1980
        elif low_bound >= 1980:
            low_bound = 1950
        if not True in ["SSP" in one for one in self.scenario]:
            if "Historical Reconstruction" in self.scenario:
                upper_bound = 2022
            else:
                upper_bound = 2015
        elif upper_bound <= 2022:
            upper_bound = 2100

        self.time_slice = (low_bound, upper_bound)

    area_average = param.Boolean(default=False)

    @param.depends("time_slice", "scenario", "append_historical", watch=False)
    def view(self):
        """
        Displays a timeline to help the user visualize the time ranges available,
        and the subset of time slice selected.
        """
        fig0 = Figure(figsize=(3, 2))
        ax = fig0.add_subplot(111)
        ax.spines["right"].set_color("none")
        ax.spines["left"].set_color("none")
        ax.yaxis.set_major_locator(ticker.NullLocator())
        ax.spines["top"].set_color("none")
        ax.xaxis.set_ticks_position("bottom")
        ax.set_xlim(1950, 2100)
        ax.set_ylim(0, 1)
        ax.xaxis.set_major_locator(ticker.AutoLocator())
        ax.xaxis.set_minor_locator(ticker.AutoMinorLocator())
        mpl_pane = pn.pane.Matplotlib(fig0, dpi=144)

        def update_bars(scenario, y_offset):
            """
            Displays the time range of available data for each scenario above the timeline.
            """
            if scenario == "Historical Reconstruction":
                color = "g"
                center = 1986  # 1950-2022
                x_width = 36
            elif scenario == "Historical Climate":
                color = "c"
                center = 1997.5  # 1980-2014
                x_width = 17.5
            else:
                center = 2057.5  # 2015-2100
                x_width = 42.5
                if "2-4.5" in one:
                    color = "y"
                elif "3-7.0" in one:
                    color = "orange"
                elif "5-8.5" in one:
                    color = "r"
                if self.append_historical:
                    ax.errorbar(x=1997.5, y=y_offset, xerr=17.5, linewidth=8, color="c")
            ax.errorbar(x=center, y=y_offset, xerr=x_width, linewidth=8, color=color)
            ax.annotate(scenario[:10], xy=(center - x_width, y_offset + 0.06))

        y_offset = 0.15
        if self.scenario is not None:
            for one in self.scenario:
                update_bars(one, y_offset)
                y_offset += 0.15

        ax.fill_betweenx([0, 1], 1950, self.time_slice[0], alpha=0.8, facecolor="grey")
        ax.fill_betweenx([0, 1], self.time_slice[1], 2100, alpha=0.8, facecolor="grey")

        return mpl_pane


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

    first_row = pn.Row(
        pn.Column(
            selections.param.timescale,
            selections.param.time_slice,
            pn.layout.VSpacer(),
            selections.param.variable,
            pn.widgets.RadioButtonGroup.from_param(selections.param.resolution),
            pn.layout.VSpacer(),
            selections.param.area_average,
        ),
        pn.Column(
            selections.view,
            pn.widgets.CheckBoxGroup.from_param(selections.param.scenario),
            selections.param.append_historical,
        ),
    )
    return pn.Column(first_row, location_chooser)


# === For export functionality ==========================================================


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
    from which to select the export file format. Modifies 'user_export_format' object, which is used
    by data_export() to export data to the user in their specified format.
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
