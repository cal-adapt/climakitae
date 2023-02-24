"""Backend functions related to providing and dynamically setting data selections.
Boundaries function for storing parquet geometries for spatial subsetting."""

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
    _downscaling_method_to_activity_id,
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

stations = pkg_resources.resource_filename("climakitae", "data/hadisd_stations.csv")
stations_df = pd.read_csv(stations)
stations_gpd = gpd.GeoDataFrame(
    stations_df,
    crs="EPSG:4326",
    geometry=gpd.points_from_xy(stations_df.LON_X, stations_df.LAT_Y),
)


# =========================== LOCATION SELECTIONS ==============================


class Boundaries:
    """Get geospatial polygon data from the AE catalog.
    Used to access boundaries for subsetting data by state, county, etc.
    """

    def __init__(self):
        """
        Parameters
        -----------
        _cat: intake.catalog.local.YAMLFileCatalog
            Catalog for parquet files
        _us_states: pd.DataFrame
            Table of US state names and geometries
        _ca_counties: pd.DataFrame
            Table of California county names and geometries
            Sorted by county name alphabetical order
        _ca_watersheds: pd.DataFrame
            Table of California watershed names and geometries
            Sorted by watershed name alphabetical order
        _ca_utilities: pd.DataFrame
            Table of California IOUs and POUs, names and geometries
        _ca_forecast_zones: pd.DataFrame
            Table of California Demand Forecast Zones
        """
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

    def get_us_states(self):
        """
        Returns a custom sorted dictionary of state abbreviations and indices.

        Returns
        -------
        dict

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

        Returns
        -------
        dict

        """
        return pd.Series(
            self._ca_counties.index, index=self._ca_counties["NAME"]
        ).to_dict()

    def get_ca_watersheds(self):
        """
        Returns a lookup dictionary for CA watersheds that references
        the geoparquet file.

        Returns
        -------
        dict

        """
        return pd.Series(
            self._ca_watersheds.index, index=self._ca_watersheds["Name"]
        ).to_dict()

    def get_forecast_zones(self):
        """
        Returns a lookup dictionary for CA watersheds that references
        the geoparquet file.

        Returns
        -------
        dict

        """
        return pd.Series(
            self._ca_forecast_zones.index, index=self._ca_forecast_zones["FZ_Name"]
        ).to_dict()

    def get_ious_pous(self):
        """
        Returns a lookup dictionary for IOUs & POUs that references
        the geoparquet file.

        Returns
        -------
        dict

        """
        put_at_top = [  # Put in the order you want it to appear in the dropdown
            "Pacific Gas & Electric Company",
            "San Diego Gas & Electric",
            "Southern California Edison",
            "Los Angeles Department of Water & Power",
            "Sacramento Municipal Utility District",
        ]
        other_IOUs_POUs_list = [
            ut for ut in self._ca_utilities["Utility"] if ut not in put_at_top
        ]
        other_IOUs_POUs_list = sorted(other_IOUs_POUs_list)  # Put in alphabetical order
        ordered_list = put_at_top + other_IOUs_POUs_list
        _subset = self._ca_utilities.query("Utility in @ordered_list")[["Utility"]]
        _subset["Utility"] = pd.Categorical(_subset["Utility"], categories=ordered_list)
        _subset.sort_values(by="Utility", inplace=True)
        return dict(zip(_subset["Utility"], _subset.index))

    def boundary_dict(self):
        """
        This returns a dictionary of lookup dictionaries for each set of
        geoparquet files that the user might be choosing from. It is used to
        populate the selector object dynamically as the category in
        '_LocSelectorArea.area_subset' changes.

        Returns
        -------
        dict

        """
        _all_options = {
            "none": {"entire domain": 0},
            "lat/lon": {"coordinate selection": 0},
            "states": self.get_us_states(),
            "CA counties": self.get_ca_counties(),
            "CA watersheds": self.get_ca_watersheds(),
            "CA Electric Load Serving Entities (IOU & POU)": self.get_ious_pous(),
            "CA Electricity Demand Forecast Zones": self.get_forecast_zones(),
        }
        return _all_options


class _LocSelectorArea(param.Parameterized):
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

    area_subset = param.ObjectSelector(objects=dict())
    cached_area = param.ObjectSelector(objects=dict())
    latitude = param.Range(default=(32.5, 42), bounds=(10, 67))
    longitude = param.Range(default=(-125.5, -114), bounds=(-156.82317, -84.18701))
    data_type = param.ObjectSelector(default="gridded", objects=["gridded", "station"])
    station = param.ListSelector(objects=dict())

    def __init__(self, **params):
        super().__init__(**params)

        # Get geography boundaries and selection options
        self._geographies = Boundaries()
        self._geography_choose = self._geographies.boundary_dict()

        # Set params
        self.area_subset = "none"
        self.param["area_subset"].objects = list(self._geography_choose.keys())
        self.param["cached_area"].objects = list(
            self._geography_choose[self.area_subset].keys()
        )

        if self.data_type == "station":
            overlapping_stations = _get_overlapping_station_names(
                stations_gpd,
                self.area_subset,
                self.cached_area,
                self.latitude,
                self.longitude,
                self._geographies,
                self._geography_choose,
            )
            self.param["station"].objects = overlapping_stations
            self.station = overlapping_stations
        elif self.data_type == "gridded":
            self.param["station"].objects = []
            self.station = []

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
        Makes the dropdown options for 'cached area' reflect the type of area
        subsetting selected in 'area_subset' (currently state, county, or
        watershed boundaries).
        """
        self.param["cached_area"].objects = list(
            self._geography_choose[self.area_subset].keys()
        )
        self.cached_area = list(self._geography_choose[self.area_subset].keys())[0]

    @param.depends(
        "data_type", "area_subset", "cached_area", "latitude", "longitude", watch=True
    )
    def _update_station_list(self):
        """Update the list of weather station options if the area subset changes"""
        if self.data_type == "station":
            overlapping_stations = _get_overlapping_station_names(
                stations_gpd,
                self.area_subset,
                self.cached_area,
                self.latitude,
                self.longitude,
                self._geographies,
                self._geography_choose,
            )
            self.param["station"].objects = overlapping_stations
            self.station = overlapping_stations
        elif self.data_type == "gridded":
            self.param["station"].objects = []
            self.station = []


def _get_overlapping_station_names(
    stations_gpd,
    area_subset,
    cached_area,
    latitude,
    longitude,
    _geographies,
    _geography_choose,
):
    """Wrapper function that gets the string names of any overlapping weather stations"""
    subarea = _get_subarea(
        area_subset, cached_area, latitude, longitude, _geographies, _geography_choose
    )
    overlapping_stations_gpd = _get_overlapping_stations(stations_gpd, subarea)
    overlapping_stations_names = sorted(
        list(overlapping_stations_gpd["station"].values)
    )
    return overlapping_stations_names


def _get_overlapping_stations(stations, polygon):
    """Get weather stations contained within a geometry
    Both stations and polygon MUST have the same projection

    Parameters
    ----------
    stations: gpd.GeoDataFrame
        Weather station names and coordinates, with geometry column
    polygon: gpd.GeoDataFrame
        Polygon geometry, must be a gpd.GeoDataFrame object

    Returns
    -------
    gpd.GeoDataFrame
        stations gpd subsetted to include only points contained within polygon

    """
    return gpd.sjoin(stations, polygon, op="within")


def _get_subarea(
    area_subset, cached_area, latitude, longitude, _geographies, _geography_choose
):
    """Get geometry from input settings
    Used for plotting or determining subset of overlapping weather stations in subsequent steps

    Returns
    -------
    gpd.GeoDataFrame

    """

    def _get_subarea_from_shape_index(boundary_dataset, shape_index):
        return boundary_dataset[boundary_dataset.index == shape_index]

    if area_subset == "lat/lon":
        geometry = box(
            longitude[0],
            latitude[0],
            longitude[1],
            latitude[1],
        )
        df_ae = gpd.GeoDataFrame(
            pd.DataFrame({"subset": ["coords"], "geometry": [geometry]}),
            crs="EPSG:4326",
        )
    elif area_subset != "none":
        shape_index = int(_geography_choose[area_subset][cached_area])
        if area_subset == "states":
            df_ae = _get_subarea_from_shape_index(_geographies._us_states, shape_index)
        elif area_subset == "CA counties":
            df_ae = _get_subarea_from_shape_index(
                _geographies._ca_counties, shape_index
            )
        elif area_subset == "CA watersheds":
            df_ae = _get_subarea_from_shape_index(
                _geographies._ca_watersheds, shape_index
            )
        elif area_subset == "CA Electric Load Serving Entities (IOU & POU)":
            df_ae = _get_subarea_from_shape_index(
                _geographies._ca_utilities, shape_index
            )
        elif area_subset == "CA Electricity Demand Forecast Zones":
            df_ae = _get_subarea_from_shape_index(
                _geographies._ca_forecast_zones, shape_index
            )

    else:  # If no subsetting, make the geometry a big box so all stations are included
        df_ae = gpd.GeoDataFrame(
            pd.DataFrame(
                {
                    "subset": ["coords"],
                    "geometry": [box(-150, -88, 8, 66)],  # Super big box
                }
            ),
            crs="EPSG:4326",
        )

    return df_ae


def _add_res_to_ax(
    poly, ax, rotation, xy, label, color="black", crs=ccrs.PlateCarree()
):
    """Add resolution line and label to axis

    Parameters
    ----------
    poly: geometry to plot
    ax: matplotlib axis
    color: matplotlib color
    rotation: int
    xy: tuple
    label: str
    crs: projection

    """
    ax.add_geometries(
        [poly], crs=ccrs.PlateCarree(), edgecolor=color, facecolor="white"
    )
    ax.annotate(
        label,
        xy=xy,
        rotation=rotation,
        color="black",
        xycoords=crs._as_mpl_transform(ax),
    )


class _ViewLocationSelections(param.Parameterized):
    """View the current location selections on a map
    Updates dynamically

    Parameters
    ----------
    location: LocSelectorArea
        User location selections
    selections: DataSelector
        User data selections

    """

    def __init__(self, **params):
        super().__init__(**params)

    @param.depends(
        "selections.downscaling_method",
        "selections.resolution",
        "location.latitude",
        "location.longitude",
        "location.area_subset",
        "location.cached_area",
        "location.data_type",
        "location.station",
        watch=True,
    )
    def view(self):
        fig0 = Figure(figsize=(2.9, 2.9))
        proj = ccrs.Orthographic(-118, 40)
        crs_proj4 = proj.proj4_init  # used below
        xy = ccrs.PlateCarree()
        ax = fig0.add_subplot(111, projection=proj)

        if "Statistical" in self.selections.downscaling_method:
            # 3km LOCA grid shown whenever LOCA is selected, even if WRF is also selected
            _add_res_to_ax(
                poly=self.location._wrf_bb["3 km"],
                ax=ax,
                color="magenta",
                rotation=32,
                xy=(-127, 39),
                label="3-km statistical",
            )

        elif self.selections.downscaling_method == ["Dynamical"]:
            # If only WRF is selected (indicated by list with only WRF
            if self.selections.resolution == "45 km":
                _add_res_to_ax(
                    poly=self.location._wrf_bb["45 km"],
                    ax=ax,
                    color="green",
                    rotation=28,
                    xy=(-154, 33.8),
                    label="45-km dynamical",
                )
            elif self.selections.resolution == "9 km":
                _add_res_to_ax(
                    poly=self.location._wrf_bb["9 km"],
                    ax=ax,
                    color="dodgerblue",
                    rotation=32,
                    xy=(-135, 42),
                    label="9-km dynamical",
                )
            elif self.selections.resolution == "3 km":
                _add_res_to_ax(
                    poly=self.location._wrf_bb["3 km"],
                    ax=ax,
                    color="darkorange",
                    rotation=32,
                    xy=(-127, 39),
                    label="3-km dynamical",
                )
        mpl_pane = pn.pane.Matplotlib(fig0, dpi=144)

        # Set plot extent
        if (
            (self.selections.resolution == "3 km")
            or ("CA" in self.location.area_subset)
            or (self.location.data_type == "station")
        ):
            extent = [-125, -114, 31, 43]  # Zoom in on CA
            scatter_size = 4.5  # Size of markers for stations
        elif (self.selections.resolution == "9 km") or (
            self.location.area_subset == "states"
        ):
            extent = [-130, -100, 25, 50]  # Just shows US states
            scatter_size = 2.5  # Size of markers for stations
        elif self.location.area_subset in ["none", "lat/lon"]:
            extent = [
                -150,
                -88,
                8,
                66,
            ]  # Widest extent possible-- US, some of Mexico and Canada
            scatter_size = 1.5  # Size of markers for stations
        else:  # Default for all other selections
            extent = [-125, -114, 31, 43]  # Zoom in on CA
            scatter_size = 4.5  # Size of markers for stations
        ax.set_extent(extent, crs=xy)

        subarea_gpd = _get_subarea(
            self.location.area_subset,
            self.location.cached_area,
            self.location.latitude,
            self.location.longitude,
            self.location._geographies,
            self.location._geography_choose,
        ).to_crs(crs_proj4)
        if self.location.area_subset == "lat/lon":
            ax.add_geometries(
                subarea_gpd["geometry"].values,
                crs=ccrs.PlateCarree(),
                edgecolor="b",
                facecolor="None",
            )
        elif self.location.area_subset != "none":
            subarea_gpd.plot(ax=ax, color="deepskyblue", zorder=2)
            mpl_pane.param.trigger("object")

        # Overlay the weather stations as points on the map
        if self.location.data_type == "station":
            # Subset the stations gpd to get just the user's selected stations
            # We need the stations gpd because it has the coordinates, which will be used to make the plot
            stations_selection_gpd = stations_gpd.loc[
                stations_gpd["station"].isin(self.location.station)
            ]
            stations_selection_gpd = stations_selection_gpd.to_crs(
                crs_proj4
            )  # Convert to map projection
            ax.scatter(
                stations_selection_gpd.LON_X.values,
                stations_selection_gpd.LAT_Y.values,
                transform=ccrs.PlateCarree(),
                zorder=15,
                color="black",
                s=scatter_size,  # Scatter size is dependent on extent of map
            )

        # Add state lines, international borders, and coastline
        ax.add_feature(cfeature.STATES, linewidth=0.5, edgecolor="gray")
        ax.add_feature(cfeature.COASTLINE, linewidth=0.5, edgecolor="darkgray")
        ax.add_feature(cfeature.BORDERS, edgecolor="darkgray")
        return mpl_pane


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
    with warnings.catch_warnings(record=True):
        cat_subset = cat.search(
            activity_id=activity_id,
            table_id=table_id,
            grid_label=grid_label,
            experiment_id=experiment_id,
        )

    # Get all unique simulation options from catalog selection
    try:
        simulation_options = cat_subset.unique()["source_id"]["values"]
        if "ensmean" in simulation_options:
            simulation_options.remove("ensmean")  # Remove ensemble means
    except:
        simulation_options = []
    return simulation_options


def _get_variable_options_df(
    var_catalog, unique_variable_ids, downscaling_method, timescale
):
    """Get variable options to display depending on downscaling method and timescale

    Parameters
    ----------
    var_catalog: pd.DataFrame
        Variable descriptions, units, etc in table format
    unique_variable_ids: list of strs
        List of unique variable ids from catalog.
        Used to subset var_catalog
    downscaling_method: list, one of ["Dynamical"], ["Statistical"], or ["Dynamical","Statistical"]
        Data downscaling method
    timescale: str, one of "hourly", "daily", or "monthly"
        Timescale

    Returns
    -------
    pd.DataFrame
        Subset of var_catalog for input downscaling_method and timescale
    """
    if timescale in ["daily", "monthly"]:
        timescale = "daily/monthly"
    # Catalog options and derived options together
    var_options_plus_derived = unique_variable_ids + [
        "rh_derived",
        "wind_speed_derived",
        "dew_point_derived",
    ]
    variable_options_df = var_catalog[
        (var_catalog["show"] == True)
        & (  # Make sure it's a valid variable selection
            var_catalog["variable_id"].isin(var_options_plus_derived)
            & (  # Make sure variable_id is part of the catalog options for user selections
                var_catalog["timescale"] == timescale
            )  # Make sure its the right timescale
        )
    ]

    if set(["Dynamical", "Statistical"]).issubset(downscaling_method):
        variable_options_df = variable_options_df[
            # Get shared variables
            variable_options_df["display_name"].duplicated()
        ]
    else:
        variable_options_df = variable_options_df[
            # Get variables only from one downscaling method
            variable_options_df["downscaling_method"].isin(downscaling_method)
        ]
    return variable_options_df


class _DataSelector(param.Parameterized):
    """
    An object to hold data parameters, which depends only on the 'param'
    library. Currently used in '_display_select', which uses 'panel' to draw the
    gui, but another UI could in principle be used to update these parameters
    instead.
    """

    # Defaults
    default_variable = "Air Temperature at 2m"
    time_slice = param.Range(default=(1980, 2015), bounds=(1950, 2100))
    resolution = param.ObjectSelector(
        default="45 km", objects=["45 km", "9 km", "3 km"]
    )
    timescale = param.ObjectSelector(
        default="monthly", objects=["hourly", "daily", "monthly"]
    )
    scenario_historical = param.ListSelector(
        default=["Historical Climate"],
        objects=["Historical Reconstruction", "Historical Climate"],
    )
    area_average = param.ObjectSelector(
        default="No",
        objects=["Yes", "No"],
        doc="""Compute an area average?""",
    )
    downscaling_method = param.ListSelector(
        default=["Dynamical"], objects=["Dynamical", "Statistical (available soon)"]
    )

    # Empty params, initialized in __init__
    scenario_ssp = param.ListSelector(objects=dict())
    simulation = param.ListSelector(objects=dict())
    variable = param.ObjectSelector(objects=dict())
    units = param.ObjectSelector(objects=dict())
    extended_description = param.ObjectSelector(objects=dict())
    variable_id = param.ObjectSelector(objects=dict())
    _data_warning = param.String(
        default="", doc="Warning if user has made a bad selection"
    )

    # Temporal range of each dataset
    historical_climate_range = (1980, 2015)
    historical_reconstruction_range = (1950, 2022)
    ssp_range = (2015, 2100)

    def __init__(self, **params):
        # Set default values
        super().__init__(**params)

        # Variable catalog info
        self.cat_subset = self.cat.search(
            activity_id=[
                _downscaling_method_to_activity_id(dm) for dm in self.downscaling_method
            ],
            table_id=_timescale_to_table_id(self.timescale),
            grid_label=_resolution_to_gridlabel(self.resolution),
        )
        self.unique_variable_ids = self.cat_subset.unique()["variable_id"]["values"]

        # Get variable options to display to user
        # This will further subset the variable options from
        # self.cat_subset.unique()["variable_id"]["values"], only showing
        # the user certain variables within that list dependent upon the
        # settings in var_catalog
        self.variable_options_df = _get_variable_options_df(
            var_catalog=var_catalog,
            unique_variable_ids=self.unique_variable_ids,
            downscaling_method=self.downscaling_method,
            timescale=self.timescale,
        )

        # Set scenario param
        scenario_ssp_options = [
            _scenario_to_experiment_id(scen, reverse=True)
            for scen in self.cat_subset.unique()["experiment_id"]["values"]
            if "ssp" in scen
        ]
        for scenario_i in [
            "SSP 2-4.5 -- Middle of the Road",
            "SSP 3-7.0 -- Business as Usual",
            "SSP 5-8.5 -- Burn it All",
        ]:
            if scenario_i in scenario_ssp_options:  # Reorder list
                scenario_ssp_options.remove(scenario_i)  # Remove item
                scenario_ssp_options.append(scenario_i)  # Add to back of list
        self.param["scenario_ssp"].objects = scenario_ssp_options
        self.scenario_ssp = []

        # Set variable param
        self.param["variable"].objects = self.variable_options_df.display_name.values
        self.variable = self.default_variable

        # Set simulation param
        self.simulation = _get_simulation_options(
            cat=self.cat,
            activity_id=[
                _downscaling_method_to_activity_id(dm) for dm in self.downscaling_method
            ],
            table_id=_timescale_to_table_id(self.timescale),
            grid_label=_resolution_to_gridlabel(self.resolution),
            experiment_id=[
                _scenario_to_experiment_id(scen)
                for scen in self.scenario_ssp + self.scenario_historical
            ],
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

    @param.depends("location.data_type", watch=True)
    def _update_res_and_area_average_based_on_data_type(self):
        if self.location.data_type == "station":
            self.param["resolution"].objects = ["n/a"]
            self.resolution = "n/a"
            self.param["area_average"].objects = ["n/a"]
            self.area_average = "n/a"
        elif self.location.data_type == "gridded":
            self.param["resolution"].objects = ["45 km", "9 km", "3 km"]
            self.resolution = "45 km"
            self.param["area_average"].objects = ["Yes", "No"]
            self.area_average = "No"

    @param.depends("timescale", "resolution", watch=True)
    def _update_var_options(self):
        """Update unique variable options"""
        if self.resolution == "n/a":
            pass
        else:
            self.cat_subset = self.cat.search(
                activity_id=[
                    _downscaling_method_to_activity_id(dm)
                    for dm in self.downscaling_method
                ],
                table_id=_timescale_to_table_id(self.timescale),
                grid_label=_resolution_to_gridlabel(self.resolution),
            )
            self.unique_variable_ids = self.cat_subset.unique()["variable_id"]["values"]
            self.variable_options_df = _get_variable_options_df(
                var_catalog=var_catalog,
                unique_variable_ids=self.unique_variable_ids,
                downscaling_method=self.downscaling_method,
                timescale=self.timescale,
            )

            # Reset variable dropdown
            var_options = self.variable_options_df.display_name.values
            self.param["variable"].objects = var_options
            if self.variable not in var_options:
                self.variable = var_options[0]

    @param.depends("resolution", "location.area_subset", watch=True)
    def _update_states_3km(self):
        if self.location.area_subset == "states":
            if self.resolution == "3 km":
                if "Statistical" in self.downscaling_method:
                    self.location.param["cached_area"].objects = ["CA"]
                elif self.downscaling_method == ["Dynamical"]:
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

    @param.depends("downscaling_method", watch=True)
    def _remove_hourly_LOCA(self):
        if self.downscaling_method == ["Dynamical"]:
            self.param["timescale"].objects = ["hourly", "daily", "monthly"]
        else:
            self.param["timescale"].objects = ["daily", "monthly"]
            if self.timescale == "hourly":
                self.timescale = "daily"

    @param.depends("downscaling_method", watch=True)
    def _update_res_based_on_downscaling_method(self):
        """Remove resolution options if LOCA is selected"""
        if self.downscaling_method == ["Dynamical"]:
            self.param["resolution"].objects = ["3 km", "9 km", "45 km"]

        else:  # No 45km or 9km option for LOCA grid
            self.param["resolution"].objects = ["3 km"]
            if self.resolution in ["45 km", "9 km"]:
                self.resolution = "3 km"

    @param.depends("variable", "timescale", watch=True)
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

    @param.depends("variable", "timescale", "resolution", watch=True)
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
            _scenario_to_experiment_id(scen, reverse=True)
            for scen in self.cat_subset.unique()["experiment_id"]["values"]
            if "ssp" in scen
        ]
        for scenario_i in [
            "SSP 2-4.5 -- Middle of the Road",
            "SSP 3-7.0 -- Business as Usual",
            "SSP 5-8.5 -- Burn it All",
        ]:
            if scenario_i in scenario_ssp_options:  # Reorder list
                scenario_ssp_options.remove(scenario_i)  # Remove item
                scenario_ssp_options.append(scenario_i)  # Add to back of list
        self.param["scenario_ssp"].objects = scenario_ssp_options
        self.scenario_ssp = [x for x in self.scenario_ssp if x in scenario_ssp_options]

    @param.depends("scenario_ssp", "scenario_historical", "time_slice", watch=True)
    def _update_data_warning(self):
        """Update warning raised to user based on their data selections."""
        data_warning = ""
        bad_time_slice_warning = """You've selected a time slice that is outside the temporal range 
        of the selected data."""
        # Warning based on data scenario selections
        if (  # Warn user that they cannot have SSP data and ERA5-WRF data
            True in ["SSP" in one for one in self.scenario_ssp]
        ) and ("Historical Reconstruction" in self.scenario_historical):
            data_warning = """Historical Reconstruction data is not available with SSP data.
            Try using the Historical Climate data instead."""

        elif (  # Warn user if no data is selected
            not True in ["SSP" in one for one in self.scenario_ssp]
        ) and (not True in ["Historical" in one for one in self.scenario_historical]):
            data_warning = "Please select as least one dataset."

        elif (
            (  # If both historical options are selected, warn user the data will be cut
                "Historical Reconstruction" in self.scenario_historical
            )
            and ("Historical Climate" in self.scenario_historical)
        ):
            data_warning = """The timescale of Historical Reconstruction data will be cut 
            to match that of the Historical Climate data if both are retrieved."""

        # Warnings based on time slice selections
        if (not True in ["SSP" in one for one in self.scenario_ssp]) and (
            "Historical Climate" in self.scenario_historical
        ):
            if (self.time_slice[0] < self.historical_climate_range[0]) or (
                self.time_slice[1] > self.historical_climate_range[1]
            ):
                data_warning = bad_time_slice_warning
        elif True in ["SSP" in one for one in self.scenario_ssp]:
            if not True in ["Historical" in one for one in self.scenario_historical]:
                if (self.time_slice[0] < self.ssp_range[0]) or (
                    self.time_slice[1] > self.ssp_range[1]
                ):
                    data_warning = bad_time_slice_warning
            else:
                if (self.time_slice[0] < self.historical_climate_range[0]) or (
                    self.time_slice[1] > self.ssp_range[1]
                ):
                    data_warning = bad_time_slice_warning
        elif self.scenario_historical == ["Historical Reconstruction"]:
            if (self.time_slice[0] < self.historical_reconstruction_range[0]) or (
                self.time_slice[1] > self.historical_reconstruction_range[1]
            ):
                data_warning = bad_time_slice_warning

        # Show warning
        self._data_warning = data_warning

    @param.depends("scenario_ssp", "scenario_historical", watch=True)
    def _update_time_slice_range(self):
        """
        Will discourage the user from selecting a time slice that does not exist
        for any of the selected scenarios, by updating the default range of years.
        """
        low_bound, upper_bound = self.time_slice

        if self.scenario_historical == ["Historical Climate"]:
            low_bound, upper_bound = self.historical_climate_range
        elif self.scenario_historical == ["Historical Reconstruction"]:
            low_bound, upper_bound = self.historical_reconstruction_range
        elif all(  # If both historical options are selected, and no SSP is selected
            [
                x in ["Historical Reconstruction", "Historical Climate"]
                for x in self.scenario_historical
            ]
        ) and (not True in ["SSP" in one for one in self.scenario_ssp]):
            low_bound, upper_bound = self.historical_climate_range

        if True in ["SSP" in one for one in self.scenario_ssp]:
            if (
                "Historical Climate" in self.scenario_historical
            ):  # If also append historical
                low_bound = self.historical_climate_range[0]
            else:
                low_bound = self.ssp_range[0]
            upper_bound = self.ssp_range[1]

        self.time_slice = (low_bound, upper_bound)

    @param.depends("scenario_ssp", "scenario_historical", "timescale", watch=True)
    def _update_simulation(self):
        """Simulation options will change if the scenario changes,
        or if the timescale changes, due to the fact that the ensmean
        data is available (and needs to be removed) for hourly data."""
        if self.resolution == "n/a":
            pass
        else:
            self.simulation = _get_simulation_options(
                cat=self.cat,
                activity_id=[
                    _downscaling_method_to_activity_id(dm)
                    for dm in self.downscaling_method
                ],
                table_id=_timescale_to_table_id(self.timescale),
                grid_label=_resolution_to_gridlabel(self.resolution),
                experiment_id=[
                    _scenario_to_experiment_id(scen)
                    for scen in self.scenario_ssp + self.scenario_historical
                ],
            )

    @param.depends("time_slice", "scenario_ssp", "scenario_historical", watch=False)
    def view(self):
        """
        Displays a timeline to help the user visualize the time ranges
        available, and the subset of time slice selected.
        """
        fig0 = Figure(figsize=(3, 1.75))
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
                if ["SSP" in one for one in self.scenario_ssp]:
                    if scen in [
                        "Historical Climate",
                        "Historical Reconstruction",
                    ]:
                        continue

                if scen == "Historical Reconstruction":
                    color = "darkblue"
                    if "Historical Climate" in self.scenario_historical:
                        center = 1997.5  # 1980-2014
                        x_width = 17.5
                        ax.annotate(
                            "Reconstruction", xy=(1967, y_offset + 0.06), fontsize=12
                        )
                    else:
                        center = 1986  # 1950-2022
                        x_width = 36
                        ax.annotate(
                            "Reconstruction", xy=(1955, y_offset + 0.06), fontsize=12
                        )

                elif scen == "Historical Climate":
                    color = "c"
                    center = 1997.5  # 1980-2014
                    x_width = 17.5
                    ax.annotate("Historical", xy=(1979, y_offset + 0.06), fontsize=12)

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
                    if "Historical Climate" in self.scenario_historical:
                        ax.errorbar(
                            x=1997.5, y=y_offset, xerr=17.5, linewidth=8, color="c"
                        )
                        ax.annotate(
                            "Historical", xy=(1979, y_offset + 0.06), fontsize=12
                        )
                    ax.annotate(scen[:10], xy=(2035, y_offset + 0.06), fontsize=12)

                ax.errorbar(
                    x=center, y=y_offset, xerr=x_width, linewidth=8, color=color
                )

                y_offset += 0.28

        ax.fill_betweenx(
            [0, 1],
            self.time_slice[0],
            self.time_slice[1],
            alpha=0.8,
            facecolor="lightgrey",
        )
        return mpl_pane


# ================ PRINT STATEMENT EXPLAINING CURRENT USER SELECTIONS ===================


def _get_data_selection_description(selections, location):
    """
    Make a long string to output to the user to show all their current selections.
    Updates whenever any of the input values are changed.
    """

    # Edit how the scenarios are printed in the description to make it reader-friendly
    if True in ["SSP" in one for one in selections.scenario_ssp]:
        if "Historical Climate" in selections.scenario_historical:
            scenario_print = [
                "Historical + " + ssp[:9] for ssp in selections.scenario_ssp
            ]
        else:
            scenario_print = [ssp[:9] for ssp in selections.scenario_ssp]
    else:
        scenario_print = selections.scenario_ssp + selections.scenario_historical

    # Show lat/lon selection only if area_subset == lat/lon
    if location.area_subset == "lat/lon":
        # bbox = min Longitude , min Latitude , max Longitude , max Latitude
        cached_area_print = (
            "bounding box <br>"
            "({:.2f}".format(location.longitude[0])
            + ", {:.2f}".format(location.latitude[0])
            + ", {:.2f}".format(location.longitude[1])
            + ", {:.2f}".format(location.latitude[1])
            + ")"
        )
    elif location.area_subset == "none":
        cached_area_print = "entire " + str(selections.resolution) + " grid"
    else:
        cached_area_print = str(location.cached_area)

    _data_selection_description = (
        "<font size='+0.10'>Data selections: </font><br>"
        "<ul>"
        "<li><b>data type: </b>" + str(location.data_type) + "</li>"
        "<li><b>downscaling method: </b>"
        + ", ".join(selections.downscaling_method)
        + "</li>"
        "<li><b>variable: </b>" + str(selections.variable) + "</li>"
        "<li><b>units: </b>" + str(selections.units) + "</li>"
        "<li><b>temporal resolution: </b>" + str(selections.timescale) + "</li>"
        "<li><b>model resolution: </b>" + str(selections.resolution) + "</li>"
        "<li><b>timeslice: </b>"
        + str(selections.time_slice[0])
        + " - "
        + str(selections.time_slice[1])
        + "</li>"
        "<li><b>datasets: </b>" + ", ".join(scenario_print) + "</li>"
        "</ul>"
    )
    _location_selection_description = (
        "<font size='+0.10'>Location selections: </font><br>"
        "<ul>"
        "<li><b>location: </b>" + cached_area_print + "</li>"
        "<li><b>compute area average? </b>" + str(selections.area_average) + "</li>"
        "</ul>"
    )
    return _data_selection_description + _location_selection_description


class _SelectionDescription(param.Parameterized):
    """
    Make a long string to output to the user to show all their current selections.
    Updates whenever any of the input values are changed.
    """

    _data_selection_description = param.String(
        default="", doc="Description of the user data selections."
    )

    def __init__(self, **params):
        super().__init__(**params)

        self._data_selection_description = _get_data_selection_description(
            selections=self.selections,
            location=self.location,
        )

    @param.depends(
        "selections.units",
        "selections.variable",
        "selections.scenario_historical",
        "selections.scenario_ssp",
        "selections.timescale",
        "selections.resolution",
        "selections.time_slice",
        "selections.area_average",
        "selections.downscaling_method",
        "location.area_subset",
        "location.cached_area",
        "location.data_type",
        "location.longitude",
        "location.latitude",
        watch=True,
    )
    def _update_data_selection_description(self):
        self._data_selection_description = _get_data_selection_description(
            selections=self.selections,
            location=self.location,
        )


# ================ DISPLAY LOCATION/DATA SELECTIONS IN PANEL ===================


def _selections_param_to_panel(selections):
    """For the _DataSelector object, get parameters and parameter
    descriptions formatted as panel widgets
    """
    area_average_text = pn.widgets.StaticText(
        value="Compute an area average across grid cells within your selected region?",
        name="",
    )
    area_average = pn.widgets.RadioBoxGroup.from_param(
        selections.param.area_average, inline=True
    )
    data_warning = pn.widgets.StaticText.from_param(
        selections.param._data_warning, name="", style={"color": "red"}
    )
    downscaling_method_text = pn.widgets.StaticText(value="", name="Downscaling method")
    downscaling_method = pn.widgets.CheckBoxGroup.from_param(
        selections.param.downscaling_method,
        inline=True,
        #### REMOVE THIS ONCE THE LOCA DATA IS AVAILABLE
        disabled=True,
    )
    historical_selection_text = pn.widgets.StaticText(
        value="<br>Estimates of recent historical climatic conditions",
        name="Historical Data",
    )
    historical_selection = pn.widgets.CheckBoxGroup.from_param(
        selections.param.scenario_historical
    )
    ssp_selection_text = pn.widgets.StaticText(
        value="<br> Shared Socioeconomic Pathways (SSPs) represent different global emissions scenarios",
        name="Future Model Data",
    )
    ssp_selection = pn.widgets.CheckBoxGroup.from_param(selections.param.scenario_ssp)
    resolution_text = pn.widgets.StaticText(
        value="Model resolution",
        name="",
    )
    resolution = pn.widgets.RadioBoxGroup.from_param(selections.param.resolution, inline=False)
    timescale_text = pn.widgets.StaticText(value="", name="Timescale")
    timescale = pn.widgets.RadioBoxGroup.from_param(selections.param.timescale, name="",inline=False)
    time_slice = pn.widgets.RangeSlider.from_param(selections.param.time_slice, name="")
    units_text = pn.widgets.StaticText(name="Variable Units", value="")
    units = pn.widgets.RadioBoxGroup.from_param(selections.param.units, inline=False)
    variable = pn.widgets.Select.from_param(selections.param.variable, name="")
    variable_text = pn.widgets.StaticText(name="Variable", value="")
    variable_description = pn.widgets.StaticText.from_param(
        selections.param.extended_description, name=""
    )

    widgets_dict = {
        "area_average": area_average,
        "data_warning": data_warning,
        "downscaling_method": downscaling_method,
        "historical_selection": historical_selection,
        "resolution": resolution,
        "ssp_selection": ssp_selection,
        "resolution": resolution,
        "timescale": timescale,
        "time_slice": time_slice,
        "units": units,
        "variable": variable,
        "variable_description": variable_description,
    }
    text_dict = {
        "area_average_text": area_average_text,
        "downscaling_method_text": downscaling_method_text,
        "historical_selection_text": historical_selection_text,
        "resolution_text": resolution_text,
        "ssp_selection_text": ssp_selection_text,
        "units_text": units_text,
        "timescale_text": timescale_text,
        "variable_text": variable_text,
    }

    return widgets_dict | text_dict


def _location_param_to_panel(location):
    """For the _LocSelectorArea object, get parameters and parameter
    descriptions formatted as panel widgets
    """
    data_type_text = pn.widgets.StaticText(
        value="",
        name="Data type",
    )
    data_type = pn.widgets.RadioBoxGroup.from_param(
        location.param.data_type, inline=True, name=""
    )
    area_subset = pn.widgets.Select.from_param(
        location.param.area_subset, name="Subset the data by..."
    )
    cached_area = pn.widgets.Select.from_param(
        location.param.cached_area, name="Location selection"
    )
    return {
        "area_subset": area_subset,
        "cached_area": cached_area,
        "data_type": data_type,
        "data_type_text": data_type_text,
        "latitude": location.param.latitude,
        "longitude": location.param.longitude,
    }


def _display_select(selections, location, map_view):
    """
    Called by 'select' at the beginning of the workflow, to capture user
    selections. Displays panel of widgets from which to make selections.
    Modifies 'selections' object, which is used by retrieve() to build an
    appropriate xarray Dataset.
    """

    selection_description = _SelectionDescription(
        selections=selections, location=location
    )

    # Get formatted panel widgets for each parameter
    selections_widgets = _selections_param_to_panel(selections)
    location_widgets = _location_param_to_panel(location)
    
    data_choices = pn.Column(
        selections_widgets["variable_text"],
        selections_widgets["variable"],
        selections_widgets["variable_description"],       
        pn.Row(
            pn.Column(
                selections_widgets["historical_selection_text"],
                selections_widgets["historical_selection"],
                selections_widgets["ssp_selection_text"],
                selections_widgets["ssp_selection"],
                
                pn.Column(
                    selections.view,
                    selections_widgets["time_slice"],
                    width=220,
                ),
                width=270,
            ),
            pn.Column(
                selections_widgets["units_text"],
                selections_widgets["units"],
                selections_widgets["timescale_text"],
                selections_widgets["timescale"],
                selections_widgets["resolution_text"],
                selections_widgets["resolution"],
                width=110,
            ),
        ),
        width=380,
    )

    col_1_location = pn.Column(
        map_view.view,
        location_widgets["area_subset"],
        location_widgets["cached_area"],
        location_widgets["latitude"],
        location_widgets["longitude"],
        selections_widgets["area_average_text"],
        selections_widgets["area_average"],
        width=190,
    )
    col_2_location = pn.Column(
        pn.Spacer(height=10),
        pn.widgets.CheckBoxGroup.from_param(
            location.param.station, name="Weather station"
        ),
        width=200,
    )
    loc_choices = pn.Row(col_1_location,col_2_location)
        
    everything_else = pn.Row(
        data_choices,
        pn.layout.HSpacer(width=10),
        loc_choices
    )
    
    # Panel overall structure:
    all_things = pn.Column(
        pn.Row(
            pn.Column(
                location_widgets["data_type_text"],
                location_widgets["data_type"],
                width = 150,
            ),
            pn.Column(
                selections_widgets["downscaling_method_text"],
                selections_widgets["downscaling_method"],
                width = 220,
            ),
            pn.Column(
                selections_widgets["data_warning"],
                width=400, 
            ),
        ),
        pn.Spacer(background='black',height=1),
        everything_else,
    )   

    return pn.Card(all_things,title="Choose Data Available with the Cal-Adapt Analytics Engine",collapsible=False)

# =============================== EXPORT DATA ==================================


class _UserFileChoices:
    # reserved for later: text boxes for dataset to export
    # as well as a file name
    # data_var_name = param.String()
    # output_file_name = param.String()

    def __init__(self):
        self._export_format_choices = ["Pick a file format", "CSV", "GeoTIFF", "NetCDF"]


class _FileTypeSelector(param.Parameterized):
    """
    If the user wants to export an xarray dataset, they can choose
    their preferred format here. Produces a panel from which to select a
    supported file type.
    """

    user_options = _UserFileChoices()
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
