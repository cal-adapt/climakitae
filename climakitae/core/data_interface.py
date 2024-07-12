import os
import xarray as xr
import pandas as pd
import geopandas as gpd
from shapely.geometry import box, Polygon
import intake
import param
import panel as pn
import numpy as np
import warnings
from matplotlib.figure import Figure
import matplotlib.ticker as ticker
import difflib
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from climakitae.core.paths import (
    variable_descriptions_csv_path,
    stations_csv_path,
    data_catalog_url,
    boundary_catalog_url,
)
from climakitae.core.boundaries import Boundaries
from climakitae.util.unit_conversions import get_unit_conversion_options
from climakitae.core.data_load import (
    read_catalog_from_csv,
    read_catalog_from_select,
)

from climakitae.util.utils import (
    scenario_to_experiment_id,
    resolution_to_gridlabel,
    timescale_to_table_id,
    downscaling_method_to_activity_id,
    downscaling_method_as_list,
    read_csv_file,
)

# Warnings raised by function get_subsetting_options, not sure why but they are silenced here
pd.options.mode.chained_assignment = None  # default='warn'

# Remove param's parameter descriptions from docstring because
# ANSI escape sequences in them complicate their rendering
param.parameterized.docstring_describe_params = False
# Docstring signatures are also hard to read and therefore removed
param.parameterized.docstring_signature = False


def _get_user_options(data_catalog, downscaling_method, timescale, resolution):
    """Using the data catalog, get a list of appropriate scenario and simulation options given a user's
    selections for downscaling method, timescale, and resolution.
    Unique variable ids for user selections are returned, then limited further in subsequent steps.

    Parameters
    ----------
    cat: intake catalog
    downscaling_method: str, one of "Dynamical", "Statistical", or "Dynamical+Statistical"
        Data downscaling method
    timescale: str, one of "hourly", "daily", or "monthly"
        Timescale
    resolution: str, one of "3 km", "9 km", "45 km"
        Model grid resolution

    Returns
    -------
    scenario_options: list
        Unique scenario values for input user selections
    simulation_options: list
        Unique simulation values for input user selections
    unique_variable_ids: list
        Unique variable id values for input user selections
    """

    method_list = downscaling_method_as_list(downscaling_method)

    # Get catalog subset from user inputs
    with warnings.catch_warnings(record=True):
        cat_subset = data_catalog.search(
            activity_id=[downscaling_method_to_activity_id(dm) for dm in method_list],
            table_id=timescale_to_table_id(timescale),
            grid_label=resolution_to_gridlabel(resolution),
        )

    # For LOCA grid we need to use the UCSD institution ID
    # This comes into play whenever Statistical is selected
    # WRF data on LOCA grid is tagged with UCSD institution ID
    if "Statistical" in downscaling_method:
        cat_subset = cat_subset.search(institution_id="UCSD")

    # Limit scenarios if both LOCA and WRF are selected
    # We just want the scenarios that are present in both datasets
    if downscaling_method == "Dynamical+Statistical":  # If both are selected
        loca_scenarios = cat_subset.search(
            activity_id="LOCA2"
        ).df.experiment_id.unique()  # LOCA unique member_ids
        wrf_scenarios = cat_subset.search(
            activity_id="WRF"
        ).df.experiment_id.unique()  # WRF unique member_ids
        overlapping_scenarios = list(set(loca_scenarios) & set(wrf_scenarios))
        cat_subset = cat_subset.search(experiment_id=overlapping_scenarios)

    elif downscaling_method == "Statistical":
        cat_subset = cat_subset.search(activity_id="LOCA2")

    # Get scenario options
    scenario_options = list(cat_subset.df["experiment_id"].unique())

    # Get all unique simulation options from catalog selection
    try:
        simulation_options = list(cat_subset.df["source_id"].unique())

        # Remove ensemble means
        if "ensmean" in simulation_options:
            simulation_options.remove("ensmean")
    except:
        simulation_options = []

    # Get variable options
    unique_variable_ids = list(cat_subset.df["variable_id"].unique())

    return scenario_options, simulation_options, unique_variable_ids


def _get_variable_options_df(
    variable_descriptions, unique_variable_ids, downscaling_method, timescale
):
    """Get variable options to display depending on downscaling method and timescale

    Parameters
    ----------
    variable_descriptions: pd.DataFrame
        Variable descriptions, units, etc in table format
    unique_variable_ids: list of strs
        List of unique variable ids from catalog.
        Used to subset var_config
    downscaling_method: str, one of "Dynamical", "Statistical", or "Dynamical+Statistical"
        Data downscaling method
    timescale: str, one of "hourly", "daily", or "monthly"
        Timescale

    Returns
    -------
    pd.DataFrame
        Subset of var_config for input downscaling_method and timescale
    """
    # Catalog options and derived options together
    derived_variables = list(
        variable_descriptions[
            variable_descriptions["variable_id"].str.contains("_derived")
        ]["variable_id"]
    )
    var_options_plus_derived = unique_variable_ids + derived_variables

    # Subset dataframe
    variable_options_df = variable_descriptions[
        (variable_descriptions["show"] == True)
        & (  # Make sure it's a valid variable selection
            variable_descriptions["variable_id"].isin(var_options_plus_derived)
            & (  # Make sure variable_id is part of the catalog options for user selections
                variable_descriptions["timescale"].str.contains(timescale)
            )  # Make sure its the right timescale
        )
    ]

    if downscaling_method == "Dynamical+Statistical":
        variable_options_df = variable_options_df[
            # Get shared variables
            variable_options_df["display_name"].duplicated()
        ]
    else:
        variable_options_df = variable_options_df[
            # Get variables only from one downscaling method
            variable_options_df["downscaling_method"]
            == downscaling_method
        ]
    return variable_options_df


def _get_var_ids(variable_descriptions, variable, downscaling_method, timescale):
    """Get variable ids that match the selected variable, timescale, and downscaling method.
    Required to account for the fact that LOCA, WRF, and various timescales use different variable id values.
    Used to retrieve the correct variables from the catalog in the backend.
    """

    method_list = downscaling_method_as_list(downscaling_method)

    var_id = variable_descriptions[
        (variable_descriptions["display_name"] == variable)
        & (  # Make sure it's a valid variable selection
            variable_descriptions["timescale"].str.contains(timescale)
        )  # Make sure its the right timescale
        & (
            variable_descriptions["downscaling_method"].isin(method_list)
        )  # Make sure it's the right downscaling method
    ]
    var_id = list(var_id.variable_id.values)
    return var_id


def _get_overlapping_station_names(
    stations_gdf,
    area_subset,
    cached_area,
    latitude,
    longitude,
    _geographies,
    _geography_choose,
):
    """Wrapper function that gets the string names of any overlapping weather stations"""
    subarea = _get_subarea(
        area_subset,
        cached_area,
        latitude,
        longitude,
        _geographies,
        _geography_choose,
    )
    overlapping_stations_gpd = _get_overlapping_stations(stations_gdf, subarea)
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
    return gpd.sjoin(stations, polygon, predicate="within")


def _get_subarea(
    area_subset,
    cached_area,
    latitude,
    longitude,
    _geographies,
    _geography_choose,
):
    """Get geometry from input settings
    Used for plotting or determining subset of overlapping weather stations in subsequent steps

    Returns
    -------
    gpd.GeoDataFrame

    """

    def _get_subarea_from_shape_index(
        boundary_dataset: Boundaries, shape_indices: list
    ) -> gpd.GeoDataFrame:
        return boundary_dataset.loc[shape_indices]

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
        # `if-condition` added for catching errors with delays in rendering cached area.
        if cached_area == None:
            shape_indices = [0]
        else:
            # Filter for indices that are selected in `Location selection` dropdown
            shape_indices = list(
                {
                    key: _geography_choose[area_subset][key] for key in cached_area
                }.values()
            )

        if area_subset == "states":
            df_ae = _get_subarea_from_shape_index(
                _geographies._us_states, shape_indices
            )
        elif area_subset == "CA counties":
            df_ae = _get_subarea_from_shape_index(
                _geographies._ca_counties, shape_indices
            )
        elif area_subset == "CA watersheds":
            df_ae = _get_subarea_from_shape_index(
                _geographies._ca_watersheds, shape_indices
            )
        elif area_subset == "CA Electric Load Serving Entities (IOU & POU)":
            df_ae = _get_subarea_from_shape_index(
                _geographies._ca_utilities, shape_indices
            )
        elif area_subset == "CA Electricity Demand Forecast Zones":
            df_ae = _get_subarea_from_shape_index(
                _geographies._ca_forecast_zones, shape_indices
            )
        elif area_subset == "CA Electric Balancing Authority Areas":
            df_ae = _get_subarea_from_shape_index(
                _geographies._ca_electric_balancing_areas, shape_indices
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


def _map_view(selections, stations_gdf):
    """View the current location selections on a map
    Updates dynamically

    Parameters
    ----------
    selections: DataSelector
        User data selections
    stations_gpd: gpd.DataFrame
        DataFrame with station coordinates

    """

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

    fig0 = Figure(figsize=(2.25, 2.25))
    proj = ccrs.Orthographic(-118, 40)
    crs_proj4 = proj.proj4_init  # used below
    xy = ccrs.PlateCarree()
    ax = fig0.add_subplot(111, projection=proj)
    mpl_pane = pn.pane.Matplotlib(fig0, dpi=1000)

    # Get geometry of selected location
    subarea_gpd = _get_subarea(
        selections.area_subset,
        selections.cached_area,
        selections.latitude,
        selections.longitude,
        selections._geographies,
        selections._geography_choose,
    )
    # Set plot extent
    ca_extent = [-125, -114, 31, 43]  # Zoom in on CA
    us_extent = [
        -130,
        -100,
        25,
        50,
    ]  # Western USA + a lil bit of baja (viva mexico)
    na_extent = [-150, -88, 8, 66]  # North America extent (largest extent)
    if selections.area_subset == "lat/lon":
        extent = na_extent  # default
        # Dynamically update extent depending on borders of lat/lon selection
        for extent_i in [ca_extent, us_extent, na_extent]:
            # Construct a polygon from the extent
            geom_extent = Polygon(
                box(extent_i[0], extent_i[2], extent_i[1], extent_i[3])
            )
            # Check if user selections for lat/lon are contained in the extent
            if geom_extent.contains(subarea_gpd.geometry.values[0]):
                # If so, set the extent to the smallest extent possible
                # Such that the lat/lon selection is contained within the map's boundaries
                extent = extent_i
                break
    elif (selections.resolution == "3 km") or ("CA" in selections.area_subset):
        extent = ca_extent
    elif (selections.resolution == "9 km") or (selections.area_subset == "states"):
        extent = us_extent
    elif selections.area_subset == "none":
        extent = na_extent
    else:  # Default for all other selections
        extent = ca_extent
    ax.set_extent(extent, crs=xy)

    # Set size of markers for stations depending on map boundaries
    if extent == ca_extent:
        scatter_size = 4.5
    elif extent == us_extent:
        scatter_size = 2.5
    elif extent == na_extent:
        scatter_size = 1.5

    if selections.resolution == "45 km":
        _add_res_to_ax(
            poly=_wrf_bb["45 km"],
            ax=ax,
            color="green",
            rotation=28,
            xy=(-154, 33.8),
            label="45 km",
        )
    elif selections.resolution == "9 km":
        _add_res_to_ax(
            poly=_wrf_bb["9 km"],
            ax=ax,
            color="red",
            rotation=32,
            xy=(-134, 42),
            label="9 km",
        )
    elif selections.resolution == "3 km":
        _add_res_to_ax(
            poly=_wrf_bb["3 km"],
            ax=ax,
            color="darkorange",
            rotation=32,
            xy=(-127, 40),
            label="3 km",
        )

    # Add user-selected geometries
    if selections.area_subset == "lat/lon":
        ax.add_geometries(
            subarea_gpd["geometry"].values,
            crs=ccrs.PlateCarree(),
            edgecolor="b",
            facecolor="None",
        )
    elif selections.area_subset != "none":
        subarea_gpd.to_crs(crs_proj4).plot(ax=ax, color="deepskyblue", zorder=2)
        mpl_pane.param.trigger("object")

    # Overlay the weather stations as points on the map
    if selections.data_type == "Station":
        # Subset the stations gpd to get just the user's selected stations
        # We need the stations gpd because it has the coordinates, which will be used to make the plot
        stations_selection_gdf = stations_gdf.loc[
            stations_gdf["station"].isin(selections.station)
        ]
        stations_selection_gdf = stations_selection_gdf.to_crs(
            crs_proj4
        )  # Convert to map projection
        ax.scatter(
            stations_selection_gdf.LON_X.values,
            stations_selection_gdf.LAT_Y.values,
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


class VariableDescriptions:
    """Load Variable Desciptions CSV only once

    This is a singleton class that needs to be called separately from DataInterface
    because variable descriptions are used without DataInterface in ck.view. Also
    ck.view is loaded on package load so this avoids loading boundary data when not
    needed.

    """

    def __new__(cls):
        if not hasattr(cls, "instance"):
            cls.instance = super(VariableDescriptions, cls).__new__(cls)
        return cls.instance

    def __init__(self):
        self.variable_descriptions = pd.DataFrame

    def load(self):
        if self.variable_descriptions.empty:
            self.variable_descriptions = read_csv_file(variable_descriptions_csv_path)


class DataInterface:
    """Load data connections into memory once

    This is a singleton class called by the various Param classes to connect to the local
    data and to the intake data catalog and parquet boundary catalog. The class attributes
    are read only so that the data does not get changed accidentially.

    """

    def __new__(cls):
        if not hasattr(cls, "instance"):
            cls.instance = super(DataInterface, cls).__new__(cls)
        return cls.instance

    def __init__(self):
        var_desc = VariableDescriptions()
        var_desc.load()
        self._variable_descriptions = var_desc.variable_descriptions
        self._stations = read_csv_file(stations_csv_path)
        self._stations_gdf = gpd.GeoDataFrame(
            self.stations,
            crs="EPSG:4326",
            geometry=gpd.points_from_xy(self.stations.LON_X, self.stations.LAT_Y),
        )
        self._data_catalog = intake.open_esm_datastore(data_catalog_url)

        # Get geography boundaries
        self._boundary_catalog = intake.open_catalog(boundary_catalog_url)
        self._geographies = Boundaries(self.boundary_catalog)

        self._geographies.load()

    @property
    def variable_descriptions(self):
        return self._variable_descriptions

    @property
    def stations(self):
        return self._stations

    @property
    def stations_gdf(self):
        return self._stations_gdf

    @property
    def data_catalog(self):
        return self._data_catalog

    @property
    def boundary_catalog(self):
        return self._boundary_catalog

    @property
    def geographies(self):
        return self._geographies


class DataParameters(param.Parameterized):
    """Python param object to hold data parameters for use in panel GUI."""

    # Unit conversion options for each unit
    unit_options_dict = get_unit_conversion_options()

    # Location defaults
    area_subset = param.Selector(objects=dict())
    cached_area = param.ListSelector(objects=dict())
    latitude = param.Range(default=(32.5, 42), bounds=(10, 67))
    longitude = param.Range(default=(-125.5, -114), bounds=(-156.82317, -84.18701))

    # Data defaults
    variable_type = param.Selector(
        default="Variable",
        objects=["Variable", "Derived Index"],
        doc="Choose between variable or AE derived index",
    )
    default_variable = "Air Temperature at 2m"
    time_slice = param.Range(default=(1980, 2015), bounds=(1950, 2100))
    resolution = param.Selector(default="9 km", objects=["3 km", "9 km", "45 km"])
    timescale = param.Selector(
        default="monthly", objects=["daily", "monthly", "hourly"]
    )
    scenario_historical = param.ListSelector(
        default=["Historical Climate"],
        objects=["Historical Climate", "Historical Reconstruction"],
    )
    area_average = param.Selector(
        default="No",
        objects=["Yes", "No"],
        doc="""Compute an area average?""",
    )
    downscaling_method = param.Selector(
        default="Dynamical",
        objects=["Dynamical", "Statistical", "Dynamical+Statistical"],
    )
    data_type = param.Selector(default="Gridded", objects=["Gridded", "Station"])
    station = param.ListSelector(objects=dict())
    _station_data_info = param.String(
        default="", doc="Information about the bias correction process and resolution"
    )

    # Empty params, initialized in __init__
    scenario_ssp = param.ListSelector(objects=dict())
    simulation = param.ListSelector(objects=dict())
    variable = param.Selector(objects=dict())
    units = param.Selector(objects=dict())
    extended_description = param.Selector(objects=dict())
    variable_id = param.ListSelector(objects=dict())

    # Temporal range of each dataset
    historical_climate_range_wrf = (1980, 2015)
    historical_climate_range_loca = (1950, 2015)
    historical_climate_range_wrf_and_loca = (1981, 2015)
    historical_reconstruction_range = (1950, 2022)
    ssp_range = (2015, 2100)

    # User warnings
    _info_about_station_data = "When you retrieve the station data, gridded model data will be bias-corrected to that point. This process can start from any model grid-spacing."
    _data_warning = param.String(
        default="", doc="Warning if user has made a bad selection"
    )

    def __init__(self, **params):
        # Set default values
        super().__init__(**params)

        self.data_interface = DataInterface()

        # Data Catalog
        self._data_catalog = self.data_interface.data_catalog

        # variable descriptions
        self._variable_descriptions = self.data_interface.variable_descriptions

        # station data
        self._stations_gdf = self.data_interface.stations_gdf

        # Get geography boundaries and selection options
        self._geographies = self.data_interface.geographies
        self._geography_choose = self._geographies.boundary_dict()

        # Set location params
        self.area_subset = "none"
        self.param["area_subset"].objects = list(self._geography_choose.keys())
        self.param["cached_area"].objects = list(
            self._geography_choose[self.area_subset].keys()
        )

        # Set data params
        (
            self.scenario_options,
            self.simulation,
            unique_variable_ids,
        ) = _get_user_options(
            data_catalog=self._data_catalog,
            downscaling_method=self.downscaling_method,
            timescale=self.timescale,
            resolution=self.resolution,
        )
        self.variable_options_df = _get_variable_options_df(
            variable_descriptions=self._variable_descriptions,
            unique_variable_ids=unique_variable_ids,
            downscaling_method=self.downscaling_method,
            timescale=self.timescale,
        )

        # Show derived index option?
        indices = True
        if self.data_type == "station":
            indices = False
        if self.downscaling_method != "Dynamical":
            indices = False
        if self.timescale == "monthly":
            indices = False
        if indices == False:
            self.param["variable_type"].objects = ["Variable"]
            self.variable_type = "Variable"
        elif indices == True:
            self.param["variable_type"].objects = ["Variable", "Derived Index"]

        # Set scenario param
        scenario_ssp_options = [
            scenario_to_experiment_id(scen, reverse=True)
            for scen in self.scenario_options
            if "ssp" in scen
        ]
        for scenario_i in [
            "SSP 3-7.0 -- Business as Usual",
            "SSP 2-4.5 -- Middle of the Road",
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

        # Set colormap, units, & extended description
        var_info = self.variable_options_df[
            self.variable_options_df["display_name"] == self.variable
        ]

        # Set params that are not selected by the user
        self.colormap = var_info.colormap.item()
        self.units = var_info.unit.item()
        self.extended_description = var_info.extended_description.item()
        self.variable_id = _get_var_ids(
            self._variable_descriptions,
            self.variable,
            self.downscaling_method,
            self.timescale,
        )
        self._data_warning = ""

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
        # Needs to be a list [] object in order to contain multiple objects for `cached_area`
        self.cached_area = [list(self._geography_choose[self.area_subset].keys())[0]]

    @param.depends("data_type", watch=True)
    def _update_area_average_based_on_data_type(self):
        """Update area average selection choices based on station vs. gridded data.
        There is no area average option if station data is selected. It will be shown as n/a.
        """
        if self.data_type == "Station":
            self.param["area_average"].objects = ["n/a"]
            self.area_average = "n/a"
        elif self.data_type == "Gridded":
            self.param["area_average"].objects = ["Yes", "No"]
            self.area_average = "No"

    @param.depends("downscaling_method", "data_type", "variable_type", watch=True)
    def _update_data_type_options_if_loca_or_derived_var_selected(self):
        """If statistical downscaling is selected, remove option for station data because we don't
        have the 2m temp variable for LOCA"""
        if (
            "Statistical" in self.downscaling_method
            or self.variable_type == "Derived Index"
        ):
            self.param["data_type"].objects = ["Gridded"]
            self.data_type = "Gridded"
        else:
            self.param["data_type"].objects = ["Gridded", "Station"]
        if "Station" in self.data_type or self.variable_type == "Derived Index":
            self.param["downscaling_method"].objects = ["Dynamical"]
            self.downscaling_method = "Dynamical"
        else:
            self.param["downscaling_method"].objects = [
                "Dynamical",
                "Statistical",
                "Dynamical+Statistical",
            ]

    @param.depends("data_type", "downscaling_method", watch=True)
    def _update_res_based_on_data_type_and_downscaling_method(self):
        """Update the grid resolution options based on the data selections."""
        if "Statistical" in self.downscaling_method:
            self.param["resolution"].objects = ["3 km"]
            self.resolution = "3 km"
        else:
            if self.data_type == "Station":
                self.param["resolution"].objects = ["3 km", "9 km"]
                if self.resolution == "45 km":
                    self.resolution = "3 km"
            elif self.data_type == "Gridded":
                self.param["resolution"].objects = ["3 km", "9 km", "45 km"]

    @param.depends(
        "data_type", "timescale", "downscaling_method", "variable_type", watch=True
    )
    def _remove_index_options_if_no_indices(self):
        """Remove derived index as an option if the current selections do not have any index options.
        UPDATE IF YOU ADD MORE INDICES."""

        ## Remove derived index as an option if the current selections do not have any index options.
        indices = True
        # Cases where we currently don't have derived indices
        if self.data_type == "station":
            # Only air temp available for station data
            indices = False
        if self.downscaling_method != "Dynamical":
            # Currently we only have indices for WRF data
            indices = False
        if self.timescale == "monthly":
            indices = False
        if indices == False:
            # Remove derived index as an option
            self.param["variable_type"].objects = ["Variable"]
            self.variable_type = "Variable"
        elif indices == True:
            self.param["variable_type"].objects = ["Variable", "Derived Index"]

    @param.depends(
        "timescale",
        "resolution",
        "downscaling_method",
        "data_type",
        "variable",
        "variable_type",
        watch=True,
    )
    def _update_user_options(self):
        """Update unique variable options"""

        # Station data is only available hourly
        if self.data_type == "Station":
            self.param["timescale"].objects = ["hourly"]
            self.timescale = "hourly"
            self.param["variable_type"].objects = ["Variable"]
            self.variable_type = "Variable"
        elif self.data_type == "Gridded":
            if self.downscaling_method == "Statistical":
                self.param["timescale"].objects = ["daily", "monthly"]
                if self.timescale == "hourly":
                    self.timescale = "daily"
            elif self.downscaling_method == "Dynamical":
                self.param["timescale"].objects = ["daily", "monthly", "hourly"]
            else:  # "Dynamical+Statistical"
                # If both are selected, only show daily data
                # We do not have WRF on LOCA grid resampled to monthly
                self.param["timescale"].objects = ["daily"]
                self.timescale = "daily"

        (
            self.scenario_options,
            self.simulation,
            unique_variable_ids,
        ) = _get_user_options(
            data_catalog=self._data_catalog,
            downscaling_method=self.downscaling_method,
            timescale=self.timescale,
            resolution=self.resolution,
        )

        if self.data_type == "Station":
            # If station is selected, the only valid option is air temperature
            temp = "Air Temperature at 2m"
            self.param["variable"].objects = [temp]
            self.variable = temp

        else:
            # Otherwise, get a list of variable options using the catalog search
            self.variable_options_df = _get_variable_options_df(
                variable_descriptions=self._variable_descriptions,
                unique_variable_ids=unique_variable_ids,
                downscaling_method=self.downscaling_method,
                timescale=self.timescale,
            )

            # Filter for derived indices
            # Depends on user selection for variable_type
            if self.variable_type == "Variable":
                # Remove indices
                self.variable_options_df = self.variable_options_df[
                    ~self.variable_options_df["variable_id"].str.contains("index")
                ]
            elif self.variable_type == "Derived Index":
                # Show only indices
                self.variable_options_df = self.variable_options_df[
                    self.variable_options_df["variable_id"].str.contains("index")
                ]
            var_options = self.variable_options_df.display_name.values
            self.param["variable"].objects = var_options
            if self.variable not in var_options:
                self.variable = var_options[0]

        var_info = self.variable_options_df[
            self.variable_options_df["display_name"] == self.variable
        ]  # Get info for just that variable
        self.extended_description = var_info.extended_description.item()
        self.variable_id = _get_var_ids(
            self._variable_descriptions,
            self.variable,
            self.downscaling_method,
            self.timescale,
        )
        self.colormap = var_info.colormap.item()

    @param.depends("resolution", "area_subset", watch=True)
    def _update_states_3km(self):
        if self.area_subset == "states":
            if self.resolution == "3 km":
                if "Statistical" in self.downscaling_method:
                    self.param["cached_area"].objects = ["CA"]
                elif self.downscaling_method == "Dynamical":
                    self.param["cached_area"].objects = [
                        "CA",
                        "NV",
                        "OR",
                        "UT",
                        "AZ",
                    ]
                self.cached_area = ["CA"]
            else:
                self.param["cached_area"].objects = self._geography_choose[
                    "states"
                ].keys()

    @param.depends("variable", "timescale", "downscaling_method", watch=True)
    def _update_unit_options(self):
        """Update unit options and native units for selected variable."""
        var_info = self.variable_options_df[
            self.variable_options_df["display_name"] == self.variable
        ]
        native_unit = var_info.unit.item()
        if native_unit in ["mm/d", "mm/h"]:
            # Show same unit options for all mm
            native_unit = "mm"
        if (
            native_unit in self.unit_options_dict.keys()
        ):  # See if there's unit conversion options for native variable
            self.param["units"].objects = self.unit_options_dict[native_unit]
            if self.units not in self.unit_options_dict[native_unit]:
                self.units = native_unit
        else:  # Just use native units if no conversion options available
            self.param["units"].objects = [native_unit]
            self.units = native_unit

    @param.depends("resolution", "downscaling_method", "data_type", watch=True)
    def _update_scenarios(self):
        """
        Update scenario options. Raise data warning if a bad selection is made.
        """
        # Set incoming scenario_historical
        _scenario_historical = self.scenario_historical

        # Get scenario options in catalog format
        scenario_ssp_options = [
            scenario_to_experiment_id(scen, reverse=True)
            for scen in self.scenario_options
            if "ssp" in scen
        ]
        for scenario_i in [
            "SSP 3-7.0 -- Business as Usual",
            "SSP 2-4.5 -- Middle of the Road",
            "SSP 5-8.5 -- Burn it All",
        ]:
            if scenario_i in scenario_ssp_options:  # Reorder list
                scenario_ssp_options.remove(scenario_i)  # Remove item
                scenario_ssp_options.append(scenario_i)  # Add to back of list
        self.param["scenario_ssp"].objects = scenario_ssp_options
        self.scenario_ssp = [x for x in self.scenario_ssp if x in scenario_ssp_options]

        historical_scenarios = ["historical", "reanalysis"]
        scenario_historical_options = [
            scenario_to_experiment_id(scen, reverse=True)
            for scen in self.scenario_options
            if scen in historical_scenarios
        ]
        self.param["scenario_historical"].objects = scenario_historical_options

        # check if input historical scenarios match new available scenarios
        # if no reanalysis scenario then return False
        def _check_inputs(a, b):
            chk = False
            if len(b) < 2:
                return chk
            for i in a:
                if i in a:
                    chk = True
            return chk

        # check if new selection has the historical scenario options and if not select the first new option
        if _check_inputs(_scenario_historical, scenario_historical_options):
            self.scenario_historical = _scenario_historical
        else:
            self.scenario_historical = [scenario_historical_options[0]]

    @param.depends(
        "scenario_ssp",
        "scenario_historical",
        "downscaling_method",
        "time_slice",
        watch=True,
    )
    def _update_data_warning(self):
        """Update warning raised to user based on their data selections."""
        data_warning = ""
        bad_time_slice_warning = """You've selected a time slice that is outside the temporal range 
        of the selected data."""

        # Set time range of historical data
        if self.downscaling_method == "Statistical":
            historical_climate_range = self.historical_climate_range_loca
        elif self.downscaling_method == "Dynamical+Statistical":
            historical_climate_range = self.historical_climate_range_wrf_and_loca
        else:
            historical_climate_range = self.historical_climate_range_wrf

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
            if (self.time_slice[0] < historical_climate_range[0]) or (
                self.time_slice[1] > historical_climate_range[1]
            ):
                data_warning = bad_time_slice_warning
        elif True in ["SSP" in one for one in self.scenario_ssp]:
            if not True in ["Historical" in one for one in self.scenario_historical]:
                if (self.time_slice[0] < self.ssp_range[0]) or (
                    self.time_slice[1] > self.ssp_range[1]
                ):
                    data_warning = bad_time_slice_warning
            else:
                if (self.time_slice[0] < historical_climate_range[0]) or (
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

    @param.depends(
        "scenario_ssp", "scenario_historical", "downscaling_method", watch=True
    )
    def _update_time_slice_range(self):
        """
        Will discourage the user from selecting a time slice that does not exist
        for any of the selected scenarios, by updating the default range of years.
        """
        low_bound, upper_bound = self.time_slice

        # Set time range of historical data
        if self.downscaling_method == "Statistical":
            historical_climate_range = self.historical_climate_range_loca
        elif self.downscaling_method == "Dynamical+Statistical":
            historical_climate_range = self.historical_climate_range_wrf_and_loca
        else:
            historical_climate_range = self.historical_climate_range_wrf

        if self.scenario_historical == ["Historical Climate"]:
            low_bound, upper_bound = historical_climate_range
        elif self.scenario_historical == ["Historical Reconstruction"]:
            low_bound, upper_bound = self.historical_reconstruction_range
        elif all(  # If both historical options are selected, and no SSP is selected
            [
                x in ["Historical Reconstruction", "Historical Climate"]
                for x in self.scenario_historical
            ]
        ) and (not True in ["SSP" in one for one in self.scenario_ssp]):
            low_bound, upper_bound = historical_climate_range

        if True in ["SSP" in one for one in self.scenario_ssp]:
            if (
                "Historical Climate" in self.scenario_historical
            ):  # If also append historical
                low_bound = historical_climate_range[0]
            else:
                low_bound = self.ssp_range[0]
            upper_bound = self.ssp_range[1]

        self.time_slice = (low_bound, upper_bound)

    @param.depends("data_type", watch=True)
    def _update_textual_description(self):
        if self.data_type == "Gridded":
            self._station_data_info = ""
        elif self.data_type == "Station":
            self._station_data_info = self._info_about_station_data

    @param.depends(
        "data_type",
        "area_subset",
        "cached_area",
        "latitude",
        "longitude",
        watch=True,
    )
    def _update_station_list(self):
        """Update the list of weather station options if the area subset changes"""
        if self.data_type == "Station":
            overlapping_stations = _get_overlapping_station_names(
                self._stations_gdf,
                self.area_subset,
                self.cached_area,
                self.latitude,
                self.longitude,
                self._geographies,
                self._geography_choose,
            )
            if len(overlapping_stations) == 0:
                notice = "No stations available at this location"
                self.param["station"].objects = [notice]
                self.station = [notice]
            else:
                self.param["station"].objects = overlapping_stations
                self.station = overlapping_stations
        elif self.data_type == "Gridded":
            notice = "Set data type to 'Station' to see options"
            self.param["station"].objects = [notice]
            self.station = [notice]

    def retrieve(self, config=None, merge=True):
        """Retrieve data from catalog

        By default, DataParameters determines the data retrieved.
        To retrieve data using the settings in a configuration csv file, set config to the local
        filepath of the csv.
        Grabs the data from the AWS S3 bucket, returns lazily loaded dask array.
        User-facing function that provides a wrapper for read_catalog_from_csv and read_catalog_from_select.

        Parameters
        ----------
        config: str, optional
            Local filepath to configuration csv file
            Default to None-- retrieve settings in selections
        merge: bool, optional
            If config is TRUE and multiple datasets desired, merge to form a single object?
            Defaults to True.

        Returns
        -------
        xr.DataArray
            Lazily loaded dask array
            Default if no config file provided
        xr.Dataset
            If multiple rows are in the csv, each row is a data_variable
            Only an option if a config file is provided
        list of xr.DataArray
            If multiple rows are in the csv and merge=True,
            multiple DataArrays are returned in a single list.
            Only an option if a config file is provided.

        """

        def warnoflargefilesize(da):
            if da.nbytes >= int(1e9) and da.nbytes < int(5e9):
                print(
                    "!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!\n"
                    "! Returned data array is large. Operations could take up to 5x longer than 1GB of data!\n"
                    "!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!\n"
                )
            elif da.nbytes >= int(5e9) and da.nbytes < int(1e10):
                print(
                    "!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!\n"
                    "!! Returned data array is very large. Operations could take up to 8x longer than 1GB of data !!\n"
                    "!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!\n"
                )
            elif da.nbytes >= int(1e10):
                print(
                    "!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!\n"
                    "!!! Returned data array is huge. Operations could take 10x to infinity longer than 1GB of data !!!\n"
                    "!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!\n"
                )

        if config is not None:
            if type(config) == str:
                data_return = read_catalog_from_csv(self, config, merge)
            else:
                raise ValueError(
                    "To retrieve data specified in a configuration file, please input the path to your local configuration csv as a string"
                )
        data_return = read_catalog_from_select(self)

        if isinstance(data_return, list):
            for l in data_return:
                warnoflargefilesize(l)
        else:
            warnoflargefilesize(data_return)
        return data_return


class DataParametersWithPanes(DataParameters):
    """Extends DataParameters class to include panel widgets that display the time scale and a map overview"""

    def __init__(self, **params):
        # Set default values
        super().__init__(**params)

    @param.depends(
        "time_slice",
        "scenario_ssp",
        "scenario_historical",
        "downscaling_method",
        watch=False,
    )
    def scenario_view(self):
        """
        Displays a timeline to help the user visualize the time ranges
        available, and the subset of time slice selected.
        """
        # Set time range of historical data
        if self.downscaling_method == "Statistical":
            historical_climate_range = self.historical_climate_range_loca
        elif self.downscaling_method == "Dynamical+Statistical":
            historical_climate_range = self.historical_climate_range_wrf_and_loca
        else:
            historical_climate_range = self.historical_climate_range_wrf
        historical_central_year = sum(historical_climate_range) / 2
        historical_x_width = historical_central_year - historical_climate_range[0]

        fig0 = Figure(figsize=(2, 2))
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
        mpl_pane = pn.pane.Matplotlib(fig0, dpi=1000)

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
                        center = historical_central_year
                        x_width = historical_x_width
                        ax.annotate(
                            "Reconstruction", xy=(1967 - 6, y_offset + 0.06), fontsize=9
                        )
                    else:
                        center = 1986  # 1950-2022
                        x_width = 36
                        ax.annotate(
                            "Reconstruction", xy=(1955 - 6, y_offset + 0.06), fontsize=9
                        )

                elif scen == "Historical Climate":
                    color = "c"
                    center = historical_central_year
                    x_width = historical_x_width
                    ax.annotate(
                        "Historical",
                        xy=(historical_climate_range[0] - 6, y_offset + 0.06),
                        fontsize=9,
                    )

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
                            x=historical_central_year,
                            y=y_offset,
                            xerr=historical_x_width,
                            linewidth=8,
                            color="c",
                        )
                        ax.annotate(
                            "Historical",
                            xy=(historical_climate_range[0] - 6, y_offset + 0.06),
                            fontsize=9,
                        )

                    ax.annotate(scen[:10], xy=(2035, y_offset + 0.06), fontsize=9)

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

    @param.depends(
        "downscaling_method",
        "resolution",
        "latitude",
        "longitude",
        "area_subset",
        "cached_area",
        "data_type",
        "station",
        watch=False,
    )
    def map_view(self):
        """Create a map of the location selections"""
        return _map_view(selections=self, stations_gdf=self._stations_gdf)


class Select(DataParametersWithPanes):
    def show(self):
        # Show panel visualually
        select_panel = _display_select(self)
        return select_panel


def _selections_param_to_panel(self):
    """For the _DataSelector object, get parameters and parameter
    descriptions formatted as panel widgets
    """
    area_subset = pn.widgets.Select.from_param(
        self.param.area_subset, name="Subset the data by..."
    )
    area_average_text = pn.widgets.StaticText(
        value="Compute an area average across grid cells within your selected region?",
        name="",
    )
    area_average = pn.widgets.RadioBoxGroup.from_param(
        self.param.area_average, inline=True
    )
    cached_area = pn.widgets.MultiSelect.from_param(
        self.param.cached_area, name="Location selection"
    )
    data_type_text = pn.widgets.StaticText(
        value="",
        name="Data type",
    )
    data_type = pn.widgets.RadioBoxGroup.from_param(
        self.param.data_type, inline=True, name=""
    )
    data_warning = pn.widgets.StaticText.from_param(
        self.param._data_warning, name="", style={"color": "red"}
    )
    downscaling_method_text = pn.widgets.StaticText(value="", name="Downscaling method")
    downscaling_method = pn.widgets.RadioBoxGroup.from_param(
        self.param.downscaling_method, inline=True
    )
    historical_selection_text = pn.widgets.StaticText(
        value="<br>Estimates of recent historical climatic conditions",
        name="Historical Data",
    )
    historical_selection = pn.widgets.CheckBoxGroup.from_param(
        self.param.scenario_historical
    )
    station_data_info = pn.widgets.StaticText.from_param(
        self.param._station_data_info, name="", style={"color": "red"}
    )
    ssp_selection_text = pn.widgets.StaticText(
        value="<br> Shared Socioeconomic Pathways (SSPs) represent different global emissions scenarios",
        name="Future Model Data",
    )
    ssp_selection = pn.widgets.CheckBoxGroup.from_param(self.param.scenario_ssp)
    resolution_text = pn.widgets.StaticText(
        value="",
        name="Model Grid-Spacing",
    )
    resolution = pn.widgets.RadioBoxGroup.from_param(
        self.param.resolution, inline=False
    )
    timescale_text = pn.widgets.StaticText(value="", name="Timescale")
    timescale = pn.widgets.RadioBoxGroup.from_param(
        self.param.timescale, name="", inline=False
    )
    time_slice = pn.widgets.RangeSlider.from_param(self.param.time_slice, name="")
    units_text = pn.widgets.StaticText(name="Variable Units", value="")
    units = pn.widgets.RadioBoxGroup.from_param(self.param.units, inline=False)
    variable = pn.widgets.Select.from_param(self.param.variable, name="")
    variable_text = pn.widgets.StaticText(name="Variable", value="")
    variable_description = pn.widgets.StaticText.from_param(
        self.param.extended_description, name=""
    )
    variable_type = pn.widgets.RadioBoxGroup.from_param(
        self.param.variable_type, inline=True, name=""
    )

    widgets_dict = {
        "area_average": area_average,
        "area_subset": area_subset,
        "cached_area": cached_area,
        "data_type": data_type,
        "data_type_text": data_type_text,
        "data_warning": data_warning,
        "downscaling_method": downscaling_method,
        "historical_selection": historical_selection,
        "latitude": self.param.latitude,
        "longitude": self.param.longitude,
        "resolution": resolution,
        "station_data_info": station_data_info,
        "ssp_selection": ssp_selection,
        "resolution": resolution,
        "timescale": timescale,
        "time_slice": time_slice,
        "units": units,
        "variable": variable,
        "variable_description": variable_description,
        "variable_type": variable_type,
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


def _display_select(self):
    """
    Called by 'select' at the beginning of the workflow, to capture user
    selections. Displays panel of widgets from which to make selections.
    Modifies 'selections' object, which is used by retrieve() to build an
    appropriate xarray Dataset.
    """
    # Get formatted panel widgets for each parameter
    widgets = _selections_param_to_panel(self)

    data_choices = pn.Column(
        widgets["variable_text"],
        widgets["variable_type"],
        widgets["variable"],
        widgets["variable_description"],
        pn.Row(
            pn.Column(
                widgets["historical_selection_text"],
                widgets["historical_selection"],
                widgets["ssp_selection_text"],
                widgets["ssp_selection"],
                pn.Column(
                    self.scenario_view,
                    widgets["time_slice"],
                    width=220,
                ),
                width=250,
            ),
            pn.Column(
                widgets["units_text"],
                widgets["units"],
                widgets["timescale_text"],
                widgets["timescale"],
                widgets["resolution_text"],
                widgets["resolution"],
                widgets["station_data_info"],
                width=150,
            ),
        ),
        width=380,
    )

    col_1_location = pn.Column(
        self.map_view,
        widgets["area_subset"],
        widgets["cached_area"],
        widgets["latitude"],
        widgets["longitude"],
        widgets["area_average_text"],
        widgets["area_average"],
        width=220,
    )
    col_2_location = pn.Column(
        pn.Spacer(height=10),
        pn.widgets.StaticText(
            value="",
            name="Weather station",
        ),
        pn.widgets.CheckBoxGroup.from_param(self.param.station, name=""),
        width=270,
    )
    loc_choices = pn.Row(col_1_location, col_2_location)

    everything_else = pn.Row(data_choices, pn.layout.HSpacer(width=10), loc_choices)

    # Panel overall structure:
    all_things = pn.Column(
        pn.Row(
            pn.Column(
                widgets["data_type_text"],
                widgets["data_type"],
                width=150,
            ),
            pn.Column(
                widgets["downscaling_method_text"],
                widgets["downscaling_method"],
                width=325,
            ),
            pn.Column(
                widgets["data_warning"],
                width=400,
            ),
        ),
        pn.Spacer(background="black", height=1),
        everything_else,
    )

    return pn.Card(
        all_things,
        title="Choose Data Available with the Cal-Adapt Analytics Engine",
        collapsible=False,
    )


## -------------- Data access without GUI -------------------


def _get_user_friendly_catalog(intake_catalog, variable_descriptions):
    """Get a user-friendly version of the intake data catalog using climakitae naming conventions"""

    # Get the catalog as a dataframe
    cat_df = intake_catalog.df.copy()

    # Create new, user friendly catalog in pandas DataFrame format
    cat_df_cleaned = pd.DataFrame()
    cat_df_cleaned["downscaling_method"] = cat_df["activity_id"].apply(
        lambda x: downscaling_method_to_activity_id(x, reverse=True)
    )
    cat_df_cleaned["resolution"] = cat_df["grid_label"].apply(
        lambda x: resolution_to_gridlabel(x, reverse=True)
    )
    cat_df_cleaned["timescale"] = cat_df["table_id"].apply(
        lambda x: timescale_to_table_id(x, reverse=True)
    )
    cat_df_cleaned["scenario"] = cat_df["experiment_id"].apply(
        lambda x: scenario_to_experiment_id(x, reverse=True)
    )
    cat_df_cleaned["variable_id"] = cat_df["variable_id"]

    # Get user-friendly variable names from variable_descriptions.csv and add to dataframe
    cat_df_cleaned["variable"] = cat_df_cleaned.apply(
        lambda x: _get_var_name_from_table(
            x["variable_id"],
            x["downscaling_method"],
            x["timescale"],
            variable_descriptions,
        ),
        axis=1,
    )

    # We dont' show the users all the available variables in the catalog
    # These variables aren't defined in variable_descriptions.csv
    # Here, we just remove all those variables
    cat_df_cleaned = cat_df_cleaned[cat_df_cleaned["variable"] != "NONE"]

    # Move variable column to first position
    col = cat_df_cleaned.pop("variable")
    cat_df_cleaned.insert(0, col.name, col)

    # Remove variable_id column
    cat_df_cleaned.pop("variable_id")

    # Remove duplicate columns
    # Duplicates occur due to the many unique member_ids
    cat_df_cleaned = cat_df_cleaned.drop_duplicates(ignore_index=True)

    return cat_df_cleaned


def _get_var_name_from_table(variable_id, downscaling_method, timescale, var_pd):
    """Get the variable name corresponding to its ID, downscaling method, and timescale
    Enables the _get_user_friendly_catalog function to get the name of a variable corresponding to a set of user inputs
    i.e we have several different precip variables, corresponding to different downscaling methods (WRF vs. LOCA)

    Parameters
    ----------
    variable_id: str
    downscaling_method: str
    timescale: str
    var_pd: pd.DataFrame
        Variable descriptions table

    Returns
    -------
    var_name: str
        Display name of variable from variable descriptions table
        Will match what the user would see in the selections GUI
    """
    # Query the table based on input values
    var_pd_query = var_pd[
        (var_pd["variable_id"] == variable_id)
        & (var_pd["downscaling_method"] == downscaling_method)
    ]

    # Timescale in table needs to be handled differently
    # This is because the monthly variables are derived from daily variables, so they are listed in the table as "daily, monthly"
    # Hourly variables may be different
    # Querying the data needs special handling due to the layout of the csv file
    var_pd_query = var_pd_query[var_pd_query["timescale"].str.contains(timescale)]

    # This might return nothing if the variable is one we don't want to show the users
    # If so, set the var_name to nan
    # The row will later be dropped
    if len(var_pd_query) == 0:
        var_name = "NONE"

    # If a variable name is found, grab and return its proper name
    else:
        var_name = var_pd_query["display_name"].item()

    return var_name


def get_data_options(
    variable=None,
    downscaling_method=None,
    resolution=None,
    timescale=None,
    scenario=None,
    tidy=True,
):
    """Get data options, in the same format as the Select GUI, given a set of possible inputs.
    Allows the user to access the data using the same language as the GUI, bypassing the sometimes unintuitive naming in the catalog.
    If no function inputs are provided, the function returns the entire AE catalog that is available via the Select GUI

    Parameters
    ----------
    variable: str, optional
        Default to None
    downscaling_method: str, optional
        Default to None
    resolution: str, optional
        Default to None
    timescale: str, optional
        Default to None
    scenario: str, optional
        Default to None
    tidy: boolean, optional
        Format the pandas dataframe? This creates a DataFrame with a MultiIndex that makes it easier to parse the options.
        Default to True

    Returns
    -------
    cat_subset: pd.DataFrame
        Catalog options for user-provided inputs

    """

    # Get intake catalog and variable descriptions from DataInterface object
    data_interface = DataInterface()
    var_pd = data_interface.variable_descriptions
    catalog = data_interface.data_catalog

    cat_df = _get_user_friendly_catalog(
        intake_catalog=catalog, variable_descriptions=var_pd
    )

    # Raise error for bad input from user
    for user_input in [variable, downscaling_method, resolution, timescale, scenario]:
        if (user_input is not None) and (type(user_input) != str):
            print("Function arguments require a single string value for your inputs")
            return None

    d = {
        "variable": variable,
        "timescale": timescale,
        "downscaling_method": downscaling_method,
        "scenario": scenario,
        "resolution": resolution,
    }

    for key, val in zip(
        d.keys(), d.values()
    ):  # Loop through each key, value pair in the dictionary
        # Use the catalog to find the valid values in the list
        valid_options = np.unique(cat_df[key].values)
        if (
            val == None
        ):  # If the user didn't input anything for that key, set the values to all the valid options
            d[key] = valid_options
            continue  # Don't finish the loop
        # If the input value is not in the valid options, see if you can help the user out
        elif val not in valid_options:

            # Perhaps they didn't capitalize it correctly?
            if val.lower().capitalize() in valid_options:
                d[key] = [val.lower().capitalize()]
                continue  # Don't finish the loop

            # This catches any common bad inputs for resolution: i.e. "3KM" or "3km" instead of "3 km"
            if key == "resolution":
                try:
                    good_resolution_input = val.lower().split("km")[0] + " km"
                    if good_resolution_input in valid_options:
                        d[key] = [good_resolution_input]
                        continue
                except:
                    pass

            print("Input " + key + "='" + val + "' is not a valid option.")
            # Use difflib package to make a guess for what the input might have been
            # For example, if they input "statistikal" instead of "Statistical", difflib will find "Statistical"
            # Change the cutoff to increase/decrease the flexibility of the function
            closest_option = difflib.get_close_matches(val, valid_options, cutoff=0.59)
            if len(closest_option) == 0:
                # Sad! No closest options found. Just set the key to all valid options
                print("Valid options: " + ", ".join(valid_options))
                raise ValueError("Bad input")
            else:  # difflib made a guess! Set the key to the guess (or, guesses)
                if len(closest_option) == 1:
                    print("Did you mean '" + closest_option[0] + "'?")
                else:
                    print(
                        "Did you mean: \n- "
                        + "\n- ".join([x + "?" for x in closest_option])
                    )
                d[key] = closest_option
    # Dictionary values should be a list for the pandas logic to work correctly
    for key, val in zip(d.keys(), d.values()):
        if type(val) not in (list, np.ndarray):
            d[key] = [val]

    # Subset the catalog with the user's inputs
    cat_subset = cat_df[
        (cat_df["variable"].isin(d["variable"]))
        & (cat_df["downscaling_method"].isin(d["downscaling_method"]))
        & (cat_df["resolution"].isin(d["resolution"]))
        & (cat_df["timescale"].isin(d["timescale"]))
        & (cat_df["scenario"].isin(d["scenario"]))
    ].reset_index(drop=True)
    if len(cat_subset) == 0:
        print("No data found for your input values")
        return None

    if tidy:
        cat_subset = cat_subset.set_index(
            ["downscaling_method", "scenario", "timescale"]
        )
    return cat_subset


def get_subsetting_options(area_subset="all"):
    """Get all geometry options for spatial subsetting.
    Options match those in selections GUI

    Parameters
    -----------
    area_subset: str, one of "all", "states", "CA counties", "CA Electricity Demand Forecast Zones", "CA watersheds", "CA Electric Balancing Authority Areas", "CA Electric Load Serving Entities (IOU & POU)"
        Defaults to "all", which shows all the geometry options with area_subset as a multiindex

    Returns
    -------
    geom_df: pd.DataFrame
        Geometry options
        Shows only options for one area_subset if input is provided that is not "all"
        i.e. if area_subset = "states", only the options for states will be returned
    """
    # Get geographies from DataInterface object
    data_interface = DataInterface()
    geographies = data_interface._geographies
    boundary_dict = geographies.boundary_dict()

    # Get geometries and labels from Boundaries object
    df_dict = {
        "states": geographies._us_states[["abbrevs", "geometry"]].rename(
            columns={"abbrevs": "NAME"}
        ),
        "CA counties": geographies._ca_counties[["NAME", "geometry"]],
        "CA Electricity Demand Forecast Zones": geographies._ca_forecast_zones.rename(
            columns={"FZ_Name": "NAME"}
        )[["NAME", "geometry"]],
        "CA watersheds": geographies._ca_watersheds.rename(columns={"Name": "NAME"})[
            ["NAME", "geometry"]
        ],
        "CA Electric Balancing Authority Areas": geographies._ca_electric_balancing_areas[
            ["NAME", "geometry"]
        ],
        "CA Electric Load Serving Entities (IOU & POU)": geographies._ca_utilities.rename(
            columns={"Utility": "NAME"}
        )[
            ["NAME", "geometry"]
        ],
    }

    # Confirm that input for argument "area_subset" is valid
    # Raise error and print helpful statements if bad input
    valid_inputs = list(df_dict.keys()) + ["all"]
    if area_subset not in valid_inputs:
        print(
            "'"
            + str(area_subset)
            + "' is not a valid option for function argument 'area_subset'.\nChoose one of the following: "
            + ", ".join(valid_inputs)
        )
        print("Default argument 'all' will show all valid geometry options.")
        raise ValueError("Bad input for argument 'area_subset'")

    # Some of the geometry options are limited further by the selections.show() GUI
    # i.e. not all US states are an option in the GUI, even though the parquet file provided by geographies._us_states contains all US states
    # Here, we limit the output to return the same options as the GUI
    for name, df in df_dict.items():
        df["area_subset"] = [name] * len(
            df
        )  # Add area subset as a column. Used to create multiindex if area_subset = "all"
        df = df[df["NAME"].isin(list(boundary_dict[name].keys()))]
        df_dict[name] = df  # Replace the DataFrame with the new, reduced DataFrame

    if area_subset != "all":
        # Only return the desired area subset
        geoms_df = (
            df_dict[area_subset]
            .drop(columns="area_subset")
            .rename(columns={"NAME": "cached_area"})
            .set_index("cached_area")
        )
    else:
        geoms_df = pd.concat(list(df_dict.values())).rename(
            columns={"NAME": "cached_area"}
        )
        geoms_df = geoms_df.set_index(
            ["area_subset", "cached_area"]
        )  # Create multiindex

    return geoms_df


def get_data(
    variable,
    downscaling_method,
    resolution,
    timescale,
    scenario,
    units=None,
    area_subset="none",
    cached_area=["entire domain"],
    area_average="No",
):
    # Need to add error handing for bad variable input
    """Retrieve data from the catalog using a simple function.
    Contrasts with selections.retrieve(), which retrieves data from the user imputs in climakitae's selections GUI.

    Parameters
    ----------
    variable: str
    downscaling_method: str
    resolution: str
    timescale: str
    scenario: str
    units: None, optional
        Defaults to native units of data
    area_subset: str, optional
        Area category: i.e "CA counties"
        Defaults to entire domain ("none")
    cached_area: list, optional
        Area: i.e "Alameda county"
        Defaults to entire domain ("none")
    area_average: str, one of "No" or "Yes", optional
        Take an average over spatial domain?
        Default to No

    Returns
    --------
    data: xarray.core.dataarray.DataArray

    """

    # Get variable descriptions from DataInterface object
    data_interface = DataInterface()
    var_pd = data_interface.variable_descriptions

    if type(cached_area) == str:
        cached_area = [cached_area]

    # Maybe the user put an input for cached area but not for area subset
    # We need to have the matching/correct area subset in order for selections.retrieve() to actually subset the data
    # Here, we load in the geometry options to set area_subset to the correct value
    # This also raises an appropriate error if the user has a bad input
    if area_subset == "none" and cached_area != ["entire domain"]:
        geom_df = get_subsetting_options(area_subset="all").reset_index()
        area_subset_vals = geom_df[geom_df["cached_area"] == cached_area[0]][
            "area_subset"
        ].values
        if len(area_subset_vals) == 0:
            print(
                "'"
                + str(cached_area[0])
                + "' is not a valid option for function argument 'cached_area'"
            )
            raise ValueError("Bad input for argument 'cached_area'")
        else:
            area_subset = area_subset_vals[0]

    # Query the table based on input values
    var_pd_query = var_pd[
        (var_pd["display_name"] == variable)
        & (var_pd["downscaling_method"] == downscaling_method)
    ]

    # Timescale in table needs to be handled differently
    # This is because the monthly variables are derived from daily variables, so they are listed in the table as "daily, monthly"
    # Hourly variables may be different
    # Querying the data needs special handling due to the layout of the csv file
    var_pd_query = var_pd_query[var_pd_query["timescale"].str.contains(timescale)]

    if units is None:
        units = var_pd_query["unit"].item()

    # Create selections object
    selections = Select()

    # Defaults
    selections.area_average = area_average
    selections.area_subset = area_subset
    selections.cached_area = cached_area

    # Function not currently flexible enough to allow for appending historical
    # Need to add in error control
    selections.append_historical = False

    # User selections
    selections.downscaling_method = downscaling_method
    selections.variable = variable
    selections.resolution = resolution
    selections.timescale = timescale
    selections.units = units

    if scenario in ["Historical Climate", "Historical Reconstruction"]:
        selections.scenario_historical = [scenario]
        selections.scenario_ssp = []
    else:
        selections.scenario_historical = []
        selections.scenario_ssp = [scenario]

    # Retrieve data
    data = selections.retrieve()
    return data
