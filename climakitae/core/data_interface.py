import pandas as pd
import geopandas as gpd
from shapely.geometry import box
import intake
import param
import numpy as np
import warnings
import difflib
import cartopy.crs as ccrs
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
    downscaling_method_as_list,
    read_csv_file,
    scenario_to_experiment_id,
    resolution_to_gridlabel,
    timescale_to_table_id,
    downscaling_method_to_activity_id,
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
    data_catalog: intake_esm.source.ESMDataSource
        Intake ESM data catalog
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

    Parameters
    ----------
    variable_descriptions: pd.DataFrame
        Variable descriptions, units, etc in table format
    variable: str
        variable display name from catalog.
    downscaling_method: str, one of "Dynamical", "Statistical", or "Dynamical+Statistical"
        Data downscaling method
    timescale: str, one of "hourly", "daily", or "monthly"
        Timescale

    Returns
    -------
    list
        variable ids from intake catalog matching incoming query
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
    """Wrapper function that gets the string names of any overlapping weather stations

    Parameters
    ----------
    stations_gdf: gpd.GeoDataFrame
        geopandas GeoDataFrame of station locations
    area_subset: str
        DataParameters.area_subset param value
    cached_area: str
        DataParameters.cached_area param value
    latitude: float
        DataParameters.latitude param value
    longitude: float
        DataParameters.longitude param value
    _geographies: Boundaries
        reference to Boundaries class
    _geography_choose: dict
        dict of dicts containing boundary attributes
    """
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

    Parameters
    ----------
    area_subset: str
        DataParameters.area_subset param value
    cached_area: str
        DataParameters.cached_area param value
    latitude: tuple
        DataParameters.latitude param value
    longitude: tuple
        DataParameters.longitude param value
    _geographies: Boundaries
        reference to Boundaries class
    _geography_choose: dict
        dict of dicts containing boundary attributes

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


class VariableDescriptions:
    """Load Variable Desciptions CSV only once

    This is a singleton class that needs to be called separately from DataInterface
    because variable descriptions are used without DataInterface in ck.view. Also
    ck.view is loaded on package load so this avoids loading boundary data when not
    needed.

    Attributes
    ----------
    variable_descriptions: pd.DataFrame
        pandas dataframe that stores available data variables usable with the package

    """

    def __new__(cls):
        if not hasattr(cls, "instance"):
            cls.instance = super(VariableDescriptions, cls).__new__(cls)
        return cls.instance

    def __init__(self):
        self.variable_descriptions = pd.DataFrame

    def load(self):
        """Read the variable descriptions csv into class variable."""
        if self.variable_descriptions.empty:
            self.variable_descriptions = read_csv_file(variable_descriptions_csv_path)


class DataInterface:
    """Load data connections into memory once

    This is a singleton class called by the various Param classes to connect to the local
    data and to the intake data catalog and parquet boundary catalog. The class attributes
    are read only so that the data does not get changed accidentially.

    Attributes
    ----------
    variable_descriptions: pd.DataFrame
        variable descriptions pandas data frame
    stations: gpd.DataFrame
        station locations pandas data frame
    stations_gdf: gpd.GeoDataFrame
        station locations geopandas data frame
    data_catalog: intake_esm.source.ESMDataSource
        intake ESM data catalog
    boundary_catalog: intake.catalog.Catalog
        parquet boundary catalog
    geographies: Boundaries
        boundary dictionaries class
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
    """Python param object to hold data parameters for use in panel GUI.
    Call DataParameters when you want to select and retrieve data from the
    climakitae data catalog without using the ck.Select GUI. ck.Select uses
    this class to store selections and retrieve data.

    DataParameters calls DataInterface, a singleton class that makes the connection
    to the intake-esm data store in S3 bucket.

    Attributes
    ----------
    unit_options_dict: dict
        options dictionary for converting unit to other units
    area_subset: str
        dataset to use from Boundaries for sub area selection
    cached_area: list of strs
        one or more features from area_subset datasets to use for selection
    latitude: tuple
        latitude range of selection box
    longitude: tuple
        longitude range of selection box
    variable_type: str
        toggle raw or derived variable selection
    default_variable: str
        initial variable to have selected in widget
    time_slice: tuple
        year range to select
    resolution: str
        resolution of data to select ("3 km", "9 km", "45 km")
    timescale: str
        frequency of dataset ("hourly", "daily", "monthly")
    scenario_historical: list of strs
        historical scenario selections
    area_average: str
        whether to comput area average ("Yes", "No")
    downscaling_method: str
        whether to choose WRF or LOCA2 data or both ("Dynamical", "Statistical", "Dynamical+Statistical")
    data_type: str
        whether to choose gridded or station based data ("Gridded", "Station")
    station: list or strs
        list of stations that can be filtered by cached_area
    _station_data_info: str
        informational statement when station data selected with data_type
    scenario_ssp: list of strs
        list of future climate scenarios selected (availability depends on other params)
    simulation: list of strs
        list of simulations (models) selected (availability depends on other params)
    variable: str
        variable long display name
    units: str
        unit abbreviation currently of the data (native or converted)
    extended_description: str
        extended description of the data variable
    variable_id: list of strs
        list of variable ids that match the variable (WRF and LOCA2 can have different codes for same type of variable)
    historical_climate_range_wrf: tuple
        time range of historical WRF data
    historical_climate_range_loca: tuple
        time range of historical LOCA2 data
    historical_climate_range_wrf_and_loca: tuple
        time range of historical WRF and LOCA2 data combined
    historical_reconstruction_range: tuple
        time range of historical reanalysis data
    ssp_range: tuple
        time range of future scenario SSP data
    _info_about_station_data: str
        warning message about station data
    _data_warning: str
        warning about selecting unavailable data combination
    data_interface: DataInterface
        data connection singleton class that provides data
    _data_catalog: intake_esm.source.ESMDataSource
        shorthand alias to DataInterface.data_catalog
    _variable_descriptions: pd.DataFrame
        shorthand alias to DataInterface.variable_descriptions
    _stations_gdf: gpd.GeoDataFrame
        shorthand alias to DataInterface.stations_gdf
    _geographies: Boundaries
        shorthand alias to DataInterface.geographies
    _geography_choose: dict
        shorthand alias to Boundaries.boundary_dict()
    colormap: str
        default colormap to render the currently selected data
    scenario_options: list of strs
        list of available scenarios (historical and ssp) for selection
    variable_options_df: pd.DataFrame
        filtered variable descriptions for the downscaling_method and timescale
    """

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


## -------------- Data access without GUI -------------------


def _get_user_friendly_catalog(intake_catalog, variable_descriptions):
    """Get a user-friendly version of the intake data catalog using climakitae naming conventions

    Parameters
    ----------
    intake_catalog: intake_esm.source.ESMDataSource
    variable_descriptions: pd.DataFrame

    Returns
    -------
    cat_df_cleaned: intake_esm.source.ESMDataSource
    """

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

    # Move variable row to first position
    col = cat_df_cleaned.pop("variable")
    cat_df_cleaned.insert(0, col.name, col)

    # Remove variable_id row
    cat_df_cleaned.pop("variable_id")

    # Remove duplicate rows
    # Duplicates occur due to the many unique member_ids
    cat_df_cleaned = cat_df_cleaned.drop_duplicates(ignore_index=True)

    return cat_df_cleaned


def _get_var_name_from_table(variable_id, downscaling_method, timescale, var_df):
    """Get the variable name corresponding to its ID, downscaling method, and timescale
    Enables the _get_user_friendly_catalog function to get the name of a variable corresponding to a set of user inputs
    i.e we have several different precip variables, corresponding to different downscaling methods (WRF vs. LOCA)

    Parameters
    ----------
    variable_id: str
    downscaling_method: str
    timescale: str
    var_df: pd.DataFrame
        Variable descriptions table

    Returns
    -------
    var_name: str
        Display name of variable from variable descriptions table
        Will match what the user would see in the selections GUI
    """
    # Query the table based on input values
    var_df_query = var_df[
        (var_df["variable_id"] == variable_id)
        & (var_df["downscaling_method"] == downscaling_method)
    ]

    # Timescale in table needs to be handled differently
    # This is because the monthly variables are derived from daily variables, so they are listed in the table as "daily, monthly"
    # Hourly variables may be different
    # Querying the data needs special handling due to the layout of the csv file
    var_df_query = var_df_query[var_df_query["timescale"].str.contains(timescale)]

    # This might return nothing if the variable is one we don't want to show the users
    # If so, set the var_name to nan
    # The row will later be dropped
    if len(var_df_query) == 0:
        var_name = "NONE"

    # If a variable name is found, grab and return its proper name
    else:
        var_name = var_df_query["display_name"].item()

    return var_name


def _get_closest_options(val, valid_options):
    """If the user inputs a bad option, find the closest option from a list of valid options

    Parameters
    ----------
    val: str
        User input
    valid_options: list
        Valid options for that key from the catalog

    Returns
    -------
    closest_options: list or None
        List of best guesses, or None if nothing close is found
    """

    # Perhaps the user just capitalized it wrong?
    is_it_just_capitalized_wrong = [
        i for i in valid_options if val.lower() == i.lower()
    ]

    # Perhaps the input is a substring of a valid option?
    is_it_a_substring = [i for i in valid_options if val.lower() in i.lower()]

    # Use difflib package to make a guess for what the input might have been
    # For example, if they input "statistikal" instead of "Statistical", difflib will find "Statistical"
    # Change the cutoff to increase/decrease the flexibility of the function
    maybe_difflib_can_find_something = difflib.get_close_matches(
        val, valid_options, cutoff=0.59
    )

    if len(is_it_just_capitalized_wrong) > 0:
        closest_options = is_it_just_capitalized_wrong

    elif len(is_it_a_substring) > 0:
        closest_options = is_it_a_substring

    elif len(maybe_difflib_can_find_something) > 0:
        closest_options = maybe_difflib_can_find_something

    else:
        closest_options = None

    return closest_options


def _check_if_good_input(d, cat_df):
    """Check if inputs are valid and makes a "guess" using cat_df if the input is not valid

    Parameters
    ----------
    d: dict
        Dictionary of str: list
        The keys should correspond to valid column names in cat_df
        THE ITEMS NEED TO BE LISTS, even if its just a single length list
        i.e {"scenario": ["Historical Climate"]}
    cat_df: pd.DataFrame
        User-friendly catalog

    Returns
    -------
    d: dict
        Cleaned up dictionary

    """
    # Check that inputs are valid, make guess if not valid
    for key, val in zip(
        d.keys(), d.values()
    ):  # Loop through each key, value pair in the dictionary
        # Use the catalog to find the valid values in the list
        valid_options = np.unique(cat_df[key].values)
        if val in [
            [None],
            None,
        ]:  # If the user didn't input anything for that key, set the values to all the valid options
            d[key] = valid_options
            continue  # Don't finish the loop
        # If the input value is not in the valid options, see if you can help the user out
        key_updated = []
        for val_i in val:

            # This catches any common bad inputs for resolution: i.e. "3KM" or "3km" instead of "3 km"
            if key == "resolution":
                try:
                    good_resolution_input = val_i.lower().split("km")[0] + " km"
                    if good_resolution_input in valid_options:
                        print(
                            "Input " + key + "='" + val_i + "' is not a valid option."
                        )
                        print(
                            "Outputting data for "
                            + key
                            + "='"
                            + good_resolution_input
                            + "'\n"
                        )
                        key_updated.append(good_resolution_input)
                        continue
                except:
                    pass

            if val_i not in valid_options:

                print("Input " + key + "='" + val_i + "' is not a valid option.")

                closest_options = _get_closest_options(val_i, valid_options)

                # Sad! No closest options found. Just set the key to all valid options
                if closest_options is None:
                    print("Valid options: " + ", ".join(valid_options))
                    raise ValueError("Bad input")

                # Just one option in the list
                elif len(closest_options) == 1:
                    print("Closest option: '" + closest_options[0] + "'")

                elif len(closest_options) > 1:
                    print("Closest options: \n- " + "\n- ".join(closest_options))

                # Set key to closest option
                print("Outputting data for " + key + "='" + closest_options[0] + "'\n")
                key_updated.append(closest_options[0])
            else:
                key_updated.append(val_i)
        d[key] = key_updated
    return d


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
    scenario: str or list, optional
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
    var_df = data_interface.variable_descriptions
    catalog = data_interface.data_catalog
    cat_df = _get_user_friendly_catalog(
        intake_catalog=catalog, variable_descriptions=var_df
    )

    # Raise error for bad input from user
    for user_input in [variable, downscaling_method, resolution, timescale]:
        if (user_input is not None) and (type(user_input) != str):
            print("Function arguments require a single string value for your inputs")
            return None

    def _list(x):
        """Convert x to a list if its not a list"""
        if type(x) == list:
            return x
        elif type(x) != list:
            return [x]

    d = {
        "variable": _list(variable),
        "timescale": _list(timescale),
        "downscaling_method": _list(downscaling_method),
        "scenario": _list(scenario),
        "resolution": _list(resolution),
    }

    d = _check_if_good_input(d, cat_df)

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
    ----------
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
    Contrasts with selections.retrieve(), which retrieves data from the user inputs in climakitae's selections GUI.

    Parameters
    ----------
    variable: str
    downscaling_method: str
    resolution: str
    timescale: str
    scenario: str or list of str
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
    -------
    data: xr.DataArray
    """

    # Get intake catalog and variable descriptions from DataInterface object
    data_interface = DataInterface()
    var_df = data_interface.variable_descriptions
    catalog = data_interface.data_catalog
    cat_df = _get_user_friendly_catalog(
        intake_catalog=catalog, variable_descriptions=var_df
    )

    # Raise error for bad input from user
    for user_input in [variable, downscaling_method, resolution, timescale]:
        if (user_input is not None) and (type(user_input) not in [str, list]):
            print("Function arguments require a single string value for your inputs")
            return None

    d = {
        "variable": variable if type(variable) == list else [variable],
        "timescale": timescale if type(timescale) == list else [timescale],
        "downscaling_method": (
            downscaling_method
            if type(downscaling_method) == list
            else [downscaling_method]
        ),
        "scenario": scenario if type(scenario) == list else [scenario],
        "resolution": resolution if type(resolution) == list else [resolution],
    }

    d = _check_if_good_input(d, cat_df)

    # Convert list items back to single str to match formatting in data
    # Except for scenario which requires a list
    d["variable"] = d["variable"][0]
    d["timescale"] = d["timescale"][0]
    d["downscaling_method"] = d["downscaling_method"][0]
    d["resolution"] = d["resolution"][0]

    # We need a list for cached area
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
    var_df_query = var_df[
        (var_df["display_name"] == d["variable"])
        & (var_df["downscaling_method"] == d["downscaling_method"])
    ]

    # Timescale in table needs to be handled differently
    # This is because the monthly variables are derived from daily variables, so they are listed in the table as "daily, monthly"
    # Hourly variables may be different
    # Querying the data needs special handling due to the layout of the csv file
    var_df_query = var_df_query[var_df_query["timescale"].str.contains(d["timescale"])]

    if units is None:
        units = var_df_query["unit"].item()

    # Create selections object
    selections = DataParameters()

    # Defaults
    selections.area_average = area_average
    selections.area_subset = area_subset
    selections.cached_area = cached_area

    # Function not currently flexible enough to allow for appending historical
    # Need to add in error control
    selections.append_historical = False

    # User selections
    selections.downscaling_method = d["downscaling_method"]
    selections.variable = d["variable"]
    selections.resolution = d["resolution"]
    selections.timescale = d["timescale"]
    selections.units = units

    if "Historical Reconstruction" in d["scenario"]:
        selections.scenario_historical = ["Historical Reconstruction"]
        selections.scenario_ssp = []
        if len(d["scenario"]) != 1:
            print(
                "WARNING: Historical Reconstruction data cannot be retrieved in the same data object as other scenario options. Only returning Historical Reconstruction data."
            )

    else:
        if "Historical Climate" in d["scenario"]:
            selections.scenario_historical = ["Historical Climate"]
        selections.scenario_ssp = [x for x in d["scenario"] if "Historical" not in x]

    # Retrieve data
    data = selections.retrieve()
    return data
