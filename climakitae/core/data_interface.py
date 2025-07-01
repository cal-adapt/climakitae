"""
This module provides the core data interface to access climate data. It contains
several key components:

1. `VariableDescriptions`: A singleton class to load and provide access to available
climate variables.
2. `DataInterface`: A singleton class that manages connections to the data catalog,
boundary data, and stations.
3. `DataParameters`: A parameterized class that handles data selection, filtering, and
retrieval.

The module also includes several utility functions to:
- Get available data options and subsetting options
- Handle spatial subsetting by different boundaries (states, counties, watersheds, etc.)
- Retrieve data with simplified parameter specification
- Validate user inputs and provide helpful error messages
- Convert between different naming conventions in the catalog

This interface serves as the foundation for both programmatic access to climate data and
the interactive GUI selection interface.
"""

import difflib
import warnings
from typing import Iterable, List, Union

import geopandas as gpd
import intake
import intake_esm
import numpy as np
import pandas as pd
import param
import xarray as xr
from shapely.geometry import box

from climakitae.core.boundaries import Boundaries
from climakitae.core.constants import SSPS, WARMING_LEVELS
from climakitae.core.data_load import read_catalog_from_select
from climakitae.core.paths import (
    boundary_catalog_url,
    data_catalog_url,
    gwl_1850_1900_file,
    stations_csv_path,
    variable_descriptions_csv_path,
)
from climakitae.util.unit_conversions import get_unit_conversion_options
from climakitae.util.utils import (
    downscaling_method_as_list,
    downscaling_method_to_activity_id,
    read_csv_file,
    resolution_to_gridlabel,
    scenario_to_experiment_id,
    timescale_to_table_id,
)
from climakitae.util.warming_levels import create_ae_warming_trajectories

# Warnings raised by function get_subsetting_options, not sure why but they are silenced here
pd.options.mode.chained_assignment = None  # default='warn'

# Remove param's parameter descriptions from docstring because
# ANSI escape sequences in them complicate their rendering
param.parameterized.docstring_describe_params = False
# Docstring signatures are also hard to read and therefore removed
param.parameterized.docstring_signature = False


def _get_user_options(
    data_catalog: intake_esm.source.ESMDataSource,
    downscaling_method: str,
    timescale: str,
    resolution: str,
) -> tuple[list[str], list[str], list[str]]:
    """
    Using the data catalog, get a list of appropriate scenario and simulation options
    given a user's selections for downscaling method, timescale, and resolution.
    Unique variable ids for user selections are returned, then limited further in
    subsequent steps.

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
    match downscaling_method:
        case "Dynamical+Statistical":  # If both are selected
            loca_scenarios = cat_subset.search(
                activity_id="LOCA2"
            ).df.experiment_id.unique()  # LOCA unique member_ids
            wrf_scenarios = cat_subset.search(
                activity_id="WRF"
            ).df.experiment_id.unique()  # WRF unique member_ids
            overlapping_scenarios = list(set(loca_scenarios) & set(wrf_scenarios))
            cat_subset = cat_subset.search(experiment_id=overlapping_scenarios)
        case "Statistical":
            cat_subset = cat_subset.search(activity_id="LOCA2")

    # Get scenario options
    scenario_options = list(cat_subset.df["experiment_id"].unique())

    # Get all unique simulation options from catalog selection
    try:
        simulation_options = list(cat_subset.df["source_id"].unique())

        # Remove ensemble means
        if "ensmean" in simulation_options:
            simulation_options.remove("ensmean")
    except KeyError:
        simulation_options = []

    # Get variable options
    unique_variable_ids = list(cat_subset.df["variable_id"].unique())

    return scenario_options, simulation_options, unique_variable_ids


def _get_variable_options_df(
    variable_descriptions: pd.DataFrame,
    unique_variable_ids: list[str],
    downscaling_method: str,
    timescale: str,
    enable_hidden_vars: bool = False,
) -> pd.DataFrame:
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
    enable_hidden_vars: boolean, default to False
        Return all variables, including the ones in which "show" is set to False?

    Returns
    -------
    pd.DataFrame
        Subset of var_config for input downscaling_method and timescale
    """

    # Based on logic in the code and the name of the variable this needs to be the
    # opposite of the variable named enable_hidden_vars
    hide_hidden_vars = not enable_hidden_vars

    # Catalog options and derived options together
    derived_variables = list(
        variable_descriptions[
            variable_descriptions["variable_id"].str.contains("_derived")
        ]["variable_id"]
    )
    var_options_plus_derived = unique_variable_ids + derived_variables

    # Subset dataframe
    variable_options_df = variable_descriptions[
        (variable_descriptions["show"] == hide_hidden_vars)
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


def _get_var_ids(
    variable_descriptions: pd.DataFrame,
    variable: str,
    downscaling_method: str,
    timescale: str,
) -> list[str]:
    """
    Get variable ids that match the selected variable, timescale, and downscaling
    method. Required to account for the fact that LOCA, WRF, and various timescales use
    different variable id values. Used to retrieve the correct variables from the
    catalog in the backend.

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
    stations_gdf: gpd.GeoDataFrame,
    area_subset: str,
    cached_area: str,
    latitude: tuple[float, float],
    longitude: tuple[float, float],
    _geographies: Boundaries,
    _geography_choose: dict,
) -> list[str]:
    """Wrapper function that gets the string names of any overlapping weather stations

    Parameters
    ----------
    stations_gdf: gpd.GeoDataFrame
        geopandas GeoDataFrame of station locations
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


def _get_overlapping_stations(
    stations: gpd.GeoDataFrame, polygon: gpd.GeoDataFrame
) -> gpd.GeoDataFrame:
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
    area_subset: str,
    cached_area: str,
    latitude: tuple[float, float],
    longitude: tuple[float, float],
    _geographies: Boundaries,
    _geography_choose: dict,
) -> gpd.GeoDataFrame:
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

    df_ae = gpd.GeoDataFrame()

    def _get_subarea_from_shape_index(
        boundary_dataset: Boundaries, shape_indices: list
    ) -> gpd.GeoDataFrame:
        return boundary_dataset.loc[shape_indices]

    match area_subset:
        case "lat/lon":
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
        case _ if area_subset != "none":
            # `if-condition` added for catching errors with delays in rendering cached area.
            if cached_area is None:
                shape_indices = [0]
            else:
                # Filter for indices that are selected in `Location selection` dropdown
                shape_indices = list(
                    {
                        key: _geography_choose[area_subset][key] for key in cached_area
                    }.values()
                )

            match area_subset:
                case "states":
                    df_ae = _get_subarea_from_shape_index(
                        _geographies._us_states, shape_indices
                    )
                case "CA counties":
                    df_ae = _get_subarea_from_shape_index(
                        _geographies._ca_counties, shape_indices
                    )
                case "CA watersheds":
                    df_ae = _get_subarea_from_shape_index(
                        _geographies._ca_watersheds, shape_indices
                    )
                case "CA Electric Load Serving Entities (IOU & POU)":
                    df_ae = _get_subarea_from_shape_index(
                        _geographies._ca_utilities, shape_indices
                    )
                case "CA Electricity Demand Forecast Zones":
                    df_ae = _get_subarea_from_shape_index(
                        _geographies._ca_forecast_zones, shape_indices
                    )
                case "CA Electric Balancing Authority Areas":
                    df_ae = _get_subarea_from_shape_index(
                        _geographies._ca_electric_balancing_areas, shape_indices
                    )
                case _:
                    raise ValueError("area_subset not set correctly")
        case _:  # If no subsetting, make the geometry a big box to include all stations
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
    warming_level_times: pd.DataFrame
        table of when each simulation/scenario reaches each warming level
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
        self._warming_level_times = read_csv_file(
            gwl_1850_1900_file, index_col=[0, 1, 2]
        )

        # Get geography boundaries
        self._boundary_catalog = intake.open_catalog(boundary_catalog_url)
        self._geographies = Boundaries(self.boundary_catalog)

        self._geographies.load()

    @property
    def variable_descriptions(self):
        """Get the variable descriptions dataframe"""
        return self._variable_descriptions

    @property
    def stations(self):
        """Get the stations dataframe"""
        return self._stations

    @property
    def stations_gdf(self):
        """Get the stations geopandas dataframe"""
        return self._stations_gdf

    @property
    def data_catalog(self):
        """Get the data catalog"""
        return self._data_catalog

    @property
    def warming_level_times(self):
        """Get the warming level times dataframe"""
        return self._warming_level_times

    @property
    def boundary_catalog(self):
        """Get the boundary catalog"""
        return self._boundary_catalog

    @property
    def geographies(self):
        """Get the geographies object"""
        return self._geographies


class DataParameters(param.Parameterized):
    """Python param object to hold data parameters for use in panel GUI.
    Call DataParameters when you want to select and retrieve data from the
    climakitae data catalog without using the ckg.Select GUI. ckg.Select uses
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
        whether to choose WRF or LOCA2 data or both ("Dynamical", "Statistical",
        "Dynamical+Statistical")
    data_type: str
        whether to choose gridded or station based data ("Gridded", "Stations")
    stations: list or strs
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
    enable_hidden_vars: boolean
        enable selection of variables that are hidden from the GUI?
    extended_description: str
        extended description of the data variable
    variable_id: list of strs
        list of variable ids that match the variable (WRF and LOCA2 can have different
        codes for same type of variable)
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
    _warming_level_times: pd.DataFrame
        shorthand alias to DataInterface.warming_level_times
    colormap: str
        default colormap to render the currently selected data
    scenario_options: list of strs
        list of available scenarios (historical and ssp) for selection
    variable_options_df: pd.DataFrame
        filtered variable descriptions for the downscaling_method and timescale
    warming_level: array
        global warming level(s)
    warming_level_window: integer
        years around Global Warming Level (+/-) \n (e.g. 15 means a 30yr window)
    approach: str, "Warming Level" or "Time"
        how do you want the data to be retrieved?
    warming_level_months: array
        months of year to use for computing warming levels
        default to entire calendar year: 1,2,3,4,5,6,7,8,9,10,11,12
    all_touched: boolean
        spatial subset option for within or touching selection
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
    data_type = param.Selector(default="Gridded", objects=["Gridded", "Stations"])
    stations = param.ListSelector(objects=dict())
    _station_data_info = param.String(
        default="", doc="Information about the bias correction process and resolution"
    )
    enable_hidden_vars = param.Boolean(False)
    approach = param.Selector(default="Time", objects=["Time", "Warming Level"])

    # Empty params, initialized in __init__
    scenario_ssp = param.ListSelector(objects=dict())
    simulation = param.ListSelector(objects=dict())
    variable = param.Selector(objects=dict())
    units = param.Selector(objects=dict())
    extended_description = param.Selector(objects=dict())
    variable_id = param.ListSelector(objects=dict())
    warming_level = param.ListSelector(objects=dict())

    # Temporal range of each dataset
    historical_climate_range_wrf = (1980, 2015)
    historical_climate_range_loca = (1950, 2015)
    historical_climate_range_wrf_and_loca = (1981, 2015)
    historical_reconstruction_range = (1950, 2022)
    ssp_range = (2015, 2100)

    # Warming level options
    wl_options = WARMING_LEVELS
    wl_time_option = ["n/a"]
    warming_level = param.List(default=[1.0], item_type=Union[float, str])
    warming_level_window = param.Integer(
        default=15,
        bounds=(5, 25),
        doc="Years around Global Warming Level (+/-) \n (e.g. 15 means a 30yr window)",
    )
    warming_level_months = param.ListSelector(
        default=list(np.arange(1, 13)),  # All 12 months of the year
        objects=list(np.arange(1, 13)),  # All 12 months of the year
    )

    all_touched = param.Boolean(False)

    # User warnings
    _info_about_station_data = "This method retrieves gridded model data that is bias-corrected using historical weather station data at that point. This process can start from any model grid-spacing."
    _data_warning = param.String(
        default="", doc="Warning if user has made a bad selection"
    )

    def __init__(self, **params):
        # Set default values
        super().__init__(**params)

        self.data_interface = DataInterface()

        # Data Catalog
        self._data_catalog = self.data_interface.data_catalog

        # Warming Levels Table
        self._warming_level_times = self.data_interface.warming_level_times

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

        self.all_touched = False

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
            enable_hidden_vars=self.enable_hidden_vars,
        )

        # Show derived index option?
        indices = True
        if self.data_type == "Stations":
            indices = False
        if self.downscaling_method != "Dynamical":
            indices = False
        if self.timescale == "monthly":
            indices = False
        if not indices:
            self.param["variable_type"].objects = ["Variable"]
            self.variable_type = "Variable"
        else:
            self.param["variable_type"].objects = ["Variable", "Derived Index"]

        # Set scenario param
        scenario_ssp_options = [
            scenario_to_experiment_id(scen, reverse=True)
            for scen in self.scenario_options
            if "ssp" in scen
        ]
        for scenario_i in SSPS:
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

    @param.depends("approach", watch=True)
    def _update_scenarios_approach(self):
        """
        Update the scenario options shown based on retrieval method.
        If warming level is selected, there should be no scenario options shown.
        If time-based is selected, there should be no warming levels options shown.
        """
        match self.approach:
            case "Warming Level":
                self.warming_level = [2.0]

                self.param["scenario_ssp"].objects = ["n/a"]
                self.scenario_ssp = ["n/a"]

                self.param["scenario_historical"].objects = ["n/a"]
                self.scenario_historical = ["n/a"]

            case "Time":
                self.warming_level = ["n/a"]

                self.param["scenario_ssp"].objects = SSPS
                self.scenario_ssp = []

                self.param["scenario_historical"].objects = [
                    "Historical Climate",
                    "Historical Reconstruction",
                ]
                self.scenario_historical = ["Historical Climate"]
            case _:
                raise ValueError(
                    'approach needs to be either "Warming Level" or "Time"'
                )

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
        match self.data_type:
            case "Stations":
                self.param["area_average"].objects = ["n/a"]
                self.area_average = "n/a"
            case "Gridded":
                self.param["area_average"].objects = ["Yes", "No"]
                self.area_average = "No"
            case _:
                raise ValueError('data_type needs to either "Stations" or "Gridded"')

    @param.depends(
        "downscaling_method",
        "data_type",
        "variable_type",
        "approach",
        watch=True,
    )
    def _update_data_type_option_for_some_selections(self):
        """
        Station data selection not permitted for the following selections:
        - If statistical downscaling is selected, remove option for station data because
        we don't have the 2m temp variable for LOCA.
        - No station data (yet) for warming levels-- can explore adding in the future.
        Order of operations for station based retrieval using a warming levels approach
        should be: quantile mapping first to adjust to observations, then retrieve the
        sliced data.
        - No station data (yet) for derived indices-- can explore adding in the future

        """
        if (
            "Statistical" in self.downscaling_method
            or self.variable_type == "Derived Index"
            or self.approach == "Warming Level"
        ):
            self.param["data_type"].objects = ["Gridded"]
            self.data_type = "Gridded"
        else:
            self.param["data_type"].objects = ["Gridded", "Stations"]
        if self.variable_type == "Derived Index":
            # Haven't built into the code to retrieve derive index for statistically downscaled data yet. Derived indices at the moment only work for hourly data.
            self.param["downscaling_method"].objects = ["Dynamical"]
            self.downscaling_method = "Dynamical"

            self.param["approach"].objects = ["Time", "Warming Level"]

        elif "Stations" in self.data_type:
            self.param["downscaling_method"].objects = ["Dynamical"]
            self.downscaling_method = "Dynamical"

            self.param["approach"].objects = ["Time"]
            self.approach = "Time"
        else:
            self.param["downscaling_method"].objects = [
                "Dynamical",
                "Statistical",
                "Dynamical+Statistical",
            ]
            self.param["approach"].objects = ["Time", "Warming Level"]

    @param.depends("data_type", "downscaling_method", watch=True)
    def _update_res_based_on_data_type_and_downscaling_method(self):
        """Update the grid resolution options based on the data selections."""
        if "Statistical" in self.downscaling_method:
            self.param["resolution"].objects = ["3 km"]
            self.resolution = "3 km"
        else:
            match self.data_type:
                case "Stations":
                    self.param["resolution"].objects = ["3 km", "9 km"]
                    if self.resolution == "45 km":
                        self.resolution = "3 km"
                case "Gridded":
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
        if self.data_type == "Stations":
            # Only air temp available for station data
            indices = False
        if self.downscaling_method != "Dynamical":
            # Currently we only have indices for WRF data
            indices = False
        if self.timescale == "monthly":
            indices = False
        if not indices:
            # Remove derived index as an option
            self.param["variable_type"].objects = ["Variable"]
            self.variable_type = "Variable"
        else:
            self.param["variable_type"].objects = ["Variable", "Derived Index"]

    @param.depends(
        "timescale",
        "resolution",
        "downscaling_method",
        "data_type",
        "variable",
        "variable_type",
        "enable_hidden_vars",
        watch=True,
    )
    def _update_user_options(self):
        """Update unique variable options"""

        # Station data is only available hourly
        match (self.data_type, self.downscaling_method):
            case ("Stations", _):
                self.param["timescale"].objects = ["hourly"]
                self.timescale = "hourly"
                self.param["variable_type"].objects = ["Variable"]
                self.variable_type = "Variable"
            case ("Gridded", "Statistical"):
                self.param["timescale"].objects = ["daily", "monthly"]
                if self.timescale == "hourly":
                    self.timescale = "daily"
            case ("Gridded", "Dynamical"):
                self.param["timescale"].objects = ["daily", "monthly", "hourly"]
            case ("Gridded", "Dynamical+Statistical"):
                # If both are selected, only show daily data
                # We do not have WRF on LOCA grid resampled to monthly
                self.param["timescale"].objects = ["daily"]
                self.timescale = "daily"
            case _:
                raise ValueError(
                    "data_type and downscaling_method combination not correct"
                )
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

        if self.data_type == "Stations":
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
                enable_hidden_vars=self.enable_hidden_vars,
            )

            # Filter for derived indices
            # Depends on user selection for variable_type
            match self.variable_type:
                case "Variable":
                    # Remove indices
                    self.variable_options_df = self.variable_options_df[
                        ~self.variable_options_df["variable_id"].str.contains("index")
                    ]
                case "Derived Index":
                    # Show only indices
                    self.variable_options_df = self.variable_options_df[
                        self.variable_options_df["variable_id"].str.contains("index")
                    ]
                case _:
                    raise ValueError(
                        'variable_type needs to be either "Variable" or "Derived Index"'
                    )
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
                match self.downscaling_method:
                    case "Statistical" | "Dynamical+Statistical":
                        self.param["cached_area"].objects = ["CA"]
                    case "Dynamical":
                        self.param["cached_area"].objects = [
                            "CA",
                            "NV",
                            "OR",
                            "UT",
                            "AZ",
                        ]
                    case _:
                        raise ValueError(
                            'downscaling_method needs to be "Statistical", "Dynamical", or "Dynamical+Statistical"'
                        )
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

    @param.depends(
        "resolution", "downscaling_method", "data_type", "approach", watch=True
    )
    def _update_scenarios(self):
        """
        Update scenario options. Raise data warning if a bad selection is made.
        """

        if self.approach == "Time":
            # Set incoming scenario_historical
            _scenario_historical = self.scenario_historical

            # Get scenario options in catalog format
            scenario_ssp_options = [
                scenario_to_experiment_id(scen, reverse=True)
                for scen in self.scenario_options
                if "ssp" in scen
            ]
            for scenario_i in SSPS:
                if scenario_i in scenario_ssp_options:  # Reorder list
                    scenario_ssp_options.remove(scenario_i)  # Remove item
                    scenario_ssp_options.append(scenario_i)  # Add to back of list
            self.param["scenario_ssp"].objects = scenario_ssp_options
            self.scenario_ssp = [
                x for x in self.scenario_ssp if x in scenario_ssp_options
            ]

            historical_scenarios = ["historical", "reanalysis"]
            scenario_historical_options = [
                scenario_to_experiment_id(scen, reverse=True)
                for scen in self.scenario_options
                if scen in historical_scenarios
            ]
            self.param["scenario_historical"].objects = scenario_historical_options

            # check if input historical scenarios match new available scenarios
            # if no reanalysis scenario then return False
            def _check_inputs(a: list[str], b: list[str]) -> bool:
                """Check if any element in list a also exists in list b."""
                if len(b) < 2:
                    return False

                # Use set intersection for efficient membership checking
                return bool(set(a) & set(b))

            # check if new selection has the historical scenario options and if not select the first new option
            if _check_inputs(_scenario_historical, scenario_historical_options):
                self.scenario_historical = _scenario_historical
            else:
                self.scenario_historical = [scenario_historical_options[0]]

        else:
            pass

    @param.depends(
        "approach",
        "scenario_ssp",
        "scenario_historical",
        "downscaling_method",
        "time_slice",
        watch=True,
    )
    def _update_data_warning(self):
        """Update warning raised to user based on their data selections.
        No warming shown if approach is set to warming level.
        """
        data_warning = ""
        bad_time_slice_warning = """You've selected a time slice that is outside the temporal range 
        of the selected data."""

        if self.approach == "Warming Level":
            data_warning = ""

        else:
            # Set time range of historical data
            match self.downscaling_method:
                case "Dynamical":
                    historical_climate_range = self.historical_climate_range_wrf
                case "Statistical":
                    historical_climate_range = self.historical_climate_range_loca
                case "Dynamical+Statistical":
                    historical_climate_range = (
                        self.historical_climate_range_wrf_and_loca
                    )
                case _:
                    raise ValueError(
                        'downscaling_method needs to be "Statistical", "Dynamical", or "Dynamical+Statistical"'
                    )

            # Warning based on data scenario selections
            if (  # Warn user that they cannot have SSP data and ERA5-WRF data
                True in ["SSP" in one for one in self.scenario_ssp]
            ) and ("Historical Reconstruction" in self.scenario_historical):
                data_warning = """Historical Reconstruction data is not available with SSP data.
                Try using the Historical Climate data instead."""

            elif (
                (  # Warn user if no data is selected
                    all("SSP" not in one for one in self.scenario_ssp)
                )
                and (
                    not True
                    in ["Historical" in one for one in self.scenario_historical]
                )
                and (self.scenario_ssp != ["n/a"])
                and (self.scenario_historical != ["n/a"])
            ):
                data_warning = "Please select as least one dataset."

            elif (  # If both historical options are selected, warn user the data will be cut
                "Historical Reconstruction" in self.scenario_historical
            ) and (
                "Historical Climate" in self.scenario_historical
            ):
                data_warning = """The timescale of Historical Reconstruction data will be cut 
                to match that of the Historical Climate data if both are retrieved."""

            # Warnings based on time slice selections
            if (all("SSP" not in one for one in self.scenario_ssp)) and (
                "Historical Climate" in self.scenario_historical
            ):
                if (self.time_slice[0] < historical_climate_range[0]) or (
                    self.time_slice[1] > historical_climate_range[1]
                ):
                    data_warning = bad_time_slice_warning
            elif True in ["SSP" in one for one in self.scenario_ssp]:
                if not True in [
                    "Historical" in one for one in self.scenario_historical
                ]:
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
        match self.downscaling_method:
            case "Dynamical":
                historical_climate_range = self.historical_climate_range_wrf
            case "Statistical":
                historical_climate_range = self.historical_climate_range_loca
            case "Dynamical+Statistical":
                historical_climate_range = self.historical_climate_range_wrf_and_loca
            case _:
                raise ValueError(
                    'downscaling_method needs to be "Statistical", "Dynamical", or "Dynamical+Statistical"'
                )
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
        match self.data_type:
            case "Gridded":
                self._station_data_info = ""
            case "Stations":
                self._station_data_info = self._info_about_station_data
            case _:
                raise ValueError('data_type needs to either "Stations" or "Gridded"')

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
        match self.data_type:
            case "Stations":
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
                    self.param["stations"].objects = [notice]
                    self.stations = [notice]
                else:
                    self.param["stations"].objects = overlapping_stations
                    self.stations = overlapping_stations
            case "Gridded":
                notice = "Set data type to 'Station' to see options"
                self.param["stations"].objects = [notice]
                self.stations = [notice]
            case _:
                raise ValueError('data_type needs to either "Stations" or "Gridded"')

    def retrieve(
        self, config: str = None, merge: bool = True
    ) -> Union[xr.DataArray, xr.Dataset, List[xr.DataArray]]:
        """Retrieve data from catalog

        By default, DataParameters determines the data retrieved.
        Grabs the data from the AWS S3 bucket, returns lazily loaded dask array.
        User-facing function that provides a wrapper for read_catalog_from_select.

        Returns
        -------
        data_return : xr.DataArray | xr.Dataset | list of xr.DataArray
            DataArray or Dataset object
        """

        def _warn_of_large_file_size(da: xr.DataArray):
            """Warn user if the data array is large"""
            nbytes = da.nbytes
            match nbytes:
                case nbytes if nbytes >= int(1e9) and nbytes < int(5e9):
                    print(
                        "!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!\n"
                        "! Returned data array is large. Operations could take up to 5x longer than 1GB of data!\n"
                        "!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!\n"
                    )
                case nbytes if nbytes >= int(5e9) and nbytes < int(1e10):
                    print(
                        "!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!\n"
                        "!! Returned data array is very large. Operations could take up to 8x longer than 1GB of data !!\n"
                        "!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!\n"
                    )
                case nbytes if nbytes >= int(1e10):
                    print(
                        "!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!\n"
                        "!!! Returned data array is huge. Operations could take 10x to infinity longer than 1GB of data !!!\n"
                        "!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!\n"
                    )

        def _warn_of_empty_data(self):
            if self.approach == "Warming Level" and (len(self.warming_level) > 1):
                print(
                    "WARNING FOR WARMING LEVELS APPROACH\n-----------------------------------\nThere may be NaNs in your data for certain simulation/warming level combinations if the warming level is not reached for that particular simulation before the year 2100. \n\nThis does not mean you have missing data, but rather a feature of how the data is combined in retrieval to return a single data object. \n\nIf you want to remove these empty simulations, it is recommended to first subset the data object by each individual warming level and then dropping NaN values."
                )
            elif (self.approach == "Time") and (len(self.scenario_ssp) > 1):
                print(
                    "WARNING\n-------\nYou have retrieved data for more than one SSP, but not all ensemble members for each GCM are available for all SSPs.\n\nAs a result, some scenario and simulation combinations may contain NaN values.\n\nIf you want to remove these empty simulations, it is recommended to first subset the data object by each individual scenario and then dropping NaN values."
                )

        data_return = read_catalog_from_select(self)

        if isinstance(data_return, list):
            for l in data_return:
                _warn_of_large_file_size(l)
        else:
            _warn_of_large_file_size(data_return)

        # Warn about empty simulations for certain selections
        _warn_of_empty_data(self)

        return data_return


## -------------- Data access without GUI -------------------


def _get_user_friendly_catalog(
    intake_catalog: intake_esm.source.ESMDataSource, variable_descriptions: pd.DataFrame
) -> pd.DataFrame:
    """Get a user-friendly version of the intake data catalog using climakitae naming conventions

    Parameters
    ----------
    intake_catalog: intake_esm.source.ESMDataSource
    variable_descriptions: pd.DataFrame

    Returns
    -------
    cat_df_cleaned: intake_esm.source.ESMDataSource
    """

    def _expand(row: pd.DataFrame, cat_df: pd.DataFrame) -> pd.DataFrame:
        """Used for adding climakitae derived variables into catalog options
        Expand the row for each individual derived variable to include the appropriate resolution and scenario options from it's variable dependency.
        For example, the fosberg fire index is dependent on temperature.
        We don't store info in the variable_descriptions csv on the options for scenario and resolution for derived variables
        So, we need to pull those options from the valid options from the catalog variables that the derived variables are built from

        Parameters
        ----------
        row: pd.DataFrame
            Row of derived variable
        cat_df: pd.DataFrame
            Catalog dataframe

        Returns
        -------
        dependency_options: pd.DataFrame
            Dataframe with the options for the derived variable
        """
        # Just get the first dependency
        # Ideally it should look at all the dependent variables but it's a lot of messy code and I'm not sure if it actually matters
        first_dependency = row["dependencies"].split(", ")[0]

        # Get the dependency info from the catalog
        dependency_options = cat_df[
            (cat_df["variable_id"] == first_dependency)
            & (cat_df["downscaling_method"] == row["downscaling_method"])
            & (cat_df["timescale"] == row["timescale"])
        ].reset_index(drop=True)

        # Basically, replace the info from the dependent variable with the derived variable
        # We assume they are the same
        dependency_options["variable"] = row.variable  # Add derived var as variable
        return dependency_options

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

    # Remove duplicate rows
    # Duplicates occur due to the many unique member_ids
    cat_df_cleaned = cat_df_cleaned.drop_duplicates(ignore_index=True)

    # Get the derived variables
    derived_vars = _get_and_reformat_derived_variables(variable_descriptions)

    # For each row (with unique variable, timescale, and downscaling method), get the options for scenario and resolution
    # This will "expand" the row because there will be multiple combinations possible (likely)
    derived_vars_all = pd.DataFrame()
    for _, row in derived_vars.iterrows():  # Loop through each row
        derived_vars_all = pd.concat(
            [derived_vars_all, _expand(row, cat_df_cleaned)], ignore_index=True
        )

    # Combine derived and catalog variables into one!
    options_all = pd.concat([cat_df_cleaned, derived_vars_all], ignore_index=True)
    options_all.pop("variable_id")  # Remove variable id column

    return options_all


def _get_var_name_from_table(
    variable_id: str, downscaling_method: str, timescale: str, var_df: pd.DataFrame
) -> str:
    """Get the variable name corresponding to its ID, downscaling method, and timescale
    Enables the _get_user_friendly_catalog function to get the name of a variable corresponding to a set of user inputs
    i.e we have several different precip variables, corresponding to different downscaling methods (WRF vs. LOCA)

    Parameters
    ----------
    variable_id : str
    downscaling_method : str
    timescale : str
    var_df : pd.DataFrame
        Variable descriptions table

    Returns
    -------
    var_name : str
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


def _get_closest_options(
    val: str, valid_options: list[str], cutoff: float = 0.59
) -> list[str] | None:
    """If the user inputs a bad option, find the closest option from a list of valid options

    Parameters
    ----------
    val: str
        User input
    valid_options: list
        Valid options for that key from the catalog
    cutoff: a float in the range [0, 1]
        See difflib.get_close_matches
        Possibilities that don't score at least that similar to word are ignored.

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
        val, valid_options, cutoff=cutoff
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


def _check_if_good_input(d: dict, cat_df: pd.DataFrame) -> dict:
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
                except AttributeError:
                    pass

            if val_i not in valid_options:
                print("Input " + key + "='" + val_i + "' is not a valid option.")

                closest_options = _get_closest_options(val_i, valid_options)

                # Sad! No closest options found. Just set the key to all valid options
                match closest_options:
                    case None:
                        print("Valid options: \n- ", end="")
                        print("\n- ".join(valid_options))
                        raise ValueError("Bad input")

                    # Just one option in the list
                    case closest_options if len(closest_options) == 1:
                        print("Closest option: '" + closest_options[0] + "'")

                    case closest_options if len(closest_options) > 1:
                        print("Closest options: \n- " + "\n- ".join(closest_options))

                # Set key to closest option
                print("Outputting data for " + key + "='" + closest_options[0] + "'\n")
                key_updated.append(closest_options[0])
            else:
                key_updated.append(val_i)
        d[key] = key_updated
    return d


def get_data_options(
    variable: str = None,
    downscaling_method: str = None,
    resolution: str = None,
    timescale: str = None,
    scenario: Union[str, list[str]] = None,
    tidy: bool = True,
) -> pd.DataFrame:
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
            print(
                _format_error_print_message(
                    "Function arguments require a single string value for your inputs"
                )
            )
            return None

    def _list(x: Union[str, list]) -> list:
        """Convert x to a list if its not a list"""
        return x if isinstance(x, list) else [x]

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
        print(
            _format_error_print_message(
                "No data found for your input values. Please modify your data request."
            )
        )
        return None

    if tidy:
        cat_subset = cat_subset.set_index(
            ["downscaling_method", "scenario", "timescale"]
        )
    return cat_subset


def get_subsetting_options(area_subset: str = "all") -> pd.DataFrame:
    """Get all geometry options for spatial subsetting.
    Options match those in selections GUI

    Parameters
    ----------
    area_subset: str
        One of "all", "states", "CA counties", "CA Electricity Demand Forecast Zones", "CA watersheds", "CA Electric Balancing Authority Areas", "CA Electric Load Serving Entities (IOU & POU)", "Stations"
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
        "Stations": data_interface._stations_gdf.sort_values("station").rename(
            columns={"station": "NAME"}
        )[["NAME", "geometry"]],
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
        if name == "Stations":  # This logic doesn't apply to weather stations
            pass  # do nothing
        else:  # Limit options
            df = df[df["NAME"].isin(list(boundary_dict[name].keys()))]
        df_dict[name] = df  # Replace the dictionary with the new, reduced dictionary

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


def _format_error_print_message(error_message: str) -> str:
    """Format error message using the same format"""
    return f"ERROR: {error_message} \nReturning None"


def get_data(
    variable: str,
    resolution: str,
    timescale: str,
    downscaling_method: str = "Dynamical",
    data_type: str = "Gridded",
    approach: str = "Time",
    scenario: str = None,
    units: str = None,
    warming_level: list[float] = None,
    area_subset: str = "none",
    latitude: tuple[float, float] = None,
    longitude: tuple[float, float] = None,
    cached_area: list[str] = None,
    area_average: str = None,
    time_slice: tuple = None,
    stations: list[str] = None,
    warming_level_window: int = None,
    warming_level_months: list[int] = None,
    all_touched=False,
) -> xr.DataArray:
    # Need to add error handing for bad variable input
    """Retrieve formatted data from the Analytics Engine data catalog using a simple function.
    Contrasts with DataParameters().retrieve(), which retrieves data from the user inputs in climakitaegui's selections GUI.

    Parameters
    ----------
    variable: str
        String name of climate variable
    resolution: str, one of ["3 km", "9 km", "45 km"]
        Resolution of data in kilometers
    timescale: str, one of ["hourly", "daily", "monthly"]
        Temporal frequency of dataset
    downscaling_method: str, one of ["Dynamical", "Statistical", "Dynamical+Statistical"], optional
        Downscaling method of the data:
        WRF ("Dynamical"), LOCA2 ("Statistical"), or both "Dynamical+Statistical"
        Default to "Dynamical"
    data_type: str, one of ["Gridded", "Stations"], optional
        Whether to choose gridded data or weather station data
        Default to "Gridded"
    approach: one of ["Time", "Warming Level"], optional
        Default to "Time"
    scenario: str or list of str, optional
        SSP scenario ["SSP 3-7.0", "SSP 2-4.5","SSP 5-8.5"] and/or historical data selection ["Historical Climate", "Historical Reconstruction"]
        If approach = "Time", you need to set a valid option
        If approach = "Warming Level", scenario is ignored
    units: str, optional
        Variable units.
        Defaults to native units of data
    area_subset: str, optional
        Area category: i.e "CA counties"
        Defaults to entire domain ("none")
    cached_area: list, optional
        Area: i.e "Alameda county"
        Defaults to entire domain (["entire domain"])
    area_average: one of ["Yes","No"], optional
        Take an average over spatial domain?
        Default to "No".
    latitude: None or tuple of float, optional
        Tuple of valid latitude bounds
        Default to entire domain
    longitude: None or tuple of float, optional
        Tuple of valid longitude bounds
        Default to entire domain
    time_slice: tuple, optional
        Time range for retrieved data
        Only valid for approach = "Time"
    stations: list of str, optional
        Which weather stations to retrieve data for
        Only valid for data_type = "Stations"
        Default to all stations
    warming_level: list of float, optional
        Must be one of the warming levels available in `clmakitae.core.constants`
        Only valid for approach = "Warming Level" and data_type = "Stations"
    warming_level_window: int in range (5,25), optional
        Years around Global Warming Level (+/-) \n (e.g. 15 means a 30yr window)
        Only valid for approach = "Warming Level" and data_type = "Stations"
    warming_level_months: list of int, optional
        Months of year for which to perform warming level computation
        Default to all months in a year: [1,2,3,4,5,6,7,8,9,10,11,12]
        For example, you may want to set warming_level_months=[12,1,2] to perform the analysis for the winter season.
        Only valid for approach = "Warming Level" and data_type = "Stations"
    all_touched: boolean
        spatial subset option for within or touching selection

    Returns
    -------
    data: xr.DataArray

    Notes
    -----
    Errors aren't raised by the function. Rather, an appropriate informative message is printed, and the function returns None. This is due to the fact that the AE Jupyter Hub raises a strange Pieces Mismatch Error for some bad inputs; instead, that error is ignored and a more informative error message is printed instead.

    """

    def _check_valid_input_station(
        stations: list[str], station_options_all: list[str]
    ) -> list[str]:
        """Check that the user input a valid value for station
        If invalid input, the function will "guess" a close-ish station using difflib
        See _get_closest_option function for more info
        If invalid input and no guesses found, the function will print an informative
        error message and raise a ValueError

        Parameters
        ----------
        stations: list of str
        station_options_all: list of string
            All the possible station options
            Can be retrieved from DataParameters()._stations_gdf.station.values

        Returns
        -------
        stations: list of str

        """
        station_options_all = sorted(
            station_options_all
        )  # sorted() puts the list in alphabetical order

        # Keep track of if error was raised and message was printed to user
        # If more than one station prints errors to the console, print a space between each station
        printed_warning = False

        for i, station_i in enumerate(stations):  # Go through all the stations
            # If the station is a valid option, don't do anything
            if station_i in station_options_all:
                continue

            if printed_warning:
                print(
                    "\n", end=""
                )  # Add a space between stations for better readability

            # If the station isn't a valid option...
            print("Input station='" + station_i + "' is not a valid option.")
            closest_options = _get_closest_options(
                station_i, station_options_all
            )  # See if theres any similar options

            # Sad! No closest options found. Just set the key to all valid options
            match closest_options:
                case None:
                    print("Valid options: \n- ", end="")
                    print("\n- ".join(station_options_all))
                    raise ValueError("Bad input")

                # Just one option in the list
                case closest_options if len(closest_options) == 1:
                    print("Closest option: '" + closest_options[0] + "'")

                case closest_options if len(closest_options) > 1:
                    print("Closest options: \n- " + "\n- ".join(closest_options))

            print("Outputting data for station='" + closest_options[0] + "'")
            stations[i] = closest_options[
                0
            ]  # Replace that value in the list with the best option :)

            printed_warning = True

        return stations

    # Internal functions
    def _error_handling_warming_level_inputs(
        wl: Union[list[float], list[int]],
        argument_name: str,
        downscaling_method: str,
        resolution: str,
    ):
        """
        Error handling for arguments: warming_level and warming_level_month
        Both require a list of either floats or ints
        argument_name is either "warming_level" or "warming_level_months" and is used to
        print an appropriate error message for bad input
        """
        # Find the WL bounds for LOCA and WRF
        loca, wrf = create_ae_warming_trajectories(resolution)
        loca_max = round(loca.max().max(), 2)
        wrf_max = round(wrf.max().max(), 2)

        match downscaling_method:
            case "Statistical":
                max_val = loca_max
            case "Dynamical":
                max_val = wrf_max
            case "Dynamical+Statistical":
                max_val = min(loca_max, wrf_max)
            case _:
                raise ValueError(
                    "Downscaling method be 'Statistical', 'Dynamical', or 'Dynamical+Statistical'"
                )

        if (wl is not None) and not isinstance(wl, list):
            if isinstance(wl, (float, int)):  # Convert float to a singleton list
                wl = [wl]
            if not isinstance(wl, list):
                raise ValueError(
                    f"""Function argument {argument_name} requires a float/int or list 
                    of floats/ints input. Your input: {type(wl)}"""
                )
        if isinstance(wl, list):
            for x in wl:
                if not isinstance(x, (float, int)):
                    raise ValueError(
                        f"Each item in '{argument_name}' must be a float or int. Got: {type(x)}"
                    )
                if argument_name == "warming_level":
                    if x < 0 or x > max_val:
                        raise ValueError(
                            f"{argument_name} value {x}. "
                            f"Allowed range for {downscaling_method}-downscaled data at {resolution} resolution is 0 to {max_val:.2f}."
                        )
        return wl

    def _error_handling_approach_inputs(
        approach: str, scenario: str, warming_level: list[float], time_slice: tuple
    ) -> tuple[str, str, list[float], tuple]:
        """Error handling for approach and scenario inputs"""
        _valid_options_approach = ["Time", "Warming Level"]
        if approach not in _valid_options_approach:
            # Maybe the user just capitalized it wrong
            # If so, fix it for them-- don't raise an error
            if approach.lower().title() in _valid_options_approach:
                approach = approach.lower().title()
            else:
                # An error will be raised later when you try to set selections
                pass

        # Print a warming if scenario is set but approach is Warming Level
        if approach == "Warming Level" and scenario not in [None, ["n/a"], "n/a"]:
            print(
                'WARNING: "scenario" argument will be ignored for warming levels approach'
            )
            scenario = None
        if approach == "Warming Level" and time_slice != None:
            print(
                'WARNING: "time_slice" argument will be ignored for warming levels approach'
            )
            time_slice = None

        if approach == "Time":
            warming_level = ["n/a"]

        return approach, scenario, warming_level, time_slice

    def _error_handling_location_settings(
        area_subset: list[str], cached_area: list[str]
    ) -> list[str]:
        """Maybe the user put an input for cached area but not for area subset
        We need to have the matching/correct area subset in order for selections.retrieve() to actually subset the data
        Here, we load in the geometry options to set area_subset to the correct value
        This also raises an appropriate error if the user has a bad input
        """
        if area_subset == "none" and cached_area != ["entire domain"]:
            geom_df = get_subsetting_options(area_subset="all").reset_index()
            area_subset_vals = geom_df[geom_df["cached_area"] == cached_area[0]][
                "area_subset"
            ].values
            if len(area_subset_vals) == 0:
                raise ValueError("Invalid input for argument 'cached_area'")
            else:
                area_subset = area_subset_vals[0]
        return area_subset

    def _get_scenario_ssp_scenario_historical(
        approach: str, scenario: str
    ) -> tuple[str, str]:
        """Get scenario_ssp, scenario_historical depending on user inputs"""
        match approach:
            case "Warming Level":
                scenario_ssp = ["n/a"]
                scenario_historical = ["n/a"]
            case "Time":
                if (
                    "Historical Reconstruction" in scenario
                ):  # Handling for Historical Reconstruction option
                    scenario_historical = [x for x in scenario if "Historical" in x]
                    scenario_ssp = []
                    if (
                        len(scenario) != 1
                    ):  # No SSP options for Historical Reconstruction data
                        print(
                            "WARNING: Historical Reconstruction data cannot be retrieved in the same data object as SSP scenario options. SSP data will not be retrieved."
                        )
                else:
                    scenario_ssp = [
                        x for x in scenario if "Historical" not in x
                    ]  # Add non-historical SSPs to scenario_ssp key
                    if "Historical Climate" in scenario:
                        scenario_historical = ["Historical Climate"]
                    else:
                        scenario_historical = []
            case _:
                scenario_ssp, scenario_historical = None, None
        return scenario_ssp, scenario_historical

    # default values set as lists are dangerous, so set them to None and then set to
    # default value later
    if cached_area is None:
        cached_area = ["entire domain"]
    # Get intake catalog and variable descriptions from DataInterface object
    data_interface = DataInterface()
    var_df = data_interface.variable_descriptions.rename(
        columns={"variable": "display_name"}
    )  # Rename column so that it can be merged with cat_df

    ## --------- ERROR HANDLING ----------
    # Deal with bad or missing users inputs

    # Station data error handling
    if data_type == "Stations":
        # dictionary with { argument name : [valid option, user input]}
        d = {
            "downscaling_method": ["Dynamical", downscaling_method],
            "timescale": ["hourly", timescale],
            "variable": ["Air Temperature at 2m", variable],
        }
        # Go through the users inputs
        # See if they match the required value for that argument
        # If not, print a warning to the user.
        for key, vals in zip(d.keys(), d.values()):
            if vals[0] != vals[1]:
                print(
                    "Weather station data can only be retrieved for {0}={1} \nYour input: {2} \nRetrieving data for {0}={1}".format(
                        key, vals[0], vals[1]
                    )
                )

        downscaling_method = "Dynamical"
        timescale = "hourly"
        variable = "Air Temperature at 2m"

        # Deal with scenario and time_slice arguments
        # Handle various use-cases of user inputs/errors
        if scenario is None:
            if time_slice is None:
                # Default
                scenario = ["Historical Climate"]
            else:
                scenario = []

        if resolution == "3 km":
            # Neither SSP 2-4.5 nor SSP 5-8.5 are valid options for scenario... need to remove
            for bad_scenario_choice in ["SSP 2-4.5", "SSP 5-8.5"]:
                if bad_scenario_choice in scenario:
                    error_message = f"{bad_scenario_choice} is not a valid scenario input for resolution = {resolution}"
                    print(_format_error_print_message(error_message))
                    return None
        if time_slice is not None:
            # Make sure time_slice and scenario match each other
            # If time_slice is not assigned by the user, it will be auto-set by the DataInterface object
            if any(value < 2015 for value in time_slice) and (
                ("Historical Climate") not in scenario
            ):
                # Add Historical Climate to scenario if the time scale includes historical period
                scenario.append("Historical Climate")
            if any(value >= 2015 for value in time_slice) and not any(
                "SSP" in item for item in scenario
            ):
                # If the time scale includes the future period and no SSP data is selected, add SSP 3-7.0
                scenario.append("SSP 3-7.0")

        if stations is None:
            # Print a warning if the user wants to retrieve station data but they don't input a value for station
            # The function will return all the stations by default
            print(
                "WARNING: You haven't set a particular station/s to retrieve data for; the function will default to retrieving all available stations in the domain"
            )
        if (stations is not None) and (type(stations) == str):
            # Catch easy user mistake without raising an error: Inputting a string instead of a list of list
            # I imagine this could happen if you just wanted to retrieve data for a single station
            stations = [stations]

    # If lat/lon input, change cached_area and area_subset
    if (latitude is not None) and (longitude is not None):
        area_subset = "lat/lon"
        cached_area = ["coordinate selection"]

    # Check warming level inputs
    try:
        warming_level = _error_handling_warming_level_inputs(
            warming_level, "warming_level", downscaling_method, resolution
        )
        warming_level_months = _error_handling_warming_level_inputs(
            warming_level_months, "warming_level_months", downscaling_method, resolution
        )
    except ValueError as error_message:
        print(_format_error_print_message(error_message))
        return None

    # Make sure the inputs are a valid type (no floats, ints, dictionaries, etc)
    for user_input in [
        variable,
        downscaling_method,
        resolution,
        timescale,
        area_subset,
        area_average,
        approach,
        scenario,
    ]:
        if (user_input is not None) and (type(user_input) not in [str, list]):
            error_message = (
                "Function arguments require a single string value for your inputs"
            )
            print(_format_error_print_message(error_message))
            return None

    # Maybe area average was capitalized wrong
    # Fix it instead of raising an error
    if area_average is not None:
        if area_average.lower().title() in ["Yes", "No"]:
            area_average = area_average.lower().title()

    # Cached area should be a list even if its just a single string value (i.e. [str])
    cached_area = [cached_area] if type(cached_area) != list else cached_area

    # If all_touched is None set to False
    if all_touched == None:
        all_touched = False

    # Check if all_touched boolean
    if all_touched not in [True, False]:
        raise ValueError("all_touched must be a boolean")

    # Make sure approach matches the scenario setting
    # See function documentation for more details
    approach, scenario, warming_level, time_slice = _error_handling_approach_inputs(
        approach, scenario, warming_level, time_slice
    )

    # Make sure the area subset is set to a valid input
    # See function documentation for more details
    try:
        area_subset = _error_handling_location_settings(area_subset, cached_area)
    except ValueError as error_message:
        print(_format_error_print_message(error_message))
        return None

    ## --------- ADD ARGUMENTS TO A DICTIONARY ----------
    # A dictionary is used for all the inputs in selections because it enables better error handling and cleaner code when we set selections.thing = thing
    # It also makes parsing through the arguments easier
    # The inputs here need to be a list so that they can be parsed easier by the _check_if_good_input function when comparing with the valid catalog options to confirm the user input is valid
    scenario_user_input = scenario  # What the user originally input for scenario

    check_input_df = get_data_options(
        variable=variable,
        downscaling_method=downscaling_method,
        resolution=resolution,
        timescale=timescale,
        scenario=scenario,
        tidy=False,
    )

    if check_input_df is None:
        # Does this print an informative error message? I think so but I'm not sure.
        return None

    # Merge with variable dataframe to get all the info about the data in one place
    check_input_df = check_input_df.merge(var_df, how="left")

    # Convert to a dictionary so it can be easily parsed by the function
    cat_dict = check_input_df.to_dict(orient="list")
    for key, values in cat_dict.items():
        # Remove non-unique values
        # This happens because we converted a pandas dataframe to a dictionary
        cat_dict[key] = list(np.unique(values))

    # _check_if_good_input will default fill the scenario options with EVERY possible option
    # It will in most cases give a list of all the available SSPs and the two historical data options (Historical Climate AND Historical Reconstruction)
    # I'd like the function to just default to Historical Climate + SSPs
    # So, if the user input None for scenario, I just remove Historical Reconstruction from the list
    if scenario_user_input == None:
        if "Historical Reconstruction" in cat_dict["scenario"]:
            cat_dict["scenario"] = [
                item
                for item in cat_dict["scenario"]
                if item != "Historical Reconstruction"
            ]

    # Check if it's an index
    variable_id = (
        var_df[var_df["display_name"] == cat_dict["variable"][0]].iloc[0].variable_id
    )
    variable_type = "Derived Index" if "_index" in variable_id else "Variable"

    # Settings for selections
    selections_dict = {
        "variable": cat_dict["variable"][0],
        "timescale": cat_dict["timescale"][0],
        "downscaling_method": cat_dict["downscaling_method"][0],
        "resolution": cat_dict["resolution"][0],
        "data_type": data_type,
        "scenario": cat_dict["scenario"],
        "area_average": area_average,
        "area_subset": area_subset,
        "cached_area": cached_area,
        "approach": approach,
        "warming_level": warming_level,
        "warming_level_window": warming_level_window,
        "warming_level_months": warming_level_months,
        "variable_type": variable_type,
        "time_slice": time_slice,
        "latitude": latitude,
        "longitude": longitude,
        "stations": stations,
        "all_touched": all_touched,
    }

    scenario_ssp, scenario_historical = _get_scenario_ssp_scenario_historical(
        selections_dict["approach"], selections_dict["scenario"]
    )
    selections_dict["scenario_ssp"] = scenario_ssp
    selections_dict["scenario_historical"] = scenario_historical

    ## ----- SET THE UNITS ------

    # Query the table based on input values
    # Timescale in table needs to be handled differently
    # This is because the monthly variables are derived from daily variables, so they are listed in the table as "daily, monthly"
    # Hourly variables may be different
    # Querying the data needs special handling due to the layout of the csv file
    var_df_query = var_df[
        (var_df["display_name"] == selections_dict["variable"])
        & (var_df["downscaling_method"] == selections_dict["downscaling_method"])
    ]
    var_df_query = var_df_query[
        var_df_query["timescale"].str.contains(selections_dict["timescale"])
    ]

    selections_dict["units"] = (
        units if units is not None else var_df_query["unit"].item()
    )  # Set units if user doesn't set them manually

    ## ------ CREATE SELECTIONS OBJECT --------
    selections = DataParameters()

    # Error handling for stations
    # If the user input a value for the station argument, check that it exists
    # If it doesn't exist, see if you can find something close... if not, throw an error
    # Need to do the error handling here since it requires the selections object
    if data_type == "Stations" and stations is not None:
        stations = _check_valid_input_station(
            stations, selections._stations_gdf.station.values
        )

    ## ------- SET EACH ATTRIBUTE -------

    try:
        selections.data_type = selections_dict["data_type"]
        selections.approach = selections_dict["approach"]
        selections.scenario_ssp = selections_dict["scenario_ssp"]
        selections.scenario_historical = selections_dict["scenario_historical"]
        selections.area_subset = selections_dict["area_subset"]
        selections.cached_area = selections_dict["cached_area"]
        selections.downscaling_method = selections_dict["downscaling_method"]
        selections.resolution = selections_dict["resolution"]
        selections.timescale = selections_dict["timescale"]
        selections.variable_type = selections_dict["variable_type"]
        selections.variable = selections_dict["variable"]
        selections.units = selections_dict["units"]
        selections.all_touched = selections_dict["all_touched"]

        # Setting the values like this enables us to take advantage of the default settings in DataParameters without having to manually set defaults in this function
        if selections_dict["warming_level"] is not None:
            selections.warming_level = selections_dict["warming_level"]
        if selections_dict["warming_level_window"] is not None:
            selections.warming_level_window = selections_dict["warming_level_window"]
        if selections_dict["area_average"] is not None:
            selections.area_average = selections_dict["area_average"]
        if selections_dict["time_slice"] is not None:
            selections.time_slice = selections_dict["time_slice"]
        if selections_dict["warming_level_months"] is not None:
            selections.warming_level_months = selections_dict["warming_level_months"]
        if selections_dict["latitude"] is not None:
            selections.latitude = selections_dict["latitude"]
        if selections_dict["longitude"] is not None:
            selections.longitude = selections_dict["longitude"]
        if selections_dict["stations"] is not None:
            selections.stations = selections_dict["stations"]
    except ValueError as error_message:
        # The error message is really long
        # And sometimes has a confusing Attribute Error: Pieces mismatch that is hard to interpret
        # Here we just print the error message and return None instead of allowing the long error to be raised by default
        print(_format_error_print_message(error_message))
        return None

    # Retrieve data
    data = selections.retrieve()
    return data


def _get_and_reformat_derived_variables(
    variable_descriptions: pd.DataFrame,
) -> pd.DataFrame:
    """(1) Get the just derived variables from the variables_descriptions csv and
    (2) Reformat the data such that the timescales are split into separate rows

    This is such that it can match the formatting in the catalog, where each independent data option is separated into a distinct row
    i.e. if derived variable "var" has "timescale" = "hourly, monthly", it is separated into two separate rows with one row having "timescale" = "hourly" and the second row having "timescale" = "monthly"

    Backend helper function for retrieving a user-friendly version of the intake catalog that contains our added climakitae derived variables

    Arguments
    ---------
    variable_descriptions: pd.DataFrame
        Variable descriptions, units, etc in table format

    Returns
    -------
    derived_vars_df: pd.DataFrame
        Subset of variable_descriptions containing only variables with variable_id containing the substring "_derived"
        "timescale" column separated into unique timescales for each row

    """
    # Just get subset of derived variables
    derived_variables = (
        variable_descriptions[
            variable_descriptions["variable_id"].str.contains("_derived")
        ]
        .reset_index(drop=True)
        .rename(columns={"display_name": "variable"})
    )

    # Get the derived variables that are valid across multile timescales
    # They will have a comma-separated timescale i.e. "daily, monthly"
    derived_variables_multi_timescale = derived_variables[
        derived_variables["timescale"].str.contains(", ")
    ]

    # loop through each row in the DataFrame
    derived_variables_split = []
    for i in range(len(derived_variables_multi_timescale)):
        # Get single row, containing one variable
        row = derived_variables_multi_timescale.iloc[i]

        # Split the comma-separated timescale into a list
        # i.e. "daily, monthly" becomes ["daily","monthly"]
        timescale = row["timescale"].split(", ")

        # Create a dataframe with one row per timescale
        # i.e. "derived index" for a timescale of "daily, monthly" becomes a two-row dataframe...
        # i.e. row 1 containing "derived index" and "daily" and row 2 containing "derived index" and "monthly"
        row_split_i = pd.DataFrame([row] * len(timescale))
        row_split_i["timescale"] = timescale
        derived_variables_split.append(row_split_i)

    # Now, append a pandas dataframe with remaining derived variables
    # These are the ones that have a single timescale (i.e. "hourly" or "daily")
    derived_variables_single_timescale = derived_variables[
        ~derived_variables["timescale"].str.contains(", ")
    ]
    derived_variables_split.append(derived_variables_single_timescale)
    derived_vars_df = pd.concat(derived_variables_split, ignore_index=True)

    return derived_vars_df
