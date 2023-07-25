import os
import pkg_resources
import pandas as pd
import geopandas as gpd
from shapely.geometry import box, Polygon
import intake
import param
import warnings
from matplotlib.figure import Figure
import matplotlib.ticker as ticker
from climakitae.core.constants import (
    variable_descriptions_csv_path,
    stations_csv_path,
    data_catalog_url,
)
from climakitae.util.utils import read_csv_file
from climakitae.core.boundaries import Boundaries
from climakitae.util.unit_conversions import _get_unit_conversion_options
from climakitae.core.catalog_convert import (
    _downscaling_method_to_activity_id,
    _resolution_to_gridlabel,
    _timescale_to_table_id,
    _scenario_to_experiment_id,
)


class DataInterface:
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(DataInterface, cls).__new__(cls)
            return cls._instance

    def __init__(self):
        self.variable_descriptions = read_csv_file(variable_descriptions_csv_path)
        self.stations = read_csv_file(stations_csv_path)
        self.stations_gdf = gpd.GeoDataFrame(
            self.stations,
            crs="EPSG:4326",
            geometry=gpd.points_from_xy(self.stations.LON_X, self.stations.LAT_Y),
        )
        self.data_catalog = intake.open_esm_datastore(data_catalog_url)

        # Get geography boundaries
        self.geographies = Boundaries()


class DataParameters(param.Parameterized):
    """
    An object to hold data parameters, which depends only on the 'param'
    library. Currently used in '_display_select', which uses 'panel' to draw the
    gui, but another UI could in principle be used to update these parameters
    instead.
    """

    # Unit conversion options for each unit
    unit_options_dict = _get_unit_conversion_options()

    # Location defaults
    area_subset = param.Selector(objects=dict())
    cached_area = param.Selector(objects=dict())
    latitude = param.Range(default=(32.5, 42), bounds=(10, 67))
    longitude = param.Range(default=(-125.5, -114), bounds=(-156.82317, -84.18701))

    # Data defaults
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
    downscaling_method = param.ListSelector(
        default=["Dynamical"], objects=["Dynamical", "Statistical"]
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

    def _get_user_options(
        self, data_catalog, downscaling_method, timescale, resolution
    ):
        """Using the data catalog, get a list of appropriate scenario and simulation options given a user's
        selections for downscaling method, timescale, and resolution.
        Unique variable ids for user selections are returned, then limited further in subsequent steps.

        Parameters
        ----------
        cat: intake catalog
        downscaling_method: list, one of ["Dynamical"], ["Statistical"], or ["Dynamical","Statistical"]
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

        # Get catalog subset from user inputs
        with warnings.catch_warnings(record=True):
            cat_subset = data_catalog.search(
                activity_id=[
                    _downscaling_method_to_activity_id(dm) for dm in downscaling_method
                ],
                table_id=_timescale_to_table_id(timescale),
                grid_label=_resolution_to_gridlabel(resolution),
            )

        # For LOCA grid we need to use the UCSD institution ID
        # This comes into play whenever Statistical is selected
        # WRF data on LOCA grid is tagged with UCSD institution ID
        if "Statistical" in downscaling_method:
            cat_subset = cat_subset.search(institution_id="UCSD")

        # Limit scenarios if both LOCA and WRF are selected
        # We just want the scenarios that are present in both datasets
        if set(["Dynamical", "Statistical"]).issubset(
            downscaling_method
        ):  # If both are selected
            loca_scenarios = cat_subset.search(
                activity_id="LOCA2"
            ).df.experiment_id.unique()  # LOCA unique member_ids
            wrf_scenarios = cat_subset.search(
                activity_id="WRF"
            ).df.experiment_id.unique()  # WRF unique member_ids
            overlapping_scenarios = list(set(loca_scenarios) & set(wrf_scenarios))
            cat_subset = cat_subset.search(experiment_id=overlapping_scenarios)

        elif downscaling_method == ["Statistical"]:
            cat_subset = cat_subset.search(activity_id="LOCA2")

        # Get scenario options
        scenario_options = list(cat_subset.df["experiment_id"].unique())

        # Get all unique simulation options from catalog selection
        try:
            simulation_options = list(cat_subset.df["source_id"].unique())

            # Remove troublesome simulations
            simulation_options = [
                sim
                for sim in simulation_options
                if sim not in ["HadGEM3-GC31-LL", "KACE-1-0-G"]
            ]

            # Remove ensemble means
            if "ensmean" in simulation_options:
                simulation_options.remove("ensmean")
        except:
            simulation_options = []

        # Get variable options
        unique_variable_ids = list(cat_subset.df["variable_id"].unique())

        return scenario_options, simulation_options, unique_variable_ids

    def _get_variable_options_df(
        self, variable_descriptions, unique_variable_ids, downscaling_method, timescale
    ):
        """Get variable options to display depending on downscaling method and timescale

        Parameters
        ----------
        var_config: pd.DataFrame
            Variable descriptions, units, etc in table format
        unique_variable_ids: list of strs
            List of unique variable ids from catalog.
            Used to subset var_config
        downscaling_method: list, one of ["Dynamical"], ["Statistical"], or ["Dynamical","Statistical"]
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

    def _get_var_ids(
        self, variable_descriptions, variable, downscaling_method, timescale
    ):
        """Get variable ids that match the selected variable, timescale, and downscaling method.
        Required to account for the fact that LOCA, WRF, and various timescales use different variable id values.
        Used to retrieve the correct variables from the catalog in the backend.
        """
        var_id = variable_descriptions[
            (variable_descriptions["display_name"] == variable)
            & (  # Make sure it's a valid variable selection
                variable_descriptions["timescale"].str.contains(timescale)
            )  # Make sure its the right timescale
            & (
                variable_descriptions["downscaling_method"].isin(downscaling_method)
            )  # Make sure it's the right downscaling method
        ]
        var_id = list(var_id.variable_id.values)
        return var_id

    def _get_overlapping_station_names(
        self,
        stations_gdf,
        area_subset,
        cached_area,
        latitude,
        longitude,
        _geographies,
        _geography_choose,
    ):
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
                    df_ae = _get_subarea_from_shape_index(
                        _geographies._us_states, shape_index
                    )
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
                elif area_subset == "CA Electric Balancing Authority Areas":
                    df_ae = _get_subarea_from_shape_index(
                        _geographies._ca_electric_balancing_areas, shape_index
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
        ) = self._get_user_options(
            data_catalog=self._data_catalog,
            downscaling_method=self.downscaling_method,
            timescale=self.timescale,
            resolution=self.resolution,
        )
        self.variable_options_df = self._get_variable_options_df(
            variable_descriptions=self._variable_descriptions,
            unique_variable_ids=unique_variable_ids,
            downscaling_method=self.downscaling_method,
            timescale=self.timescale,
        )

        # Set scenario param
        scenario_ssp_options = [
            _scenario_to_experiment_id(scen, reverse=True)
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
        self.variable_id = self._get_var_ids(
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
        self.cached_area = list(self._geography_choose[self.area_subset].keys())[0]

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

    @param.depends("downscaling_method", "data_type", watch=True)
    def _update_data_type_options_if_loca_selected(self):
        """If statistical downscaling is selected, remove option for station data because we don't
        have the 2m temp variable for LOCA"""
        if "Statistical" in self.downscaling_method:
            self.param["data_type"].objects = ["Gridded"]
            self.data_type = "Gridded"
        else:
            self.param["data_type"].objects = ["Gridded", "Station"]
        if "Station" in self.data_type:
            self.param["downscaling_method"].objects = ["Dynamical"]
            if "Statistical" in self.downscaling_method:
                self.downscaling_method.remove("Statistical")
        else:
            self.param["downscaling_method"].objects = ["Dynamical", "Statistical"]

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
        "timescale",
        "resolution",
        "downscaling_method",
        "data_type",
        "variable",
        watch=True,
    )
    def _update_user_options(self):
        """Update unique variable options"""

        if self.data_type == "Station":
            self.param["timescale"].objects = ["hourly"]
            self.timescale = "hourly"
        elif self.data_type == "Gridded":
            if self.downscaling_method == ["Statistical"]:
                self.param["timescale"].objects = ["daily", "monthly"]
                if self.timescale == "hourly":
                    self.timescale = "daily"
            elif self.downscaling_method == ["Dynamical"]:
                self.param["timescale"].objects = ["daily", "monthly", "hourly"]
            else:
                # If both are selected, only show daily data
                # We do not have WRF on LOCA grid resampled to monthly
                self.param["timescale"].objects = ["daily"]
                self.timescale = "daily"

        if self.downscaling_method == []:
            # Default options to show if nothing is selected
            downscaling_method = ["Dynamical"]
        else:
            downscaling_method = self.downscaling_method

        (
            self.scenario_options,
            self.simulation,
            unique_variable_ids,
        ) = self._get_user_options(
            data_catalog=self._data_catalog,
            downscaling_method=downscaling_method,
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
            self.variable_options_df = self._get_variable_options_df(
                variable_descriptions=self._variable_descriptions,
                unique_variable_ids=unique_variable_ids,
                downscaling_method=self.downscaling_method,
                timescale=self.timescale,
            )
            var_options = self.variable_options_df.display_name.values
            self.param["variable"].objects = var_options
            if self.variable not in var_options:
                self.variable = var_options[0]

        var_info = self.variable_options_df[
            self.variable_options_df["display_name"] == self.variable
        ]  # Get info for just that variable
        self.extended_description = var_info.extended_description.item()
        self.variable_id = self._get_var_ids(
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
                elif (
                    self.downscaling_method == ["Dynamical"]
                    or self.downscaling_method == []
                ):
                    self.param["cached_area"].objects = [
                        "CA",
                        "NV",
                        "OR",
                        "UT",
                        "AZ",
                    ]
                self.cached_area = "CA"
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
        # Get scenario options in catalog format
        scenario_ssp_options = [
            _scenario_to_experiment_id(scen, reverse=True)
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
            _scenario_to_experiment_id(scen, reverse=True)
            for scen in self.scenario_options
            if scen in historical_scenarios
        ]
        self.param["scenario_historical"].objects = scenario_historical_options
        if self.scenario_historical not in scenario_historical_options:
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
        if self.downscaling_method == ["Statistical"]:
            historical_climate_range = self.historical_climate_range_loca
        elif set(["Dynamical", "Statistical"]).issubset(self.downscaling_method):
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
        if self.downscaling_method == ["Statistical"]:
            historical_climate_range = self.historical_climate_range_loca
        elif set(["Dynamical", "Statistical"]).issubset(self.downscaling_method):
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
            overlapping_stations = self._get_overlapping_station_names(
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
