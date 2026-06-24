"""
Calculates the Standard Year Climate Profiles using a warming level approach and designated
quantiles. The historical baseline for relative profile computation is a warming level of 1.2 C.
User specified warming level will be calculated relative to this baseline unless the "no_delta" option
is set to True, in which case the raw profile(s) for the requested warming level(s) will
be returned.
"""

from typing import Tuple
from typing import Any, Dict

import numpy as np
import pandas as pd
import xarray as xr
from tqdm.auto import tqdm  # Progress bar

from climakitae.core.constants import UNSET, WRF_BA_MODELS
from climakitae.core.data_interface import DataInterface, get_data
from climakitae.core.paths import HADISD_STATIONS_URL, VARIABLE_DESCRIPTIONS_CSV_PATH
from climakitae.explore.typical_meteorological_year import is_HadISD, match_str_to_wl
from climakitae.util.utils import julianDay_to_date, read_csv_file
from climakitae.util.warming_levels import get_gwl_at_year

xr.set_options(keep_attrs=True)  # Keep attributes when mutating xr objects


def _get_station_coordinates(station_name: str) -> Tuple[float, float]:
    """
    Look up the latitude and longitude coordinates for a given station name.

    Reads directly from the stations CSV to avoid the overhead of
    instantiating DataInterface (which eagerly loads all boundary data).

    Parameters
    ----------
    station_name : str
        Name of the weather station to look up.

    Returns
    -------
    Tuple[float, float]
        (latitude, longitude) coordinates of the station.

    Raises
    ------
    ValueError
        If the station name is not found in the stations CSV.

    Examples
    --------
    >>> lat, lon = _get_station_coordinates("San Diego Lindbergh Field (KSAN)")
    >>> print(f"Latitude: {lat}, Longitude: {lon}")
    """
    stations_df = pd.read_csv(HADISD_STATIONS_URL)

    # Look up the station in the DataFrame
    station_row = stations_df[stations_df["station"] == station_name]

    if station_row.empty:
        raise ValueError(
            f"Station '{station_name}' not found. "
            f"Please check the station name and try again."
        )

    # Extract coordinates
    lat = station_row["LAT_Y"].iloc[0]
    lon = station_row["LON_X"].iloc[0]

    return lat, lon


def _convert_stations_to_lat_lon(
    stations: list[str], buffer: float = 0.02
) -> Tuple[Tuple[float, float], Tuple[float, float]]:
    """
    Convert a list of station names to lat/lon bounds with a buffer.

    Parameters
    ----------
    stations : list[str]
        List of weather station names to convert.
    buffer : float, optional
        Buffer to add around station coordinates in degrees (default: 0.02).

    Returns
    -------
    Tuple[Tuple[float, float], Tuple[float, float]]
        (latitude_bounds, longitude_bounds) where each bounds is a tuple of (min, max).

    Raises
    ------
    ValueError
        If any station name is not found in the DataInterface.

    Examples
    --------
    >>> stations = ["San Diego Lindbergh Field (KSAN)"]
    >>> lat_bounds, lon_bounds = _convert_stations_to_lat_lon(stations)
    >>> print(f"Latitude: {lat_bounds}, Longitude: {lon_bounds}")
    """
    if not stations:
        raise ValueError("No stations provided for coordinate conversion.")

    # Get coordinates for all stations
    lats = []
    lons = []
    for station in stations:
        lat, lon = _get_station_coordinates(station)
        lats.append(lat)
        lons.append(lon)

    # Calculate bounds with buffer
    min_lat = min(lats) - buffer
    max_lat = max(lats) + buffer
    min_lon = min(lons) - buffer
    max_lon = max(lons) + buffer

    return (min_lat, max_lat), (min_lon, max_lon)


def _get_clean_standardyr_filename(
    var_id: str,
    q: float,
    location: str,
    gwl: float,
    warming_level_window: int | None,
    no_delta: bool,
    approach: str | None,
    centered_year: int | None,
    scenario: str | None,
    ba_models: bool,
) -> str:
    """
    Standardizes filename export for standard year files

    Parameters
    ----------
    var_id : str
        Name of variable used in profile
    q : float
        Percentile used in profile
    location: str
        String describing profile location
    gwl : float
        Single gwl for csv file name
    warming_level_window: int
        Years around Global Warming Level (+/-) (e.g. 15 means a 30yr window)
    no_delta : bool
        no_delta value used to generate profile
    approach (Optional) : str, "Warming Level" or "Time"
        The climate profile approach to use, either
            - "Time" (which is actually a warming level approach, but centered on an input) or
            - "Warming Level" (default)
    centered_year (Optional) : int in range 1980,2099]
        For approach="Time", the year for which to find a corresponding warming level
    time_profile_scenario (Optional) : str, default "SSP 3-7.0"
        SSP scenario from ["SSP 3-7.0", "SSP 2-4.5","SSP 5-8.5"]
    ba_models (optional) : bool, default False,
        If True only return bias-adjusted WRF models

    Returns
    -------
    str
        A cleaned up file name
    """

    # clean arguments for filenaming
    clean_loc_name = location.replace(" ", "_").replace("(", "").replace(")", "")
    clean_q_name = f"{q:.2f}".split(".")[1].lower()
    clean_var_name = var_id.lower()

    clean_gwl_name = ""
    if gwl:
        clean_gwl_name = match_str_to_wl(gwl).lower().replace(".", "pt")
        clean_gwl_name = f"_{clean_gwl_name}"

    delta_str = ""
    if no_delta is False:
        delta_str = "_delta_from_historical"

    # default 30yr window (corresponds to default 15)
    window_str = "_30yr_window"
    if warming_level_window:
        # custom window size provided
        window = warming_level_window * 2
        window_str = f"_{window}yr_window"

    approach_str = ""
    if approach:
        approach_str = approach.lower().replace(" ", "_")
        approach_str = f"_{approach_str}"

    centered_year_str = ""
    if centered_year:
        centered_year_str = f"_{centered_year}"

    scenario_str = ""
    if scenario:
        scenario_str = (
            scenario.lower().replace("-", "").replace(" ", "").replace(".", "")
        )
        scenario_str = f"_{scenario_str}"
    elif approach == "Time":
        # if time-based profile being generated, include default value in filename
        scenario_str = "_ssp370"
    ba_models_str = ""
    if ba_models:
        ba_models_str = "_ba_models"

    filename = f"stdyr_{clean_var_name}_{clean_q_name}ptile_{clean_loc_name}{clean_gwl_name}{delta_str}{window_str}{approach_str}{centered_year_str}{scenario_str}{ba_models_str}.csv"
    return filename


# helper functions
def _check_cached_area(location_str: str, **kwargs: Any) -> str:
    """
    Check cached area input to profile selections
    """
    cached_area = kwargs.get("cached_area")

    match cached_area:
        case str():
            location_str = cached_area.lower()
        case list():
            location_str = "_".join(c.lower() for c in cached_area)
        case _:
            return location_str
    return location_str


def _check_lat_lon(location_str: str, **kwargs: Any) -> str:
    """
    Check latitude and longitude inputs to profile selections
    """

    latitude = kwargs.get("latitude")
    longitude = kwargs.get("longitude")

    match latitude, longitude, len(location_str):
        # lat/lon provided, no cached area
        case tuple(), tuple(), 0:
            latitude = latitude[0] + 0.02
            longitude = longitude[0] + 0.02

            lat_str = str(round(latitude, 6)).replace(".", "-")
            lon_str = str(round(abs(longitude), 6)).replace(".", "-")

            location_str = f"{lat_str}N_{lon_str}W"
        case _:
            return location_str

    return location_str


def _check_stations(location_str: str, **kwargs: Any) -> str:
    """
    Check station name input to profile selections
    """
    stations = kwargs.get("stations")

    match stations, len(location_str):
        # only station(s) provided
        case list(), 0:
            # if only one station in the list
            if len(stations) == 1:
                # if that station is a HadISD station
                if is_HadISD(stations[0]):
                    location_str = stations[0].lower()
                # if not a HadISD station
                else:
                    raise ValueError("Station must be a HadISD station.")
            # if there are multiple station names
            else:
                # if all are HadISD stations
                if all(is_HadISD(s) for s in stations):
                    location_str = "_".join(s.lower() for s in stations)
                # if at least one is not a HadISD station
                else:
                    raise ValueError(f"All stations must be HadISD stations.")
        # no station(s), other location parameters provided
        case (object(), length) if length > 0:
            return location_str
        case _:
            raise TypeError(
                "Location must be provided as either `station_name` or `cached_area` or `latitude` plus `longitude`."
            )

    return location_str


def export_profile_to_csv(profile: pd.DataFrame, **kwargs: Any) -> None:
    """
    Export profile to csv file with a descriptive file name.

    Each warming level is saved in a separate file.

    Parameters
    ----------
    profile: pd.DataFrame
        Standard year profile with MultiIndex columns

    **kwargs : dict
        Keyword arguments for data selection. Allowed keys:
            variable : str
                Name of variable used in profile
            q : float
                Percentile used in profile
            global_warming_levels : list[float]
                List of global warming levels in profile
            warming_level_winow: int in range (5,25), optional
                Years around Global Warming Level (+/-) (e.g. 15 means a 30yr window)
            location : str, tuple(float | int), list[str | float | int]
                Station name, cached_area, or coordinates
            cached_area : str, optional
                Name of cached area used in profile
            no_delta : bool, default False, optional
                True if no_delta=True when generating profile
            approach (Optional) : str, "Warming Level" or "Time"
                The climate profile approach to use, either
                    - "Time" (which is actually a warming level approach, but centered on an input) or
                    - "Warming Level" (default)
            centered_year (Optional) : int in range [1980,2099]
                For approach="Time", the year for which to find a corresponding warming level
            time_profile_scenario (Optional) : str, default "SSP 3-7.0"
                SSP scenario from ["SSP 3-7.0", "SSP 2-4.5","SSP 5-8.5"]
            bias_adjusted_models (optional) : bool, default False, if True only return bias-adjusted WRF models
    """

    # Get required parameter values
    variable = kwargs.get("variable")
    q = kwargs.get("q")

    # Get warming_level, no_delta, warming_level_window, approach, centered_year inputs, and scenario
    no_delta = kwargs.get("no_delta", False)
    warming_level_window = kwargs.get("warming_level_window", None)
    approach = kwargs.get("approach", None)
    centered_year = kwargs.get("centered_year", None)
    global_warming_levels = kwargs.get("warming_level", None)
    scenario = kwargs.get("time_profile_scenario", None)
    ba_models = kwargs.get("bias_adjusted_models", False)

    # Get variable id string to use in file name
    variable_descriptions = read_csv_file(VARIABLE_DESCRIPTIONS_CSV_PATH)
    var_id = variable_descriptions[
        (variable_descriptions["display_name"] == variable)
        & (variable_descriptions["timescale"] == "hourly")
    ]["variable_id"].item()

    # Get location string based on combination of location variables
    kwargs = _handle_location_params(**kwargs)
    func_list = [_check_cached_area, _check_lat_lon, _check_stations]
    location_str = ""
    for func in func_list:
        location_str = func(location_str, **kwargs)

    # Check profile MultiIndex to pull out data by Global Warming Level
    match profile.keys().nlevels:
        case 1:  # Single WL
            # If 'warming_level' provided, fetch the value within the input list
            if global_warming_levels:
                gwl = global_warming_levels[0]
            else:
                # For time-based approach, do not include gwl in the filename - set it to None
                if approach == "Time":
                    gwl = global_warming_levels
                # Otherwise, use the default
                else:
                    gwl = 1.2

            filename = _get_clean_standardyr_filename(
                var_id,
                q,
                location_str,
                gwl,
                warming_level_window,
                no_delta,
                approach,
                centered_year,
                scenario,
                ba_models,
            )
            profile.to_csv(filename, index=False)
        case 2:  # Multiple WL (WL included in MultiIndex)
            for gwl in global_warming_levels:  # Single file per WL
                filename = _get_clean_standardyr_filename(
                    var_id,
                    q,
                    location_str,
                    gwl,
                    warming_level_window,
                    no_delta,
                    approach,
                    centered_year,
                    scenario,
                    ba_models,
                )
                profile.xs(f"WL_{gwl}", level="Warming_Level", axis=1).to_csv(
                    filename, index=False
                )
        case _:
            raise ValueError(
                f"Profile MultiIndex should have two or three levels. Found {profile.keys().nlevels} levels."
            )


def _get_buffer_from_resolution(resolution: str) -> float:
    """Return a buffer in degrees, used to select closest gridcell.

    Parameters
    ----------
    resolution: str
        The grid resolution, in km

    Returns
    -------
    buffer: float
        The buffer, in degrees, to select around a lat/lon point
    """
    match resolution:
        case "3 km":
            buffer = 0.02
        case "9 km":
            buffer = 0.08
        case "45 km":
            buffer = 0.35
        case _:
            # Use 3 km value as buffer
            buffer = 0.02
    return buffer


def _handle_location_params(**kwargs: Dict[str, Any]) -> Dict[str, Any]:
    """Parse the `location` parameter to determine what kind of location is being selected.

    Parameters
    ----------
    **kwargs : dict
        Keyword arguments for data selection. Allowed keys:
        - variable (Optional) : str, default "Air Temperature at 2m"
        - resolution (Optional) : str, default "3 km"
        - approach (Optional) : str, "Warming Level" or "Time"
        - centered_year (Optional) : int in range [1980,2099]
        - time_profile_scenario (Optional): str, default "SSP 3-7.0"
        - warming_levels (Optional) : List[float], default [1.2]
        - warming_level_window (Optional): int in range [5,25]
        - location (Optional) : str, List[str|float|int], or Tuple[str|float|int]
        - units (Optional) : str, default "degF"
        - no_delta (optional) : bool, default False, if True, do not retrieve historical data, return raw future profile
        - bias_adjusted_models (optional) : bool, default False, if True only return bias-adjusted WRF models

    Returns
    -------
    **kwargs : dict
        Arguments with updated "cached_area", "stations", or "latitude"/"longitude" parameters
    """
    # location is not required, could be unset
    location = kwargs.pop("location", None)
    match location:
        case str():  # Can be single HadISD station or cached area
            if is_HadISD(location):
                kwargs["stations"] = [location]
            else:
                kwargs["cached_area"] = location
        case (
            list() | tuple()
        ):  # Can be list/tuple of HadISD stations, cached area, or lat/lon
            # Catch cases where someone provides invalid data - a list of coordinate tuples, for example
            if not isinstance(location[0], (str, float, int)):
                formatted_param_type = type(location).__name__
                formatted_bad_type = type(location[0]).__name__
                raise TypeError(
                    f"The location {formatted_param_type} may only contain string or numeric values, not {formatted_bad_type}."
                )
            # Now work through the valid options
            if all(is_HadISD(loc) for loc in location):  # stations
                kwargs["stations"] = location
            elif all(isinstance(loc, str) for loc in location):  # cached area
                kwargs["cached_area"] = location
            elif all(isinstance(loc, (float, int)) for loc in location):  # coordinates
                if len(location) == 2:
                    if all(loc <= 0 for loc in location) or all(
                        loc >= 0 for loc in location
                    ):
                        raise ValueError(
                            "Expected a positive-valued latitude coordinate and negative-valued longitude coordinate."
                        )
                    else:
                        # In western US on our grids, latitude will always be positive
                        # and longitude always negative
                        latitude = [loc for loc in location if loc > 0][0]
                        longitude = [loc for loc in location if loc < 0][0]
                        # Add buffer around lat/lon point to help get nearest cell
                        buffer = _get_buffer_from_resolution(
                            kwargs.get("resolution", None)
                        )
                        kwargs["latitude"] = (latitude - buffer, latitude + buffer)
                        kwargs["longitude"] = (longitude - buffer, longitude + buffer)
                else:  # Location len != 2
                    raise ValueError(
                        "Length of `location` parameter must be two if providing a coordinate pair."
                    )
            else:  # Location parameter didn't match any expected format
                raise ValueError(f"Unable to parse location parameter.")
        case None:
            # Do nothing here. Users will see a warning about the lack of location in retrieve_profile_data.
            pass
        case _:
            formatted_param_type = type(location).__name__
            raise TypeError(
                f"The `location` parameter type should str, List, or Tuple if set. Got type {formatted_param_type}."
            )
    return kwargs


def _handle_approach_params(**kwargs: Dict[str, Any]) -> Dict[str, Any]:
    """
    Helper function that
    1. performs validation on variables related to approach ('approach','centered_year','warming_level')
    2. carries out the time-based approach, for valid user inputs

    Parameters
    ----------
    **kwargs : dict
        Keyword arguments for data selection. Allowed keys:
        - variable (Optional) : str, default "Air Temperature at 2m"
        - resolution (Optional) : str, default "3 km"
        - approach (Optional) : str, "Warming Level" or "Time"
        - centered_year (Optional) : int in range [1980,2099]
        - time_profile_scenario (Optional): str, default "SSP 3-7.0"
        - warming_levels (Optional) : List[float], default [1.2]
        - warming_level_window (Optional): int in range [5,25]
        - cached_area (Optional) : str or List[str]
        - latitude (Optional) : float or tuple
        - longitude (Optional) : float or tuple
        - stations (Optional) : list[str], default None
        - units (Optional) : str, default "degF"
        - no_delta (optional) : bool, default False, if True, do not retrieve historical data, return raw future profile
        - bias_adjusted_models (optional) : bool, default False, if True only return bias-adjusted WRF models

    Returns
    -------
    **kwargs : dict
        Arguments with updated :approach" and "warming_level" parameters

    """

    approach = kwargs.get("approach")
    centered_year = kwargs.get("centered_year")
    warming_level = kwargs.get("warming_level", None)
    scenario = kwargs.get("time_profile_scenario", None)
    resolution = kwargs.get("resolution", "3 km")
    ba_models = kwargs.get("bias_adjusted_models", False)

    # catch invalid scenario and resolution combinations
    if scenario in ["SSP 5-8.5", "SSP 2-4.5"]:
        if resolution == "3 km":
            raise ValueError(
                f"if 'time_profile_scenario' is '{scenario}', resolution must be '9 km' or '45 km' - got {resolution}."
            )

    # catch invalid scenario and ba_models combinations
    if scenario in ["SSP 5-8.5", "SSP 2-4.5"]:
        if ba_models:
            raise ValueError(
                f"No bias-adjusted models available for 'time_profile_scenario' = {scenario}."
            )

    # handle approach, centered year, and scenario combindations
    match approach, centered_year, scenario:
        # If 'approach'="Time" and 'centered_year' is provided
        case "Time", int(), _:
            # Throw error if 'centered_year' not in acceptable range
            if centered_year not in range(1980, 2100):
                raise ValueError(
                    f"Only years 1980-2099 are valid inputs for 'centered_year'. Received {centered_year}."
                )
            # Throw error if 'warming_level' provided
            elif warming_level is not None:
                raise ValueError(
                    f"Do not input warming level(s) if using a time-based approach."
                )
            # otherwise:
            # get warming level based on year and set 'warming_level' to this value
            else:
                print(
                    f"You have chosen to produce a time-based Standard Year climate profile centered around {centered_year}. \n"
                    "Standard Year functionality for time-based profiles: \n"
                    "   1. Input centered_year and scenario are used by the get_gwl_at_year function to identify the warming level corresponding to centered_year. \n"
                    "   2. Data is retrieved at the corresponding warming level with all available simulations across all scenarios, following GWL best practices. \n"
                    "   3. Standard Year profile is generated at the corresponding warming level. \n"
                )

                print(
                    f"Step 1: The corresponding global warming level for input centered year {centered_year} will now be determined.\n"
                )
                if centered_year < 2015:
                    warming_level_scenario = "Historical"
                    if scenario is None:
                        scenario = "SSP 3-7.0"
                elif scenario is None:
                    scenario = "SSP 3-7.0"
                    warming_level_scenario = scenario
                    print(
                        f"Using default '{warming_level_scenario}' to find corresponding warming level."
                    )

                else:
                    warming_level_scenario = scenario
                    f"Using input '{warming_level_scenario}' to find corresponding warming level."

                gwl_options = get_gwl_at_year(centered_year, warming_level_scenario)
                new_warming_level = [
                    float(gwl_options.loc[warming_level_scenario, "Mean"])
                ]
                print(
                    f"Corresponding warming level for 'centered_year'= {centered_year} and '{warming_level_scenario}' scenario is {new_warming_level}. \n"
                )

                kwargs["warming_level"] = new_warming_level
                kwargs["approach"] = "Warming Level"
                kwargs["time_profile_scenario"] = scenario

        # If 'approach'="Time" and 'centered_year' is not provided
        case "Time", object(), _:
            # throw error
            raise ValueError("If 'approach'='Time', 'centered_year' must be provided.")
        # If 'approach'="Warming Level" or None
        case None | "Warming Level", _, _:
            # and if 'centered_year' provided, throw error
            if isinstance(centered_year, int):
                raise ValueError(
                    f"If 'centered_year' provided, 'approach' must be 'Time'. Received '{approach}.'"
                )
            # else, nothing happens - user is using warming level approach from the get-go
            else:
                None
            # check if scenario provided
            if isinstance(scenario, str):
                raise ValueError(
                    f"If 'scenario' provided, 'approach' must be 'Time'. Received '{approach}.'"
                )
        # Catches invalid 'approach' parameter inputs
        case _, _, _:
            raise ValueError(
                f"Only 'Time' or 'Warming Level' accepted as inputs for 'approach'. Received '{approach}'."
            )

    return kwargs


def _filter_ba_models(
    data: xr.DataArray,
) -> xr.DataArray:
    """
    Filter data to include only bias-adjusted WRF models.
    This function filters the input data to retain only simulations that correspond to
    bias-adjusted Weather Research and Forecasting (WRF) models.

    Parameters
    ----------
    data : xr.DataArray
        Input climate data array containing simulation data across multiple models.

    Returns
    -------
    xr.DataArray
        Filtered data array containing only bias-adjusted WRF model simulations
        when conditions are met, otherwise returns the original data unchanged.

    Notes
    -----
    The function uses WRF_BA_MODELS constant to identify which simulations
    correspond to bias-adjusted WRF models by checking if any of the model
    names appear as substrings in the simulation names.
    """
    # Filter only for BC-WRF models
    # This check is to see which simulations have these BC models as the root part
    # of their simulation name, which is to accomodate for simulation names being
    # the same across SSPs.
    data = data.sel(
        simulation=[
            sim
            for sim in data.simulation.values
            if any(s in sim for s in WRF_BA_MODELS)
        ]
    )
    return data


def retrieve_profile_data(**kwargs: Any) -> Tuple[xr.DataArray, xr.DataArray]:
    """
    Backend function for retrieving data needed for computing climate profiles.

    Reads in the full hourly data for the 8760 analysis, including all warming levels.

    Parameters
    ----------
    **kwargs : dict
        Keyword arguments for data selection. Allowed keys:
        - variable (Optional) : str, default "Air Temperature at 2m"
        - resolution (Optional) : str, default "3 km"
        - approach (Optional) : str, "Warming Level" or "Time"
        - centered (Optional) : int
        - warming_levels (Optional) : List[float], default [1.2]
        - warming_level_window (Optional): int in range [5,25]
        - cached_area (Optional) : str or List[str]
        - latitude (Optional) : float or tuple
        - longitude (Optional) : float or tuple
        - stations (Optional) : list[str], default None
        - units (Optional) : str, default "degF"
        - no_delta (optional) : bool, default False, if True, do not retrieve historical data, return raw future profile
        - bias_adjusted_models (optional) : bool, default False, if True only return bias-adjusted WRF models

    Returns
    -------
    Tuple[xr.Dataset, xr.Dataset]
        (historic_data, future_data, get_data_params) - Historical data at 1.2°C warming,
        and future data at specified warming levels.
    Raises
    ------
    ValueError
        If invalid parameter keys are provided.

    Example
    -------
    >>> historic_data, future_data = retrieve_profile_data(
    ...     variable="Air Temperature at 2m",
    ...     resolution="45 km",
    ...     time_profile_scenario="SSP 2-4.5",
    ...     warming_level=[1.5, 2.0, 3.0],
    ...     units="degF"
    ... )

    >>> historic_data, future_data = retrieve_profile_data(
    ...     warming_level=[2.0]
    ... )

    Notes
    -----
    Historical data is always retrieved for warming level = 1.2°C.
    Future data uses user-specified warming levels or defaults.

    The function prioritizes location parameters in the following order:
    1. cached_area
    2. latitude/longitude
    3. stations
    Each parameter will override the lower-priority ones if provided. So if cached_area
    is given, lat/lon and stations are ignored. If lat/lon are given, stations are
    ignored. If stations are given, they are used only if neither cached_area nor lat/lon
    are provided.

    If no location parameters are provided, a warning is issued about retrieving the
    entire CA dataset.
    """
    no_delta = kwargs.pop("no_delta", False)
    # Define allowed inputs with types and defaults
    # Compute units default separately to avoid runtime evaluation in dictionary
    units_default = (
        "degF"  # Default to degF if user hasn't specified both variable and units
        if kwargs.get("variable", None) is None and kwargs.get("units", None) is None
        else None  # otherwise default to None and let get_data decide
    )

    ALLOWED_INPUTS = {
        "variable": (str, "Air Temperature at 2m"),
        "resolution": (str, "3 km"),
        "approach": (str, "Warming Level"),
        "centered_year": (int, None),
        "time_profile_scenario": (str, "SSP 3-7.0"),
        "warming_level": (list, [1.2]),
        "warming_level_window": (int, None),
        "cached_area": ((str, list), None),
        "latitude": ((float, tuple), None),
        "longitude": ((float, tuple), None),
        "stations": (list, None),
        "units": (str, units_default),
        "bias_adjusted_models": (bool, False),
    }

    # if the user does not enter warming level the analysis is a moot point
    # because the historical data is always at 1.2C
    REQUIRED_INPUTS = []
    for req in REQUIRED_INPUTS:
        if req not in kwargs:
            raise ValueError(f"Missing required input: '{req}'")

    # Validate input keys
    invalid_keys = set(kwargs.keys()) - set(ALLOWED_INPUTS.keys())
    if invalid_keys:
        raise ValueError(
            f"Invalid input(s): {list(invalid_keys)}. "
            f"Allowed inputs are: {list(ALLOWED_INPUTS.keys())}"
        )

    # Validate input types
    for key, value in kwargs.items():
        expected_type, _ = ALLOWED_INPUTS[key]
        # Handle union types (tuples of types)
        if isinstance(expected_type, tuple):
            if not isinstance(value, expected_type):
                type_names = [t.__name__ for t in expected_type]
                raise TypeError(
                    f"Parameter '{key}' must be of type {' or '.join(type_names)}, "
                    f"got {type(value).__name__}"
                )
        else:
            if not isinstance(value, expected_type):
                raise TypeError(
                    f"Parameter '{key}' must be of type {expected_type.__name__}, "
                    f"got {type(value).__name__}"
                )
        # check that warming_level_window is between 5 and 25
        if key == "warming_level_window":
            if value not in range(5, 26):
                raise ValueError(
                    f"Parameter '{key}' must be an integer between 5 and 25, "
                    f"got {value}"
                )
        # check that time_profile_scenario is within ["SSP 3-7.0", "SSP 2-4.5","SSP 5-8.5"]
        if key == "time_profile_scenario":
            if value not in ["SSP 3-7.0", "SSP 2-4.5", "SSP 5-8.5"]:
                raise ValueError(
                    f"Parameter '{key}' must be 'SSP 3-7.0', 'SSP 2-4.5', or 'SSP 5-8.5', "
                    f"received {value}."
                )

    # Validate and update approach parameters
    kwargs = _handle_approach_params(**kwargs)

    # Validate location parameters
    # the bahavior will be to use cached_area if provided
    # otherwise use lat/lon if provided
    # otherwise use stations if provided
    location_params = ["cached_area", "latitude", "longitude", "stations"]
    provided_location_params = [
        key for key in location_params if kwargs.get(key) is not None
    ]

    if "cached_area" in provided_location_params:
        # If cached_area is provided, unset lat/lon and stations
        if "latitude" in kwargs or "longitude" in kwargs:
            kwargs.pop("latitude", None)
            kwargs.pop("longitude", None)
            print("   ⚠️  Note: Using cached_area, ignoring provided latitude/longitude")
        if "stations" in kwargs:
            kwargs.pop("stations", None)
            print("   ⚠️  Note: Using cached_area, ignoring provided stations")
    elif (
        "latitude" in provided_location_params
        or "longitude" in provided_location_params
    ):
        # If lat/lon provided, unset stations
        if "stations" in kwargs:
            kwargs.pop("stations", None)
            print("   ⚠️  Note: Using latitude/longitude, ignoring provided stations")
    elif "stations" in provided_location_params:
        # Stations provided - convert to lat/lon with buffer
        stations = kwargs.pop("stations")
        buffer = _get_buffer_from_resolution(kwargs.get("resolution", "3 km"))
        print(
            f"   📍 Converting {len(stations)} station(s) to lat/lon coordinates with ±{buffer:.2f}° buffer"
        )
        try:
            lat_bounds, lon_bounds = _convert_stations_to_lat_lon(
                stations, buffer=buffer
            )
            kwargs["latitude"] = lat_bounds
            kwargs["longitude"] = lon_bounds
            print(f"      Latitude range: {lat_bounds[0]:.4f} to {lat_bounds[1]:.4f}")
            print(f"      Longitude range: {lon_bounds[0]:.4f} to {lon_bounds[1]:.4f}")
        except ValueError as e:
            raise ValueError(f"Error converting stations to coordinates: {e}")
    else:
        # No location parameters provided - warn about entire CA dataset
        print(
            "   ⚠️  WARNING: No location parameter provided (cached_area, latitude/longitude, or stations)"
        )
        print(
            "      The entire California dataset will be retrieved, which may be very large and slow."
        )
        print(
            "      Consider specifying a cached_area, lat/lon bounds, or specific stations for better performance."
        )

    # Set default parameters for data retrieval
    # Note: if stations were provided, they've been converted to lat/lon above
    get_data_params = {
        "variable": kwargs.get("variable", "Air Temperature at 2m"),
        "resolution": kwargs.get("resolution", "3 km"),
        "downscaling_method": "Dynamical",  # must be WRF, cannot be LOCA
        "timescale": "hourly",  # must be hourly for 8760 analysis
        "area_average": "Yes",
        "units": kwargs.get(
            "units",
            (
                "degF"  # Default to degF if user hasn't specified both variable and units
                if kwargs.get("variable", None) is None
                and kwargs.get("units", None) is None
                else None  # otherwise default to None and let get_data decide
            ),
        ),
        "approach": "Warming Level",
        "warming_level": [1.2],  # Historic global warming level
        "warming_level_window": kwargs.get(
            "warming_level_window", 15
        ),  # Use user input warming level window, if provided. Otherwise, default to 15.
        "cached_area": kwargs.get("cached_area", None),
        "latitude": kwargs.get("latitude", None),
        "longitude": kwargs.get("longitude", None),
    }

    warming_level = kwargs.get("warming_level", [1.2])
    print(
        f"Step 2: Data is being retrieved for warming level {warming_level} across all available simulations and scenarios."
    )

    historic_data = None
    if not no_delta:
        # Retrieve historical data at 1.2°C warming level
        historic_data = get_data(**get_data_params)

    # Update with any user-provided parameters for future data retrieval
    get_data_params.update(kwargs)
    future_data = get_data(**get_data_params)

    # Filter for only bias-adjusted WRF models, if user indicates this
    ba_models = kwargs.get("bias_adjusted_models", False)
    if ba_models:
        print("Filtering data for bias-adjusted models.")
        future_data = _filter_ba_models(future_data)
        if historic_data is not None:
            historic_data = _filter_ba_models(historic_data)
    else:
        None

    print(
        f"Step 3: The climate profile will now be produced at warming level {warming_level}."
    )

    return historic_data, future_data


def get_climate_profile(**kwargs: Dict[str, Any]) -> pd.DataFrame:
    """
    High-level function to compute standard year climate profiles using warming level data.

    This function retrieves climate data and computes standard year
    profiles using the 8760 analysis approach. It combines data retrieval
    and profile computation in a single call.

    Parameters
    ----------
    **kwargs : dict
        Keyword arguments for data selection. Allowed keys:
        - variable (Optional) : str, default "Air Temperature at 2m"
        - resolution (Optional) : str, default "3 km"
        - approach (Optional) : str, "Warming Level" or "Time", default "Warming Level"
        - centered_year (Optional) : int
        - time_profile_scenario (Optional) : str, default "SSP 3-7.0"
        - warming_level (Required) : List[float], default [1.2]
        - warming_level_window (Optional): int in range [5,25], default 15
        - location (Optional) : str, List[str|float|int], or Tuple[str|float|int]
        - units (Optional) : str, default "degF"
        - q (Optional) : float | list[float], default 0.5, quantile for profile calculation
        - no_delta (optional) : bool, default False, if True, do not apply baseline subtraction, return raw future profile
        - bias_adjusted_models (optional) : bool, default False, if True only return bias-adjusted WRF models

    Returns
    -------
    pd.DataFrame
        Standard year table for each warming level, with days of year as
        the index and hour of day as the columns. If multiple warming levels exist,
        they will be included as additional column levels. Units and metadata are
        preserved in the DataFrame's attrs dictionary.

    Examples
    --------
    >>> profile = get_climate_profile(
    ...     variable="Air Temperature at 2m",
    ...     warming_level=[1.5, 2.0, 3.0],
    ...     units="degF"
    ... )

    >>> profile = get_climate_profile(warming_level=[2.0])
    """
    # Extract parameters for compute_profile
    q = kwargs.pop("q", 0.5)
    no_delta = kwargs.get("no_delta", False)

    # Parse the location parameter
    kwargs = _handle_location_params(**kwargs)

    # Retrieve the climate data
    print("📊 Retrieving climate data...")
    with tqdm(
        total=2 if not no_delta else 1, desc="Data retrieval", unit="dataset"
    ) as pbar:
        historic_data, future_data = retrieve_profile_data(**kwargs)
        pbar.update(2)

    # Notify users of default values being used in the absence of input parameters
    # relevant for warming_level_window and warming_level
    defaults = {
        "warming_level": [1.2],
        "warming_level_window": 15,
        "approach": "Warming Level",
        "variable": "Air Temperature at 2m",
        "q": 0.5,
        "resolution": "3 km",
        "units": "degF",
    }
    for key, default_val in defaults.items():
        if key in kwargs:
            # skip this key
            continue
        if key == "warming_level":
            # if approach=Time, then default warming level is not used
            if kwargs.get("approach") == "Time":
                continue
            else:
                print(f"Using default '{key}': {default_val}")
                kwargs[key] = default_val
        else:
            if key == "q" and q != 0.5:
                continue
            else:
                print(f"Using default '{key}': {default_val}")
                kwargs[key] = default_val

    # catch invalid selections that return None
    if future_data is None and historic_data is None:
        raise ValueError(
            "No data returned for either historical or future datasets.\nPlease review your data selection parameters."
        )

    # Call compute_profile with the processed data
    # Compute profiles for both historical and future data
    def _fetch_primary_data_variable(ds: xr.Dataset) -> xr.DataArray:
        """
        Helper to extract the primary data variable from a Dataset.

        Parameters
        ----------
        ds : xr.Dataset
            The xarray Dataset from which to extract the primary data variable.

        Returns
        -------
        xr.DataArray
            The primary data variable as a DataArray.
        """
        if not isinstance(ds, xr.Dataset):
            return ds
        var_name = list(ds.data_vars.keys())[0]
        return ds[var_name]

    future_profile_data = _fetch_primary_data_variable(future_data)
    historic_profile_data = _fetch_primary_data_variable(historic_data)

    # Compute profiles for both datasets
    print("⚙️  Computing climate profiles...")

    future_profile = compute_profile(future_profile_data, q=q)
    if no_delta:
        historic_profile = None
    else:
        historic_profile = compute_profile(historic_profile_data, q=q)

    if no_delta:
        print("   ✓ No baseline subtraction requested, returning raw future profile")
        return future_profile

    # Compute the difference profile
    difference_profile = _compute_difference_profile(future_profile, historic_profile)

    print(
        f"✅ Climate profile computation complete! Final shape: {difference_profile.shape}"
    )
    print(
        f"   (Days: {difference_profile.shape[0]}, Hours/Columns: {difference_profile.shape[1]})"
    )

    return difference_profile


def _compute_difference_profile(
    future_profile: pd.DataFrame, historic_profile: pd.DataFrame
) -> pd.DataFrame:
    """
    Compute the difference between future and historic climate profiles.

    Parameters
    ----------
    future_profile : pd.DataFrame
        Future climate profile DataFrame
    historic_profile : pd.DataFrame
        Historic climate profile DataFrame

    Returns
    -------
    pd.DataFrame
        Difference profile (future - historic)
    """
    future_has_multiindex = isinstance(future_profile.columns, pd.MultiIndex)
    historic_has_multiindex = isinstance(historic_profile.columns, pd.MultiIndex)
    if (
        future_has_multiindex == historic_has_multiindex
    ):  # either both contain a MultiIndex, or both do not
        return _compute_paired_difference(future_profile, historic_profile)
    else:  # multiple warming levels in future profile, while historic profile always contains one warming level - 1.2
        # add MultiIndex to historic profile, then perform paired difference, as done above
        historic_profile_reformatted = historic_profile.copy()
        historic_profile_reformatted.columns = pd.MultiIndex.from_arrays(
            [
                ["1.2"] * len(historic_profile_reformatted.columns),
                historic_profile_reformatted.columns,
            ],
            names=["Warming_Level", "Simulation"],
        )
        return _compute_paired_difference(future_profile, historic_profile_reformatted)


def _compute_paired_difference(
    future_profile: pd.DataFrame,
    historic_profile: pd.DataFrame,
) -> pd.DataFrame:
    """
    Compute difference for profiles with matching simulations.

    Parameters
    ----------
    future_profile : pd.DataFrame
        Future profile with Simulation level
    historic_profile : pd.DataFrame
        Historic profile with Simulation level

    Returns
    -------
    pd.DataFrame
        Difference profile with paired simulations
    """
    # Check for duplicate columns and handle them
    if not future_profile.columns.is_unique:
        print(
            "   ⚠️  Warning: Found duplicate columns in future profile. Removing duplicates."
        )
        future_profile = future_profile.loc[:, ~future_profile.columns.duplicated()]

    if not historic_profile.columns.is_unique:
        print(
            "   ⚠️  Warning: Found duplicate columns in historic profile. Removing duplicates."
        )
        historic_profile = historic_profile.loc[
            :, ~historic_profile.columns.duplicated()
        ]

    difference_profile = future_profile.copy()

    # Get unique simulations from both profiles
    future_sims = future_profile.columns.unique()
    historic_sims = historic_profile.columns.unique()

    # Find common simulations
    common_sims = set(future_sims) & set(historic_sims)

    historic_mean = historic_profile.T.mean()

    if not common_sims:
        print(
            "   ⚠️  Warning: No matching simulations found between future and historic profiles!"
        )
        print(f"      Future simulations: {list(future_sims)}")
        print(f"      Historic simulations: {list(historic_sims)}")
        # Fall back to using mean of historic
        # Note: axis parameter removed in pandas 2.2, use level-based groupby instead
        for col in future_profile.columns:
            difference_profile.loc[:, col] = future_profile[col] - historic_mean
    else:
        # Compute differences for matching simulations
        n_cols = len(future_profile.columns)
        with tqdm(
            total=n_cols, desc="   Computing paired differences", unit="column"
        ) as pbar:
            for col in future_profile.columns:
                # if the column in future_profile is also in historic_column
                if col in historic_profile.columns:
                    # find the difference between them
                    difference_profile.loc[:, col] = (
                        future_profile[col] - historic_profile[col]
                    )
                else:
                    # if not, subtract the per-hours historic means from the future_profile column
                    difference_profile.loc[:, col] = future_profile[col] - historic_mean
                pbar.update(1)

    return difference_profile


def compute_profile(data: xr.DataArray, q=0.5) -> pd.DataFrame:
    """
    Calculates the standard year climate profile for warming level data using 8760
    analysis.

    This function handles global warming levels approach using time_delta coordinate.
    Processes all 30 years of warming level data centered around the year a warming level
    is reached, computes the specified quantile for each hour of the year across all years,
    then selects the actual data value closest to that quantile (not interpolated),
    and returns a characteristic profile of 8760 hours (one year) for each warming level
    and simulation combination.

    Parameters
    ----------
    data : xr.DataArray
        Hourly base-line subtracted data for one variable with warming_level,
        time_delta, and simulation dimensions. Expected to contain ~30 years
        (262,800 hours) of data for each warming level and simulation.

    q : float, optional
        Quantile value for selecting representative values (0.0 to 1.0).
        Default is 0.5 (median).

    Returns
    -------
    pd.DataFrame
        Standard year table for each warming level and simulation,
        with days of year as the index and hour of day as the columns.
        Multi-index columns include Hour, Warming_Level, and Simulation dimensions.

    """
    # Check for simulation dimension
    has_simulation = "simulation" in data.dims
    if has_simulation:
        n_simulations = len(data.simulation)
        simulations = data.simulation.values
    else:
        n_simulations = 1
        simulations = [None]

    # Get all available time_delta data (all 30 years)
    hours_per_year = 8760
    total_hours = len(data.time_delta)
    n_years = total_hours // hours_per_year

    print(f"      📊 Processing {total_hours:,} hours ({n_years} years) of data")
    print(f"      🎯 Computing {q*100:.0f}th percentile for each hour of year")

    # Create hour-of-year coordinate for all data (cycling through 1-8760)
    hour_of_year_all = np.tile(np.arange(1, hours_per_year + 1), n_years)[:total_hours]
    data = data.assign_coords(hour_of_year=("time_delta", hour_of_year_all))

    warming_levels = data.warming_level.values

    # Create helper function to extract meaningful simulation labels
    def _get_simulation_label(sim: str | int | None, sim_idx: int) -> str:
        """Extract meaningful simulation label from simulation identifier."""
        if sim is None:
            return f"Sim_{sim_idx+1}"

        sim_str = str(sim)
        if "WRF_" in sim_str:
            # Parse simulation name format: WRF_GCM_params_scenario
            # Example: WRF_CESM2_r11i1p1f1_historical+ssp245
            parts = sim_str.split("_")
            if len(parts) >= 4:
                gcm = parts[1]  # e.g., CESM2, CNRM-ESM2-1
                params = parts[2]  # e.g., r11i1p1f1
                scenario = parts[3]  # e.g., historical+ssp245

                # Extract SSP from scenario (e.g., ssp245 from historical+ssp245)
                if "ssp" in scenario:
                    ssp_part = scenario.split("ssp")[-1]  # Get part after 'ssp'
                    ssp = f"ssp{ssp_part}"
                else:
                    ssp = "hist"  # fallback for historical-only

                return f"{gcm}-{params}-{ssp}"
            elif len(parts) >= 2:
                # Fallback for shorter format
                return f"{parts[1]}-{sim_idx+1}"
            else:
                return f"Sim_{sim_idx+1}"
        else:
            # Ensure uniqueness by adding index for non-WRF format
            base_name = sim_str.split("_")[0] if "_" in sim_str else sim_str
            return f"{base_name}-{sim_idx+1}"

    # Process all data using quantile computation across years
    print(
        f"      ⚙️ Computing quantiles for {len(warming_levels)} warming level(s) and {n_simulations} simulation(s)"
    )

    # Eagerly compute all dask data at once (one round-trip to scheduler)
    if hasattr(data.data, "chunks"):
        print("      📥 Loading data into memory...")
        from dask.diagnostics import ProgressBar

        with ProgressBar():
            data = data.compute()

    # Initialize storage for profiles
    profile_data = {}

    # Progress tracking
    total_combinations = len(warming_levels) * n_simulations
    with tqdm(
        total=total_combinations,
        desc="      Computing profiles",
        unit="combo",
        leave=False,
    ) as pbar:

        for wl_idx, wl in enumerate(warming_levels):
            for sim_idx, sim in enumerate(simulations):
                # Get simulation label
                sim_label = _get_simulation_label(sim, sim_idx)
                # Select data for this warming level and simulation combination
                if has_simulation:
                    subset_data = data.isel(warming_level=wl_idx, simulation=sim_idx)
                else:
                    subset_data = data.isel(warming_level=wl_idx)
                # Vectorized quantile computation using numpy
                # Reshape raw values into (n_years, hours_per_year) then compute
                # the quantile across years for each hour-of-year position
                values = subset_data.values
                n_total = len(values)
                usable = (n_total // hours_per_year) * hours_per_year
                year_hour_matrix = values[:usable].reshape(-1, hours_per_year)

                # Compute quantile targets for each of the 8760 hour positions
                quantile_targets = np.nanquantile(
                    year_hour_matrix, q, axis=0
                )  # shape: (8760,)

                # For each hour position, find the actual year whose value is
                # closest to the quantile (avoids interpolation)
                diffs = np.abs(
                    year_hour_matrix - quantile_targets[np.newaxis, :]
                )  # (n_years, 8760)
                closest_year_idx = np.nanargmin(diffs, axis=0)  # (8760,)
                profile_1d = year_hour_matrix[
                    closest_year_idx, np.arange(hours_per_year)
                ]

                # Store the profile
                key = (f"WL_{wl}", sim_label)
                profile_data[key] = profile_1d

                pbar.update(1)

    df_profile = _construct_profile_dataframe(
        profile_data=profile_data,
        warming_levels=warming_levels,
        simulations=simulations,
        sim_label_func=_get_simulation_label,
        hours_per_year=hours_per_year,
    )

    # Prepare metadata dictionary
    metadata = {
        "quantile": q,
        "method": "8760 analysis - actual data closest to quantile across 30 years",
        "description": f"Climate profile computed using actual data values closest to the {q*100:.0f}th percentile of hourly data",
    }

    # Add original data attributes if available
    if hasattr(data, "attrs"):
        if "units" in data.attrs:
            metadata["units"] = data.attrs["units"]
        if "extended_description" in data.attrs:
            metadata["extended_description"] = data.attrs["extended_description"]
        if "variable_id" in data.attrs:
            metadata["variable_name"] = data.attrs["variable_id"]
        elif hasattr(data, "name") and data.name:
            metadata["variable_name"] = data.name

    # Set all metadata using the helper function
    set_profile_metadata(df_profile, metadata)

    print(f"      ✅ Profile computation complete! Final shape: {df_profile.shape}")
    print(
        f"         With index: {df_profile.index.name}, columns: {df_profile.columns.names}"
    )
    if hasattr(data, "attrs") and "units" in data.attrs:
        print(f"         Units: {data.attrs['units']}")

    return df_profile


def _construct_profile_dataframe(
    profile_data: dict,
    warming_levels: np.ndarray,
    simulations: list,
    sim_label_func: callable,
    hours_per_year: int,
) -> pd.DataFrame:
    """
    Construct a DataFrame from profile data based on warming level and simulation dimensions.

    Parameters
    ----------
    profile_data : dict
        Dictionary with (warming_level, simulation) keys and profile arrays as values
    warming_levels : np.ndarray
        Array of warming level values
    simulations : list
        List of simulation identifiers
    sim_label_func : callable
        Function to extract simulation labels
    hours_per_year : int
        Hours per year (8760)

    Returns
    -------
    pd.DataFrame
        Structured DataFrame with appropriate column structure based on dimensions
    """

    n_warming_levels = len(warming_levels)
    n_simulations = len(simulations)

    if n_warming_levels == 1 and n_simulations == 1:
        return _create_simple_dataframe(
            profile_data,
            warming_levels[0],
            simulations[0],
            sim_label_func,
            hours_per_year,
        )
    elif n_warming_levels == 1 and n_simulations > 1:

        return _create_single_wl_multi_sim_dataframe(
            profile_data, warming_levels[0], simulations, sim_label_func, hours_per_year
        )

    elif n_warming_levels > 1 and n_simulations == 1:
        return _create_multi_wl_single_sim_dataframe(
            profile_data, warming_levels, simulations[0], sim_label_func, hours_per_year
        )
    else:
        return _create_multi_wl_multi_sim_dataframe(
            profile_data, warming_levels, simulations, sim_label_func, hours_per_year
        )


def _create_simple_dataframe(
    profile_data: dict,
    warming_level: float,
    simulation: any,
    sim_label_func: callable,
    hours_per_year: int,
) -> pd.DataFrame:
    """
    Create a simple DataFrame for single warming level and single simulation.

    Parameters
    ----------
    profile_data : dict
        Profile data dictionary
    warming_level : float
        Single warming level value
    simulation : any
        Single simulation identifier
    sim_label_func : callable
        Function to get simulation label
    hours_per_year : int
        Hours per year (8760)

    Returns
    -------
    pd.DataFrame
        Simple DataFrame with a single simulation column
    """

    wl_key = warming_level
    sim_key = sim_label_func(simulation, 0)

    # Stack data
    all_data = _stack_profile_data(
        profile_data=profile_data, wl_names=[f"WL_{wl_key}"], sim_names=[sim_key]
    )

    return pd.DataFrame(
        all_data,
        columns=[sim_key],
        index=np.arange(1, hours_per_year + 1, 1),
    )


def _create_single_wl_multi_sim_dataframe(
    profile_data: dict,
    warming_level: float,
    simulations: list,
    sim_label_func: callable,
    hours_per_year: int,
) -> pd.DataFrame:
    """
    Create DataFrame for single warming level with multiple simulations.

    Parameters
    ----------
    profile_data : dict
        Profile data dictionary
    warming_level : float
        Single warming level value
    simulations : list
        List of simulation identifiers
    sim_label_func : callable
        Function to get simulation labels
    hours_per_year : int
        Hours per year (8760)

    Returns
    -------
    pd.DataFrame
        DataFrame with Simulation columns
    """
    wl = warming_level
    sim_names = [sim_label_func(sim, i) for i, sim in enumerate(simulations)]

    # Ensure simulation names are unique
    if len(sim_names) != len(set(sim_names)):
        print(
            "   ⚠️  Warning: Duplicate simulation names detected, adding uniqueness suffixes"
        )
        unique_sim_names = []
        name_counts = {}
        for name in sim_names:
            if name not in name_counts:
                name_counts[name] = 0
                unique_sim_names.append(name)
            else:
                name_counts[name] += 1
                unique_sim_names.append(f"{name}_v{name_counts[name]}")
        sim_names = unique_sim_names

    # Stack data
    all_data = _stack_profile_data(
        profile_data=profile_data, wl_names=[f"WL_{wl}"], sim_names=sim_names
    )

    return pd.DataFrame(
        all_data,
        columns=sim_names,
        index=np.arange(1, hours_per_year + 1, 1),
    )


def _create_multi_wl_single_sim_dataframe(
    profile_data: dict,
    warming_levels: np.ndarray,
    simulation: any,
    sim_label_func: callable,
    hours_per_year: int,
) -> pd.DataFrame:
    """
    Create DataFrame for multiple warming levels with single simulation.

    Parameters
    ----------
    profile_data : dict
        Profile data dictionary
    warming_levels : np.ndarray
        Array of warming level values
    simulation : any
        Single simulation identifier
    sim_label_func : callable
        Function to get simulation label
    hours_per_year : int
        Hours per year (8760)

    Returns
    -------
    pd.DataFrame
        DataFrame with Warming_Level columns
    """

    sim_name = sim_label_func(simulation, 0)
    wl_names = [f"WL_{wl}" for wl in warming_levels]

    # Create MultiIndex columns
    col_tuples = [(wl_name, sim) for wl_name in wl_names for sim in [sim_name]]

    multi_cols = pd.MultiIndex.from_tuples(
        col_tuples, names=["Warming_Level", "Simulation"]
    )

    # Stack data
    all_data = _stack_profile_data(
        profile_data=profile_data, wl_names=wl_names, sim_names=[sim_name]
    )

    return pd.DataFrame(
        all_data,
        columns=multi_cols,
        index=np.arange(1, hours_per_year + 1, 1),
    )


def _create_multi_wl_multi_sim_dataframe(
    profile_data: dict,
    warming_levels: np.ndarray,
    simulations: list,
    sim_label_func: callable,
    hours_per_year: int,
) -> pd.DataFrame:
    """
    Create DataFrame for multiple warming levels and multiple simulations.

    Parameters
    ----------
    profile_data : dict
        Profile data dictionary
    warming_levels : np.ndarray
        Array of warming level values
    simulations : list
        List of simulation identifiers
    sim_label_func : callable
        Function to get simulation labels
    hours_per_year : int
        Hours per year (8760)

    Returns
    -------
    pd.DataFrame
        DataFrame with (Warming_Level, Simulation) MultiIndex columns
    """

    wl_names = [f"WL_{wl}" for wl in warming_levels]
    sim_names = [sim_label_func(sim, i) for i, sim in enumerate(simulations)]

    # Ensure simulation names are unique
    if len(sim_names) != len(set(sim_names)):
        print(
            "   ⚠️  Warning: Duplicate simulation names detected, adding uniqueness suffixes"
        )
        unique_sim_names = []
        name_counts = {}
        for name in sim_names:
            if name not in name_counts:
                name_counts[name] = 0
                unique_sim_names.append(name)
            else:
                name_counts[name] += 1
                unique_sim_names.append(f"{name}_v{name_counts[name]}")
        sim_names = unique_sim_names

    # Create MultiIndex columns
    col_tuples = [(wl_name, sim_name) for wl_name in wl_names for sim_name in sim_names]
    multi_cols = pd.MultiIndex.from_tuples(
        col_tuples, names=["Warming_Level", "Simulation"]
    )

    # Stack data
    all_data = _stack_profile_data(
        profile_data=profile_data, wl_names=wl_names, sim_names=sim_names
    )

    return pd.DataFrame(
        all_data,
        columns=multi_cols,
        index=np.arange(1, hours_per_year + 1, 1),
    )


def _stack_profile_data(
    profile_data: dict,
    wl_names: list,
    sim_names: list,
) -> np.ndarray:
    """
    Stack profile data into a single array for DataFrame construction.

    Parameters
    ----------
    profile_data : dict
        Dictionary with (wl_name, sim_name) keys and profile arrays as values
    wl_names : list
        List of warming level names
    sim_names : list
        List of simulation names

    Returns
    -------
    np.ndarray
        Stacked data array ready for DataFrame construction
    """
    all_data = []

    for wl_name in wl_names:
        for sim_name in sim_names:
            key = (wl_name, sim_name)
            all_data.append(profile_data[key])

    return np.column_stack(all_data)


def get_profile_units(profile_df: pd.DataFrame) -> str:
    """
    Extract units information from a climate profile DataFrame.

    Parameters
    ----------
    profile_df : pd.DataFrame
        Climate profile DataFrame with units stored in attrs

    Returns
    -------
    str
        Units string, or 'Unknown' if not found

    Examples
    --------
    >>> profile = get_climate_profile(variable="Air Temperature at 2m", warming_level=[2.0])
    >>> units = get_profile_units(profile)
    >>> print(f"Temperature units: {units}")
    """
    return profile_df.attrs.get("units", "Unknown")


def get_profile_metadata(profile_df: pd.DataFrame) -> dict:
    """
    Extract all metadata from a climate profile DataFrame.

    Parameters
    ----------
    profile_df : pd.DataFrame
        Climate profile DataFrame with metadata stored in attrs

    Returns
    -------
    dict
        Dictionary containing all available metadata

    Examples
    --------
    >>> profile = get_climate_profile(variable="Air Temperature at 2m", warming_level=[2.0])
    >>> metadata = get_profile_metadata(profile)
    >>> print(f"Variable: {metadata.get('variable_name')}")
    >>> print(f"Units: {metadata.get('units')}")
    >>> print(f"Method: {metadata.get('method')}")
    """
    return dict(profile_df.attrs)


def set_profile_metadata(profile_df: pd.DataFrame, metadata: dict) -> None:
    """
    Set or update metadata in a climate profile DataFrame.

    Parameters
    ----------
    profile_df : pd.DataFrame
        Climate profile DataFrame to update
    metadata : dict
        Dictionary containing metadata key-value pairs to set

    Returns
    -------
    None
        The function modifies the DataFrame in place

    Examples
    --------
    >>> profile = get_climate_profile(variable="Air Temperature at 2m", warming_level=[2.0])
    >>> new_metadata = {
    ...     "source": "Custom Dataset",
    ...     "author": "Jane Doe",
    ...     "notes": "This profile was generated for testing purposes."
    ... }
    >>> set_profile_metadata(profile, new_metadata)
    >>> print(profile.attrs)
    """
    if not isinstance(metadata, dict):
        raise ValueError("Metadata must be provided as a dictionary.")

    for key, value in metadata.items():
        profile_df.attrs[key] = value
