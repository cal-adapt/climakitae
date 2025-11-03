"""
Calculates the Standard Year Climate Profiles using a warming level approach and designated
quantiles. The historical baseline for relative profile computation is a warming level of 1.2 C.
User specified warming level will be calculated relative to this baseline unless the "no_delta" option
is set to True, in which case the raw profile(s) for the requested warming level(s) will
be returned.
"""

from typing import Tuple

import numpy as np
import pandas as pd
import xarray as xr
from tqdm.auto import tqdm  # Progress bar

from climakitae.core.constants import UNSET
from climakitae.core.data_interface import DataInterface, get_data
from climakitae.core.paths import VARIABLE_DESCRIPTIONS_CSV_PATH
from climakitae.explore.typical_meteorological_year import match_str_to_wl, is_HadISD
from climakitae.util.utils import julianDay_to_date, read_csv_file

xr.set_options(keep_attrs=True)  # Keep attributes when mutating xr objects


def _get_station_coordinates(station_name: str) -> Tuple[float, float]:
    """
    Look up the latitude and longitude coordinates for a given station name.

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
        If the station name is not found in the DataInterface.

    Examples
    --------
    >>> lat, lon = _get_station_coordinates("San Diego Lindbergh Field (KSAN)")
    >>> print(f"Latitude: {lat}, Longitude: {lon}")
    """
    data_interface = DataInterface()
    stations_gdf = data_interface.stations_gdf

    # Look up the station in the GeoDataFrame
    station_row = stations_gdf[stations_gdf["station"] == station_name]

    if station_row.empty:
        raise ValueError(
            f"Station '{station_name}' not found in the DataInterface. "
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
    var_id: str, q: float, location: str, gwl: float, no_delta: bool
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
    no_delta : bool
        no_delta value used to generate profile

    Returns
    -------
    str
        A cleaned up file name
    """

    # clean arguments for filenaming
    clean_loc_name = location.replace(" ", "_").replace("(", "").replace(")", "")
    clean_q_name = f"{q:.2f}".split(".")[1].lower()
    clean_var_name = var_id.lower()
    clean_gwl_name = match_str_to_wl(gwl).lower().replace(".", "pt")
    if no_delta:
        delta_str = ""
    else:
        delta_str = "_delta_from_historical"

    filename = f"stdyr_{clean_var_name}_{clean_q_name}ptile_{clean_loc_name}_{clean_gwl_name}{delta_str}.csv"
    return filename


def export_profile_to_csv(
    profile: pd.DataFrame,
    variable: str,
    q: float,
    global_warming_levels: list[float],
    station_name: str = UNSET,
    cached_area: str = UNSET,
    latitude: float | int = UNSET,
    longitude: float | int = UNSET,
    no_delta: bool = False,
):
    """Export profile to csv file with a descriptive file name.

    Each warming level is saved in a separate file.

    Parameters
    ----------
    profile : pd.DataFrame
        Standard year profile with MultiIndex columns
    variable : str
        Name of variable used in profile
    q : float
        Percentile used in profile
    global_warming_levels : list[float]
        List of global warming levels in profile
    latitude : float | int, optional
        Latitude coordinate from profile location
    longitude : float | int, optional
        Longitude coordinate from profile location
    station_name : str, optional
        Name of HadISD station or custom location used in profile
    cached_area : str, optional
        Name of cached area used in profile
    coord_name : str, optional
        Name of location, used with latitude and longitude
    no_delta : bool, optional
        True if no_delta=True when generating profile

    Notes
    -----
    `station_name` can be used with `latitude` and `longitude` as long as
    `station_name` is not set to a HadISD station.

    If `cached_area` is set along with `latitude` and `longitude`,
    `latitude` and `longitude` take priority and `cached_area` will be dropped.

    """

    # Get variable id string to use in file name
    variable_descriptions = read_csv_file(VARIABLE_DESCRIPTIONS_CSV_PATH)
    var_id = variable_descriptions[
        (variable_descriptions["display_name"] == variable)
        & (variable_descriptions["timescale"] == "hourly")
    ]["variable_id"].item()

    # Get location string based on combination of
    # location variables
    location_str = ""
    match station_name, cached_area, latitude, longitude:
        # Station name and lat/lon
        case str(), object(), float() | int(), float() | int():
            if is_HadISD(station_name):
                raise ValueError(
                    "Do not set `latitude` and `longitude` when using a HadISD station for `station_name`. Change `station_name` value if using custom location."
                )
            lat_str = str(round(latitude, 6)).replace(".", "-")
            lon_str = str(round(abs(longitude), 6)).replace(".", "-")
            location_str = f"{station_name.lower()}_{lat_str}N_{lon_str}W"
        # Lat lon only
        case object(), object(), float() | int(), float() | int():
            lat_str = str(round(latitude, 6)).replace(".", "-")
            lon_str = str(round(abs(longitude), 6)).replace(".", "-")
            location_str = f"{lat_str}N_{lon_str}W"
        # Only station name
        case str(), object(), object(), object():
            location_str = station_name.lower()
        # Only cached area
        case object(), str(), object(), object():
            location_str = cached_area.lower()
        # Something else
        case _:
            raise TypeError(
                "Location must be provided as either `station_name` or `cached_area` or `latitude` plus `longitude`."
            )

    # Check profile MultiIndex to pull out data by Global Warming Level
    match profile.keys().nlevels:
        case 2:  # Single WL
            gwl = global_warming_levels[0]
            filename = _get_clean_standardyr_filename(
                var_id, q, location_str, gwl, no_delta
            )
            profile.to_csv(filename)
        case 3:  # Multiple WL (WL included in MultiIndex)
            for gwl in global_warming_levels:  # Single file per WL
                filename = _get_clean_standardyr_filename(
                    var_id, q, location_str, gwl, no_delta
                )
                profile.xs(f"WL_{gwl}", level="Warming_Level", axis=1).to_csv(filename)
        case _:
            raise ValueError(
                f"Profile MultiIndex should have two or three levels. Found {profile.keys().nlevels} levels."
            )


def retrieve_profile_data(**kwargs: any) -> Tuple[xr.Dataset, xr.Dataset]:
    """
    Backend function for retrieving data needed for computing climate profiles.

    Reads in the full hourly data for the 8760 analysis, including all warming levels.

    Parameters
    ----------
    **kwargs : dict
        Keyword arguments for data selection. Allowed keys:
        - variable (Optional) : str, default "Air Temperature at 2m"
        - resolution (Optional) : str, default "3 km"
        - warming_levels (Optional) : List[float], default [1.2]
        - cached_area (Optional) : str or List[str]
        - latitude (Optional) : float or tuple
        - longitude (Optional) : float or tuple
        - stations (Optional) : list[str], default None
        - units (Optional) : str, default "degF"
        - no_delta (optional) : bool, default False, if True, do not retrieve historical data, return raw future profile

    Returns
    -------
    Tuple[xr.Dataset, xr.Dataset]
        (historic_data, future_data) - Historical data at 1.2°C warming,
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
    ...     scenario=["SSP 3-7.0"],
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
        "warming_level": (list, [1.2]),
        "cached_area": ((str, list), None),
        "latitude": ((float, tuple), None),
        "longitude": ((float, tuple), None),
        "stations": (list, None),
        "units": (str, units_default),
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
        print(
            f"   📍 Converting {len(stations)} station(s) to lat/lon coordinates with ±0.02° buffer"
        )
        try:
            lat_bounds, lon_bounds = _convert_stations_to_lat_lon(stations, buffer=0.02)
            kwargs["latitude"] = lat_bounds
            kwargs["longitude"] = lon_bounds
            print(f"      Latitude range: {lat_bounds[0]:.4f} to {lat_bounds[1]:.4f}")
            print(f"      Longitude range: {lon_bounds[0]:.4f} to {lon_bounds[1]:.4f}")
        except ValueError as e:
            raise ValueError(f"Error converting stations to coordinates: {e}")
    else:
        # No location parameters provided - warn about entire CA dataset
        print(
            "   ⚠️  WARNING: No location parameters provided (cached_area, latitude/longitude, or stations)"
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
        "warming_level": [1.2],
        "cached_area": kwargs.get("cached_area", None),
        "latitude": kwargs.get("latitude", None),
        "longitude": kwargs.get("longitude", None),
    }

    historic_data = None
    if not no_delta:
        # Retrieve historical data at 1.2°C warming level
        historic_data = get_data(**get_data_params)

    # Update with any user-provided parameters for future data retrieval
    get_data_params.update(kwargs)
    future_data = get_data(**get_data_params)

    return historic_data, future_data


def get_climate_profile(**kwargs) -> pd.DataFrame:
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
        - warming_level (Required) : List[float], default [1.2]
        - cached_area (Optional) : str or List[str]
        - units (Optional) : str, default "degF"
        - latitude (Optional) : float or tuple
        - longitude (Optional) : float or tuple
        - stations (Optional) : list[str], default None
        - days_in_year (Optional) : int, default 365
        - q (Optional) : float | list[float], default 0.5, quantile for profile calculation
        - no_delta (optional) : bool, default False, if True, do not apply baseline subtraction, return raw future profile

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
    days_in_year = kwargs.pop("days_in_year", 365)
    q = kwargs.pop("q", 0.5)
    no_delta = kwargs.get("no_delta", False)

    # Retrieve the climate data
    print("📊 Retrieving climate data...")
    with tqdm(
        total=2 if not no_delta else 1, desc="Data retrieval", unit="dataset"
    ) as pbar:
        historic_data, future_data = retrieve_profile_data(**kwargs)
        pbar.update(2)

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

    future_profile = compute_profile(
        future_profile_data, days_in_year=days_in_year, q=q
    )
    if no_delta:
        historic_profile = None
    else:
        historic_profile = compute_profile(
            historic_profile_data, days_in_year=days_in_year, q=q
        )

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

    if future_has_multiindex and historic_has_multiindex:
        return _compute_multiindex_difference(future_profile, historic_profile)
    elif future_has_multiindex and not historic_has_multiindex:
        return _compute_mixed_index_difference(future_profile, historic_profile)
    else:
        return _compute_simple_difference(future_profile, historic_profile)


def _compute_multiindex_difference(
    future_profile: pd.DataFrame, historic_profile: pd.DataFrame
) -> pd.DataFrame:
    """
    Compute difference when both profiles have MultiIndex columns.

    Parameters
    ----------
    future_profile : pd.DataFrame
        Future profile with MultiIndex columns
    historic_profile : pd.DataFrame
        Historic profile with MultiIndex columns

    Returns
    -------
    pd.DataFrame
        Difference profile
    """
    future_levels = future_profile.columns.names
    historic_levels = historic_profile.columns.names

    if "Simulation" in future_levels and "Simulation" in historic_levels:
        return _compute_simulation_paired_difference(
            future_profile, historic_profile, future_levels, historic_levels
        )
    elif "Warming_Level" in future_levels and "Simulation" not in future_levels:
        return _compute_warming_level_difference(
            future_profile, historic_profile, future_levels, historic_levels
        )
    else:
        # Default to simple difference if structure is unexpected
        return _compute_simple_difference(future_profile, historic_profile)


def _compute_simulation_paired_difference(
    future_profile: pd.DataFrame,
    historic_profile: pd.DataFrame,
    future_levels: list,
    historic_levels: list,
) -> pd.DataFrame:
    """
    Compute difference for profiles with matching simulations.

    Parameters
    ----------
    future_profile : pd.DataFrame
        Future profile with Simulation level
    historic_profile : pd.DataFrame
        Historic profile with Simulation level
    future_levels : list
        Names of future profile column levels
    historic_levels : list
        Names of historic profile column levels

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
    future_sims = future_profile.columns.get_level_values("Simulation").unique()
    historic_sims = historic_profile.columns.get_level_values("Simulation").unique()

    # Find common simulations
    common_sims = set(future_sims) & set(historic_sims)

    if not common_sims:
        print(
            "   ⚠️  Warning: No matching simulations found between future and historic profiles!"
        )
        print(f"      Future simulations: {list(future_sims)}")
        print(f"      Historic simulations: {list(historic_sims)}")
        # Fall back to using mean of historic
        historic_mean = historic_profile.groupby(level="Hour", axis=1).mean()
        for col in future_profile.columns:
            hour = col[0] if "Hour" in future_levels else col[-1]
            difference_profile.loc[:, col] = future_profile[col] - historic_mean[hour]
    else:
        # Compute differences for matching simulations
        n_cols = len(future_profile.columns)
        with tqdm(
            total=n_cols, desc="   Computing paired differences", unit="column"
        ) as pbar:
            for col in future_profile.columns:
                historic_col = _find_matching_historic_column(
                    col, future_levels, historic_profile, historic_levels
                )
                if historic_col and historic_col in historic_profile.columns:
                    difference_profile.loc[:, col] = (
                        future_profile[col] - historic_profile[historic_col]
                    )
                else:
                    # Use mean of historic for that hour
                    hour = col[0]  # Assuming hour is first level
                    historic_hour_mean = _get_historic_hour_mean(
                        historic_profile, historic_levels, hour
                    )
                    difference_profile.loc[:, col] = (
                        future_profile[col] - historic_hour_mean
                    )
                pbar.update(1)

    return difference_profile


def _compute_warming_level_difference(
    future_profile: pd.DataFrame,
    historic_profile: pd.DataFrame,
    future_levels: list,
    historic_levels: list,
) -> pd.DataFrame:
    """
    Compute difference for profiles with warming levels but no simulations.

    Parameters
    ----------
    future_profile : pd.DataFrame
        Future profile with Warming_Level
    historic_profile : pd.DataFrame
        Historic profile
    future_levels : list
        Names of future profile column levels
    historic_levels : list
        Names of historic profile column levels

    Returns
    -------
    pd.DataFrame
        Difference profile
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

    n_cols = len(future_profile.columns)
    with tqdm(total=n_cols, desc="   Computing differences", unit="column") as pbar:
        for col in future_profile.columns:
            # Use future_levels to determine which position contains the hour
            if "Hour" in future_levels:
                hour_idx = future_levels.index("Hour")
                hour = col[hour_idx]
            else:
                # Fallback: assume first position is hour
                hour = col[0]

            if hour in historic_profile.columns:
                difference_profile.loc[:, col] = (
                    future_profile[col] - historic_profile[hour]
                )
            else:
                # Try to find corresponding hour in historic MultiIndex
                if historic_levels and "Hour" in historic_levels:
                    try:
                        historic_hour = historic_profile.xs(
                            hour, level="Hour", axis=1
                        ).iloc[:, 0]
                    except (KeyError, IndexError):
                        # If xs fails, fall back to first column
                        historic_hour = historic_profile.iloc[:, 0]
                else:
                    historic_hour = historic_profile.iloc[
                        :, 0
                    ]  # Fall back to first column
                difference_profile.loc[:, col] = future_profile[col] - historic_hour
            pbar.update(1)

    return difference_profile


def _compute_mixed_index_difference(
    future_profile: pd.DataFrame, historic_profile: pd.DataFrame
) -> pd.DataFrame:
    """
    Compute difference when future has MultiIndex and historic has simple index.

    Parameters
    ----------
    future_profile : pd.DataFrame
        Future profile with MultiIndex columns
    historic_profile : pd.DataFrame
        Historic profile with simple columns

    Returns
    -------
    pd.DataFrame
        Difference profile
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

    n_cols = len(future_profile.columns)
    with tqdm(total=n_cols, desc="   Computing differences", unit="column") as pbar:
        for col in future_profile.columns:
            historic_value = _find_matching_historic_value(
                col, future_profile, historic_profile
            )
            difference_profile.loc[:, col] = future_profile[col] - historic_value
            pbar.update(1)

    return difference_profile


def _compute_simple_difference(
    future_profile: pd.DataFrame, historic_profile: pd.DataFrame
) -> pd.DataFrame:
    """
    Compute difference for profiles with simple (non-MultiIndex) columns.

    Parameters
    ----------
    future_profile : pd.DataFrame
        Future profile
    historic_profile : pd.DataFrame
        Historic profile

    Returns
    -------
    pd.DataFrame
        Difference profile
    """
    if list(future_profile.columns) == list(historic_profile.columns):
        print("   ✓ Columns match - computing element-wise difference")
        return future_profile - historic_profile
    else:
        print("   ⚠️  Warning: Column mismatch between future and historic profiles")
        print(f"      Future columns: {list(future_profile.columns)[:5]}...")
        print(f"      Historic columns: {list(historic_profile.columns)[:5]}...")
        # Try to align by position
        min_cols = min(len(future_profile.columns), len(historic_profile.columns))
        return future_profile.iloc[:, :min_cols] - historic_profile.iloc[:, :min_cols]


def _find_matching_historic_column(
    future_col: tuple,
    future_levels: list,
    historic_profile: pd.DataFrame,
    historic_levels: list,
) -> tuple | None:
    """
    Find the matching historic column for a future column.

    Parameters
    ----------
    future_col : tuple
        Future column tuple
    future_levels : list
        Names of future column levels
    historic_profile : pd.DataFrame
        Historic profile DataFrame
    historic_levels : list
        Names of historic column levels

    Returns
    -------
    tuple | None
        Matching historic column or None
    """
    # Check that both have required levels for matching
    if "Hour" in future_levels and "Simulation" in future_levels:
        # Identify positions in future column
        hour_idx = future_levels.index("Hour")
        sim_idx = future_levels.index("Simulation")

        hour = future_col[hour_idx]
        sim = future_col[sim_idx]

        # Check historic structure using historic_levels
        if "Hour" in historic_levels and "Simulation" in historic_levels:
            # Build historic column tuple based on historic_levels structure
            hist_hour_idx = historic_levels.index("Hour")
            hist_sim_idx = historic_levels.index("Simulation")

            # Create tuple in the order of historic_levels
            if hist_hour_idx < hist_sim_idx:
                historic_col = (hour, sim)
            else:
                historic_col = (sim, hour)

            if historic_col in historic_profile.columns:
                return historic_col

    return None


def _get_historic_hour_mean(
    historic_profile: pd.DataFrame, historic_levels: list, hour: any
) -> pd.Series:
    """
    Get the mean of historic profile for a specific hour.

    Parameters
    ----------
    historic_profile : pd.DataFrame
        Historic profile DataFrame
    historic_levels : list
        Names of historic column levels
    hour : any
        Hour identifier

    Returns
    -------
    pd.Series
        Mean values for the specified hour
    """
    if "Simulation" in historic_levels:
        return historic_profile.xs(hour, level="Hour", axis=1).mean()
    else:
        return historic_profile[hour] if hour in historic_profile.columns else 0


def _find_matching_historic_value(
    future_col: tuple, future_profile: pd.DataFrame, historic_profile: pd.DataFrame
) -> pd.Series:
    """
    Find matching historic value for a future column with mixed index types.

    Parameters
    ----------
    future_col : tuple
        Future column identifier
    future_profile : pd.DataFrame
        Future profile DataFrame
    historic_profile : pd.DataFrame
        Historic profile DataFrame

    Returns
    -------
    pd.Series
        Matching historic values
    """
    if "Hour" in future_profile.columns.names:
        hour_idx = future_profile.columns.names.index("Hour")
        hour = future_col[hour_idx]

        # Try direct match
        if hour in historic_profile.columns:
            return historic_profile[hour]

        # Try numeric hour matching
        hour_num = (
            int(hour.replace("am", "").replace("pm", ""))
            if isinstance(hour, str)
            else hour
        )
        if hour_num in historic_profile.columns:
            return historic_profile[hour_num]

        # Fall back to positional matching
        col_position = future_profile.columns.get_loc(future_col)
        if isinstance(col_position, int):
            historic_col_idx = col_position % len(historic_profile.columns)
            return historic_profile.iloc[:, historic_col_idx]

    # No hour level, use positional matching
    col_position = future_profile.columns.get_loc(future_col)
    if isinstance(col_position, int):
        historic_col_idx = col_position % len(historic_profile.columns)
        return historic_profile.iloc[:, historic_col_idx]

    # Default fallback
    return historic_profile.iloc[:, 0]


def compute_profile(data: xr.DataArray, days_in_year: int = 365, q=0.5) -> pd.DataFrame:
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

    days_in_year : int, optional
        Either 366 or 365, depending on whether or not the year is a leap year.
        Default to 365 days

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
    hours_per_day = 24
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

                # Group by hour_of_year and find the actual data value closest to the quantile
                # This gives us the actual data point closest to the q-th quantile for each of the 8760 hours
                # Load data to avoid dask chunking issues with quantile
                if hasattr(subset_data.data, "chunks"):
                    # If it's a dask array, load it into memory
                    subset_data = subset_data.compute()

                def _closest_to_quantile(dat: xr.DataArray) -> xr.DataArray:
                    """Find the actual data value closest to the specified quantile."""
                    # Stack all dimensions except time_delta into a single dimension
                    stacked = dat.stack(all_dims=list(dat.dims))
                    # Compute the target quantile value
                    target_quantile = stacked.quantile(q, dim="all_dims")
                    # Find the index of the value closest to the quantile
                    closest_idx = abs(stacked - target_quantile).argmin(dim="all_dims")
                    # Return the actual data value at that index
                    return xr.DataArray(stacked.isel(all_dims=closest_idx).values)

                profile_1d = subset_data.groupby("hour_of_year").map(
                    _closest_to_quantile
                )

                # Reshape to (days_in_year, 24) for the final DataFrame
                profile_reshaped = profile_1d.values.reshape(
                    days_in_year, hours_per_day
                )

                # Store the profile
                key = (f"WL_{wl}", sim_label)
                profile_data[key] = profile_reshaped

                pbar.update(1)

    # Create the multi-index DataFrame structure
    df_profile = _construct_profile_dataframe(
        profile_data=profile_data,
        warming_levels=warming_levels,
        simulations=simulations,
        sim_label_func=_get_simulation_label,
        days_in_year=days_in_year,
        hours_per_day=hours_per_day,
    )

    # Determine which formatting function to use based on the structure
    _format_based_on_structure(df_profile)

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


def _format_based_on_structure(df: pd.DataFrame):
    """
    Format the DataFrame based on whether it has single-level or multi-level columns.

    Parameters
    ----------
    df : pd.DataFrame
        The DataFrame to format.

    Returns
    -------
    pd.DataFrame
        The formatted DataFrame.
    """
    if not isinstance(df.columns, pd.MultiIndex):
        # Simple single-level columns
        df = _format_meteo_yr_df(df)
    else:
        # Multi-level columns - need special formatting
        # For now, just format the index (Day of Year)
        year = 2024 if len(df) == 366 else 2023
        new_index = [
            julianDay_to_date(julday, year=year, str_format="%b-%d")
            for julday in df.index
        ]
        df.index = pd.Index(new_index, name="Day of Year")


def _construct_profile_dataframe(
    profile_data: dict,
    warming_levels: np.ndarray,
    simulations: list,
    sim_label_func: callable,
    days_in_year: int,
    hours_per_day: int,
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
    days_in_year : int
        Number of days in the year (365 or 366)
    hours_per_day : int
        Number of hours per day (24)

    Returns
    -------
    pd.DataFrame
        Structured DataFrame with appropriate column structure based on dimensions
    """
    hours = np.arange(1, 25, 1)
    n_warming_levels = len(warming_levels)
    n_simulations = len(simulations)

    if n_warming_levels == 1 and n_simulations == 1:
        return _create_simple_dataframe(
            profile_data,
            warming_levels[0],
            simulations[0],
            sim_label_func,
            days_in_year,
            hours,
        )
    elif n_warming_levels == 1 and n_simulations > 1:
        return _create_single_wl_multi_sim_dataframe(
            profile_data,
            warming_levels[0],
            simulations,
            sim_label_func,
            days_in_year,
            hours,
            hours_per_day,
        )
    elif n_warming_levels > 1 and n_simulations == 1:
        return _create_multi_wl_single_sim_dataframe(
            profile_data,
            warming_levels,
            simulations[0],
            sim_label_func,
            days_in_year,
            hours,
            hours_per_day,
        )
    else:
        return _create_multi_wl_multi_sim_dataframe(
            profile_data,
            warming_levels,
            simulations,
            sim_label_func,
            days_in_year,
            hours,
            hours_per_day,
        )


def _create_simple_dataframe(
    profile_data: dict,
    warming_level: float,
    simulation: any,
    sim_label_func: callable,
    days_in_year: int,
    hours: np.ndarray,
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
    days_in_year : int
        Number of days in year
    hours : np.ndarray
        Array of hour values

    Returns
    -------
    pd.DataFrame
        Simple DataFrame with hour columns
    """
    wl_key = f"WL_{warming_level}"
    sim_key = sim_label_func(simulation, 0)
    profile_matrix = profile_data[(wl_key, sim_key)]

    return pd.DataFrame(
        profile_matrix,
        columns=hours,
        index=np.arange(1, days_in_year + 1, 1),
    )


def _create_single_wl_multi_sim_dataframe(
    profile_data: dict,
    warming_level: float,
    simulations: list,
    sim_label_func: callable,
    days_in_year: int,
    hours: np.ndarray,
    hours_per_day: int,
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
    days_in_year : int
        Number of days in year
    hours : np.ndarray
        Array of hour values
    hours_per_day : int
        Hours per day (24)

    Returns
    -------
    pd.DataFrame
        DataFrame with (Hour, Simulation) MultiIndex columns
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

    # Create MultiIndex columns
    col_tuples = [(hour, sim_name) for hour in hours for sim_name in sim_names]
    multi_cols = pd.MultiIndex.from_tuples(col_tuples, names=["Hour", "Simulation"])

    # Stack data
    all_data = _stack_profile_data(
        profile_data=profile_data,
        hours_per_day=hours_per_day,
        wl_names=[f"WL_{wl}"],
        sim_names=sim_names,
        hour_first=True,
    )

    return pd.DataFrame(
        all_data,
        columns=multi_cols,
        index=np.arange(1, days_in_year + 1, 1),
    )


def _create_multi_wl_single_sim_dataframe(
    profile_data: dict,
    warming_levels: np.ndarray,
    simulation: any,
    sim_label_func: callable,
    days_in_year: int,
    hours: np.ndarray,
    hours_per_day: int,
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
    days_in_year : int
        Number of days in year
    hours : np.ndarray
        Array of hour values
    hours_per_day : int
        Hours per day (24)

    Returns
    -------
    pd.DataFrame
        DataFrame with (Hour, Warming_Level) MultiIndex columns
    """
    sim_name = sim_label_func(simulation, 0)
    wl_names = [f"WL_{wl}" for wl in warming_levels]

    # Create MultiIndex columns
    col_tuples = [(hour, wl_name) for hour in hours for wl_name in wl_names]
    multi_cols = pd.MultiIndex.from_tuples(col_tuples, names=["Hour", "Warming_Level"])

    # Stack data
    all_data = _stack_profile_data(
        profile_data=profile_data,
        hours_per_day=hours_per_day,
        wl_names=wl_names,
        sim_names=[sim_name],
        hour_first=True,
    )

    return pd.DataFrame(
        all_data,
        columns=multi_cols,
        index=np.arange(1, days_in_year + 1, 1),
    )


def _create_multi_wl_multi_sim_dataframe(
    profile_data: dict,
    warming_levels: np.ndarray,
    simulations: list,
    sim_label_func: callable,
    days_in_year: int,
    hours: np.ndarray,
    hours_per_day: int,
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
    days_in_year : int
        Number of days in year
    hours : np.ndarray
        Array of hour values
    hours_per_day : int
        Hours per day (24)

    Returns
    -------
    pd.DataFrame
        DataFrame with (Hour, Warming_Level, Simulation) MultiIndex columns
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
    col_tuples = [
        (hour, wl_name, sim_name)
        for hour in hours
        for wl_name in wl_names
        for sim_name in sim_names
    ]
    multi_cols = pd.MultiIndex.from_tuples(
        col_tuples, names=["Hour", "Warming_Level", "Simulation"]
    )

    # Stack data with all three dimensions
    all_data = _stack_profile_data(
        profile_data=profile_data,
        hours_per_day=hours_per_day,
        wl_names=wl_names,
        sim_names=sim_names,
        hour_first=True,
        three_level=True,
    )

    return pd.DataFrame(
        all_data,
        columns=multi_cols,
        index=np.arange(1, days_in_year + 1, 1),
    )


def _stack_profile_data(
    profile_data: dict,
    hours_per_day: int,
    wl_names: list,
    sim_names: list,
    hour_first: bool = True,
    three_level: bool = False,
) -> np.ndarray:
    """
    Stack profile data into a single array for DataFrame construction.

    Parameters
    ----------
    profile_data : dict
        Dictionary with (wl_name, sim_name) keys and profile arrays as values
    hours_per_day : int
        Number of hours per day (24)
    wl_names : list
        List of warming level names
    sim_names : list
        List of simulation names
    hour_first : bool, optional
        Whether hour should be the first level in iteration order
    three_level : bool, optional
        Whether this is a three-level MultiIndex (Hour, WL, Sim)

    Returns
    -------
    np.ndarray
        Stacked data array ready for DataFrame construction
    """
    all_data = []

    if three_level:
        # For three-level index: iterate hour -> wl -> sim
        for hour in range(hours_per_day):
            for wl_name in wl_names:
                for sim_name in sim_names:
                    key = (wl_name, sim_name)
                    all_data.append(profile_data[key][:, hour])
    elif hour_first:
        # For two-level with hour first
        for hour in range(hours_per_day):
            for wl_name in wl_names:
                for sim_name in sim_names:
                    key = (wl_name, sim_name)
                    all_data.append(profile_data[key][:, hour])
    else:
        # For other two-level cases
        for wl_name in wl_names:
            for sim_name in sim_names:
                for hour in range(hours_per_day):
                    key = (wl_name, sim_name)
                    all_data.append(profile_data[key][:, hour])

    return np.column_stack(all_data)


def _format_meteo_yr_df(df: pd.DataFrame) -> pd.DataFrame:
    """
    Format meteorological yearly dataframe for display with readable time labels.

    This function reformats a dataframe output from compute_profile
    by reordering columns to PST time format, converting numeric hour columns to
    12-hour AM/PM format, and converting Julian day indices to Month-Day format.

    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe with meteorological data. Expected to have 24 columns
        representing hours (0-23) and Julian day indices (1-365 or 1-366).

    Returns
    -------
    pd.DataFrame
        Formatted dataframe with:
        - Columns reordered and labeled in 12-hour AM/PM format (12am, 1am, ..., 11pm)
        - Column name set to "Hour"
        - Index converted from Julian days to "Month-Day" format (e.g., "Jan-01")
        - Index name set to "Day of Year"
        - Uses leap year (2024) for 366-day datasets, otherwise uses 2023

    Notes
    -----
    The function assumes the input dataframe has exactly 24 columns representing
    hourly data, with the first 7 columns corresponding to hours 17-23 (5pm-11pm)
    and the last 17 columns corresponding to hours 0-16 (12am-4pm).
    """
    ## Re-order columns for PST, with easy to read time labels
    cols = df.columns.tolist()
    cols = cols[7:] + cols[:7]
    df = df[cols]

    n_col_lst = []
    for ampm in ["am", "pm"]:
        hr_lst = []
        for hr in range(1, 13, 1):
            hr_lst.append(str(hr) + ampm)
        hr_lst = hr_lst[-1:] + hr_lst[:-1]
        n_col_lst = n_col_lst + hr_lst
    df.columns = n_col_lst
    df.columns.name = "Hour"

    # Convert Julian date index to Month-Day format
    # Use 2024 as year if we have 366 days (leap year), otherwise use 2023
    year = 2024 if len(df) == 366 else 2023
    new_index = [
        julianDay_to_date(julday, year=year, str_format="%b-%d") for julday in df.index
    ]
    df.index = pd.Index(new_index, name="Day of Year")
    return df


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
