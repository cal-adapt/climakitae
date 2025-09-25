"""
Calculates the Average Meterological Year (AMY) and Severe Meteorological Year (SMY) for the Cal-Adapt: Analytics Engine using a standard climatological period (1981-2010) for the historical baseline, and uses a 30-year window around when a designated warming level is exceeded for the SSP3-7.0 future scenario for 1.5°C, 2°C, and 3°C.
The AMY is comparable to a typical meteorological year, but not quite the same full methodology.
"""

## PROCESS: average meteorological year
# for each hour, the average variable over the whole climatological period is determined
# data for that hour that has the value most closely equal (smallest absolute difference)
# to the hourly average over the whole measurement period is chosen as the AMY data for that hour
# process is repeated for each hour in the year
# repeat values (where multiple years have the same smallest abs value) are
# removed, earliest occurrence selected for AMY
# hours are added together to provide a full year of hourly samples

## Produces 3 different kinds of AMY
## 1: Absolute/unbias-corrected raw AMY, either historical or warming level-centered future
## 2: Future-minus-historical warming level AMY (see warming_levels)
## 3: Severe AMY based upon historical baseline and a designated threshold/percentile

from typing import Tuple

import numpy as np
import pandas as pd
import xarray as xr
from tqdm.auto import tqdm  # Progress bar

from climakitae.core.data_interface import DataParameters, get_data
from climakitae.core.data_load import read_catalog_from_select
from climakitae.util.utils import julianDay_to_date

xr.set_options(keep_attrs=True)  # Keep attributes when mutating xr objects


# =========================== HELPER FUNCTIONS: DATA RETRIEVAL ==============================


# !DEPRECATED
def _set_amy_year_inputs(year_start: int, year_end: int) -> tuple[int, int]:
    """Helper function for _retrieve_meteo_yr_data.
    Checks that the user has input valid values.
    Sets year end if it hasn't been set; default is 30 year range (year_start + 30). Minimum is 5 year range.

    """
    match year_end:
        case None:
            year_end = (
                year_start + 30 if (year_start + 30 < 2100) else 2100
            )  # Default is +30 years
        case _ if year_end > 2100:
            print("Your end year cannot exceed 2100. Resetting end year to 2100.")
            year_end = 2100
    if year_end - year_start < 5:
        raise ValueError(
            """To compute an Average Meteorological Year, you must input a date range with a difference
            of at least 5 years, where the end year is no later than 2100 and the start year is no later than
            2095."""
        )
    if year_start < 1980:
        raise ValueError(
            """You've input an invalid start year. The start year must be 1980 or later."""
        )
    return (year_start, year_end)


# !DEPRECATED
def retrieve_meteo_yr_data(
    data_params: DataParameters,
    ssp: str = None,
    year_start: int = 2015,
    year_end: int = None,
) -> xr.DataArray:
    """Backend function for retrieving data needed for computing a meteorological year.

    Reads in the hourly ensemble means instead of the hourly data.
    Reads in future SSP data, historical climate data, or a combination
    of both, depending on year_start and year_end

    Parameters
    ----------
    self : AverageMetYearParameters
    ssp : str
        one of "SSP 2-4.5", "SSP 3-7.0", "SSP 5-8.5"
        Shared Socioeconomic Pathway. Defaults to SSP 3-7.0
    year_start : int, optional
        Year between 1980-2095. Default to 2015
    year_end : int, optional
        Year between 1985-2100. Default to year_start+30

    Returns
    -------
    xr.DataArray
        Hourly ensemble means from year_start-year_end for the ssp specified.

    """
    # Ensure only WRF data is being used
    data_params.downscaling_method = "Dynamical"

    # Save units. Sometimes they get lost.
    units = data_params.units

    # Check year start and end inputs
    year_start, year_end = _set_amy_year_inputs(year_start, year_end)

    # Set scenario selections
    if (ssp is not None) and (year_end >= 2015):
        data_params.scenario_ssp = [ssp]
    if year_end < 2015:
        data_params.scenario_ssp = []
    elif (year_end >= 2015) and (data_params.scenario_ssp) == []:
        data_params.scenario_ssp = ["SSP 3-7.0"]  # Default
    if year_start < 2015:  # Append historical data
        data_params.scenario_historical = ["Historical Climate"]
    else:
        data_params.scenario_historical = []
    if len(data_params.scenario_ssp) > 1:
        # If multiple SSPs are selected, only use the first one
        data_params.scenario_ssp = [data_params.scenario_ssp[0]]

    # Set other data parameters
    data_params.simulation = ["ensmean"]
    data_params.time_slice = (year_start, year_end)
    data_params.area_average = "Yes"
    data_params.timescale = "hourly"
    data_params.units = units

    # Grab data from the catalog
    amy_data = read_catalog_from_select(data_params)
    if amy_data is None:
        # Catch small spatial resolutions
        raise ValueError(
            "COULD NOT RETRIEVE DATA: For the provided data selections, there is not sufficient data to retrieve. Try selecting a larger spatial area, or a higher resolution. Returning None."
        )
    return amy_data.isel(scenario=0, simulation=0)


def retrieve_profile_data(**kwargs: any) -> Tuple[xr.Dataset, xr.Dataset]:
    """
    Backend function for retrieving data needed for computing climate profiles.

    Reads in the full hourly data for the 8760 analysis, including all warming levels.

    Parameters
    ----------
    **kwargs : dict
        Keyword arguments for data selection. Allowed keys:
        - variable (Optional) : str, default "Air Temperature at 2m"
        - resolution (Optional) : str, default "4km"
        - scenario (Optional) : List[str], default ["SSP 3-7.0"]
        - warming_levels (Required) : List[float], default [1.2]
        - cached_area (Optional) : str or List[str]
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
    """
    no_delta = kwargs.pop("no_delta", False)
    # Define allowed inputs with types and defaults
    ALLOWED_INPUTS = {
        "variable": (str, "Air Temperature at 2m"),
        "resolution": (str, "3 km"),
        "warming_level": (list, [1.2]),
        "cached_area": ((str, list), None),
        "units": (str, "degF"),
        "latitude": ((float, tuple), None),
        "longitude": ((float, tuple), None),
    }

    # if the user does not enter warming level the analysis is a moot point
    # because the historical data is always at 1.2C
    REQUIRED_INPUTS = ["warming_level"]
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
        if not isinstance(value, expected_type):
            raise TypeError(
                f"Parameter '{key}' must be of type {expected_type.__name__}, "
                f"got {type(value).__name__}"
            )

    # Set default parameters for data retrieval
    get_data_params = {
        "variable": kwargs.get("variable", "Air Temperature at 2m"),
        "resolution": kwargs.get("resolution", "3 km"),
        "downscaling_method": "Dynamical",  # must be WRF, cannot be LOCA
        "timescale": "hourly",  # must be hourly for 8760 analysis
        "area_average": "Yes",
        "units": "degF",
        "approach": "Warming Level",
        "warming_level": [1.2],
        "cached_area": kwargs.get("cached_area", None),
    }

    historic_data = None
    if not no_delta:
        # Retrieve historical data at 1.2°C warming level
        historic_data = get_data(**get_data_params)

    # Update with any user-provided parameters for future data retrieval
    get_data_params.update(kwargs)
    future_data = get_data(**get_data_params)

    return historic_data, future_data


# =========================== HELPER FUNCTIONS: AMY/TMY CALCULATION ==============================


# ! DEPRECATED
def compute_amy(data: xr.DataArray, days_in_year: int = 366) -> pd.DataFrame:
    """Calculates the average meteorological year based on a designated period of time

    Applicable for both the historical and future periods.

    Parameters
    ----------
    data : xr.DataArray
        Hourly data for one variable
    days_in_year : int, optional
        Either 366 or 365, depending on whether or not the year is a leap year.
        Default to 366 days (leap year)

    Returns
    -------
    pd.DataFrame
        Average meteorological year table, with days of year as
        the index and hour of day as the columns.

    """

    def _closest_to_mean(dat: xr.DataArray) -> xr.DataArray:
        """Find the value closest to the mean of the data."""
        stacked = dat.stack(allofit=list(dat.dims))
        index = abs(stacked - stacked.quantile("allofit")).argmin(dim="allofit").values
        return xr.DataArray(stacked.isel(allofit=index).values)

    def _return_diurnal(y: xr.DataArray) -> xr.DataArray:
        """Return the diurnal cycle of the data."""
        return y.groupby("time.hour").map(_closest_to_mean)

    hourly_da = data.groupby("time.dayofyear").map(_return_diurnal)

    # Funnel data into pandas DataFrame object
    df_amy = pd.DataFrame(
        hourly_da,  # hourly DataArray,
        columns=np.arange(1, 25, 1),
        index=np.arange(1, days_in_year + 1, 1),
    )
    # Format dataframe
    df_amy = _format_meteo_yr_df(df_amy)
    return df_amy


def get_climate_profile(**kwargs) -> pd.DataFrame:
    """
    High-level function to compute climate profiles using warming level data.

    This function retrieves climate data and computes average meteorological year
    profiles using the 8760 analysis approach. It combines data retrieval and
    profile computation in a single call.

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
        - days_in_year (Optional) : int, default 365
        - q (Optional) : float | list[float], default [], quantile for profile calculation
        - no_delta (optional) : bool, default False, if True, do not apply baseline subtraction, return raw future profile

    Returns
    -------
    pd.DataFrame
        Average meteorological year table for each warming level, with days of year as
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
    with tqdm(total=2, desc="Data retrieval", unit="dataset") as pbar:
        historic_data, future_data = retrieve_profile_data(**kwargs)
        pbar.update(2)

    # Call compute_profile with the processed data
    # Compute profiles for both historical and future data
    if isinstance(future_data, xr.Dataset):
        var_name = list(future_data.data_vars.keys())[0]
        future_profile_data = future_data[var_name]
    else:
        future_profile_data = future_data

    if isinstance(historic_data, xr.Dataset):
        var_name = list(historic_data.data_vars.keys())[0]
        historic_profile_data = historic_data[var_name]
    else:
        historic_profile_data = historic_data

    # Compute profiles for both datasets
    print("⚙️  Computing climate profiles...")

    with tqdm(total=2, desc="Profile computation", unit="profile") as pbar:
        future_profile = compute_profile(
            future_profile_data, days_in_year=days_in_year, q=q
        )
        pbar.update(1)
        if no_delta:
            historic_profile = None
        else:
            historic_profile = compute_profile(
                historic_profile_data, days_in_year=days_in_year, q=q
            )
        pbar.update(1)

    if no_delta or historic_profile is None:
        print("   ✓ No baseline subtraction requested, returning raw future profile")
        return future_profile

    # Check the structure of both profiles to handle simulation dimension properly
    future_has_multiindex = isinstance(future_profile.columns, pd.MultiIndex)
    historic_has_multiindex = isinstance(historic_profile.columns, pd.MultiIndex)

    if future_has_multiindex and historic_has_multiindex:
        # Both have MultiIndex - need to handle carefully
        future_levels = future_profile.columns.names
        historic_levels = historic_profile.columns.names

        if "Simulation" in future_levels and "Simulation" in historic_levels:
            # Both have simulations - compute difference for matching simulations
            difference_profile = future_profile.copy()

            # Get unique simulations from both profiles
            future_sims = future_profile.columns.get_level_values("Simulation").unique()
            historic_sims = historic_profile.columns.get_level_values(
                "Simulation"
            ).unique()

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
                    difference_profile[col] = future_profile[col] - historic_mean[hour]
            else:
                # Compute differences for matching simulations
                n_cols = len(future_profile.columns)
                with tqdm(
                    total=n_cols, desc="   Computing paired differences", unit="column"
                ) as pbar:
                    for col in future_profile.columns:
                        # Extract levels from the column
                        if "Hour" in future_levels and "Simulation" in future_levels:
                            if len(col) == 3:  # Hour, Warming_Level, Simulation
                                hour, wl, sim = col
                                historic_col = (
                                    (hour, sim)
                                    if (hour, sim) in historic_profile.columns
                                    else None
                                )
                            else:  # Hour, Simulation
                                hour, sim = col
                                historic_col = (
                                    (hour, sim)
                                    if (hour, sim) in historic_profile.columns
                                    else None
                                )

                        if historic_col and historic_col in historic_profile.columns:
                            difference_profile[col] = (
                                future_profile[col] - historic_profile[historic_col]
                            )
                        else:
                            # If no matching simulation, use mean of historic for that hour
                            hour = col[0]  # Assuming hour is first level
                            if "Simulation" in historic_levels:
                                historic_hour_mean = historic_profile.xs(
                                    hour, level="Hour", axis=1
                                ).mean()
                            else:
                                historic_hour_mean = (
                                    historic_profile[hour]
                                    if hour in historic_profile.columns
                                    else 0
                                )
                            difference_profile[col] = (
                                future_profile[col] - historic_hour_mean
                            )
                        pbar.update(1)

        elif "Warming_Level" in future_levels and "Simulation" not in future_levels:
            # Future has warming levels but no simulations
            difference_profile = future_profile.copy()

            n_cols = len(future_profile.columns)
            with tqdm(
                total=n_cols, desc="   Computing differences", unit="column"
            ) as pbar:
                for col in future_profile.columns:
                    hour = col[0]  # Assuming (Hour, Warming_Level) structure
                    if hour in historic_profile.columns:
                        # Use .loc to properly assign to MultiIndex columns
                        difference_profile.loc[:, col] = (
                            future_profile[col] - historic_profile[hour]
                        )
                    else:
                        # Try to find corresponding hour in historic MultiIndex
                        if historic_has_multiindex and "Hour" in historic_levels:
                            historic_hour = historic_profile.xs(
                                hour, level="Hour", axis=1
                            ).iloc[:, 0]
                        else:
                            historic_hour = historic_profile.iloc[
                                :, 0
                            ]  # Fall back to first column
                        difference_profile.loc[:, col] = (
                            future_profile[col] - historic_hour
                        )
                    pbar.update(1)

    elif future_has_multiindex and not historic_has_multiindex:
        # Future has MultiIndex, historic doesn't
        difference_profile = future_profile.copy()

        n_cols = len(future_profile.columns)
        with tqdm(total=n_cols, desc="   Computing differences", unit="column") as pbar:
            for col in future_profile.columns:
                # Try to match hour
                if "Hour" in future_profile.columns.names:
                    hour_idx = future_profile.columns.names.index("Hour")
                    hour = col[hour_idx]
                    # Find corresponding hour in historic (considering PST shift)
                    if hour in historic_profile.columns:
                        difference_profile.loc[:, col] = (
                            future_profile[col] - historic_profile[hour]
                        )
                    else:
                        # Try numeric hour matching (1-24)
                        hour_num = (
                            int(hour.replace("am", "").replace("pm", ""))
                            if isinstance(hour, str)
                            else hour
                        )
                        if hour_num in historic_profile.columns:
                            difference_profile.loc[:, col] = (
                                future_profile[col] - historic_profile[hour_num]
                            )
                        else:
                            # Fall back to positional matching
                            col_position = future_profile.columns.get_loc(col)
                            if isinstance(col_position, int):
                                historic_col_idx = col_position % len(
                                    historic_profile.columns
                                )
                                difference_profile.loc[:, col] = (
                                    future_profile[col]
                                    - historic_profile.iloc[:, historic_col_idx]
                                )
                else:
                    # No hour level, use positional matching
                    col_position = future_profile.columns.get_loc(col)
                    if isinstance(col_position, int):
                        historic_col_idx = col_position % len(historic_profile.columns)
                        difference_profile.loc[:, col] = (
                            future_profile[col]
                            - historic_profile.iloc[:, historic_col_idx]
                        )
                pbar.update(1)

    else:
        # Both have single-level columns
        # Check if columns match
        if list(future_profile.columns) == list(historic_profile.columns):
            print("   ✓ Columns match - computing element-wise difference")
            difference_profile = future_profile - historic_profile
        else:
            print("   ⚠️  Warning: Column mismatch between future and historic profiles")
            print(f"      Future columns: {list(future_profile.columns)[:5]}...")
            print(f"      Historic columns: {list(historic_profile.columns)[:5]}...")
            # Try to align by position
            min_cols = min(len(future_profile.columns), len(historic_profile.columns))
            difference_profile = (
                future_profile.iloc[:, :min_cols] - historic_profile.iloc[:, :min_cols]
            )

    print(
        f"✅ Climate profile computation complete! Final shape: {difference_profile.shape}"
    )
    print(
        f"   (Days: {difference_profile.shape[0]}, Hours/Columns: {difference_profile.shape[1]})"
    )
    return difference_profile


def compute_profile(data: xr.DataArray, days_in_year: int = 365, q=0.5) -> pd.DataFrame:
    """
    Calculates the average meteorological year profile for warming level data using 8760
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
        Average meteorological year table for each warming level and simulation,
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
    def _get_simulation_label(sim, sim_idx):
        """Extract meaningful simulation label from simulation identifier."""
        if sim is None:
            return f"Sim_{sim_idx+1}"

        sim_str = str(sim)
        if "WRF_" in sim_str:
            # Extract the GCM model name (e.g., CESM2, CNRM-ESM2-1, etc.)
            parts = sim_str.split("_")
            if len(parts) >= 2:
                return parts[1]  # Get the GCM name
            else:
                return f"Sim_{sim_idx+1}"
        else:
            return sim_str.split("_")[0] if "_" in sim_str else sim_str

    # Process all data using quantile computation across years
    print(
        f"      ⚙️  Computing quantiles for {len(warming_levels)} warming level(s) and {n_simulations} simulation(s)"
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
    hours = np.arange(1, 25, 1)  # Hours 1-24

    if len(warming_levels) == 1 and n_simulations == 1:
        # Single warming level, single simulation - simple columns
        wl_key = f"WL_{warming_levels[0]}"
        sim_key = _get_simulation_label(simulations[0], 0)
        profile_matrix = profile_data[(wl_key, sim_key)]

        df_profile = pd.DataFrame(
            profile_matrix,
            columns=hours,
            index=np.arange(1, days_in_year + 1, 1),
        )

    elif len(warming_levels) == 1 and n_simulations > 1:
        # Single warming level, multiple simulations
        wl = warming_levels[0]
        sim_names = [_get_simulation_label(sim, i) for i, sim in enumerate(simulations)]

        col_tuples = [(hour, sim_name) for hour in hours for sim_name in sim_names]
        multi_cols = pd.MultiIndex.from_tuples(col_tuples, names=["Hour", "Simulation"])

        # Stack all data horizontally
        all_data = []
        for hour in range(hours_per_day):
            for sim_name in sim_names:
                key = (f"WL_{wl}", sim_name)
                all_data.append(profile_data[key][:, hour])

        all_data_array = np.column_stack(all_data)
        df_profile = pd.DataFrame(
            all_data_array,
            columns=multi_cols,
            index=np.arange(1, days_in_year + 1, 1),
        )

    elif len(warming_levels) > 1 and n_simulations == 1:
        # Multiple warming levels, single simulation
        sim_name = _get_simulation_label(simulations[0], 0)
        wl_names = [f"WL_{wl}" for wl in warming_levels]

        col_tuples = [(hour, wl_name) for hour in hours for wl_name in wl_names]
        multi_cols = pd.MultiIndex.from_tuples(
            col_tuples, names=["Hour", "Warming_Level"]
        )

        # Stack all data horizontally
        all_data = []
        for hour in range(hours_per_day):
            for wl_name in wl_names:
                key = (wl_name, sim_name)
                all_data.append(profile_data[key][:, hour])

        all_data_array = np.column_stack(all_data)
        df_profile = pd.DataFrame(
            all_data_array,
            columns=multi_cols,
            index=np.arange(1, days_in_year + 1, 1),
        )

    else:
        # Multiple warming levels AND multiple simulations
        wl_names = [f"WL_{wl}" for wl in warming_levels]
        sim_names = [_get_simulation_label(sim, i) for i, sim in enumerate(simulations)]

        col_tuples = [
            (hour, wl_name, sim_name)
            for hour in hours
            for wl_name in wl_names
            for sim_name in sim_names
        ]
        multi_cols = pd.MultiIndex.from_tuples(
            col_tuples, names=["Hour", "Warming_Level", "Simulation"]
        )

        # Stack all data horizontally
        all_data = []
        for hour in range(hours_per_day):
            for wl_name in wl_names:
                for sim_name in sim_names:
                    key = (wl_name, sim_name)
                    all_data.append(profile_data[key][:, hour])

        all_data_array = np.column_stack(all_data)
        df_profile = pd.DataFrame(
            all_data_array,
            columns=multi_cols,
            index=np.arange(1, days_in_year + 1, 1),
        )

    # Determine which formatting function to use based on the structure
    if not isinstance(df_profile.columns, pd.MultiIndex):
        # Simple single-level columns
        df_profile = _format_meteo_yr_df(df_profile)
    else:
        # Multi-level columns - need special formatting
        # For now, just format the index (Day of Year)
        year = 2024 if len(df_profile) == 366 else 2023
        new_index = [
            julianDay_to_date(julday, year=year, str_format="%b-%d")
            for julday in df_profile.index
        ]
        df_profile.index = pd.Index(new_index, name="Day of Year")

    # Preserve units information from the original data
    if hasattr(data, "attrs") and "units" in data.attrs:
        df_profile.attrs["units"] = data.attrs["units"]
        df_profile.attrs["display_name"] = data.attrs.get("display_name", "N/A")
        df_profile.attrs["variable_name"] = data.attrs.get("variable_id", data.name)

    # Add metadata about the profile computation
    df_profile.attrs["quantile"] = q
    df_profile.attrs["method"] = (
        "8760 analysis - actual data closest to quantile across 30 years"
    )
    df_profile.attrs["description"] = (
        f"Climate profile computed using actual data values closest to the {q*100:.0f}th percentile of hourly data"
    )

    print(f"      ✅ Profile computation complete! Final shape: {df_profile.shape}")
    print(
        f"         With index: {df_profile.index.name}, columns: {df_profile.columns.names}"
    )
    if hasattr(data, "attrs") and "units" in data.attrs:
        print(f"         Units: {data.attrs['units']}")

    return df_profile


def _format_meteo_yr_df(df: pd.DataFrame) -> pd.DataFrame:
    """Format dataframe output from compute_amy and compute_severe_yr"""
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


def compute_severe_yr(data: xr.DataArray, days_in_year: int = 366) -> pd.DataFrame:
    """Calculate the severe meteorological year based on the 90th percentile of data.

    Applicable for both the historical and future periods.

    Parameters
    ----------
    data : xr.DataArray
        Hourly data for one variable
    days_in_year : int, optional
        Either 366 or 365, depending on whether or not the year is a leap year.
        Default to 366 days (leap year)

    Returns
    -------
    pd.DataFrame
        Severe meteorological year table, with days of year as
        the index and hour of day as the columns.

    """

    def closest_to_quantile(dat):
        stacked = dat.stack(allofit=list(dat.dims))
        index = (
            abs(stacked - stacked.quantile(q=0.90, dim="allofit"))
            .argmin(dim="allofit")
            .values
        )
        return xr.DataArray(stacked.isel(allofit=index).values)

    def return_diurnal(y):
        return y.groupby("time.hour").map(closest_to_quantile)

    hourly_da = data.groupby("time.dayofyear").map(return_diurnal)

    ## Funnel data into pandas DataFrame object
    df_severe_yr = pd.DataFrame(
        hourly_da,
        columns=np.arange(1, 25, 1),
        index=np.arange(1, days_in_year + 1, 1),
    )

    # Format dataframe
    df_severe_yr = _format_meteo_yr_df(df_severe_yr)
    return df_severe_yr


# =========================== HELPER FUNCTIONS: MISC ==============================


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


def compute_mean_monthly_meteo_yr(
    tmy_df: pd.DataFrame, col_name: str = "mean_value"
) -> pd.DataFrame:
    """Compute mean monthly values for input meteorological year data.

    Parameters
    ----------
    tmy_df : pd.DataFrame
        Matrix with day of year as index and hour as columns
        Output of either compute_severe_yr or compute_meteo_yr
    col_name : str, optional
        Name to give single output column
        It may be informative to assign this to the name of the data variable

    Returns
    -------
    pd.DataFrame
        Table with month as index and monthly mean as column

    """
    # Convert from matrix --> hour and data as individual columns
    tmy_stacked = (
        pd.DataFrame(tmy_df.stack()).rename(columns={0: col_name}).reset_index()
    )
    # Combine Hour and Day of Year to get combined date. Assign as index
    tmy_stacked["Date"] = tmy_stacked["Day of Year"] + " " + tmy_stacked["Hour"]
    tmy_stacked = tmy_stacked.drop(columns=["Day of Year", "Hour"]).set_index("Date")

    # Reformat index to datetime so that you can resample the data monthly
    reformatted_idx = pd.to_datetime(
        ["2024." + idx for idx in tmy_stacked.index], format="%Y.%b-%d %I%p"
    )
    tmy_monthly_mean = tmy_stacked.set_index(reformatted_idx).resample("MS").mean()

    # Reset index to be user-friendly month strings
    tmy_monthly_mean = tmy_monthly_mean.set_index(tmy_monthly_mean.index.strftime("%b"))
    tmy_monthly_mean.index.name = "Month"
    return tmy_monthly_mean
