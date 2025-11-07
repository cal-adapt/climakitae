"""Utility functions for processing data arrays in climakitae."""

import re
import warnings
from typing import Dict, Union

import numpy as np
import xarray as xr

from climakitae.core.constants import UNSET
from climakitae.explore.threshold_tools import calculate_ess


# Constants for effective sample size calculations
MIN_ESS_THRESHOLD = 25  # Minimum effective sample size for reliable statistics
FALLBACK_ESS_VALUE = 25.0  # Default ESS value when calculation fails
MIN_TIME_POINTS = 10  # Minimum time points required for ESS calculation

# Constants for chunking and memory management
DEFAULT_TIME_CHUNK_DAYS = 365  # Default time chunk size in days
MIN_TIME_CHUNK_DIVISOR = 10  # Divisor for calculating minimum time chunk
BYTES_PER_GB = 1e9  # Conversion factor from GB to bytes

# Constants for ESS approximation
LARGE_DATASET_THRESHOLD = 1000  # Threshold for using approximation in ESS
LARGE_TIMESERIES_THRESHOLD = 500  # Threshold for large timeseries approximation
MAX_LAG_DIVISOR = 4  # Divisor for maximum lag in autocorrelation
MAX_LAG_SAMPLES_GRIDDED = 100  # Maximum lag samples for gridded data
MAX_LAG_SAMPLES_TIMESERIES = 50  # Maximum lag samples for timeseries data
AUTOCORR_LAG_SAMPLES_GRIDDED = 20  # Number of lag samples for gridded autocorrelation
AUTOCORR_LAG_SAMPLES_TIMESERIES = (
    15  # Number of lag samples for timeseries autocorrelation
)

# Constants for spatial sampling
DASK_SPATIAL_SAMPLE_STEP = 20  # Spatial sampling step for Dask arrays
DASK_MAX_SPATIAL_SAMPLES = 100  # Maximum spatial samples for Dask arrays
MEMORY_SPATIAL_SAMPLE_STEP = 10  # Spatial sampling step for in-memory arrays
MEMORY_MAX_SPATIAL_SAMPLES = 50  # Maximum spatial samples for in-memory arrays


def is_station_identifier(value: str) -> bool:
    """
    Check if a string looks like a station identifier.

    This function uses heuristics to determine if a string appears to be
    a weather station identifier based on common patterns.

    Parameters
    ----------
    value : str
        String to check

    Returns
    -------
    bool
        True if the value looks like a station code or station name

    Notes
    -----
    Recognizes two patterns:
    1. 4-character codes starting with 'K' (common US airport weather stations)
       Examples: KSAC (Sacramento), KBFL (Bakersfield), KSFO (San Francisco)
    2. Strings with parentheses containing a code with 'K'
       Examples: "Sacramento (KSAC)", "San Francisco International (KSFO)"

    Examples
    --------
    >>> is_station_identifier("KSAC")
    True
    >>> is_station_identifier("Sacramento (KSAC)")
    True
    >>> is_station_identifier("CA")
    False
    >>> is_station_identifier("Kern County")
    False
    """
    # Check if it's a 4-character code starting with 'K' (common US airport codes)
    if len(value) == 4 and value[0].upper() == "K" and value.isalnum():
        return True

    # Check if it contains parentheses with a code (e.g., "Sacramento (KSAC)")
    if "(" in value and ")" in value and "K" in value.upper():
        return True

    return False


def find_station_match(station_identifier: str, stations_df):
    """
    Find matching station(s) in the stations DataFrame.

    This function centralizes the station matching logic used by both the Clip
    processor and the clip parameter validator. It tries multiple matching strategies
    in order of specificity:
    1. Exact match on station ID column
    2. Exact match on station name column
    3. Partial match on station name column

    Parameters
    ----------
    station_identifier : str
        Station identifier to search for (e.g., "KSAC", "Sacramento (KSAC)", "Sacramento")
    stations_df : pd.DataFrame
        DataFrame containing station data with columns: ID, station, city, state, LAT_Y, LON_X

    Returns
    -------
    pd.DataFrame
        DataFrame containing matching station(s). May have 0, 1, or multiple rows:
        - Empty (len=0): No matches found
        - Single row (len=1): Exact match found
        - Multiple rows (len>1): Multiple stations match the identifier

    Notes
    -----
    The caller is responsible for:
    - Checking if stations_df is None or empty before calling
    - Handling the different match scenarios (no match, single match, multiple matches)
    - Providing appropriate error messages or warnings based on context

    Examples
    --------
    >>> # For validation (clip_param_validator.py)
    >>> match = find_station_match("KSAC", stations_df)
    >>> if len(match) == 0:
    ...     # Handle no match - provide suggestions
    >>> elif len(match) > 1:
    ...     # Handle multiple matches - ask user to be more specific
    >>> else:
    ...     # Valid single match
    ...     return True

    >>> # For coordinate extraction (clip.py)
    >>> match = find_station_match("KSAC", stations_df)
    >>> if len(match) == 0:
    ...     # Raise ValueError with suggestions
    >>> elif len(match) > 1:
    ...     # Raise ValueError asking for more specific identifier
    >>> else:
    ...     # Extract coordinates and metadata
    ...     lat = float(match.iloc[0]["LAT_Y"])
    ...     lon = float(match.iloc[0]["LON_X"])
    """
    # Normalize the input
    station_id_upper = station_identifier.upper().strip()

    # Handle empty string after normalization
    if not station_id_upper:
        # Return empty DataFrame with same structure
        return stations_df.iloc[0:0]

    # Try exact match on ID column
    match = stations_df[stations_df["ID"].str.upper() == station_id_upper]

    if len(match) == 0:
        # Try matching on station name (full station column)
        match = stations_df[stations_df["station"].str.upper() == station_id_upper]

    if len(match) == 0:
        # Try partial match on station name
        match = stations_df[
            stations_df["station"].str.upper().str.contains(station_id_upper, na=False)
        ]

    return match


def _get_block_maxima_optimized(
    da_series: xr.DataArray,
    extremes_type: str = "max",
    duration: tuple[int, str] = UNSET,
    groupby: tuple[int, str] = UNSET,
    grouped_duration: tuple[int, str] = UNSET,
    check_ess: bool = True,
    block_size: int = 1,
    chunk_spatial: bool = True,
    max_memory_gb: float = 2.0,
) -> xr.DataArray:
    """Optimized, vectorized, and Dask-compatible version of get_block_maxima.

    This function converts data into block maximums, defaulting to annual maximums
    (default block size = 1 year). Optimized for large datasets and Dask arrays with
    improved memory management and vectorized operations.

    Parameters
    ----------
    da_series : xarray.DataArray
        DataArray from retrieve
    extremes_type : str
        option for max or min. Defaults to max
    duration : tuple
        length of extreme event, specified as (4, 'hour')
    groupby : tuple
        group over which to look for max occurrence, specified as (1, 'day')
    grouped_duration : tuple
        length of event after grouping, specified as (5, 'day')
    check_ess : bool
        optional flag specifying whether to check the effective sample size (ESS)
        within the blocks of data, and throw a warning if the average ESS is too small.
        can be silenced with check_ess=False.
    block_size : int
        block size in years. default is 1 year.
    chunk_spatial : bool
        whether to rechunk spatial dimensions for optimal performance
    max_memory_gb : float
        maximum memory to use for computation in GB

    Returns
    -------
    xarray.DataArray
        Block maxima with optimized processing

    """
    # Validate inputs
    valid_extremes = ["max", "min"]
    if extremes_type not in valid_extremes:
        raise ValueError(f"invalid extremes type. expected one of: {valid_extremes}")

    # Optimize chunking for Dask arrays
    if hasattr(da_series.data, "chunks"):
        da_series = _optimize_chunking_for_block_maxima(
            da_series, max_memory_gb, chunk_spatial
        )

    # Process duration events using vectorized rolling operations
    if duration is not UNSET:
        da_series = _apply_duration_filter_vectorized(
            da_series, duration, extremes_type
        )

    # Process groupby events using efficient resampling
    if groupby is not UNSET:
        da_series = _apply_groupby_filter_vectorized(da_series, groupby, extremes_type)

    # Process grouped duration events
    if grouped_duration is not UNSET:
        if groupby is UNSET:
            raise ValueError(
                "To use `grouped_duration` option, must first use groupby."
            )
        da_series = _apply_grouped_duration_filter_vectorized(
            da_series, grouped_duration, extremes_type
        )

    # Extract block maxima using optimized resampling
    bms = _extract_block_extremes_vectorized(da_series, extremes_type, block_size)

    # Calculate effective sample size if requested
    if check_ess:
        _check_effective_sample_size_optimized(da_series, block_size)

    # Set attributes efficiently
    bms = _set_block_maxima_attributes(
        bms, duration, groupby, grouped_duration, extremes_type, block_size
    )

    # Handle NaN values efficiently
    bms = _handle_nan_values_optimized(bms)

    return bms


def _optimize_chunking_for_block_maxima(
    da: xr.DataArray, max_memory_gb: float, chunk_spatial: bool = True
) -> xr.DataArray:
    """Optimize chunking for block maxima calculation with Dask arrays.

    This function optimizes the chunk sizes for efficient block maxima computation,
    balancing memory usage with computational performance. For temporal operations,
    larger time chunks are preferred to reduce overhead.

    Parameters
    ----------
    da : xr.DataArray
        Input DataArray with Dask backing (if applicable).
    max_memory_gb : float
        Maximum memory to use for computation in GB.
    chunk_spatial : bool, default True
        Whether to rechunk spatial dimensions for optimal performance.

    Returns
    -------
    xr.DataArray
        DataArray with optimized chunking configuration.

    Notes
    -----
    For non-Dask arrays, the original array is returned unchanged.
    The function calculates optimal chunk sizes based on memory constraints
    and the structure of temporal vs spatial dimensions.

    """
    if not hasattr(da.data, "chunks"):
        return da

    # Calculate optimal chunk sizes
    time_size = da.sizes.get("time", 1)
    spatial_dims = [dim for dim in da.dims if dim != "time"]

    # For temporal operations, prefer larger time chunks
    optimal_time_chunk = min(
        time_size, max(DEFAULT_TIME_CHUNK_DAYS, time_size // MIN_TIME_CHUNK_DIVISOR)
    )

    # Rechunk if beneficial
    chunk_dict = {"time": optimal_time_chunk}

    if chunk_spatial and spatial_dims:
        # Calculate spatial chunk size based on memory constraints
        element_size = da.dtype.itemsize
        max_elements = int((max_memory_gb * BYTES_PER_GB) / element_size)

        total_spatial = np.prod([da.sizes[dim] for dim in spatial_dims])
        if total_spatial > 0:
            spatial_chunk_size = int(np.sqrt(max_elements / optimal_time_chunk))

            for dim in spatial_dims:
                chunk_dict[dim] = min(da.sizes[dim], spatial_chunk_size)

    return da.chunk(chunk_dict)


def _apply_duration_filter_vectorized(
    da: xr.DataArray, duration: tuple[int, str], extremes_type: str
) -> xr.DataArray:
    """Apply duration filter using vectorized rolling operations.

    This function applies a duration-based filter to identify extreme events
    that persist for a specified duration. Currently only supports hourly
    frequency data with hour-based durations.

    Parameters
    ----------
    da : xr.DataArray
        Input DataArray with time dimension.
    duration : tuple[int, str]
        Duration specification as (length, unit), e.g., (4, 'hour').
    extremes_type : str
        Type of extreme ('max' or 'min').

    Returns
    -------
    xr.DataArray
        Filtered DataArray with duration-based rolling operations applied.

    Raises
    ------
    ValueError
        If duration type is not 'hour' or data frequency is not hourly,
        or if extremes_type is not 'max' or 'min'.

    Notes
    -----
    For maximum extremes, the function applies a rolling minimum to identify
    events where values remain above a threshold for the specified duration.
    For minimum extremes, the logic is reversed.

    """
    dur_len, dur_type = duration

    if dur_type != "hour" or getattr(da, "frequency", None) not in ["1hr", "hourly"]:
        raise ValueError(
            "Current specifications not implemented. `duration` options only implemented for `hour` frequency."
        )

    # Use vectorized rolling operations
    if extremes_type == "max":
        return da.rolling(time=dur_len, center=False).min()
    elif extremes_type == "min":
        return da.rolling(time=dur_len, center=False).max()
    else:
        raise ValueError('extremes_type needs to be either "max" or "min"')


def _apply_groupby_filter_vectorized(
    da: xr.DataArray, groupby: tuple[int, str], extremes_type: str
) -> xr.DataArray:
    """Apply groupby filter using efficient resampling.

    This function groups data over specified time periods and extracts
    extremes within each group. Currently only supports day-based groupings.

    Parameters
    ----------
    da : xr.DataArray
        Input DataArray with time dimension.
    groupby : tuple[int, str]
        Grouping specification as (length, unit), e.g., (1, 'day').
    extremes_type : str
        Type of extreme ('max' or 'min').

    Returns
    -------
    xr.DataArray
        Resampled DataArray with extremes extracted for each group.

    Raises
    ------
    ValueError
        If groupby type is not 'day' or extremes_type is not 'max' or 'min'.

    Notes
    -----
    Uses pandas resampling with 'left' labeling for consistent time indexing.

    """
    group_len, group_type = groupby

    if group_type != "day":
        raise ValueError(
            "`groupby` specifications only implemented for 'day' groupings."
        )

    # Use efficient resampling
    resample_rule = f"{group_len}D"
    resampler = da.resample(time=resample_rule, label="left")

    if extremes_type == "max":
        return resampler.max()
    elif extremes_type == "min":
        return resampler.min()
    else:
        raise ValueError('extremes_type needs to be either "max" or "min"')


def _apply_grouped_duration_filter_vectorized(
    da: xr.DataArray, grouped_duration: tuple[int, str], extremes_type: str
) -> xr.DataArray:
    """Apply grouped duration filter using vectorized operations.

    This function applies a duration filter after groupby operations to identify
    events that persist for a specified duration within grouped periods.
    Currently only supports day-based durations.

    Parameters
    ----------
    da : xr.DataArray
        Input DataArray that has already been processed by groupby operations.
    grouped_duration : tuple[int, str]
        Duration specification as (length, unit), e.g., (3, 'day').
    extremes_type : str
        Type of extreme ('max' or 'min').

    Returns
    -------
    xr.DataArray
        DataArray with grouped duration filter applied using rolling operations.

    Raises
    ------
    ValueError
        If duration type is not 'day' or extremes_type is not 'max' or 'min'.

    Notes
    -----
    This filter is typically applied after groupby operations to further
    refine the identification of persistent extreme events.

    """
    dur2_len, dur2_type = grouped_duration

    if dur2_type != "day":
        raise ValueError(
            "`grouped_duration` specification must be in days. example: `grouped_duration = (3, 'day')`."
        )

    # Use vectorized rolling operations
    if extremes_type == "max":
        return da.rolling(time=dur2_len, center=False).min()
    elif extremes_type == "min":
        return da.rolling(time=dur2_len, center=False).max()
    else:
        raise ValueError('extremes_type needs to be either "max" or "min"')


def _extract_block_extremes_vectorized(
    da: xr.DataArray, extremes_type: str, block_size: int
) -> xr.DataArray:
    """Extract block extremes using optimized resampling.

    This function extracts extreme values (maxima or minima) from blocks
    of data, typically annual blocks, using efficient pandas resampling.

    Parameters
    ----------
    da : xr.DataArray
        Input DataArray with time dimension.
    extremes_type : str
        Type of extreme to extract ('max' or 'min').
    block_size : int
        Size of blocks in years for resampling.

    Returns
    -------
    xr.DataArray
        DataArray containing block extremes with appropriate attributes.

    Raises
    ------
    ValueError
        If extremes_type is not 'max' or 'min'.

    Notes
    -----
    Uses year-end resampling ('YE') to ensure consistent block boundaries.
    Preserves original attributes and adds extremes type information.

    """
    resample_rule = f"{block_size}YE"
    resampler = da.resample(time=resample_rule)

    if extremes_type == "max":
        bms = resampler.max(keep_attrs=True)
        bms.attrs["extremes type"] = "maxima"
    elif extremes_type == "min":
        bms = resampler.min(keep_attrs=True)
        bms.attrs["extremes type"] = "minima"
    else:
        raise ValueError('extremes_type needs to be either "max" or "min"')

    return bms


def _check_effective_sample_size_optimized(da: xr.DataArray, block_size: int) -> None:
    """Optimized effective sample size calculation for large datasets.

    This function calculates the effective sample size (ESS) to assess
    the statistical independence of data points within blocks. ESS accounts
    for temporal autocorrelation that reduces the effective number of
    independent observations.

    Parameters
    ----------
    da : xr.DataArray
        Input DataArray with time dimension.
    block_size : int
        Size of blocks in years for analysis.

    Warns
    -----
    UserWarning
        If the average ESS is below the recommended threshold, indicating
        potential bias in extreme value statistics.

    Notes
    -----
    The function handles both gridded (x, y, time) and timeseries (time) data.
    For gridded data, spatial sampling is used to estimate representative ESS.
    A warning is issued if ESS falls below the minimum threshold.

    """
    try:
        if "x" in da.dims and "y" in da.dims:
            # For gridded data, use chunked computation
            average_ess = _calc_average_ess_gridded_optimized(da, block_size)
        elif da.dims == ("time",):
            # For timeseries data
            average_ess = _calc_average_ess_timeseries_optimized(da, block_size)
        else:
            print(
                f"WARNING: the effective sample size can only be checked for timeseries or spatial data. "
                f"You provided data with the following dimensions: {da.dims}."
            )
            return

        if average_ess < MIN_ESS_THRESHOLD:
            print(
                f"WARNING: The average effective sample size in your data is {round(average_ess, 2)} per block, "
                f"which is lower than a standard target of around {MIN_ESS_THRESHOLD}. This may result in biased estimates of "
                f"extreme value distributions when calculating return values, periods, and probabilities from this data. "
                f"Consider using a longer block size to increase the effective sample size in each block of data."
            )
    except (ValueError, RuntimeError) as e:
        print(f"WARNING: Could not calculate effective sample size: {e}")


def _calc_average_ess_gridded_optimized(data: xr.DataArray, block_size: int) -> float:
    """Optimized ESS calculation for gridded data using vectorized operations.

    This function calculates the effective sample size for gridded datasets
    by sampling spatial locations and computing ESS for each time series.
    Uses efficient algorithms that adapt to dataset size.

    Parameters
    ----------
    data : xr.DataArray
        Input gridded DataArray with time, x, and y dimensions.
    block_size : int
        Size of blocks in years for sampling temporal data.

    Returns
    -------
    float
        Average effective sample size across sampled spatial locations.
        Returns fallback value if calculation fails.

    Notes
    -----
    For large datasets (>1000 time points), uses logarithmic lag sampling
    for autocorrelation approximation. For smaller datasets, uses exact
    calculation. Spatial sampling strategy differs between Dask and
    in-memory arrays for optimal performance.

    """
    try:
        # Use xarray's efficient groupby operations
        yearly_groups = data.groupby(data.time.dt.year)

        # Calculate ESS for each year block using apply with optimized function
        def calc_ess_optimized(year_data):
            """Calculate ESS for a single year of data."""
            # Use autocorrelation function efficiently
            if (
                len(year_data.time) < MIN_TIME_POINTS
            ):  # Skip years with insufficient data
                return np.nan

            # Simplified ESS calculation for large datasets
            # Use approximate autocorrelation for speed
            n = len(year_data.time)
            if n > LARGE_DATASET_THRESHOLD:  # Use approximation for large datasets
                # Sample autocorrelation at key lags
                max_lag = min(n // MAX_LAG_DIVISOR, MAX_LAG_SAMPLES_GRIDDED)
                lags = np.logspace(
                    0, np.log10(max_lag), AUTOCORR_LAG_SAMPLES_GRIDDED
                ).astype(int)
                autocorr_sum = 0
                for lag in lags:
                    if lag < n - 1:
                        corr = np.corrcoef(
                            year_data.values[:-lag], year_data.values[lag:]
                        )[0, 1]
                        if not np.isnan(corr):
                            autocorr_sum += corr * (n - lag) / n
                ess = n / (1 + 2 * autocorr_sum)
            else:
                # Use exact calculation for smaller datasets
                ess = calculate_ess(year_data).item()

            return ess

        # Apply ESS calculation efficiently
        ess_values = []

        def process_spatial_ess(year_data, sample_step, max_samples):
            """Process spatial ESS sampling for a given year of data."""
            spatial_sample = year_data.isel(
                x=slice(None, None, max(1, year_data.sizes["x"] // sample_step)),
                y=slice(None, None, max(1, year_data.sizes["y"] // sample_step)),
            )
            stacked = spatial_sample.stack(spatial=("x", "y"))
            ess_spatial = []
            for i in range(min(max_samples, stacked.sizes["spatial"])):
                try:
                    ess_val = calc_ess_optimized(stacked.isel(spatial=i))
                    if not np.isnan(ess_val):
                        ess_spatial.append(ess_val)
                except (ValueError, RuntimeError, IndexError):
                    continue
            return ess_spatial

        if hasattr(data.data, "chunks"):
            # For Dask arrays, use map_blocks for efficiency
            for year, year_data in yearly_groups:
                if year % block_size == 0:  # Sample every block_size years
                    if "x" in year_data.dims and "y" in year_data.dims:
                        ess_spatial = process_spatial_ess(
                            year_data,
                            DASK_SPATIAL_SAMPLE_STEP,
                            DASK_MAX_SPATIAL_SAMPLES,
                        )
                        if ess_spatial:
                            ess_values.extend(ess_spatial)
        else:
            # For in-memory arrays, process directly
            for year, year_data in yearly_groups:
                if year % block_size == 0:
                    try:
                        if "x" in year_data.dims and "y" in year_data.dims:
                            ess_spatial = process_spatial_ess(
                                year_data,
                                MEMORY_SPATIAL_SAMPLE_STEP,
                                MEMORY_MAX_SPATIAL_SAMPLES,
                            )
                            if ess_spatial:
                                ess_values.extend(ess_spatial)
                        else:
                            ess_val = calc_ess_optimized(year_data)
                            if not np.isnan(ess_val):
                                ess_values.append(ess_val)
                    except (ValueError, RuntimeError, IndexError):
                        continue

        return np.nanmean(ess_values) if ess_values else FALLBACK_ESS_VALUE
    except (ValueError, RuntimeError, MemoryError):
        # Fallback to simple estimate
        return FALLBACK_ESS_VALUE


def _calc_average_ess_timeseries_optimized(
    data: xr.DataArray, block_size: int
) -> float:
    """Optimized ESS calculation for timeseries data.

    This function calculates the effective sample size for timeseries data
    by processing temporal blocks and computing autocorrelation-adjusted ESS.
    Uses efficient algorithms that adapt to time series length.

    Parameters
    ----------
    data : xr.DataArray
        Input timeseries DataArray with time dimension.
    block_size : int
        Size of blocks in years for resampling temporal data.

    Returns
    -------
    float
        Average effective sample size across temporal blocks.
        Returns fallback value if calculation fails.

    Notes
    -----
    For large time series (>500 time points), uses logarithmic lag sampling
    for autocorrelation approximation. For smaller time series, uses exact
    calculation from the threshold_tools module.

    """
    try:
        # Use efficient resampling for block-wise ESS calculation
        block_resampler = data.resample(time=f"{block_size}YS")

        ess_values = []
        for _, block_data in block_resampler:
            try:
                n = len(block_data.time)
                if n < MIN_TIME_POINTS:
                    continue

                # Simplified ESS calculation
                if (
                    n > LARGE_TIMESERIES_THRESHOLD
                ):  # Use approximation for large time series
                    # Sample autocorrelation at key lags
                    max_lag = min(n // MAX_LAG_DIVISOR, MAX_LAG_SAMPLES_TIMESERIES)
                    lags = np.logspace(
                        0, np.log10(max_lag), AUTOCORR_LAG_SAMPLES_TIMESERIES
                    ).astype(int)
                    autocorr_sum = 0
                    values = block_data.values
                    for lag in lags:
                        if lag < n - 1:
                            corr = np.corrcoef(values[:-lag], values[lag:])[0, 1]
                            if not np.isnan(corr):
                                autocorr_sum += corr * (n - lag) / n
                    ess = n / (1 + 2 * autocorr_sum)
                else:
                    # Use exact calculation for smaller datasets
                    ess = calculate_ess(block_data).item()

                if not np.isnan(ess):
                    ess_values.append(ess)
            except (ValueError, RuntimeError, IndexError):
                continue

        return np.nanmean(ess_values) if ess_values else FALLBACK_ESS_VALUE
    except (ValueError, RuntimeError, MemoryError):
        return FALLBACK_ESS_VALUE


def _set_block_maxima_attributes(
    bms: xr.DataArray,
    duration,
    groupby,
    grouped_duration,
    extremes_type: str,
    block_size: int,
) -> xr.DataArray:
    """Set attributes efficiently for block maxima DataArray.

    This function adds comprehensive metadata attributes to the block maxima
    DataArray to document the processing parameters and methods used.

    Parameters
    ----------
    bms : xr.DataArray
        Block maxima DataArray to add attributes to.
    duration : tuple or None
        Duration specification used in processing.
    groupby : tuple or None
        Groupby specification used in processing.
    grouped_duration : tuple or None
        Grouped duration specification used in processing.
    extremes_type : str
        Type of extreme ('max' or 'min').
    block_size : int
        Size of blocks in years.

    Returns
    -------
    xr.DataArray
        DataArray with comprehensive attributes added.

    Notes
    -----
    Attributes include processing parameters and method documentation
    for reproducibility and data provenance tracking.

    """
    attrs = {
        "duration": duration,
        "groupby": groupby,
        "grouped_duration": grouped_duration,
        "extreme_value_extraction_method": "block maxima",
        "block_size": f"{block_size} year",
        "timeseries_type": f"block {extremes_type} series",
    }
    return bms.assign_attrs(attrs)


def _handle_nan_values_optimized(bms: xr.DataArray) -> xr.DataArray:
    """Handle NaN values efficiently using vectorized operations.

    This function processes NaN values in block maxima DataArrays by checking
    for completely empty datasets and dropping time steps with NaN values.
    Provides informative feedback about data cleaning operations.

    Parameters
    ----------
    bms : xr.DataArray
        Block maxima DataArray that may contain NaN values.

    Returns
    -------
    xr.DataArray
        Cleaned DataArray with NaN time steps removed.

    Raises
    ------
    ValueError
        If the input DataArray contains only NaN values and cannot
        be processed for block maxima extraction.

    Notes
    -----
    For Dask arrays, the null count computation is optimized using
    the compute() method. The function provides user feedback about
    the number of dropped time steps for transparency.

    """
    if not bms.isnull().any():
        return bms

    # Check if all values are null
    total_null = bms.isnull().sum()
    if hasattr(total_null, "compute"):
        total_null = total_null.compute()

    if total_null.item() == bms.size:
        raise ValueError(
            "ERROR: The given `da_series` does not include any recorded values for this variable, "
            "and we cannot create block maximums off of an empty DataArray."
        )

    # Drop NaN values along time dimension
    dropped_bms = bms.dropna(dim="time")
    dropped_count = bms.sizes["time"] - dropped_bms.sizes["time"]

    if dropped_count > 0:
        name_str = f" {bms.name}" if bms.name else ""
        print(
            f"Dropping {dropped_count} block maxima NaNs across entire{name_str} DataArray. "
            f"Please see guidance for more information."
        )

    return dropped_bms


def extend_time_domain(
    result: Dict[str, Union[xr.Dataset, xr.DataArray]],
) -> Union[xr.Dataset, xr.DataArray]:
    """Extend the time domain of the input data to cover 1980-2100.

    This method ensures that all SSP scenarios have historical data
    included in the time series, allowing for proper warming level calculations.
    This is handled by concatenating historical data with SSP data and updating
    the attributes to that of the SSP data. Historical data is expected to be
    available in the input dictionary with keys formatted the same as SSP keys
    but with "historical" instead of r"ssp.{3}" (e.g., "ssp245" becomes "historical").

    Parameters
    ----------
    result : Dict[str, Union[xr.Dataset | xr.DataArray]]
        A dictionary containing time-series data with keys representing different scenarios.

    Returns
    -------
    Union[xr.Dataset, xr.DataArray]
        The extended time-series data.

    Notes
    -----
    - By construction, this function will drop reanalysis data.

    """
    ret = {}

    # don't run twice, check if historical data was already prepended
    # this is to avoid unnecessary processing if the data has already been extended
    if any(v.attrs.get("historical_prepended", False) for v in result.values()):
        return result  # type: ignore

    print(
        "\n\nINFO: Prepending historical data to SSP scenarios."
        "\n      This is the default concatenation strategy for retrieved data in climakitae."
        '\n      To change this behavior, set `"concat": "sim"` in your processes dictionary.\n\n'
    )

    # Process SSP scenarios by finding and prepending historical data
    for key, data in result.items():
        if "ssp" not in key:
            # drop non-SSP data since historical gets prepended
            continue

        # Find corresponding historical key by replacing SSP pattern with "historical"
        hist_key = re.sub(r"ssp.{3}", "historical", key)

        if hist_key not in result:
            warnings.warn(
                f"\n\nNo historical data found for {key} with key {hist_key}. "
                f"\nHistorical data is required for time domain extension. "
                f"\nKeeping original SSP data without historical extension.",
                UserWarning,
                stacklevel=999,
            )
            ret[key] = data
            continue

        # Concatenate historical and SSP data along time dimension
        try:
            hist_data = result[hist_key]
            # Use proper xr.concat with explicit typing
            if isinstance(data, xr.Dataset) and isinstance(hist_data, xr.Dataset):
                extended_data = xr.concat([hist_data, data], dim="time")  # type: ignore
            elif isinstance(data, xr.DataArray) and isinstance(hist_data, xr.DataArray):
                extended_data = xr.concat([hist_data, data], dim="time")  # type: ignore
            else:
                # Handle mixed types by converting to same type
                if isinstance(data, xr.Dataset):
                    if isinstance(hist_data, xr.DataArray):
                        hist_data = hist_data.to_dataset()
                    extended_data = xr.concat([hist_data, data], dim="time")  # type: ignore
                else:  # data is DataArray
                    if isinstance(hist_data, xr.Dataset):
                        data = data.to_dataset()
                    extended_data = xr.concat([hist_data, data], dim="time")  # type: ignore

            # Preserve SSP attributes
            extended_data.attrs.update(data.attrs)
            # add key attr indicating historical data was prepended
            extended_data.attrs["historical_prepended"] = True
            ret[key] = extended_data

        except (ValueError, TypeError, KeyError, AttributeError) as e:
            warnings.warn(
                f"\n\nFailed to concatenate historical and SSP data for {key}: {e}"
                f"\nSince no historical data is available, this data is dropped.",
                UserWarning,
                stacklevel=999,
            )

    return ret
