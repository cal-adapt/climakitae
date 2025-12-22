"""
DataProcessor MetricCalc
"""

import warnings
from typing import Any, Dict, Iterable, Union

import dask
import numpy as np
import pandas as pd
import scipy.stats as stats
import xarray as xr
from dask.diagnostics.progress import ProgressBar

from climakitae.core.constants import _NEW_ATTRS_KEY, UNSET
from climakitae.explore.threshold_tools import _get_distr_func, _get_fitted_distr
from climakitae.new_core.data_access.data_access import DataCatalog
from climakitae.new_core.processors.abc_data_processor import (
    DataProcessor,
    register_processor,
)
from climakitae.new_core.processors.processor_utils import _get_block_maxima_optimized

# Constants for data size thresholds and processing
BYTES_TO_MB_FACTOR = 1e6  # Conversion factor from bytes to megabytes
BYTES_TO_GB_FACTOR = 1e9  # Conversion factor from bytes to gigabytes
SMALL_ARRAY_THRESHOLD_BYTES = 1e7  # 10MB threshold for small arrays
MEDIUM_ARRAY_THRESHOLD_BYTES = 1e9  # 1GB threshold for medium arrays
PERCENTILE_TO_QUANTILE_FACTOR = 100.0  # Convert percentiles to quantiles
DEFAULT_BATCH_SIZE = 50  # Default batch size for processing simulations
MIN_VALID_DATA_POINTS = 3  # Minimum data points required for statistical fitting
SIGNIFICANCE_LEVEL = 0.05  # P-value threshold for statistical significance
MIN_WORKERS_COUNT = 2  # Minimum number of workers for parallel processing
DEFAULT_YEARLY_CHUNK_DAYS = 365  # Default chunk size for yearly data
YEARLY_CHUNK_DIVISOR = 10  # Divisor for calculating yearly chunk size
DEFAULT_SIM_CHUNK_SIZE = 2  # Default chunk size for simulations
DEFAULT_SPATIAL_CHUNK_SIZE = 50  # Default chunk size for spatial dimensions
NUMERIC_PRECISION_DECIMAL_PLACES = 2  # Decimal places for numeric output formatting
SCIENTIFIC_NOTATION_PRECISION = 3  # Precision for scientific notation formatting
MIN_SIMULATIONS_FOR_PARALLEL = 4  # Minimum simulations required for parallel processing
RETURN_VALUE_PRECISION = 5  # Decimal places for return value rounding
P_VALUE_PRECISION = 4  # Decimal places for p-value rounding
MEMORY_SAFETY_FACTOR = 0.5  # Safety factor for memory usage calculations
MB_TO_BYTES_FACTOR = 1000  # Conversion factor from MB to bytes
MEMORY_DIVISOR_FACTOR = 2  # Divisor for memory constraint calculations

# Constants for adaptive batch sizing
MEDIUM_DASK_BATCH_SIZE = 50  # Batch size for medium Dask arrays
LARGE_DATASET_MIN_BATCH_SIZE = 10  # Minimum batch size for large datasets
LARGE_DATASET_MAX_BATCH_SIZE = 200  # Maximum batch size for large datasets
MEMORY_PER_SIM_MB_ESTIMATE = (
    100  # Estimated memory usage per simulation in MB (more aggressive)
)
TARGET_MEMORY_USAGE_FRACTION = 0.85  # Target fraction of available memory to use

# Constants for return value calculations
DEFAULT_EVENT_DURATION_DAYS = 1  # Default event duration in days
DEFAULT_BLOCK_SIZE_YEARS = 1  # Default block size in years

# Import functions for 1-in-X calculations
try:
    EXTREME_VALUE_ANALYSIS_AVAILABLE = True
except ImportError as e:
    EXTREME_VALUE_ANALYSIS_AVAILABLE = False
    warnings.warn(f"Extreme value analysis functions not available: {e}")


@register_processor("metric_calc", priority=7500)
class MetricCalc(DataProcessor):
    """
    Calculate metrics (min, max, mean, median), percentiles, and 1-in-X return values on data.

    This processor applies statistical operations to xarray datasets and data arrays,
    including percentile calculations, basic metrics like min, max, mean, and median,
    and extreme value analysis for 1-in-X return period calculations.
    Multiple calculation types can be performed simultaneously.

    Parameters
    ----------
    value : dict[str, Any]
        Configuration dictionary with the following supported keys:

        Basic Metrics:
        - metric (str, optional): Metric to calculate. Supported values:
          "min", "max", "mean", "median". Default: "mean"
        - percentiles (list, optional): List of percentiles to calculate (0-100).
          Default: None
        - percentiles_only (bool, optional): If True and percentiles are specified,
          only calculate percentiles (skip metric). Default: False
        - dim (str or list, optional): Dimension(s) along which to calculate the metric/percentiles.
          Default: "time"
        - keepdims (bool, optional): Whether to keep the dimensions being reduced. Default: False
        - skipna (bool, optional): Whether to skip NaN values in calculations. Default: True

        1-in-X Return Value Analysis:
        - one_in_x (dict, optional): Configuration for 1-in-X extreme value analysis.
          If provided, performs extreme value analysis instead of basic metrics. Keys:
          - return_periods (list): List of return periods (e.g., [10, 25, 50, 100])
          - distribution (str, optional): Distribution for fitting ("gev", "genpareto", "gamma"). Default: "gev"
          - extremes_type (str, optional): "max" or "min". Default: "max"
          - event_duration (tuple, optional): Event duration as (int, str). Default: (1, "day")
          - block_size (int, optional): Block size in years. Default: 1
          - goodness_of_fit_test (bool, optional): Perform KS test. Default: True
          - print_goodness_of_fit (bool, optional): Print p-value results. Default: True
          - variable_preprocessing (dict, optional): Variable-specific preprocessing options

    Examples
    --------
    Calculate mean over time:
    >>> metric_proc = MetricCalc({"metric": "mean"})

    Calculate 25th, 50th, 75th percentiles:
    >>> metric_proc = MetricCalc({"percentiles": [25, 50, 75]})

    Calculate 1-in-X return values:
    >>> metric_proc = MetricCalc({
    ...     "one_in_x": {
    ...         "return_periods": [10, 25, 50, 100],
    ...         "distribution": "gev",
    ...         "extremes_type": "max"
    ...     }
    ... })

    Calculate both percentiles and mean:
    >>> metric_proc = MetricCalc({"metric": "mean", "percentiles": [25, 50, 75]})

    Calculate only percentiles (no metric):
    >>> metric_proc = MetricCalc({"percentiles": [25, 50, 75], "percentiles_only": True})

    Calculate max over simulation dimension:
    >>> metric_proc = MetricCalc({"metric": "max", "dim": "simulation"})

    Calculate median over multiple dimensions:
    >>> metric_proc = MetricCalc({"metric": "median", "dim": ["time", "simulation"]})

    Notes
    -----
    - By default, both percentiles (if specified) and metrics are calculated
    - Use percentiles_only=True to calculate only percentiles when both are specified
    - Percentiles are calculated using xarray's quantile method
    - The processor preserves all attributes and coordinates from the input data
    - Results maintain the same structure as input (Dataset/DataArray/Iterable)
    - When both percentiles and metrics are calculated, the results are combined into a single output
    """

    def __init__(self, value: Dict[str, Any]):
        """
        Initialize the processor.

        Parameters
        ----------
        value : dict[str, Any]
            Configuration values for the metric calculation operation. Expected keys:
            - metric (str, optional): Metric to calculate ("min", "max", "mean", "median"). Default: "mean"
            - percentiles (list, optional): List of percentiles to calculate. Default: None
            - percentiles_only (bool, optional): If True and percentiles are specified,
              only calculate percentiles (skip metric). Default: False
            - dim (str or list, optional): Dimension(s) to reduce. Default: "time"
            - keepdims (bool, optional): Keep dimensions. Default: False
            - skipna (bool, optional): Skip NaN values. Default: True
            - one_in_x (dict, optional): 1-in-X extreme value analysis configuration

        Raises
        ------
        ValueError
            If invalid metric or percentile values are provided
        """
        self.value = value
        self.name = "metric_calc"
        self._catalog = UNSET  # Initialize catalog attribute

        # Basic metric parameters
        self.metric = value.get("metric", "mean")
        self.percentiles = value.get("percentiles", UNSET)
        self.percentiles_only = value.get("percentiles_only", False)
        self.dim = value.get("dim", "time")
        self.keepdims = value.get("keepdims", False)
        self.skipna = value.get("skipna", True)

        # 1-in-X parameters
        self.one_in_x_config = value.get("one_in_x", UNSET)
        if self.one_in_x_config is not UNSET:
            self._setup_one_in_x_parameters()

        # Auto-configure Dask for large dataset processing
        if self.one_in_x_config is not UNSET:
            self.optimize_dask_performance()

    def _setup_one_in_x_parameters(self):
        """Setup parameters for 1-in-X calculations."""
        if not EXTREME_VALUE_ANALYSIS_AVAILABLE:
            raise ValueError(
                "Extreme value analysis functions are not available. Please check climakitae installation."
            )

        # Type guard to ensure one_in_x_config is not None
        if self.one_in_x_config is UNSET:
            raise ValueError(
                "one_in_x_config cannot be UNSET when calling _setup_one_in_x_parameters"
            )

        # Required parameter
        self.return_periods = self.one_in_x_config.get("return_periods")
        if self.return_periods is UNSET:
            raise ValueError("return_periods is required for 1-in-X calculations")

        # Convert to numpy array for consistency
        if not isinstance(self.return_periods, (list, np.ndarray)):
            self.return_periods = np.array([self.return_periods])
        elif isinstance(self.return_periods, list):
            self.return_periods = np.array(self.return_periods)

        # Optional parameters with defaults
        self.distribution = self.one_in_x_config.get("distribution", "gev")
        self.extremes_type = self.one_in_x_config.get("extremes_type", "max")
        self.event_duration = self.one_in_x_config.get("event_duration", (1, "day"))
        self.block_size = self.one_in_x_config.get("block_size", 1)
        self.goodness_of_fit_test = self.one_in_x_config.get(
            "goodness_of_fit_test", True
        )
        self.print_goodness_of_fit = self.one_in_x_config.get(
            "print_goodness_of_fit", True
        )
        self.variable_preprocessing = self.one_in_x_config.get(
            "variable_preprocessing", {}
        )

    def execute(
        self,
        result: Union[
            xr.Dataset, xr.DataArray, Iterable[Union[xr.Dataset, xr.DataArray]]
        ],
        context: Dict[str, Any],
    ) -> Union[xr.Dataset, xr.DataArray, Iterable[Union[xr.Dataset, xr.DataArray]]]:
        """
        Run the processor

        Parameters
        ----------
        result : xr.Dataset | xr.DataArray | Iterable[xr.Dataset | xr.DataArray]
            The data on which to calculate metrics.

        context : dict
            The context for the processor. This is not used in this
            implementation but is included for consistency with the
            DataProcessor interface.

        Returns
        -------
        Union[xr.Dataset, xr.DataArray, Iterable[xr.Dataset | xr.DataArray]]
            The data with calculated metrics. This can be a single Dataset/DataArray or
            an iterable of them.
        """
        ret = None

        match result:
            case xr.Dataset() | xr.DataArray():
                if self.one_in_x_config is not UNSET:
                    ret = self._calculate_one_in_x_single(result)
                else:
                    ret = self._calculate_metrics_single(result)
            case dict():
                if self.one_in_x_config is not UNSET:
                    ret = {
                        key: self._calculate_one_in_x_single(value)
                        for key, value in result.items()
                    }
                else:
                    ret = {
                        key: self._calculate_metrics_single(value)
                        for key, value in result.items()
                    }
            case list() | tuple():
                if self.one_in_x_config is not UNSET:
                    processed_data = [
                        self._calculate_one_in_x_single(item)
                        for item in result
                        if isinstance(item, (xr.Dataset, xr.DataArray))
                    ]
                else:
                    processed_data = [
                        self._calculate_metrics_single(item)
                        for item in result
                        if isinstance(item, (xr.Dataset, xr.DataArray))
                    ]
                ret = type(result)(processed_data) if processed_data else None
            case _:
                raise TypeError(
                    f"Expected xr.Dataset, xr.DataArray, dict, list, or tuple, got {type(result)}"
                )

        if ret is None:
            raise ValueError(
                "Metric calculation operation failed to produce valid results."
            )

        self.update_context(context)
        return ret

    def _calculate_metrics_single(
        self, data: Union[xr.Dataset, xr.DataArray]
    ) -> Union[xr.Dataset, xr.DataArray]:
        """
        Calculate metrics on a single Dataset or DataArray.

        Parameters
        ----------
        data : xr.Dataset | xr.DataArray
            The data on which to calculate metrics.

        Returns
        -------
        xr.Dataset | xr.DataArray
            The data with calculated metrics.
        """
        # Check if dimensions exist in the data
        if isinstance(self.dim, str):
            dims_to_check = [self.dim]
        else:
            dims_to_check = self.dim

        if isinstance(data, xr.Dataset):
            # For datasets, check if dimensions exist in any data variable
            available_dims = set()
            for var in data.data_vars:
                available_dims.update(data[var].dims)
        else:
            # For data arrays
            available_dims = set(data.dims)

        # Filter out dimensions that don't exist
        valid_dims = [dim for dim in dims_to_check if dim in available_dims]

        if not valid_dims:
            warnings.warn(
                f"\n\nNone of the specified dimensions {dims_to_check} exist in the data. "
                f"\nAvailable dimensions: {list(available_dims)}"
            )
            return data

        # Use valid dimensions for calculation
        calc_dim = valid_dims if len(valid_dims) > 1 else valid_dims[0]

        # Calculate percentiles if requested
        results = []
        if self.percentiles is not None:
            percentile_result = data.quantile(
                [p / PERCENTILE_TO_QUANTILE_FACTOR for p in self.percentiles],
                dim=calc_dim,
                keep_attrs=True,
                skipna=self.skipna,
            )

            # Rename quantile coordinate to percentiles for clarity
            if "quantile" in percentile_result.coords:
                percentile_result = percentile_result.rename({"quantile": "percentile"})
                # Update coordinate values to be the original percentile values
                percentile_result = percentile_result.assign_coords(
                    percentile=self.percentiles
                )

            results.append(percentile_result)

        # Calculate single metric (unless percentiles_only is True)
        if not self.percentiles_only:
            metric_functions = {
                "min": lambda x: x.min(
                    dim=calc_dim, keep_attrs=True, skipna=self.skipna
                ),
                "max": lambda x: x.max(
                    dim=calc_dim, keep_attrs=True, skipna=self.skipna
                ),
                "mean": lambda x: x.mean(
                    dim=calc_dim, keep_attrs=True, skipna=self.skipna
                ),
                "median": lambda x: x.median(
                    dim=calc_dim, keep_attrs=True, skipna=self.skipna
                ),
            }

            metric_result = metric_functions[self.metric](data)
            results.append(metric_result)

        # Return combined results or single result
        if len(results) == 1:
            return results[0]
        elif len(results) == 2 and self.percentiles is not None:
            # Combine percentiles and metric results
            percentile_result, metric_result = results

            # Create a combined dataset/dataarray
            if isinstance(data, xr.Dataset):
                # For datasets, we need to be more careful about combining
                combined_data = {}

                # Add percentile results
                for var_name in percentile_result.data_vars:
                    for i, p in enumerate(self.percentiles):
                        # Drop the percentile coordinate to avoid conflicts when combining
                        percentile_data = (
                            percentile_result[var_name]
                            .isel(percentile=i)
                            .drop_vars("percentile")
                        )
                        combined_data[f"{var_name}_p{p}"] = percentile_data

                # Add metric results
                for var_name in metric_result.data_vars:
                    combined_data[f"{var_name}_{self.metric}"] = metric_result[var_name]

                result = xr.Dataset(combined_data, attrs=data.attrs)

            else:
                # For DataArrays, create a new dimension for the different statistics
                stats_list = [f"p{p}" for p in self.percentiles] + [self.metric]

                # Stack percentile and metric results
                all_values = []
                for i in range(len(self.percentiles)):
                    all_values.append(percentile_result.isel(percentile=i))
                all_values.append(metric_result)

                result = xr.concat(all_values, dim="statistic")
                result = result.assign_coords(statistic=stats_list)

            return result
        else:
            # Should not reach here, but return the first result as fallback
            return results[0] if results else data

    def _calculate_one_in_x_single(
        self, data: Union[xr.Dataset, xr.DataArray]
    ) -> xr.Dataset:
        """
        Calculate 1-in-X return values on a single Dataset or DataArray.

        Optimized version that processes simulations in vectorized batches where possible.

        Parameters
        ----------
        data : xr.Dataset | xr.DataArray
            The data on which to calculate 1-in-X return values.

        Returns
        -------
        xr.Dataset
            Dataset with return_value and p_values DataArrays.
        """
        if not EXTREME_VALUE_ANALYSIS_AVAILABLE:
            raise ValueError("Extreme value analysis functions are not available")

        ### The DataArray may have missing gridcells, since the WRF grid is not lat/lon, or the clipping may have may this DataArray an irregular shape.
        ### To resolve this, we will select all the valid gridcells from `data`, pass them through the calculation, and then re-insert them back into the original grid shape with NaNs where appropriate.

        # Handle Dataset vs DataArray
        if isinstance(data, xr.Dataset):
            # For datasets, process the first data variable
            var_name = list(data.data_vars)[0]
            data_array = data[var_name]
        else:
            data_array = data
            var_name = str(data_array.name) if data_array.name else "data"

        # Check if we have a simulation dimension
        if "sim" not in data_array.dims:
            raise ValueError("Data must have a 'sim' dimension for 1-in-X calculations")

        # Smart memory management for Dask arrays
        if hasattr(data_array, "chunks") and data_array.chunks is not None:
            print("Detected Dask array - using optimized chunked processing...")
            # Only compute if the array is small enough, otherwise process in chunks
            total_size = data_array.nbytes  # Estimate bytes (assuming float64)
            print(
                f"Total size of data array: {total_size / BYTES_TO_MB_FACTOR:.{NUMERIC_PRECISION_DECIMAL_PLACES}f} MB"
            )

            # Use different strategies based on data size
            if (
                total_size < SMALL_ARRAY_THRESHOLD_BYTES
            ):  # Less than 10MB - load into memory
                print("Small array detected - loading into memory...")
                data_array = data_array.compute()
                use_dask_optimization = False
            elif (
                total_size < MEDIUM_ARRAY_THRESHOLD_BYTES
            ):  # 10MB - 1GB - use chunked processing with Dask
                print("Medium array detected - using Dask-aware chunked processing...")
                use_dask_optimization = (
                    "medium"  # Use special handling for medium arrays
                )
            else:  # > 1GB - use Dask optimization
                print("Large array detected - using Dask optimization...")
                use_dask_optimization = True
        else:
            use_dask_optimization = False

        # Check if we have a time dimension, and add dummy time if needed
        if "time" not in data_array.dims:
            data_array = self._add_dummy_time_if_needed(data_array, data.frequency)

        # Apply variable-specific preprocessing
        data_array = self._preprocess_variable_for_one_in_x(data_array, var_name)

        print(
            f"Calculating 1-in-{self.return_periods} year return values using {self.distribution} distribution..."
        )
        return self._calculate_one_in_x_vectorized(data_array)

    def _fit_return_values_1d(
        self,
        block_maxima_1d: np.ndarray,
        return_periods: np.ndarray,
        distr: str = "gev",
        extremes_type: str = "max",
        get_p_value: bool = False,
    ) -> np.ndarray:
        """docstring goes here"""
        n_return_periods = len(return_periods)

        # Remove NaN values
        valid_data = block_maxima_1d[~np.isnan(block_maxima_1d)]

        # Need at least 3 valid data points for meaningful distribution fitting
        if len(valid_data) < MIN_VALID_DATA_POINTS:
            return np.full_like(return_periods, np.nan, dtype=float), np.nan

        try:
            parameters, fitted_distr = _get_fitted_distr(
                valid_data, distr, _get_distr_func(distr)
            )

            match distr:
                case "gev":
                    cdf = "genextreme"
                    args = (parameters["c"], parameters["loc"], parameters["scale"])
                case "gumbel":
                    cdf = "gumbel_r"
                    args = (parameters["loc"], parameters["scale"])
                case "weibull":
                    cdf = "weibull_min"
                    args = (parameters["c"], parameters["loc"], parameters["scale"])
                case "pearson3":
                    cdf = "pearson3"
                    args = (parameters["skew"], parameters["loc"], parameters["scale"])
                case "genpareto":
                    cdf = "genpareto"
                    args = (parameters["c"], parameters["loc"], parameters["scale"])
                case "gamma":
                    cdf = "gamma"
                    args = (parameters["a"], parameters["loc"], parameters["scale"])
                case _:
                    raise ValueError(
                        'invalid distribution type. expected one of the following: ["gev", "gumbel", "weibull", "pearson3", "genpareto", "gamma"]'
                    )

            if get_p_value:
                ks = stats.kstest(valid_data, cdf, args=args)
                p_value = ks[1]

            # Calculate return values for each return period
            return_values = np.empty(n_return_periods)
            event_prob = 1.0 / return_periods  # Assuming 1-year blocks
            if extremes_type == "max":
                return_events = 1.0 - event_prob
            else:  # min
                return_events = event_prob
            return_values = np.round(
                fitted_distr.ppf(return_events), RETURN_VALUE_PRECISION
            )
            if get_p_value:
                return return_values, p_value
            else:
                return return_values, np.nan

        except (ValueError, RuntimeError, np.linalg.LinAlgError):
            return np.full_like(return_periods, np.nan), np.nan

    def _calculate_one_in_x_vectorized(self, data_array: xr.DataArray) -> xr.Dataset:
        """
        Vectorized calculation of 1-in-X values for better performance.

        This method attempts to process multiple simulations simultaneously.
        """
        if not EXTREME_VALUE_ANALYSIS_AVAILABLE:
            raise ValueError("Extreme value analysis functions are not available")

        # Extract block maxima for all simulations at once
        print("Extracting block maxima for all simulations...")

        # Configure block maxima extraction
        kwargs = {
            "extremes_type": self.extremes_type,
            "check_ess": False,
            "block_size": self.block_size,
        }

        if self.event_duration == (1, "day"):
            kwargs["groupby"] = self.event_duration
        elif self.event_duration[1] == "hour":
            kwargs["duration"] = self.event_duration

        # Process simulations in batches to manage memory
        # Use adaptive batch sizing based on available memory
        try:
            import psutil

            available_memory_gb = psutil.virtual_memory().available / BYTES_TO_GB_FACTOR

            # Calculate adaptive batch size - more conservative for vectorized processing
            estimated_batch_size = int(
                (
                    available_memory_gb
                    * TARGET_MEMORY_USAGE_FRACTION
                    * MB_TO_BYTES_FACTOR
                )
                / (MEMORY_PER_SIM_MB_ESTIMATE * 2)  # More conservative multiplier
            )

            # Clamp between reasonable values using large dataset constants
            batch_size = max(
                LARGE_DATASET_MIN_BATCH_SIZE,  # Minimum batch size
                min(
                    estimated_batch_size,
                    LARGE_DATASET_MAX_BATCH_SIZE,
                    len(data_array.sim),
                ),  # Use large dataset max
            )

            print(
                f"Using adaptive batch size: {batch_size} simulations (available memory: {available_memory_gb:.1f}GB)"
            )

        except ImportError:
            # Fallback if psutil not available
            batch_size = min(MEDIUM_DASK_BATCH_SIZE, len(data_array.sim))
            print(f"Using default batch size: {batch_size} simulations")

        # Extract simulation values and select a batch of simulations to process
        sim_values = data_array.sim.values
        batch_sims = sim_values[0 : 0 + batch_size]

        # Compute block maxima for the selected batch of simulations
        block_maxima = _get_block_maxima_optimized(
            data_array.sel(sim=batch_sims), **kwargs
        ).squeeze()

        # Determine the time dimension to reduce over (either "time" or "time_delta")
        time_dim = ["time" if "time" in block_maxima.dims else "time_delta"][0]

        # Calculate optimal chunk sizes for Dask processing
        optimal_chunks = self._get_optimal_chunks(block_maxima)
        block_maxima = block_maxima.chunk(optimal_chunks)

        # Apply the return value fitting function to each spatial location
        return_values, d_stats, p_values = (
            xr.apply_ufunc(  # Result shape: (lat/y/spatial_1, lon/x/spatial_2, return_period)
                self._fit_return_values_1d,
                block_maxima,  # (time, lat, lon) or (time, y, x) or (time, spatial_1, spatial_2)
                kwargs={
                    "return_periods": self.return_periods,
                    "distr": self.distribution,
                    "get_p_value": self.goodness_of_fit_test,
                },
                input_core_dims=[
                    [time_dim]
                ],  # "time_dim" is the dimension we reduce over
                output_core_dims=[
                    ["one_in_x"],
                    [],
                ],  # We need to process each spatial location individually in a vectorized manner
                output_sizes={
                    "one_in_x": len(self.return_periods),
                },
                output_dtypes=("float", "float"),
                vectorize=True,  # Enable vectorized processing for better performance
                dask="parallelized",  # Use Dask for parallelized computation
                dask_gufunc_kwargs={
                    "output_sizes": {"one_in_x": len(self.return_periods)},
                    "allow_rechunk": True,  # Allow rechunking for better memory management
                },
            )
        )

        # If goodness-of-fit test is not requested, set p_values to None
        if not self.goodness_of_fit_test:
            p_values = None

        # Assign return periods as coordinates to the return values
        return_values = return_values.assign_coords(one_in_x=self.return_periods)

        # Combine the results into one Dataset to be loaded into memory
        with ProgressBar():
            combined_ds = self._create_one_in_x_result_dataset(
                return_values, p_values, data_array
            ).compute()

        return combined_ds

    def _preprocess_variable_for_one_in_x(
        self, data: xr.DataArray, var_name: str
    ) -> xr.DataArray:
        """
        Apply variable-specific preprocessing for 1-in-X calculations.

        Parameters
        ----------
        data : xr.DataArray
            Input data array
        var_name : str
            Variable name for preprocessing logic

        Returns
        -------
        xr.DataArray
            Preprocessed data array
        """
        # Apply precipitation-specific preprocessing
        if (
            "precipitation" in var_name.lower()
            or "pr" in var_name.lower()
            or var_name.lower() in ["precipitation (total)", "precipitation"]
        ):

            preprocessing = self.variable_preprocessing.get("precipitation", {})

            # Aggregate to daily if needed and requested
            if (
                preprocessing.get("daily_aggregation", True)
                and "1hr" in str(data.attrs.get("frequency", "")).lower()
                or "hourly" in str(data.attrs.get("frequency", "")).lower()
            ):
                data = data.resample(time="1D").sum()

            # Remove trace precipitation
            if preprocessing.get("remove_trace", True):
                threshold = preprocessing.get("trace_threshold", 1e-10)
                data = data.where(data > threshold, drop=True)

        return data

    def update_context(self, context: Dict[str, Any]):
        """
        Update the context with information about the transformation.

        Parameters
        ----------
        context : dict[str, Any]
            Parameters for processing the data.

        Note
        ----
        The context is updated in place. This method does not return anything.
        """
        if _NEW_ATTRS_KEY not in context:
            context[_NEW_ATTRS_KEY] = {}

        # Build description based on what was calculated
        description_parts = []

        if self.one_in_x_config is not None:
            # 1-in-X calculations
            return_periods_str = ", ".join(map(str, self.return_periods))
            description_parts.append(
                f"1-in-X return values for periods [{return_periods_str}] were "
                f"calculated using {self.distribution} distribution with "
                f"{self.extremes_type} extremes over {self.event_duration[0]} {self.event_duration[1]} events"
            )
        else:
            # Regular metric calculations
            if self.percentiles is not None:
                description_parts.append(
                    f"Percentiles {self.percentiles} were calculated"
                )

            if not self.percentiles_only:
                description_parts.append(f"Metric '{self.metric}' was calculated")

        if self.one_in_x_config is not None:
            transformation_description = (
                f"Process '{self.name}' applied to the data. "
                f"{' and '.join(description_parts)}."
            )
        else:
            transformation_description = (
                f"Process '{self.name}' applied to the data. "
                f"{' and '.join(description_parts)} along dimension(s): {self.dim}."
            )

        context[_NEW_ATTRS_KEY][self.name] = transformation_description

    def _add_dummy_time_if_needed(
        self, data_array: xr.DataArray, frequency: str
    ) -> xr.DataArray:
        """
        Add dummy time dimension if data has time_delta or similar warming level dimensions.

        This mimics the behavior of add_dummy_time_to_wl from the legacy code.

        Parameters
        ----------
        data_array : xr.DataArray
            Input data array that may have time_delta or *_from_center dimensions

        Returns
        -------
        xr.DataArray
            Data array with proper time dimension
        """
        # Find the warming level time dimension
        wl_time_dim = ""

        for dim in data_array.dims:
            dim_str = str(dim)
            if dim_str == "time_delta":
                wl_time_dim = "time_delta"
                break
            elif "from_center" in dim_str:
                wl_time_dim = dim_str
                break

        if wl_time_dim == "":
            raise ValueError(
                "Data must have a 'time', 'time_delta', or '*_from_center' dimension for 1-in-X calculations"
            )

        # Determine frequency and create dummy timestamps
        if wl_time_dim == "time_delta":
            # Get frequency from data array attributes
            time_freq_name = frequency
            name_to_freq = {"1hr": "h", "day": "D", "mon": "ME"}
        else:
            # Extract frequency from dimension name (e.g., 'hours_from_center' -> 'hours')
            time_freq_name = wl_time_dim.split("_")[0]
            name_to_freq = {"hours": "h", "days": "D", "months": "ME"}

        # Create dummy timestamps starting from 2000-01-01
        freq = name_to_freq.get(time_freq_name, "D")  # Default to daily
        timestamps = pd.date_range(
            "2000-01-01",
            periods=len(data_array[wl_time_dim]),
            freq=freq,
        )

        # Replace the warming level dimension with dummy timestamps and rename to 'time'
        data_array = data_array.assign_coords({wl_time_dim: timestamps}).rename(
            {wl_time_dim: "time"}
        )

        return data_array

    def set_data_accessor(self, catalog: DataCatalog):
        """
        Set the data accessor for the processor.

        Parameters
        ----------
        catalog : DataCatalog
            The data catalog to use for accessing data.

        Note
        ----
        This processor does not require data access, so this is a placeholder.
        """
        # This processor does not require data access
        self._catalog = catalog

    def _create_one_in_x_result_dataset(
        self, ret_vals: xr.DataArray, p_vals: xr.DataArray, data_array: xr.DataArray
    ) -> xr.Dataset:
        """
        Create the final result dataset for 1-in-X calculations.

        Parameters
        ----------
        ret_vals : xr.DataArray
            Return values DataArray
        p_vals : xr.DataArray
            P-values DataArray
        data_array : xr.DataArray
            Input data array for attributes

        Returns
        -------
        xr.Dataset
            Final result dataset with return_value and p_values
        """
        if p_vals is not None:
            result = xr.Dataset({"return_value": ret_vals, "p_values": p_vals})
        else:
            result = xr.Dataset({"return_value": ret_vals})

        # Add attributes
        result.attrs.update(
            {
                "groupby": f"{self.event_duration[0]} {self.event_duration[1]}",
                "fitted_distr": self.distribution,
                "sample_size": len(data_array.time),
            }
        )

        return result

    def optimize_dask_performance(self):
        """
        Optimize Dask performance for large dataset processing.

        This method configures Dask settings to improve performance when processing
        large climate datasets with extreme value analysis.
        """
        try:
            # Configure Dask for better performance with extreme value analysis
            dask_config = {
                "array.chunk-size": "128MiB",  # Reasonable chunk size for climate data
                "array.slicing.split_large_chunks": True,  # Handle large chunks better
                "distributed.worker.memory.target": 0.8,  # Use 80% of worker memory
                "distributed.worker.memory.spill": 0.9,  # Spill at 90% memory usage
                "distributed.worker.memory.pause": 0.95,  # Pause at 95% memory usage
                "distributed.worker.memory.terminate": 0.98,  # Terminate at 98% memory usage
            }

            # Apply configuration
            dask.config.set(dask_config)
            print("Dask performance configuration applied for extreme value analysis")

        except ImportError:
            print("Dask not available - skipping performance optimization")
        except Exception as e:
            print(f"Warning: Failed to optimize Dask performance: {e}")

    def _get_optimal_chunks(self, data_array: xr.DataArray) -> dict:
        """
        Calculate optimal chunk sizes for Dask arrays.

        Parameters
        ----------
        data_array : xr.DataArray
            Input data array

        Returns
        -------
        dict
            Dictionary of optimal chunk sizes by dimension
        """
        if not hasattr(data_array, "chunks") or data_array.chunks is None:
            return {}

        # Calculate optimal chunks based on data characteristics
        optimal_chunks = {}

        # For time dimension, use yearly chunks if possible
        if "time" in data_array.dims:
            time_size = data_array.sizes["time"]
            # Aim for roughly yearly chunks (DEFAULT_YEARLY_CHUNK_DAYS days) but adapt to data size
            yearly_chunk = min(
                time_size,
                max(DEFAULT_YEARLY_CHUNK_DAYS, time_size // YEARLY_CHUNK_DIVISOR),
            )
            optimal_chunks["time"] = yearly_chunk

        # For simulation dimension, process a few at a time
        if "sim" in data_array.dims:
            sim_size = data_array.sizes["sim"]
            sim_chunk = min(
                sim_size, DEFAULT_SIM_CHUNK_SIZE
            )  # Process DEFAULT_SIM_CHUNK_SIZE simulations at a time
            optimal_chunks["sim"] = sim_chunk

        # For spatial dimensions, keep chunks reasonably sized
        spatial_dims = [dim for dim in data_array.dims if dim not in ["time", "sim"]]
        for dim in spatial_dims:
            dim_size = data_array.sizes[dim]
            # Keep spatial chunks smaller to avoid memory issues
            spatial_chunk = min(dim_size, DEFAULT_SPATIAL_CHUNK_SIZE)
            optimal_chunks[dim] = spatial_chunk

        return optimal_chunks
