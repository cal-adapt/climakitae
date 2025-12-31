"""
DataProcessor MetricCalc
"""

import gc
import logging
import time as time_module
from typing import Any, Iterable

import numpy as np
import pandas as pd
import scipy.stats as stats
import xarray as xr
from tqdm.auto import tqdm

from climakitae.core.constants import (
    _NEW_ATTRS_KEY,
    BYTES_TO_GB_FACTOR,
    BYTES_TO_MB_FACTOR,
    MEDIUM_ARRAY_THRESHOLD_BYTES,
    MIN_VALID_DATA_POINTS,
    PERCENTILE_TO_QUANTILE_FACTOR,
    RETURN_VALUE_PRECISION,
    SMALL_ARRAY_THRESHOLD_BYTES,
    UNSET,
)
from climakitae.explore.threshold_tools import _get_distr_func, _get_fitted_distr
from climakitae.new_core.data_access.data_access import DataCatalog
from climakitae.new_core.processors.abc_data_processor import (
    DataProcessor,
    register_processor,
)
from climakitae.new_core.processors.processor_utils import _get_block_maxima_optimized

logger = logging.getLogger(__name__)


@register_processor("metric_calc", priority=7500)
class MetricCalc(DataProcessor):
    """
    Calculate metrics (min, max, mean, median), percentiles, and 1-in-X return values on data.

    This processor applies statistical operations to xarray datasets and data arrays,
    including percentile calculations, basic metrics like min, max, mean, and median,
    and extreme value analysis for 1-in-X return period calculations.
    Multiple calculation types can be performed simultaneously.

    **Multi-Variable Support**: When a Dataset with multiple variables is provided:
    - Basic metrics (min/max/mean/median) and percentiles are calculated for all variables
    - 1-in-X analysis processes each variable separately and prefixes results with variable names
      (e.g., `tasmax_return_values`, `tasmin_return_values`)

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

    def __init__(self, value: dict[str, Any]):
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

    def _setup_one_in_x_parameters(self):
        """Setup parameters for 1-in-X calculations."""
        # Type guard to ensure one_in_x_config is not None
        if self.one_in_x_config is UNSET:
            raise ValueError(
                "one_in_x_config cannot be UNSET when calling _setup_one_in_x_parameters"
            )

        # Required parameter
        self.return_periods = self.one_in_x_config.get("return_periods")
        if self.return_periods is None or self.return_periods is UNSET:
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
        result: xr.Dataset | xr.DataArray | Iterable[xr.Dataset | xr.DataArray],
        context: dict[str, Any],
    ) -> xr.Dataset | xr.DataArray | Iterable[xr.Dataset | xr.DataArray]:
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
        xr.Dataset | xr.DataArray | Iterable[xr.Dataset | xr.DataArray]
            The data with calculated metrics. This can be a single Dataset/DataArray or
            an iterable of them.
        """
        # Select processing function based on configuration
        process_fn = (
            self._calculate_one_in_x_single
            if self.one_in_x_config is not UNSET
            else self._calculate_metrics_single
        )

        ret = None

        match result:
            case xr.Dataset() | xr.DataArray():
                ret = process_fn(result)
            case dict():
                if not result:
                    raise ValueError(
                        "Metric calculation operation failed to produce valid results on empty arguments."
                    )
                ret = {key: process_fn(value) for key, value in result.items()}
            case list() | tuple():
                if not result:
                    raise ValueError(
                        "Metric calculation operation failed to produce valid results on empty arguments."
                    )
                processed_data = [
                    process_fn(item)
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
        self, data: xr.Dataset | xr.DataArray
    ) -> xr.Dataset | xr.DataArray:
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
            logging.warning(
                "None of the specified dimensions %s exist in the data. "
                "Available dimensions: %s",
                dims_to_check,
                list(available_dims),
            )
            return data

        # Use valid dimensions for calculation
        calc_dim = valid_dims if len(valid_dims) > 1 else valid_dims[0]

        # Calculate percentiles if requested
        results = []
        if self.percentiles is not UNSET and self.percentiles is not None:
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
        elif len(results) == 2 and self.percentiles is not UNSET:
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
                    # Drop the percentile coordinate to avoid conflicts when combining
                    val = percentile_result.isel(percentile=i)
                    if "percentile" in val.coords:
                        val = val.drop_vars("percentile")
                    all_values.append(val)
                all_values.append(metric_result)

                result = xr.concat(all_values, dim="statistic")
                result = result.assign_coords(statistic=stats_list)

            return result
        else:
            # Should not reach here, but return the first result as fallback
            return results[0] if results else data

    def _calculate_one_in_x_single(self, data: xr.Dataset | xr.DataArray) -> xr.Dataset:
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
            Dataset with return_value and p_values DataArrays. For multi-variable datasets,
            each variable's results are prefixed (e.g., `var1_return_values`, `var2_return_values`).
        """
        ### The DataArray may have missing gridcells, since the WRF grid is not lat/lon, or the clipping may have may this DataArray an irregular shape.
        ### To resolve this, we will select all the valid gridcells from `data`, pass them through the calculation, and then re-insert them back into the original grid shape with NaNs where appropriate.

        # Handle Dataset vs DataArray
        if isinstance(data, xr.Dataset):
            # Process all variables in the dataset
            var_names = list(data.data_vars)
            if len(var_names) == 0:
                raise ValueError("Dataset has no data variables")

            if len(var_names) == 1:
                # Single variable - process as before
                var_name = var_names[0]
                data_array = data[var_name]
            else:
                # Multiple variables - process each and merge results
                logger.info("Processing %d variables: %s", len(var_names), var_names)
                results = []
                for var_name in var_names:
                    logger.info("Processing variable: %s", var_name)
                    var_result = self._calculate_one_in_x_single(data[var_name])
                    # Rename variables to include source variable name
                    var_result = var_result.rename(
                        {
                            "return_values": f"{var_name}_return_values",
                            "p_values": f"{var_name}_p_values",
                        }
                    )
                    results.append(var_result)

                # Merge all results into a single dataset
                merged = xr.merge(results)
                logger.info("Merged %d variable results", len(var_names))
                return merged
        else:
            data_array = data
            var_name = str(data_array.name) if data_array.name else "data"

        # Check if we have a simulation dimension
        if "sim" not in data_array.dims:
            raise ValueError("Data must have a 'sim' dimension for 1-in-X calculations")

        # Smart memory management for Dask arrays
        if hasattr(data_array, "chunks") and data_array.chunks is not None:
            logger.info("Detected Dask array - using optimized chunked processing...")
            # Only compute if the array is small enough, otherwise process in chunks
            total_size = data_array.nbytes  # Estimate bytes (assuming float64)
            logger.info(
                "Total size of data array: %.2f MB",
                total_size / BYTES_TO_MB_FACTOR,
            )

            # Use different strategies based on data size
            if total_size < SMALL_ARRAY_THRESHOLD_BYTES:
                logger.info("Small array detected - loading into memory...")
                data_array = data_array.compute()
            elif total_size < MEDIUM_ARRAY_THRESHOLD_BYTES:
                logger.info(
                    "Medium array detected - using Dask-aware chunked processing..."
                )
            else:
                logger.info("Large array detected - using Dask optimization...")

        # Check if we have a time dimension, and add dummy time if needed
        if "time" not in data_array.dims:
            data_array = self._add_dummy_time_if_needed(data_array, data.frequency)

        # Apply variable-specific preprocessing
        data_array = self._preprocess_variable_for_one_in_x(data_array, var_name)

        logger.info(
            "Calculating 1-in-%s year return values using %s distribution...",
            self.return_periods,
            self.distribution,
        )
        return self._calculate_one_in_x_vectorized(data_array)

    def _fit_return_values_1d(
        self,
        block_maxima_1d: np.ndarray,
        return_periods: np.ndarray,
        distr: str = "gev",
        extremes_type: str = "max",
        get_p_value: bool = False,
    ) -> tuple[np.ndarray, float]:
        """
        Fit a distribution to 1D block maxima and calculate return values.

        This function fits a statistical distribution to the block maxima series
        and calculates return values for the specified return periods.

        Parameters
        ----------
        block_maxima_1d : np.ndarray
            1D array of block maxima values (e.g., annual maxima).
        return_periods : np.ndarray
            Array of return periods in years (e.g., [10, 25, 50, 100]).
        distr : str, optional
            Distribution type for fitting. Options: "gev", "gumbel", "weibull",
            "pearson3", "genpareto", "gamma". Default: "gev".
        extremes_type : str, optional
            Type of extremes: "max" for maxima, "min" for minima. Default: "max".
        get_p_value : bool, optional
            Whether to calculate and return p-value from KS test. Default: False.

        Returns
        -------
        tuple[np.ndarray, float]
            Tuple containing:
            - return_values: Array of return values for each return period
            - p_value: P-value from Kolmogorov-Smirnov test (np.nan if not calculated)

        Notes
        -----
        Requires at least MIN_VALID_DATA_POINTS (3) non-NaN values for fitting.
        Returns arrays of NaN if fitting fails.
        """
        # Remove NaN values
        valid_data = block_maxima_1d[~np.isnan(block_maxima_1d)]

        # Need at least 3 valid data points for meaningful distribution fitting
        if len(valid_data) < MIN_VALID_DATA_POINTS:
            return np.full_like(return_periods, np.nan, dtype=float), np.nan

        try:
            # _get_fitted_distr works with array-like inputs (numpy or xarray)
            # and returns (dict[str, float], rv_continuous_frozen)
            # Type ignore needed because threshold_tools.py has incorrect type annotation
            result = _get_fitted_distr(
                valid_data, distr, _get_distr_func(distr)  # type: ignore[arg-type]
            )
            parameters: dict[str, float] = result[0]  # type: ignore[index, assignment]
            fitted_distr = result[1]  # type: ignore[index]

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
            event_prob = 1.0 / return_periods  # Assuming 1-year blocks
            if extremes_type == "max":
                return_events = 1.0 - event_prob
            else:  # min
                return_events = event_prob
            return_values = np.round(
                fitted_distr.ppf(return_events), RETURN_VALUE_PRECISION  # type: ignore[union-attr]
            )
            if get_p_value:
                return return_values, p_value
            else:
                return return_values, np.nan

        except (ValueError, RuntimeError, np.linalg.LinAlgError):
            return np.full_like(return_periods, np.nan), np.nan

    def _calculate_one_in_x_vectorized(self, data_array: xr.DataArray) -> xr.Dataset:
        """
        Vectorized calculation of 1-in-X values with proper memory-safe batching.

        This method processes simulations in sequential batches to manage memory,
        while still using vectorized operations within each batch.
        """
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

        # Calculate adaptive batch size based on available memory
        batch_size = self._calculate_adaptive_batch_size(data_array)

        n_sims = len(data_array.sim)
        sim_indices = np.arange(n_sims)

        # Split simulation indices into batches
        n_batches = int(np.ceil(n_sims / batch_size))
        sim_batches = np.array_split(sim_indices, n_batches)

        logger.info(
            "Processing %d simulations in %d sequential batches (batch size: %d)...",
            n_sims,
            n_batches,
            batch_size,
        )

        # Process each batch sequentially and collect results
        batch_results = []

        # Use tqdm progress bar when logger is too quiet to show INFO messages
        use_tqdm = logger.getEffectiveLevel() > logging.INFO
        batch_iter = (
            tqdm(enumerate(sim_batches), total=n_batches, desc="Processing batches")
            if use_tqdm
            else enumerate(sim_batches)
        )

        for batch_idx, batch_indices in batch_iter:
            logger.info(
                "Batch %d/%d: simulations %d-%d",
                batch_idx + 1,
                n_batches,
                batch_indices[0] + 1,
                batch_indices[-1] + 1,
            )

            # Select simulations for this batch
            batch_sims = data_array.sim.values[batch_indices]
            batch_data = data_array.sel(sim=batch_sims)

            # Process this batch (includes its own progress bar for dask operations)
            batch_result = self._process_simulation_batch(
                batch_data, kwargs, batch_idx + 1, n_batches
            )
            batch_results.append(batch_result)

            # Explicit garbage collection after each batch
            gc.collect()

        # Combine all batch results along the simulation dimension
        logger.info(
            "All batches complete. Combining %d batch results...", len(batch_results)
        )
        combined_ds = xr.concat(batch_results, dim="sim")
        logger.info("Final result shape: %s", dict(combined_ds.dims))

        return combined_ds

    def _calculate_adaptive_batch_size(self, data_array: xr.DataArray) -> int:
        """
        Calculate adaptive batch size based on available memory and data size.

        Parameters
        ----------
        data_array : xr.DataArray
            Input data array to estimate memory requirements.

        Returns
        -------
        int
            Recommended batch size for simulation processing.
        """
        try:
            import psutil

            available_memory_gb = psutil.virtual_memory().available / BYTES_TO_GB_FACTOR

            # Get actual dimensions for memory estimation
            n_sims = len(data_array.sim)

            # Calculate actual size per simulation based on array shape
            # This accounts for clipped data (e.g., 1590 points instead of full grid)
            sim_dim_idx = data_array.dims.index("sim")
            shape_without_sim = list(data_array.shape)
            shape_without_sim.pop(sim_dim_idx)
            elements_per_sim = np.prod(shape_without_sim)

            # Memory per sim: input data + block maxima (~1/365th the size) + distribution fitting overhead
            # Use float32 = 4 bytes
            bytes_per_element = 4
            input_size_per_sim_mb = (
                elements_per_sim * bytes_per_element
            ) / BYTES_TO_MB_FACTOR

            # Block maxima reduces time dimension significantly (10950 days -> ~30 years)
            # But we need multiple copies: input, block maxima, intermediate arrays
            # Estimate 5x overhead for safe processing
            estimated_memory_per_sim_mb = input_size_per_sim_mb * 5

            # Target using only 30% of available memory for safety on shared systems
            target_memory_mb = available_memory_gb * 1000 * 0.3

            # Calculate batch size
            if estimated_memory_per_sim_mb > 0:
                estimated_batch_size = max(
                    1, int(target_memory_mb / estimated_memory_per_sim_mb)
                )
            else:
                estimated_batch_size = 1

            # Clamp to reasonable bounds - be very conservative
            batch_size = max(
                1,  # At least 1 simulation
                min(
                    estimated_batch_size,
                    20,  # Cap at 20 simulations per batch for memory safety
                    n_sims,  # Don't exceed total simulations
                ),
            )

            logger.info(
                "Memory analysis: %.1fGB available, ~%d elements/sim, "
                "~%.0fMB estimated/sim, batch size: %d",
                available_memory_gb,
                elements_per_sim,
                estimated_memory_per_sim_mb,
                batch_size,
            )

            return batch_size

        except ImportError:
            # Fallback if psutil not available - use very conservative batch size
            logger.warning("psutil not available, using conservative batch size of 2")
            return min(2, len(data_array.sim))

    def _process_simulation_batch(
        self,
        batch_data: xr.DataArray,
        block_maxima_kwargs: dict,
        batch_num: int = 1,
        total_batches: int = 1,
    ) -> xr.Dataset:
        """
        Process a single batch of simulations for 1-in-X analysis.

        Parameters
        ----------
        batch_data : xr.DataArray
            Data array containing a subset of simulations.
        block_maxima_kwargs : dict
            Keyword arguments for block maxima extraction.
        batch_num : int
            Current batch number (for progress display).
        total_batches : int
            Total number of batches (for progress display).

        Returns
        -------
        xr.Dataset
            Dataset with return values and p-values for this batch.
        """
        import time as time_module

        batch_start = time_module.time()

        # Check if we need spatial batching (for clipped data with many points)
        # Must do this BEFORE computing block maxima to avoid OOM
        spatial_dim = None
        spatial_size_check = 0
        if "closest_cell" in batch_data.dims:
            spatial_dim = "closest_cell"
            spatial_size_check = batch_data.sizes["closest_cell"]
        elif "points" in batch_data.dims:
            spatial_dim = "points"
            spatial_size_check = batch_data.sizes["points"]

        # If we have many spatial points, process them in chunks BEFORE block maxima
        SPATIAL_BATCH_SIZE = 100  # Process 100 points at a time (reduced for safety)

        if spatial_dim and spatial_size_check > SPATIAL_BATCH_SIZE:
            # Process spatial chunks sequentially - this avoids loading all data at once
            return_values, p_values = self._fit_with_early_spatial_batching(
                batch_data, block_maxima_kwargs, spatial_dim, SPATIAL_BATCH_SIZE
            )
        else:
            # Small enough to process all at once
            # Step 1: Extract block maxima (keep as dask if possible)
            logger.info("Extracting block maxima...")
            step_start = time_module.time()
            block_maxima = _get_block_maxima_optimized(
                batch_data, **block_maxima_kwargs
            )
            block_maxima = block_maxima.squeeze()
            logger.debug(
                "Block maxima extraction took %.1fs", time_module.time() - step_start
            )

            # Show block maxima shape (estimate without computing)
            n_years = block_maxima.sizes.get(
                "time", block_maxima.sizes.get("time_delta", 0)
            )
            spatial_size = np.prod(
                [
                    s
                    for d, s in block_maxima.sizes.items()
                    if d not in ["time", "time_delta", "sim"]
                ]
            )
            logger.info(
                "Block maxima shape: %s (%d years × %d gridcells)",
                dict(block_maxima.sizes),
                n_years,
                spatial_size,
            )

            # Determine the time dimension to reduce over
            time_dim = "time" if "time" in block_maxima.dims else "time_delta"

            # Step 2: Fit distributions
            n_fits = int(
                np.prod([s for d, s in block_maxima.sizes.items() if d != time_dim])
            )
            logger.info(
                "Fitting %s distributions to %d locations...",
                self.distribution.upper(),
                n_fits,
            )

            return_values, p_values = self._fit_distributions_vectorized(
                block_maxima, time_dim
            )

        # If goodness-of-fit test is not requested, set p_values to None
        if not self.goodness_of_fit_test:
            p_values = None

        # Assign return periods as coordinates
        return_values = return_values.assign_coords(one_in_x=self.return_periods)

        # Step 4: Create result dataset
        logger.info("Creating result dataset...")
        step_start = time_module.time()
        result_ds = self._create_one_in_x_result_dataset(
            return_values, p_values, batch_data
        )
        logger.debug(
            "Result dataset creation took %.1fs", time_module.time() - step_start
        )

        # Batch summary
        batch_elapsed = time_module.time() - batch_start
        logger.info(
            "Batch %d/%d complete (%.1fs total)",
            batch_num,
            total_batches,
            batch_elapsed,
        )

        return result_ds

    def _fit_distributions_vectorized(
        self, block_maxima: xr.DataArray, time_dim: str
    ) -> tuple[xr.DataArray, xr.DataArray]:
        """
        Fit distributions using vectorized apply_ufunc.

        Parameters
        ----------
        block_maxima : xr.DataArray
            Block maxima data with time dimension
        time_dim : str
            Name of the time dimension

        Returns
        -------
        tuple[xr.DataArray, xr.DataArray]
            Return values and p-values arrays
        """
        # Compute block maxima to numpy first (small after yearly aggregation)
        if hasattr(block_maxima.data, "compute"):
            block_maxima_computed = block_maxima.compute()
        else:
            block_maxima_computed = block_maxima

        # Apply the return value fitting function
        return_values, p_values = xr.apply_ufunc(
            self._fit_return_values_1d,
            block_maxima_computed,
            kwargs={
                "return_periods": self.return_periods,
                "distr": self.distribution,
                "get_p_value": self.goodness_of_fit_test,
            },
            input_core_dims=[[time_dim]],
            output_core_dims=[["one_in_x"], []],
            output_sizes={"one_in_x": len(self.return_periods)},
            output_dtypes=("float", "float"),
            vectorize=True,
        )

        return return_values, p_values

    def _fit_with_early_spatial_batching(
        self,
        batch_data: xr.DataArray,
        block_maxima_kwargs: dict,
        spatial_dim: str,
        batch_size: int,
    ) -> tuple[xr.DataArray, xr.DataArray]:
        """
        Fit distributions with EARLY spatial batching - before computing block maxima.

        This is critical for memory safety: we select a spatial chunk FIRST,
        then compute block maxima for just that chunk, then fit distributions.
        This avoids loading the entire raw dataset into memory.

        Parameters
        ----------
        batch_data : xr.DataArray
            Raw data (before block maxima) with spatial dimension
        block_maxima_kwargs : dict
            Keyword arguments for block maxima extraction
        spatial_dim : str
            Name of the spatial dimension to batch over
        batch_size : int
            Number of spatial points to process at once

        Returns
        -------
        tuple[xr.DataArray, xr.DataArray]
            Return values and p-values arrays
        """
        n_spatial = batch_data.sizes[spatial_dim]
        n_batches = int(np.ceil(n_spatial / batch_size))

        logger.info(
            "Processing %d spatial points in %d chunks of %d (early spatial batching)...",
            n_spatial,
            n_batches,
            batch_size,
        )

        return_values_list = []
        p_values_list = []

        step_start = time_module.time()

        # Use tqdm progress bar when logger is too quiet to show INFO messages
        use_tqdm = logger.getEffectiveLevel() > logging.INFO
        spatial_iter = (
            tqdm(range(n_batches), desc="Spatial chunks")
            if use_tqdm
            else range(n_batches)
        )

        for i in spatial_iter:
            start_idx = i * batch_size
            end_idx = min((i + 1) * batch_size, n_spatial)

            # Step 1: Select spatial chunk BEFORE any computation
            chunk_data = batch_data.isel({spatial_dim: slice(start_idx, end_idx)})

            # Step 2: Compute block maxima for just this chunk (much smaller!)
            chunk_block_maxima = _get_block_maxima_optimized(
                chunk_data, **block_maxima_kwargs
            )
            chunk_block_maxima = chunk_block_maxima.squeeze()

            # Compute to memory (now it's small: e.g., 10 sims × 2 WL × 30 years × 100 points)
            if hasattr(chunk_block_maxima.data, "compute"):
                chunk_block_maxima = chunk_block_maxima.compute()

            # Determine time dimension
            time_dim = "time" if "time" in chunk_block_maxima.dims else "time_delta"

            # Step 3: Fit distributions for this spatial chunk
            chunk_return_values, chunk_p_values = xr.apply_ufunc(
                self._fit_return_values_1d,
                chunk_block_maxima,
                kwargs={
                    "return_periods": self.return_periods,
                    "distr": self.distribution,
                    "get_p_value": self.goodness_of_fit_test,
                },
                input_core_dims=[[time_dim]],
                output_core_dims=[["one_in_x"], []],
                output_sizes={"one_in_x": len(self.return_periods)},
                output_dtypes=("float", "float"),
                vectorize=True,
            )

            return_values_list.append(chunk_return_values)
            p_values_list.append(chunk_p_values)

            # Clean up chunk data
            del chunk_data, chunk_block_maxima
            gc.collect()

            # Progress update every 2 chunks or at the end
            if (i + 1) % 2 == 0 or i == n_batches - 1:
                elapsed = time_module.time() - step_start
                rate = (end_idx) / elapsed if elapsed > 0 else 0
                remaining = (n_spatial - end_idx) / rate if rate > 0 else 0
                logger.debug(
                    "Chunk %d/%d: %d/%d points (%.1fs elapsed, ~%.0fs remaining)",
                    i + 1,
                    n_batches,
                    end_idx,
                    n_spatial,
                    elapsed,
                    remaining,
                )

        # Concatenate results along spatial dimension
        logger.info("Concatenating %d chunk results...", n_batches)
        return_values = xr.concat(return_values_list, dim=spatial_dim)
        p_values = xr.concat(p_values_list, dim=spatial_dim)

        # Clean up
        del return_values_list, p_values_list
        gc.collect()

        total_time = time_module.time() - step_start
        logger.info("Spatial processing complete (%.1fs total)", total_time)

        return return_values, p_values

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
        var_lower = var_name.lower()
        is_precipitation = "precipitation" in var_lower or var_lower == "pr"
        if is_precipitation:
            preprocessing = self.variable_preprocessing.get("precipitation", {})

            # Aggregate to daily if needed and requested
            if preprocessing.get("daily_aggregation", True) and (
                "1hr" in str(data.attrs.get("frequency", "")).lower()
                or "hourly" in str(data.attrs.get("frequency", "")).lower()
            ):
                data = data.resample(time="1D").sum()

            # Remove trace precipitation
            if preprocessing.get("remove_trace", True):
                threshold = preprocessing.get("trace_threshold", 1e-10)
                data = data.where(data > threshold, drop=True)

        return data

    def update_context(self, context: dict[str, Any]):
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

        if self.one_in_x_config is not UNSET:
            # 1-in-X calculations
            return_periods_str = ", ".join(map(str, self.return_periods))
            description_parts.append(
                f"1-in-X return values for periods [{return_periods_str}] were "
                f"calculated using {self.distribution} distribution with "
                f"{self.extremes_type} extremes over {self.event_duration[0]} {self.event_duration[1]} events"
            )
        else:
            # Regular metric calculations
            if self.percentiles is not UNSET:
                description_parts.append(
                    f"Percentiles {self.percentiles} were calculated"
                )

            if not self.percentiles_only:
                description_parts.append(f"Metric '{self.metric}' was calculated")

        if self.one_in_x_config is not UNSET:
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
        self,
        ret_vals: xr.DataArray,
        p_vals: xr.DataArray | None,
        data_array: xr.DataArray,
    ) -> xr.Dataset:
        """
        Create the final result dataset for 1-in-X calculations.

        Parameters
        ----------
        ret_vals : xr.DataArray
            Return values DataArray
        p_vals : xr.DataArray | None
            P-values DataArray
        data_array : xr.DataArray
            Input data array for attributes

        Returns
        -------
        xr.Dataset
            Final result dataset with return_value and p_values
        """
        if p_vals is not None:
            result = xr.Dataset({"return_values": ret_vals, "p_values": p_vals})
        else:
            result = xr.Dataset({"return_values": ret_vals})

        # Add attributes
        result.attrs.update(
            {
                "groupby": f"{self.event_duration[0]} {self.event_duration[1]}",
                "fitted_distr": self.distribution,
                "sample_size": len(data_array.time),
            }
        )

        return result
