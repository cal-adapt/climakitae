"""
DataProcessor MetricCalc
"""

import warnings
from typing import Any, Dict, Iterable, Union

import numpy as np
import xarray as xr

from climakitae.core.constants import _NEW_ATTRS_KEY
from climakitae.explore.threshold_tools import (
    get_block_maxima,
    get_ks_stat,
    get_return_value,
)
from climakitae.new_core.data_access.data_access import DataCatalog
from climakitae.new_core.processors.abc_data_processor import (
    DataProcessor,
    register_processor,
)

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
        self._catalog = None  # Initialize catalog attribute

        # Basic metric parameters
        self.metric = value.get("metric", "mean")
        self.percentiles = value.get("percentiles", None)
        self.percentiles_only = value.get("percentiles_only", False)
        self.dim = value.get("dim", "time")
        self.keepdims = value.get("keepdims", False)
        self.skipna = value.get("skipna", True)

        # 1-in-X parameters
        self.one_in_x_config = value.get("one_in_x", None)
        if self.one_in_x_config is not None:
            self._setup_one_in_x_parameters()

        # Validate inputs during initialization
        self._validate_parameters()

    def _setup_one_in_x_parameters(self):
        """Setup parameters for 1-in-X calculations."""
        if not EXTREME_VALUE_ANALYSIS_AVAILABLE:
            raise ValueError(
                "Extreme value analysis functions are not available. Please check climakitae installation."
            )

        # Type guard to ensure one_in_x_config is not None
        if self.one_in_x_config is None:
            raise ValueError(
                "one_in_x_config cannot be None when calling _setup_one_in_x_parameters"
            )

        # Required parameter
        self.return_periods = self.one_in_x_config.get("return_periods")
        if self.return_periods is None:
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

    def _validate_parameters(self):
        """
        Validate the metric calculation parameters.

        Raises
        ------
        ValueError
            If invalid parameter values are provided
        """
        # Check that only one calculation type is specified
        # Only consider basic metrics specified if they were explicitly set (not defaults)
        explicit_metric_specified = (
            "metric" in self.value and self.value["metric"] != "mean"
        )
        percentiles_specified = self.percentiles is not None
        basic_metrics_specified = explicit_metric_specified or percentiles_specified
        one_in_x_specified = self.one_in_x_config is not None

        if basic_metrics_specified and one_in_x_specified:
            raise ValueError(
                "Cannot specify both basic metrics/percentiles and one_in_x calculations simultaneously"
            )

        # Validate basic metric parameters
        if not one_in_x_specified:
            self._validate_basic_metric_parameters()
        else:
            self._validate_one_in_x_parameters()

    def _validate_basic_metric_parameters(self):
        """Validate parameters for basic metric calculations."""
        # Validate metric
        valid_metrics = ["min", "max", "mean", "median"]
        if self.metric not in valid_metrics:
            raise ValueError(
                f'metric must be one of {valid_metrics}, got "{self.metric}"'
            )

        # Validate percentiles
        if self.percentiles is not None:
            if not isinstance(self.percentiles, list):
                raise ValueError(
                    f"percentiles must be a list, got {type(self.percentiles)}"
                )

            for p in self.percentiles:
                if not isinstance(p, (int, float)) or not (0 <= p <= 100):
                    raise ValueError(
                        f"All percentiles must be numbers between 0 and 100, got {p}"
                    )

        # Validate dim type
        if not isinstance(self.dim, (str, list)):
            raise ValueError(f"dim must be a string or list, got {type(self.dim)}")

        # Validate keepdims type
        if not isinstance(self.keepdims, bool):
            raise ValueError(f"keepdims must be a boolean, got {type(self.keepdims)}")

        # Validate skipna type
        if not isinstance(self.skipna, bool):
            raise ValueError(f"skipna must be a boolean, got {type(self.skipna)}")

        # Validate percentiles_only type
        if not isinstance(self.percentiles_only, bool):
            raise ValueError(
                f"percentiles_only must be a boolean, got {type(self.percentiles_only)}"
            )

        # Validate percentiles_only logic
        if self.percentiles_only and self.percentiles is None:
            raise ValueError(
                "percentiles_only=True requires percentiles to be specified"
            )

    def _validate_one_in_x_parameters(self):
        """Validate parameters for 1-in-X calculations."""
        if not EXTREME_VALUE_ANALYSIS_AVAILABLE:
            raise ValueError("Extreme value analysis functions are not available")

        # Validate return_periods
        if not isinstance(self.return_periods, np.ndarray):
            raise ValueError("return_periods must be convertible to numpy array")

        for rp in self.return_periods:
            if not isinstance(rp, (int, float, np.integer, np.floating)) or rp < 1:
                raise ValueError(
                    f"All return periods must be numbers >= 1, got {rp} (type: {type(rp)})"
                )

        # Validate distribution
        valid_distributions = ["gev", "genpareto", "gamma"]
        if self.distribution not in valid_distributions:
            raise ValueError(
                f"distribution must be one of {valid_distributions}, got '{self.distribution}'"
            )

        # Validate extremes_type
        valid_extremes = ["max", "min"]
        if self.extremes_type not in valid_extremes:
            raise ValueError(
                f"extremes_type must be one of {valid_extremes}, got '{self.extremes_type}'"
            )

        # Validate event_duration
        if not isinstance(self.event_duration, tuple) or len(self.event_duration) != 2:
            raise ValueError("event_duration must be a tuple of (int, str)")

        duration_num, duration_unit = self.event_duration
        if not isinstance(duration_num, int) or duration_num <= 0:
            raise ValueError("event_duration number must be a positive integer")

        if duration_unit not in ["hour", "day"]:
            raise ValueError("event_duration unit must be 'hour' or 'day'")

        # Validate block_size
        if not isinstance(self.block_size, int) or self.block_size <= 0:
            raise ValueError("block_size must be a positive integer")

        # Validate boolean parameters
        if not isinstance(self.goodness_of_fit_test, bool):
            raise ValueError("goodness_of_fit_test must be a boolean")

        if not isinstance(self.print_goodness_of_fit, bool):
            raise ValueError("print_goodness_of_fit must be a boolean")

        # Validate variable_preprocessing
        if not isinstance(self.variable_preprocessing, dict):
            raise ValueError("variable_preprocessing must be a dictionary")

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
        print(result)

        match result:
            case xr.Dataset() | xr.DataArray():
                if self.one_in_x_config is not None:
                    ret = self._calculate_one_in_x_single(result)
                else:
                    ret = self._calculate_metrics_single(result)
            case dict():
                if self.one_in_x_config is not None:
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
                if self.one_in_x_config is not None:
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
            print(
                f"Warning: None of the specified dimensions {dims_to_check} exist in the data. Available dimensions: {list(available_dims)}"
            )
            return data

        # Use valid dimensions for calculation
        calc_dim = valid_dims if len(valid_dims) > 1 else valid_dims[0]

        results = []

        # Calculate percentiles if requested
        if self.percentiles is not None:
            percentile_result = data.quantile(
                [p / 100.0 for p in self.percentiles],
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
            total_size = (
                np.prod(data_array.shape) * 8
            )  # Estimate bytes (assuming float64)
            if total_size < 1e9:  # Less than 1GB
                print("Small array detected - loading into memory...")
                data_array = data_array.compute()
            else:
                print("Large array detected - will process simulations in chunks...")

        # Check if we have a time dimension, and add dummy time if needed
        if "time" not in data_array.dims:
            data_array = self._add_dummy_time_if_needed(data_array)

        # Apply variable-specific preprocessing
        data_array = self._preprocess_variable_for_one_in_x(data_array, var_name)

        print(
            f"Calculating 1-in-{self.return_periods} year return values using {self.distribution} distribution..."
        )

        # Try vectorized processing first for better performance
        try:
            return self._calculate_one_in_x_vectorized(data_array)
        except Exception as e:
            print(
                f"Vectorized processing failed ({e}), falling back to serial processing..."
            )
            return self._calculate_one_in_x_serial(data_array)

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

        # Process all simulations at once using xarray's vectorized operations
        all_block_maxima = []
        all_return_vals = []
        all_p_vals = []

        # Process simulations in batches to manage memory
        batch_size = min(
            10, len(data_array.sim)
        )  # Process up to 10 simulations at once
        sim_values = data_array.sim.values

        for i in range(0, len(sim_values), batch_size):
            batch_sims = sim_values[i : i + batch_size]
            print(
                f"Processing simulation batch {i//batch_size + 1}/{(len(sim_values) + batch_size - 1)//batch_size}"
            )

            batch_results = []
            batch_p_vals = []

            for s in batch_sims:
                block_maxima = None  # Initialize to avoid scoping issues
                try:
                    # Check for duplicate simulations and handle appropriately
                    sim_matches = data_array.sim.values == s
                    num_matches = sim_matches.sum()

                    # Handle duplicate simulations by selecting the first occurrence
                    if num_matches > 1:
                        first_idx = np.where(sim_matches)[0][0]
                        print(
                            f"Warning: Found {num_matches} matches for simulation '{s}', selecting first occurrence"
                        )
                        sim_data = data_array.isel(sim=first_idx)
                    else:
                        # Try the selection
                        try:
                            sim_data = data_array.sel(sim=s)
                        except Exception as sel_error:
                            # Try alternative selection methods
                            try:
                                # Try using isel if sel fails
                                sim_idx = list(data_array.sim.values).index(s)
                                sim_data = data_array.isel(sim=sim_idx)
                            except Exception:
                                raise sel_error

                    # Now squeeze to remove size-1 dimensions
                    sim_data = sim_data.squeeze()

                    # Force drop the sim dimension if it still exists
                    if "sim" in sim_data.dims:
                        sim_data = sim_data.squeeze("sim", drop=True)

                    # Extract block maxima for this simulation
                    print(sim_data)
                    block_maxima = get_block_maxima(sim_data, **kwargs).squeeze()

                    # Force drop the sim dimension if it still exists in block_maxima
                    if "sim" in block_maxima.dims:
                        block_maxima = block_maxima.squeeze("sim", drop=True)

                    # Check data quality and filter out locations with insufficient data
                    if hasattr(block_maxima, "dims") and len(block_maxima.dims) > 1:
                        # For multi-dimensional block maxima, we need to check each location
                        spatial_dims = [
                            dim
                            for dim in block_maxima.dims
                            if dim not in ["time", "year"]
                        ]
                        if spatial_dims:
                            print(
                                f"Filtering locations with insufficient valid data..."
                            )
                            # Count valid data points for each location
                            count_dim = (
                                "year"
                                if "year" in block_maxima.dims
                                else str(block_maxima.dims[0])
                            )
                            valid_counts = block_maxima.count(dim=count_dim)
                            # Filter out locations with fewer than 3 valid years
                            valid_locations = valid_counts >= 3
                            if valid_locations.sum() == 0:
                                print(
                                    f"Warning: No locations have sufficient valid data for simulation {s}"
                                )
                                raise ValueError("No valid locations found")
                            elif valid_locations.sum() < len(valid_locations):
                                n_valid = int(valid_locations.sum())
                                n_total = len(valid_locations)
                                print(
                                    f"Filtering to {n_valid} valid locations out of {n_total} total locations"
                                )
                                # Filter the block maxima to only include valid locations
                                block_maxima = block_maxima.where(
                                    valid_locations, drop=True
                                )

                    # Handle spatial dimensions (e.g., closest_cell, lat, lon)
                    spatial_dims = [
                        dim for dim in block_maxima.dims if dim not in ["time", "year"]
                    ]

                    if spatial_dims:
                        # We have spatial dimensions - need to process each location separately
                        print(
                            f"Processing {len(spatial_dims)} spatial dimension(s): {spatial_dims}"
                        )

                        # Get the first spatial dimension to iterate over
                        spatial_dim = spatial_dims[0]
                        spatial_coords = block_maxima.coords[spatial_dim]

                        location_results = []
                        for loc_idx in range(len(spatial_coords)):
                            try:
                                # Extract data for this specific location
                                loc_block_maxima = block_maxima.isel(
                                    {spatial_dim: loc_idx}
                                )

                                # Check if this location has enough valid data
                                valid_data = loc_block_maxima.dropna(
                                    dim="year", how="all"
                                )
                                if (
                                    len(valid_data.year) < 3
                                ):  # Need at least 3 years for distribution fitting
                                    print(
                                        f"Warning: Location {loc_idx} has insufficient valid data ({len(valid_data.year)} years), skipping..."
                                    )
                                    raise ValueError(
                                        f"Insufficient valid data for location {loc_idx}"
                                    )

                                # Calculate return values for this location
                                loc_result = self._get_return_values_vectorized(
                                    valid_data,  # Use the filtered data
                                    return_periods=self.return_periods,
                                    distr=self.distribution,
                                )

                                # Add the spatial coordinate back to the result
                                loc_result = loc_result.assign_coords(
                                    {spatial_dim: spatial_coords[loc_idx]}
                                )
                                loc_result = loc_result.expand_dims(spatial_dim)
                                location_results.append(loc_result)

                            except Exception as loc_error:
                                print(
                                    f"Warning: Failed to process location {loc_idx} in {spatial_dim}: {loc_error}"
                                )
                                # Create NaN result for this location
                                nan_result = xr.DataArray(
                                    np.full(len(self.return_periods), np.nan),
                                    dims=["one_in_x"],
                                    coords={
                                        "one_in_x": self.return_periods,
                                        spatial_dim: spatial_coords[loc_idx],
                                    },
                                    name="return_value",
                                )
                                nan_result = nan_result.expand_dims(spatial_dim)
                                location_results.append(nan_result)

                        # Combine results across all locations
                        if location_results:
                            result = xr.concat(location_results, dim=spatial_dim)
                        else:
                            # All locations failed - create NaN result
                            result = xr.DataArray(
                                np.full(
                                    (len(spatial_coords), len(self.return_periods)),
                                    np.nan,
                                ),
                                dims=[spatial_dim, "one_in_x"],
                                coords={
                                    spatial_dim: spatial_coords,
                                    "one_in_x": self.return_periods,
                                },
                                name="return_value",
                            )
                    else:
                        # No spatial dimensions - process as before
                        result = self._get_return_values_vectorized(
                            block_maxima,
                            return_periods=self.return_periods,
                            distr=self.distribution,
                        )

                    # Store success flag for this simulation
                    pass

                except Exception:
                    # Fallback to individual calculation
                    print(
                        f"Warning: Vectorized return value calculation failed for simulation {s}, using fallback method"
                    )
                    individual_return_values = []

                    # Check if block_maxima was successfully extracted
                    if block_maxima is None:
                        print(
                            f"Warning: Block maxima extraction failed for simulation {s}, returning NaN values"
                        )
                        individual_return_values = [np.nan] * len(self.return_periods)
                    else:
                        for rp in self.return_periods.tolist():
                            try:
                                single_result = get_return_value(
                                    block_maxima,
                                    return_period=rp,  # Pass single return period
                                    multiple_points=False,
                                    distr=self.distribution,
                                )

                                # Extract the return value from the result
                                if (
                                    isinstance(single_result, dict)
                                    and "return_value" in single_result
                                ):
                                    rv = single_result["return_value"]
                                else:
                                    rv = single_result

                                # Convert to scalar if it's a DataArray
                                if isinstance(rv, xr.DataArray):
                                    rv = (
                                        rv.values.item()
                                        if rv.values.size == 1
                                        else rv.values.flat[0]
                                    )

                                individual_return_values.append(rv)

                            except Exception as single_rv_error:
                                print(
                                    f"Warning: Return value calculation failed for return period {rp}: {single_rv_error}"
                                )
                                individual_return_values.append(np.nan)

                    # Create a DataArray with the individual return values
                    result = xr.DataArray(
                        individual_return_values,
                        dims=["one_in_x"],
                        coords={"one_in_x": self.return_periods},
                        name="return_value",
                    )

                # The result is now already properly formatted as a DataArray with the correct coordinates
                return_values = result
                batch_results.append(return_values)

                # Calculate p-values if requested
                if self.goodness_of_fit_test and block_maxima is not None:
                    _, p_value = get_ks_stat(
                        block_maxima, distr=self.distribution, multiple_points=False
                    ).data_vars.values()
                    batch_p_vals.append(p_value)

                    if self.print_goodness_of_fit:
                        self._print_goodness_of_fit_result(s, p_value)
                else:
                    batch_p_vals.append(xr.DataArray(np.nan, name="p_value"))

            all_return_vals.extend(batch_results)
            all_p_vals.extend(batch_p_vals)

        # Combine all results with robust error handling
        try:
            # Validate all return values before concatenation
            validated_return_vals = []
            for i, rv in enumerate(all_return_vals):
                try:
                    # Ensure each return value has proper dimensions
                    if isinstance(rv, xr.DataArray):
                        # Check if it has the expected dimensions
                        if "one_in_x" not in rv.dims:
                            print(
                                f"Warning: Fixing missing 'one_in_x' dimension for result {i}"
                            )
                            if len(rv.dims) == 1:
                                rv = rv.rename({rv.dims[0]: "one_in_x"})
                            elif len(rv.dims) == 0:
                                rv = xr.DataArray(
                                    [rv.values.item()] * len(self.return_periods),
                                    dims=["one_in_x"],
                                    coords={"one_in_x": self.return_periods},
                                    name="return_value",
                                )
                            else:
                                # Multi-dimensional - take the first usable dimension
                                rv = rv.isel(
                                    {dim: 0 for dim in rv.dims if dim != "one_in_x"}
                                )
                                if "one_in_x" not in rv.dims and len(rv.dims) == 1:
                                    rv = rv.rename({rv.dims[0]: "one_in_x"})

                        # Ensure coordinates are correct
                        if "one_in_x" in rv.dims and len(
                            rv.coords.get("one_in_x", [])
                        ) != len(self.return_periods):
                            rv = rv.assign_coords(
                                {"one_in_x": ("one_in_x", self.return_periods)}
                            )

                        validated_return_vals.append(rv)
                    else:
                        print(
                            f"Warning: Converting non-DataArray result {i} to DataArray"
                        )
                        validated_return_vals.append(
                            xr.DataArray(
                                np.full(len(self.return_periods), np.nan),
                                dims=["one_in_x"],
                                coords={"one_in_x": self.return_periods},
                                name="return_value",
                            )
                        )
                except Exception as val_error:
                    print(f"Warning: Failed to validate return value {i}: {val_error}")
                    validated_return_vals.append(
                        xr.DataArray(
                            np.full(len(self.return_periods), np.nan),
                            dims=["one_in_x"],
                            coords={"one_in_x": self.return_periods},
                            name="return_value",
                        )
                    )

            ret_vals = xr.concat(validated_return_vals, dim="sim")
            p_vals = xr.concat(all_p_vals, dim="sim")
        except Exception as concat_error:
            print(f"Error during concatenation: {concat_error}")
            # Create fallback results
            ret_vals = xr.DataArray(
                np.full((len(data_array.sim), len(self.return_periods)), np.nan),
                dims=["sim", "one_in_x"],
                coords={"sim": data_array.sim.values, "one_in_x": self.return_periods},
                name="return_value",
            )
            p_vals = xr.DataArray(
                np.full(len(data_array.sim), np.nan),
                dims=["sim"],
                coords={"sim": data_array.sim.values},
                name="p_value",
            )

        # Ensure proper coordinates
        ret_vals = ret_vals.assign_coords(sim=data_array.sim.values)
        p_vals = p_vals.assign_coords(sim=data_array.sim.values)

        # Create result dataset
        result = xr.Dataset({"return_value": ret_vals, "p_values": p_vals})

        # Add attributes
        result.attrs.update(
            {
                "groupby": f"{self.event_duration[0]} {self.event_duration[1]}",
                "fitted_distr": self.distribution,
                "sample_size": len(data_array.time),
            }
        )

        return result

    def _calculate_one_in_x_serial(self, data_array: xr.DataArray) -> xr.Dataset:
        """
        Serial calculation of 1-in-X values (fallback method).

        This is the original implementation for cases where vectorized processing fails.
        """
        if not EXTREME_VALUE_ANALYSIS_AVAILABLE:
            raise ValueError("Extreme value analysis functions are not available")

        # Local imports
        from climakitae.explore.threshold_tools import get_ks_stat, get_return_value

        return_vals = []
        p_vals = []

        for s in data_array.sim.values:
            sim_data = data_array.sel(sim=s).squeeze()
            print(f"Processing simulation: {s}")

            try:
                # Extract block maxima
                block_maxima = self._extract_block_maxima(sim_data)

                # Use our vectorized return value calculation
                try:
                    return_values = self._get_return_values_vectorized(
                        block_maxima,
                        return_periods=self.return_periods,
                        distr=self.distribution,
                    )
                except Exception as rv_error:
                    print(
                        f"Warning: Vectorized return value calculation failed for {s}, using fallback method"
                    )
                    # Fallback to individual calculation
                    individual_return_values = []
                    for rp in self.return_periods.tolist():
                        try:
                            single_result = get_return_value(
                                block_maxima,
                                return_period=rp,  # Pass single return period
                                multiple_points=False,
                                distr=self.distribution,
                            )

                            # Extract the return value from the result
                            if (
                                isinstance(single_result, dict)
                                and "return_value" in single_result
                            ):
                                rv = single_result["return_value"]
                            else:
                                rv = single_result

                            # Convert to scalar if it's a DataArray
                            if isinstance(rv, xr.DataArray):
                                rv = (
                                    rv.values.item()
                                    if rv.values.size == 1
                                    else rv.values.flat[0]
                                )

                            individual_return_values.append(rv)

                        except Exception as single_rv_error:
                            print(
                                f"Warning: Return value calculation failed for return period {rp}: {single_rv_error}"
                            )
                            individual_return_values.append(np.nan)

                    # Create a DataArray with the individual return values
                    return_values = xr.DataArray(
                        individual_return_values,
                        dims=["one_in_x"],
                        coords={"one_in_x": self.return_periods},
                        name="return_value",
                    )

                return_vals.append(return_values)

                # Perform goodness-of-fit test if requested
                if self.goodness_of_fit_test:
                    _, p_value = get_ks_stat(
                        block_maxima, distr=self.distribution, multiple_points=False
                    ).data_vars.values()
                    p_vals.append(p_value)

                    if self.print_goodness_of_fit:
                        self._print_goodness_of_fit_result(s, p_value)
                else:
                    p_vals.append(xr.DataArray(np.nan, name="p_value"))

            except (ValueError, RuntimeError, ImportError) as e:
                print(f"Warning: Failed to process simulation {s}: {e}")
                # Create NaN results for failed simulations
                nan_return_values = xr.DataArray(
                    np.full(len(self.return_periods), np.nan),
                    dims=["one_in_x"],
                    coords={"one_in_x": self.return_periods},
                    name="return_value",
                )
                return_vals.append(nan_return_values)
                p_vals.append(xr.DataArray(np.nan, name="p_value"))

        # Combine results with robust error handling
        try:
            # Validate all return values before concatenation
            validated_return_vals = []
            for i, rv in enumerate(return_vals):
                try:
                    # Ensure each return value has proper dimensions
                    if isinstance(rv, xr.DataArray):
                        # Check if it has the expected dimensions
                        if "one_in_x" not in rv.dims:
                            print(
                                f"Warning: Fixing missing 'one_in_x' dimension for result {i}"
                            )
                            if len(rv.dims) == 1:
                                rv = rv.rename({rv.dims[0]: "one_in_x"})
                            elif len(rv.dims) == 0:
                                rv = xr.DataArray(
                                    [rv.values.item()] * len(self.return_periods),
                                    dims=["one_in_x"],
                                    coords={"one_in_x": self.return_periods},
                                    name="return_value",
                                )
                            else:
                                # Multi-dimensional - take the first usable dimension
                                rv = rv.isel(
                                    {dim: 0 for dim in rv.dims if dim != "one_in_x"}
                                )
                                if "one_in_x" not in rv.dims and len(rv.dims) == 1:
                                    rv = rv.rename({rv.dims[0]: "one_in_x"})

                        # Ensure coordinates are correct
                        if "one_in_x" in rv.dims and len(
                            rv.coords.get("one_in_x", [])
                        ) != len(self.return_periods):
                            rv = rv.assign_coords(
                                {"one_in_x": ("one_in_x", self.return_periods)}
                            )

                        validated_return_vals.append(rv)
                    else:
                        print(
                            f"Warning: Converting non-DataArray result {i} to DataArray"
                        )
                        validated_return_vals.append(
                            xr.DataArray(
                                np.full(len(self.return_periods), np.nan),
                                dims=["one_in_x"],
                                coords={"one_in_x": self.return_periods},
                                name="return_value",
                            )
                        )
                except Exception as val_error:
                    print(f"Warning: Failed to validate return value {i}: {val_error}")
                    validated_return_vals.append(
                        xr.DataArray(
                            np.full(len(self.return_periods), np.nan),
                            dims=["one_in_x"],
                            coords={"one_in_x": self.return_periods},
                            name="return_value",
                        )
                    )

            ret_vals = xr.concat(validated_return_vals, dim="sim")
            p_vals = xr.concat(p_vals, dim="sim")
        except Exception as concat_error:
            print(f"Error during concatenation: {concat_error}")
            # Create fallback results
            ret_vals = xr.DataArray(
                np.full((len(data_array.sim), len(self.return_periods)), np.nan),
                dims=["sim", "one_in_x"],
                coords={"sim": data_array.sim.values, "one_in_x": self.return_periods},
                name="return_value",
            )
            p_vals = xr.DataArray(
                np.full(len(data_array.sim), np.nan),
                dims=["sim"],
                coords={"sim": data_array.sim.values},
                name="p_value",
            )

        # Create result dataset
        result = xr.Dataset({"return_value": ret_vals, "p_values": p_vals})

        # Add attributes
        result.attrs.update(
            {
                "groupby": f"{self.event_duration[0]} {self.event_duration[1]}",
                "fitted_distr": self.distribution,
                "sample_size": len(data_array.time),
            }
        )

        return result

    def _get_return_values_vectorized(
        self, block_maxima: xr.DataArray, return_periods: np.ndarray, distr: str = "gev"
    ) -> xr.DataArray:
        """
        Vectorized implementation of return value calculation that can handle multiple return periods.

        This is a custom implementation that avoids the bug in the original get_return_value function
        where it fails to assign coordinates properly for multi-dimensional data.

        Parameters
        ----------
        block_maxima : xr.DataArray
            Block maxima time series for a single simulation
        return_periods : np.ndarray
            Array of return periods to calculate
        distr : str
            Distribution to fit ("gev", "genpareto", "gamma")

        Returns
        -------
        xr.DataArray
            DataArray with return values for each return period
        """
        # Check for sufficient valid data before proceeding
        if "year" in block_maxima.dims:
            valid_data = block_maxima.dropna(dim="year", how="all")
        else:
            # Find the primary dimension (likely 'time' or similar)
            primary_dim = (
                [
                    dim
                    for dim in block_maxima.dims
                    if dim not in ["lat", "lon", "x", "y"]
                ][0]
                if block_maxima.dims
                else None
            )
            if primary_dim:
                valid_data = block_maxima.dropna(dim=primary_dim)
            else:
                valid_data = block_maxima
        if valid_data.size == 0:
            print(
                "Warning: No valid data found for distribution fitting - returning NaN values"
            )
            return xr.DataArray(
                np.full(len(return_periods), np.nan),
                dims=["one_in_x"],
                coords={"one_in_x": return_periods},
                name="return_value",
            )

        # Count valid values - need at least 3 for meaningful fitting
        n_valid = (
            valid_data.count().values.item()
            if hasattr(valid_data.count().values, "item")
            else int(valid_data.count())
        )
        if n_valid < 3:
            print(
                f"Warning: Insufficient valid data points ({n_valid}) for distribution fitting - returning NaN values"
            )
            return xr.DataArray(
                np.full(len(return_periods), np.nan),
                dims=["one_in_x"],
                coords={"one_in_x": return_periods},
                name="return_value",
            )

        # Import the helper functions we need
        from climakitae.explore.threshold_tools import (
            _get_distr_func,
            _get_fitted_distr,
        )

        # Get the distribution function
        distr_func = _get_distr_func(distr)

        try:
            # Fit the distribution to the valid block maxima data
            _, fitted_distr = _get_fitted_distr(valid_data, distr, distr_func)

            # Calculate return values for all return periods at once
            return_values = []
            for rp in return_periods:
                # Calculate the event probability
                block_size = 1  # Assuming 1-year blocks
                event_prob = block_size / rp

                # Calculate return event probability based on extremes type
                if self.extremes_type == "max":
                    return_event = 1.0 - event_prob
                elif self.extremes_type == "min":
                    return_event = event_prob
                else:
                    raise ValueError("extremes_type must be 'max' or 'min'")

                # Calculate the return value using the inverse CDF (percentile point function)
                try:
                    # fitted_distr is a frozen scipy.stats distribution with ppf method
                    return_value = fitted_distr.ppf(return_event)  # type: ignore
                    return_values.append(np.round(return_value, 5))
                except (ValueError, ZeroDivisionError):
                    return_values.append(np.nan)

            # Create DataArray with proper coordinates
            result = xr.DataArray(
                return_values,
                dims=["one_in_x"],
                coords={"one_in_x": return_periods},
                name="return_value",
                attrs={
                    "fitted_distribution": distr,
                    "extremes_type": self.extremes_type,
                    "block_size": "1 year",
                    "units": getattr(valid_data, "units", ""),
                },
            )

            return result

        except Exception as e:
            print(f"Warning: Failed to fit distribution {distr} to block maxima: {e}")
            # Return NaN array as fallback
            return xr.DataArray(
                np.full(len(return_periods), np.nan),
                dims=["one_in_x"],
                coords={"one_in_x": return_periods},
                name="return_value",
            )

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

    def _extract_block_maxima(self, data: xr.DataArray) -> xr.DataArray:
        """
        Extract block maxima from data array.

        Parameters
        ----------
        data : xr.DataArray
            Input data array

        Returns
        -------
        xr.DataArray
            Block maxima
        """
        if not EXTREME_VALUE_ANALYSIS_AVAILABLE:
            raise ValueError(
                "Block maxima extraction requires extreme value analysis functions"
            )

        # Import check - this should be guaranteed by the above check, but helps with type checking
        from climakitae.explore.threshold_tools import get_block_maxima  # noqa: F401

        # Configure block maxima extraction based on event duration
        duration = None
        groupby = None

        if self.event_duration == (1, "day"):
            groupby = self.event_duration
        elif self.event_duration[1] == "hour":
            duration = self.event_duration

        # Call get_block_maxima with appropriate parameters
        kwargs = {
            "extremes_type": self.extremes_type,
            "check_ess": False,  # Disable ESS check for now
            "block_size": self.block_size,
        }

        if duration is not None:
            kwargs["duration"] = duration
        if groupby is not None:
            kwargs["groupby"] = groupby

        print(data)
        return get_block_maxima(data, **kwargs).squeeze()

    def _print_goodness_of_fit_result(self, simulation: str, p_value: xr.DataArray):
        """
        Print goodness-of-fit test results.

        Parameters
        ----------
        simulation : str
            Simulation name
        p_value : xr.DataArray
            P-value from KS test
        """
        # Handle both scalar and array p-values
        p_val_scalar = (
            p_value.values.item()
            if p_value.values.size == 1
            else p_value.values.flat[0]
        )

        p_val_print = (
            format(p_val_scalar, ".3e")
            if p_val_scalar < 0.05
            else round(p_val_scalar, 4)
        )
        to_print = f"The simulation {simulation} fitted with a {self.distribution} distribution has a p-value of {p_val_print}.\n"

        if p_val_scalar < 0.05:
            to_print += " Since the p-value is <0.05, the selected distribution does not fit the data well and therefore is not a good fit (see guidance)."
        print(to_print)

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

    def _add_dummy_time_if_needed(self, data_array: xr.DataArray) -> xr.DataArray:
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
        import pandas as pd

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
            time_freq_name = getattr(data_array, "frequency", "daily")
            name_to_freq = {"hourly": "h", "daily": "D", "monthly": "ME"}
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
