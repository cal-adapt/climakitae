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
            total_size = data_array.nbytes  # Estimate bytes (assuming float64)
            print(f"Total size of data array: {total_size / 1e6:.2f} MB")
            if total_size < 1e6:  # Less than 1GB
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

    def _get_return_values_vectorized(
        self,
        block_maxima: xr.DataArray,
        return_periods: np.ndarray,
        distr: str = "gev",
    ) -> xr.DataArray:
        """
        Calculate return values from block maxima using a vectorized approach.
        """
        if not EXTREME_VALUE_ANALYSIS_AVAILABLE:
            raise ValueError("Extreme value analysis functions are not available")

        from climakitae.explore.threshold_tools import get_return_value

        # Check if the input is a Dask array
        is_dask_array = (
            hasattr(block_maxima, "chunks") and block_maxima.chunks is not None
        )

        # Set dask_kwargs if it's a Dask array
        dask_kwargs = {}
        if is_dask_array:
            dask_kwargs = {
                "dask": "parallelized",
                "output_dtypes": [block_maxima.dtype],
            }

        # Apply the function to get return values
        return_values = xr.apply_ufunc(
            get_return_value,
            block_maxima,
            input_core_dims=[["time"]],
            output_core_dims=[["one_in_x"]],
            exclude_dims=set(("time",)),
            vectorize=True,
            kwargs={"return_period": return_periods, "distr": distr},
            **dask_kwargs,
        )

        # Assign coordinates to the output
        return_values = return_values.assign_coords(one_in_x=return_periods)
        return return_values

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
                    block_maxima = get_block_maxima(sim_data, **kwargs).squeeze()

                    # Force drop the sim dimension if it still exists in block_maxima
                    if "sim" in block_maxima.dims:
                        block_maxima = block_maxima.squeeze("sim", drop=True)

                    # The vectorized function will handle spatial dimensions automatically.
                    result = self._get_return_values_vectorized(
                        block_maxima,
                        return_periods=self.return_periods,
                        distr=self.distribution,
                    )

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

    def _add_dummy_time_if_needed(self, data_array: xr.DataArray) -> xr.DataArray:
        """Add a dummy time dimension if not present."""
        if "time" not in data_array.dims:
            return data_array.expand_dims(time=[0])
        return data_array

    def _extract_block_maxima(self, data_array: xr.DataArray) -> xr.DataArray:
        """Extract block maxima from a data array."""
        kwargs = {
            "extremes_type": self.extremes_type,
            "check_ess": False,
            "block_size": self.block_size,
        }
        if self.event_duration == (1, "day"):
            kwargs["groupby"] = self.event_duration
        elif self.event_duration[1] == "hour":
            kwargs["duration"] = self.event_duration
        return get_block_maxima(data_array, **kwargs).squeeze()

    def _print_goodness_of_fit_result(self, sim: str, p_value: xr.DataArray):
        """Print the goodness-of-fit test result."""
        if p_value.values < 0.05:
            print(
                f"For simulation {sim}, the p-value is {p_value.values:.2f}, which is less than 0.05, so we reject the null hypothesis that the data follows a {self.distribution} distribution."
            )
        else:
            print(
                f"For simulation {sim}, the p-value is {p_value.values:.2f}, which is greater than 0.05, so we fail to reject the null hypothesis that the data follows a {self.distribution} distribution."
            )

    def _preprocess_variable_for_one_in_x(
        self, data_array: xr.DataArray, var_name: str
    ) -> xr.DataArray:
        """Apply variable-specific preprocessing for 1-in-X calculations."""
        if var_name == "pr" and self.variable_preprocessing.get(
            "detrend_with_rolling_mean", False
        ):
            print("Detrending precipitation data with a 30-day rolling mean...")
            rolling_mean = data_array.rolling(time=30, center=True).mean()
            data_array = data_array - rolling_mean
        return data_array

    def update_context(self, context):
        """
        Update the context with information about the clipping operation, to be stored
                in the "new_attrs" attribute.

        Parameters
        ----------
        context : dict[str, Any]
            Parameters for processing the data.
        """
        if _NEW_ATTRS_KEY not in context:
            context[_NEW_ATTRS_KEY] = {}

        context[_NEW_ATTRS_KEY][
            self.name
        ] = f"""Process '{self.name}' applied to the data."""

    def set_data_accessor(self, catalog):
        # placeholder for setting data accessor
        pass
