"""
DataProcessor MetricCalc
"""

import logging
from typing import Any, Dict, Iterable, Union

import xarray as xr

from climakitae.core.constants import _NEW_ATTRS_KEY, UNSET
from climakitae.new_core.data_access.data_access import DataCatalog
from climakitae.new_core.processors.abc_data_processor import (
    DataProcessor, register_processor)

PERCENTILE_TO_QUANTILE_FACTOR = 100.0

# Module logger
logger = logging.getLogger(__name__)


@register_processor("metric_calc", priority=7500)
class MetricCalc(DataProcessor):
    """
    Calculate metrics (min, max, mean, median) and percentiles on data.

    This processor applies statistical operations to xarray datasets and data arrays,
    including percentile calculations and basic metrics like min, max, mean, and median.
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
        - skipna (bool, optional): Whether to skip NaN values in calculations. Default: True

    Examples
    --------
    Calculate mean over time:
    >>> metric_proc = MetricCalc({"metric": "mean"})

    Calculate 25th, 50th, 75th percentiles:
    >>> metric_proc = MetricCalc({"percentiles": [25, 50, 75]})

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
            - skipna (bool, optional): Skip NaN values. Default: True

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
        self.skipna = value.get("skipna", True)

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

        if result is None or (isinstance(result, (list, tuple, dict)) and not result):
            raise ValueError(
                "Metric calculation operation failed to produce valid results on empty arguments."
            )

        match result:
            case xr.Dataset() | xr.DataArray():
                ret = self._calculate_metrics_single(result)
            case dict():
                ret = {
                    key: self._calculate_metrics_single(value)
                    for key, value in result.items()
                }
            case list() | tuple():
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
            logger.warning(
                f"\n\nNone of the specified dimensions {dims_to_check} exist in the data. "
                f"\nAvailable dimensions: {list(available_dims)}"
            )
            return data

        # Use valid dimensions for calculation
        calc_dim = valid_dims if len(valid_dims) > 1 else valid_dims[0]

        # Calculate percentiles if requested
        results = []
        if self.percentiles is not UNSET:
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
        elif len(results) == 2 and (
            self.percentiles is not UNSET and self.percentiles is not None
        ):  # self.percentiles not in (UNSET, None):
            # Combine percentiles and metric results
            percentile_result, metric_result = results

            # Create a combined dataset/dataarray
            if isinstance(data, xr.Dataset):
                return self._combine_dataset_results(
                    data, percentile_result, metric_result
                )
            else:
                return self._combine_dataarray_results(percentile_result, metric_result)

        else:
            # Should not reach here, but return the first result as fallback
            return results[0] if results else data

    def _combine_dataarray_results(
        self,
        percentile_result: xr.DataArray,
        metric_result: xr.DataArray,
    ) -> xr.DataArray:
        """
        Combine percentile and metric results into a single DataArray.

        Parameters
        ----------
        percentile_result : xr.DataArray
            The DataArray containing percentile results.
        metric_result : xr.DataArray
            The DataArray containing metric results.

        Returns
        -------
        xr.DataArray
            The combined DataArray with both percentile and metric results.
        """
        # For DataArrays, create a new dimension for the different statistics
        stats_list = [f"p{p}" for p in self.percentiles] + [self.metric]

        # Stack percentile and metric results
        all_values = []
        for i in range(len(self.percentiles)):
            all_values.append(
                percentile_result.isel(percentile=i).drop_vars(
                    "percentile", errors="ignore"
                )
            )
        all_values.append(metric_result)
        result = xr.concat(all_values, dim="statistic")
        result = result.assign_coords(statistic=stats_list)
        return result

    def _combine_dataset_results(
        self,
        data: xr.Dataset,
        percentile_result: xr.Dataset,
        metric_result: xr.Dataset,
    ) -> xr.Dataset:
        """
        Combine percentile and metric results into a single Dataset.

        Parameters
        ----------
        data : xr.Dataset
            The original data.
        percentile_result : xr.Dataset
            The dataset containing percentile results.
        metric_result : xr.Dataset
            The dataset containing metric results.

        Returns
        -------
        xr.Dataset
            The combined dataset with both percentile and metric results.
        """
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
        return result

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

        # Regular metric calculations
        if self.percentiles is not UNSET and self.percentiles is not None:
            description_parts.append(f"Percentiles {self.percentiles} were calculated")

        if not self.percentiles_only:
            description_parts.append(f"Metric '{self.metric}' was calculated")

        transformation_description = (
            f"Process '{self.name}' applied to the data. "
            f"{' and '.join(description_parts)} along dimension(s): {self.dim}."
        )

        context[_NEW_ATTRS_KEY][self.name] = transformation_description

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
        ...
