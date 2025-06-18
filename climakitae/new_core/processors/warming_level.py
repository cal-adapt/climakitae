"""
Warming Level Data Processor

A simplified warming level processor that transforms time-series climate data
to a warming level centered approach following the established template pattern.
"""

import warnings
from typing import Any, Dict, Iterable, Union

import numpy as np
import pandas as pd
import xarray as xr

from climakitae.core.constants import _NEW_ATTRS_KEY
from climakitae.core.paths import GWL_1850_1900_FILE
from climakitae.new_core.data_access.data_access import DataCatalog
from climakitae.new_core.processors.abc_data_processor import (
    DataProcessor,
    register_processor,
)
from climakitae.util.utils import read_csv_file
from climakitae.util.warming_levels import calculate_warming_level, drop_invalid_sims


@register_processor("warming_level", priority=10)
class WarmingLevel(DataProcessor):
    """
    Transform time-series climate data into a warming-levels approach.

    This processor takes data with time dimensions and transforms it to data
    organized by warming levels, following the established warming level methodology.

    Parameters
    ----------
    value : Dict[str, Any]
        Configuration dictionary containing:
        - warming_levels : list[float]
            List of global warming levels in degrees C (e.g., [1.5, 2.0])
        - warming_level_months : list[int], optional
            List of months to include (1-12). Default: all months
        - warming_level_window : int, optional
            Number of years before and after the central year. Default: 15

    Methods
    -------
    execute : Transform data to warming level approach
    update_context : Update processing context with warming level information
    set_data_accessor : Set data catalog accessor

    Notes
    -----
    The input data must span from 1980-2100 and include historical climate data
    for proper warming level calculations. Data should have simulation and scenario
    dimensions or be properly configured for stacking.
    """

    def __init__(self, value: Dict[str, Any]):
        """
        Initialize the warming level processor.

        Parameters
        ----------
        value : Dict[str, Any]
            Configuration dictionary with warming level parameters.
        """
        # Validate input
        if not isinstance(value, dict):
            raise TypeError("Expected dictionary for warming level configuration")

        # Extract configuration parameters
        self.warming_levels = value.get("warming_levels", [2.0])
        self.warming_level_months = value.get(
            "warming_level_months", list(range(1, 13))
        )
        self.warming_level_window = value.get("warming_level_window", 15)

        # Initialize instance variables
        self.name = "warming_level_simple"
        self.warming_level_times = None
        self.catalog = None

    def execute(
        self,
        result: Union[
            xr.Dataset, xr.DataArray, Iterable[Union[xr.Dataset, xr.DataArray]]
        ],
        context: Dict[str, Any],
    ) -> Union[xr.Dataset, xr.DataArray, Iterable[Union[xr.Dataset, xr.DataArray]]]:
        """
        Transform time-series data to warming level approach.

        Parameters
        ----------
        result : xr.Dataset | xr.DataArray | Iterable[xr.Dataset | xr.DataArray]
            The time-series climate data to transform.
        context : dict
            The context for the processor containing metadata and configuration.

        Returns
        -------
        Union[xr.Dataset, xr.DataArray, Iterable[xr.Dataset | xr.DataArray]]
            The data transformed to warming level approach with new dimensions:
            - warming_level: The target warming levels
            - simulation: Combined simulation identifiers
            - time_delta: Time steps relative to warming level center year
        """
        # Load warming level times table if not already loaded
        if self.warming_level_times is None:
            try:
                self.warming_level_times = read_csv_file(
                    GWL_1850_1900_FILE, index_col=[0, 1, 2]
                )
            except (FileNotFoundError, pd.errors.ParserError) as e:
                raise RuntimeError(
                    f"Failed to load warming level times table: {e}"
                ) from e

        # Handle different input types
        match result:
            case list() | tuple():  # Process each item in the iterable
                processed_items = []
                for item in result:
                    processed_item = self._process_single_data(item, context)
                    processed_items.append(processed_item)
                return type(result)(processed_items)

            case dict():
                # Process each value in the dictionary
                processed_dict = {}
                for key, value in result.items():
                    processed_dict[key] = self._process_single_data(value, context)
                return processed_dict

            case _:
                # Process single DataArray or Dataset
                if not isinstance(result, (xr.DataArray, xr.Dataset)):
                    raise TypeError(
                        f"Expected xarray DataArray or Dataset, got {type(result)}"
                    )
                return self._process_single_data(result, context)

    def _process_single_data(
        self,
        data: Union[xr.DataArray, xr.Dataset],
        context: Dict[str, Any],  # noqa: ARG002
    ) -> Union[xr.DataArray, xr.Dataset, None]:
        """
        Process a single DataArray or Dataset with warming level approach.

        Parameters
        ----------
        data : Union[xr.DataArray, xr.Dataset]
            Single data object to process
        context : Dict[str, Any]
            Processing context

        Returns
        -------
        Union[xr.DataArray, xr.Dataset, None]
            Processed data organized by warming levels
        """
        # Validate input data
        if not isinstance(data, (xr.DataArray, xr.Dataset)):
            raise TypeError(f"Expected xarray DataArray or Dataset, got {type(data)}")

        # Check for required time dimension
        if "time" not in data.dims:
            warnings.warn(
                "Input data does not have a 'time' dimension. "
                "Warming level processing requires time dimension for calculations."
            )
            return None

        # Validate time span
        time_years = data.time.dt.year
        min_year, max_year = int(time_years.min()), int(time_years.max())

        if min_year > 1980 or max_year < 2100:
            raise ValueError(
                f"Data time span ({min_year}-{max_year}) insufficient for warming level analysis. "
                "Data must span 1980-2100 for proper warming level calculations."
            )

        # Stack simulation and scenario dimensions if they exist
        if "simulation" in data.dims and "scenario" in data.dims:
            data_stacked = data.stack(all_sims=["simulation", "scenario"])
        elif "simulation" in data.dims:
            data_stacked = data.rename({"simulation": "all_sims"})
        else:
            # If no simulation dimensions, create a dummy one
            data_stacked = data.expand_dims({"all_sims": ["default_sim"]})

        # Remove invalid simulations that don't exist in warming level table
        data_cleaned = drop_invalid_sims(data_stacked, self.warming_level_times)

        if data_cleaned.sizes.get("all_sims", 0) == 0:
            raise ValueError(
                "No valid simulations remain after filtering. Check that your data "
                "contains simulations present in the warming level lookup table."
            )

        # Process each warming level
        da_list = []
        for level in self.warming_levels:
            try:
                # Use the established calculate_warming_level function
                warming_data = calculate_warming_level(
                    warming_data=data_cleaned,
                    gwl_times=self.warming_level_times,
                    level=level,
                    months=self.warming_level_months,
                    window=self.warming_level_window,
                )

                if warming_data is not None:
                    da_list.append(warming_data)
                else:
                    # Create empty array with proper structure for missing data
                    empty_data = self._create_empty_warming_level_data(
                        data_cleaned, level
                    )
                    da_list.append(empty_data)

            except (ValueError, KeyError, AttributeError, IndexError) as e:
                print(f"Warning: Failed to process warming level {level}: {e}")
                # Create empty array for failed processing
                empty_data = self._create_empty_warming_level_data(data_cleaned, level)
                da_list.append(empty_data)

        if not da_list:
            raise ValueError(
                f"No warming level data could be processed for levels: {self.warming_levels}"
            )

        # Concatenate all warming levels
        result_data = xr.concat(da_list, dim="warming_level")

        # Rename dimensions for clarity
        if "all_sims" in result_data.dims:
            result_data = result_data.rename({"all_sims": "simulation"})

        if "time" in result_data.dims:
            result_data = result_data.rename({"time": "time_delta"})

        # Add coordinate attributes
        self._add_coordinate_attributes(result_data)

        return result_data

    def _create_empty_warming_level_data(
        self, template_data: Union[xr.DataArray, xr.Dataset], level: float
    ) -> xr.DataArray:
        """
        Create empty data array for missing warming level data.

        Parameters
        ----------
        template_data : Union[xr.DataArray, xr.Dataset]
            Template data to match structure
        level : float
            Warming level value

        Returns
        -------
        xr.DataArray
            Empty data array with proper structure
        """
        # Convert Dataset to DataArray if needed
        if isinstance(template_data, xr.Dataset):
            # Use the first data variable as template
            template_data = template_data[list(template_data.data_vars)[0]]
        # Calculate expected time points
        expected_points = len(self.warming_level_months) * self.warming_level_window * 2

        # Create time coordinate
        time_coords = np.linspace(
            -expected_points / 2, expected_points / 2, expected_points, endpoint=False
        )

        # Create empty data with NaN values
        shape = (
            [len(template_data.all_sims)]
            + [expected_points]
            + list(template_data.shape[2:])
        )
        empty_data = np.full(shape, np.nan)

        # Create coordinates dictionary
        coords = {"all_sims": template_data.all_sims, "time": time_coords}

        # Add spatial coordinates if they exist
        for coord_name in template_data.coords:
            if coord_name not in ["all_sims", "time"] and isinstance(coord_name, str):
                coords[coord_name] = template_data.coords[coord_name]

        # Create DataArray
        empty_da = xr.DataArray(
            empty_data,
            dims=template_data.dims,
            coords=coords,
            attrs=template_data.attrs,
        )

        # Add warming level dimension
        empty_da = empty_da.expand_dims({"warming_level": [level]})

        return empty_da

    def _add_coordinate_attributes(self, data: xr.DataArray):
        """
        Add descriptive attributes to coordinates.

        Parameters
        ----------
        data : xr.DataArray
            Data array to add attributes to
        """
        if "warming_level" in data.coords:
            data["warming_level"].attrs = {
                "description": "degrees Celsius above the historical baseline",
                "long_name": "Global warming level",
            }

        if "time_delta" in data.coords:
            data["time_delta"].attrs = {
                "description": f"time steps from center year (±{self.warming_level_window} years)",
                "long_name": "Time relative to warming level center",
            }

        if "simulation" in data.coords:
            data["simulation"].attrs = {
                "description": "simulation identifier combining model, ensemble, and scenario",
                "long_name": "Climate simulation",
            }

    def update_context(self, context: Dict[str, Any]):
        """
        Update the context with information about the warming level transformation.

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

        # Create detailed description of the transformation
        months_str = ", ".join(map(str, self.warming_level_months))
        levels_str = ", ".join(f"{level}°C" for level in self.warming_levels)

        context[_NEW_ATTRS_KEY][self.name] = (
            f"Warming level transformation applied to the data.\n"
            f"Warming levels: {levels_str}\n"
            f"Window size: ±{self.warming_level_window} years\n"
            f"Months analyzed: {months_str}\n"
            f"Data organized by warming levels instead of chronological time."
        )

        # Update approach in context
        context["approach"] = "Warming Level"
        context["warming_levels"] = self.warming_levels
        context["warming_level_window"] = self.warming_level_window
        context["warming_level_months"] = self.warming_level_months

    def set_data_accessor(self, catalog: DataCatalog):
        """
        Set data accessor for retrieving warming level information.

        Parameters
        ----------
        catalog : DataCatalog
            Data catalog for accessing warming level lookup tables
        """
        # Store catalog reference for potential future use
        self.catalog = catalog

        # The warming level times table is loaded on-demand in execute()
        # This approach ensures the processor can work independently
