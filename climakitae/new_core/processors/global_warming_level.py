"""
Warming Level Data Processor
"""

import calendar
from typing import Any, Dict, Iterable, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import xarray as xr

from climakitae.core.constants import _NEW_ATTRS_KEY, UNSET
from climakitae.core.paths import GWL_1850_1900_FILE
from climakitae.new_core.data_access import DataCatalog
from climakitae.new_core.processors.abc_data_processor import (
    DataProcessor,
    register_processor,
)
from climakitae.util.utils import _get_cat_subset, scenario_to_experiment_id


@register_processor("warming_level", priority=20)
class GlobalWarmingLevel(DataProcessor):
    """
    Transform time-based climate data into a warming-levels approach.

    This processor takes data with time dimensions and transforms it
    to data organized by warming levels, handling cases where simulation
    information is stored in attributes rather than dimensions.

    Parameters
    ----------
    value : Dict[str, Any]
        Configuration for the warming level processor.
        Requires the following keys:
            warming_levels : List[float]
                List of global warming levels in degrees C (e.g., [1.5, 2.0])
            warming_level_times : pd.DataFrame
                Table showing when each simulation reaches each warming level
            warming_level_months : List[int]
                List of months to include (1-12)
            warming_level_window : int
                Number of years before and after the central year to include

    Notes
    -----
    The input data must span from 1980-2100 and include historical climate data.
    """

    def __init__(self, value: Optional[Dict[str, Any]] | object = UNSET):
        """Initialize the warming level processor."""
        # Handle initialization with different input types
        if value is UNSET:
            # Set defaults
            self.warming_levels = [2.0]
            self.warming_level_months = list(range(1, 13))
            self.warming_level_window = 15

        elif isinstance(value, dict):
            # Extract values from the dictionary
            self.warming_levels = value.get("warming_levels", [2.0])
            self.warming_level_months = value.get(
                "warming_level_months", list(range(1, 13))
            )
            self.warming_level_window = value.get("warming_level_window", 15)

            print("INFORMATION:::")
            print("Warming levels configuration:")
            print(f"  Warming levels: {self.warming_levels} °C")
            print(f"  Months analyzed: {self.warming_level_months}")
            print(f"  Window size: +/- {self.warming_level_window} years")
            print("If this does not look correct, please check your configuration.")
        else:
            raise TypeError("Invalid input type. Expected a dictionary or UNSET.")

        self.name: str = "warming_level"
        self.warming_level_times: pd.DataFrame | object = UNSET
        self.catalog: DataCatalog | object = UNSET
        self.needs_catalog: bool = True

    def execute(
        self,
        result: xr.DataArray | xr.Dataset | Iterable[xr.DataArray | xr.Dataset],
        context: Dict[str, Any],
    ) -> Union[xr.DataArray, xr.Dataset, Dict, List, Tuple]:
        """
        Transform time-based climate data to warming-levels approach.

        Parameters
        ----------
        result : Union[xr.DataArray, xr.Dataset, Dict, List, Tuple]
            Climate data in various formats
        context : Dict[str, Any]
            Processing context

        Returns
        -------
        Union[xr.DataArray, xr.Dataset, Dict, List, Tuple]
            Data organized by warming levels in the same structure as the input
        """
        if self.warming_level_times is UNSET:
            self.warming_level_times = pd.read_csv(
                GWL_1850_1900_FILE, index_col=[0, 1, 2]
            )
            # indices are source_id, member_id, experiment_id
            # and the columns are the warming levels
        match result:
            case dict():
                # Handle dictionary of data objects
                return {
                    key: self._process_single_data_object(value, context)
                    for key, value in result.items()
                }

            case list() | tuple():
                # Handle list or tuple of data objects
                container_type = type(result)
                return container_type(
                    [self._process_single_data_object(item, context) for item in result]
                )

            case xr.DataArray() | xr.Dataset():
                # Handle single DataArray or Dataset
                return self._process_single_data_object(result, context)

            case _:
                raise TypeError(
                    f"Unsupported data type: {type(result)}. Expected xarray.DataArray, "
                    f"xarray.Dataset, dict, list, or tuple."
                )

    def _process_single_data_object(
        self, da: Union[xr.DataArray, xr.Dataset], context: Dict[str, Any]
    ) -> Union[xr.DataArray, xr.Dataset]:
        """
        Process a single DataArray or Dataset with the warming levels approach.

        Parameters
        ----------
        da : Union[xr.DataArray, xr.Dataset]
            Single data object to process
        context : Dict[str, Any]
            Processing context

        Returns
        -------
        Union[xr.DataArray, xr.Dataset]
            Processed data organized by warming levels
        """

        # Extract simulation info from attributes instead of dimensions
        # This is the key change to handle when simulation info is in attributes
        sim_info = self._extract_simulation_info_from_attributes(da, context)

        # Calculate warming level data for each level
        da_list = []
        for level in self.warming_levels:
            da_by_wl = self._calculate_warming_level_for_single_sim(
                data=da,
                sim_info=sim_info,
                gwl_times=self.warming_level_times,
                level=level,
                months=self.warming_level_months,
                window=self.warming_level_window,
            )
            if da_by_wl is not None:
                da_list.append(da_by_wl)

        # Combine results along new dimension
        if not da_list:
            raise ValueError(
                f"No data could be processed for the specified warming levels: {self.warming_levels}. "
                "Check that the simulation reaches these warming levels."
            )

        warming_data = xr.concat(da_list, dim="warming_level")

        # Add simulation info as a coordinate for clarity
        model_id = sim_info.get("model_id", "unknown_model")
        ensemble_id = sim_info.get("ensemble_id", "unknown_ensemble")
        scenario_id = sim_info.get("scenario_id", "unknown_scenario")
        activity_id = sim_info.get("activity_id", "unknown_activity")

        sim_str = f"{activity_id}_{model_id}_{ensemble_id}_historical+{scenario_id}"
        warming_data = warming_data.expand_dims({"simulation": [sim_str]})

        # Add metadata
        frequency = da.attrs.get("table_id", None)

        warming_data["time"].attrs = {"description": f"{frequency} from center year"}
        warming_data["centered_year"].attrs = {
            "description": f"central year in +/-{self.warming_level_window} year window"
        }
        warming_data["warming_level"].attrs = {
            "description": "degrees Celsius above the historical baseline"
        }
        warming_data["simulation"].attrs = {
            "description": "simulation identifier including model, ensemble, and scenario"
        }

        # Rename the time dimension for clarity
        warming_data = warming_data.rename({"time": "time_delta"})

        # Preserve important attributes from the original data
        for attr_key in ["activity_id", "source_id", "experiment_id"]:
            if attr_key in da.attrs:
                warming_data.attrs[attr_key] = da.attrs[attr_key]

        return warming_data

    def _extract_simulation_info_from_attributes(
        self, data: Union[xr.DataArray, xr.Dataset], context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Extract simulation information from attributes instead of dimensions.

        Parameters
        ----------
        data : Union[xr.DataArray, xr.Dataset]
            Climate data with simulation info in attributes
        context : Dict[str, Any]
            Processing context with additional information

        Returns
        -------
        Dict[str, str]
            Dictionary with simulation info (model_id, ensemble_id, scenario_id, activity_id)
        """
        # Extract info from attributes
        model_id = data.attrs.get("source_id", context.get("source_id"))
        ensemble_id = data.attrs.get("member_id", context.get("member_id"))
        scenario_id = data.attrs.get("experiment_id", context.get("experiment_id"))
        activity_id = data.attrs.get("activity_id", context.get("activity_id"))

        # Validate the extracted information
        if not all([model_id, scenario_id]):
            raise ValueError(
                "Could not extract required simulation information from attributes or context. "
                "Make sure 'source_id' and 'experiment_id' are available."
            )

        # Handle case where scenario contains "Historical + "
        if isinstance(scenario_id, str) and "Historical + " in scenario_id:
            scenario_id = scenario_to_experiment_id(
                scenario_id.split("Historical + ")[1]
            )

        return {
            "model_id": model_id,
            "ensemble_id": ensemble_id or "r1i1p1f1",  # Default if missing
            "scenario_id": scenario_id,
            "activity_id": activity_id or "CMIP6",  # Default if missing
        }

    def _calculate_warming_level_for_single_sim(
        self,
        data: xr.DataArray | xr.Dataset,
        sim_info: Dict[str, str],
        gwl_times: pd.DataFrame,
        level: float,
        months: List[int],
        window: int,
    ) -> Optional[xr.DataArray]:
        """
        Calculate warming level data for a specific global warming level for a single simulation.

        Parameters
        ----------
        data : xr.DataArray | xr.Dataset
            Climate data for a single simulation
        sim_info : Dict[str, str]
            Simulation information (model_id, ensemble_id, scenario_id)
        gwl_times : pd.DataFrame
            Table showing when each simulation reaches each warming level
        level : float
            Target warming level in degrees C
        months : List[int]
            Which months to include (1-12)
        window : int
            Years before/after central year to include

        Returns
        -------
        Optional[xr.DataArray]
            Climate data for the specified warming level or None if not applicable
        """
        # Find when this simulation reaches the target warming level
        # Get the specific row from the warming level times table
        try:
            model_id = sim_info["model_id"]
            ensemble_id = sim_info["ensemble_id"]
            scenario_id = sim_info["scenario_id"]

            # Look up when this simulation reaches the warming level
            # This depends on the structure of your warming level times DataFrame
            # Adjust the lookup logic based on your DataFrame structure
            gwl_times_subset = gwl_times.loc[(model_id, ensemble_id, scenario_id)]
            center_time = gwl_times_subset.loc[str(float(level))]
        except (KeyError, ValueError) as e:
            # This simulation might not reach this warming level
            return None

        # Remove leap days for consistency
        cleaned_data = self._remove_leap_days(data)

        # Process based on whether this simulation reaches the warming level
        if pd.isna(center_time):
            return None
        else:
            return self._extract_window_around_year(
                cleaned_data, pd.to_datetime(center_time).year, months, window, level
            )

    def _remove_leap_days(self, data: xr.DataArray) -> xr.DataArray:
        """Remove February 29th from the dataset for consistency."""
        return data.loc[~((data.time.dt.month == 2) & (data.time.dt.day == 29))]

    def _extract_window_around_year(
        self,
        data: xr.DataArray,
        center_year: int,
        months: List[int],
        window: int,
        level: float,
    ) -> xr.DataArray:
        """Extract data within specified window around a center year."""
        # Define the time range
        start_year = center_year - window
        end_year = center_year + (window - 1)

        # Slice data to the window
        sliced = data.sel(time=slice(str(start_year), str(end_year)))

        # Filter to keep only specified months
        valid_months_mask = sliced.time.dt.month.isin(months)
        sliced = sliced.isel(time=valid_months_mask)

        # Get frequency from data or context
        frequency = getattr(data, "frequency", "monthly")

        # Calculate expected data points based on frequency
        expected_points = self._calculate_expected_points(frequency, window, months)

        # Create new centered time axis
        # This creates a time dimension centered around 0 (the warming level year)
        n_points = len(sliced.time)
        sliced["time"] = np.linspace(
            -n_points / 2, n_points / 2, n_points, endpoint=False
        )

        # Add center year and warming level coordinates
        sliced = sliced.assign_coords({"centered_year": center_year})
        sliced = sliced.expand_dims({"warming_level": [level]})

        return sliced

    def _calculate_expected_points(
        self, frequency: str, window: int, months: List[int]
    ) -> int:
        """Calculate expected number of data points based on frequency and window."""
        # Count days per month for daily/hourly frequency
        days_per_month = {
            i: sum(calendar.monthrange(y, i)[1] for y in range(2001, 2001 + window * 2))
            // (window * 2)
            for i in range(1, 13)
        }

        match frequency:
            case "monthly":
                return len(months) * window * 2
            case "daily":
                return sum(days_per_month[month] for month in months) * window * 2
            case "hourly":
                return sum(days_per_month[month] for month in months) * 24 * window * 2
            case _:
                # Default to monthly if frequency is unknown
                return len(months) * window * 2

    def update_context(self, context: Dict[str, Any]):
        """Update the context with warming level processing information."""
        if _NEW_ATTRS_KEY not in context:
            context[_NEW_ATTRS_KEY] = {}

        # Add more descriptive metadata
        context[_NEW_ATTRS_KEY][self.name] = (
            f"Data transformed to warming levels approach.\n"
            f"Warming levels: {self.warming_levels} °C\n"
            f"Window size: +/- {self.warming_level_window} years\n"
            f"Months analyzed: {self.warming_level_months}\n"
        )

        # Update approach in context
        context["approach"] = "Warming Level"

    def set_data_accessor(self, catalog: DataCatalog):
        """Set data accessor for retrieving warming level information if needed."""
        self.catalog = catalog

        # Initialize warming_level_times from the catalog if needed
        if self.warming_level_times is UNSET and self.catalog is not None:
            # Implement logic to retrieve warming level times from catalog
            # For example:
            # self.warming_level_times = self.catalog.get_warming_level_times()
            pass
