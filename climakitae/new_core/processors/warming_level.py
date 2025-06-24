"""
Warming Level Data Processor

A simplified warming level processor that transforms time-series climate data
to a warming level centered approach following the established template pattern.
"""

import re
import warnings
from typing import Any, Dict, Iterable, Union

import numpy as np
import pandas as pd
import xarray as xr

from climakitae.core.constants import _NEW_ATTRS_KEY
from climakitae.core.paths import GWL_1981_2010_TIMEIDX_FILE
from climakitae.new_core.data_access.data_access import DataCatalog
from climakitae.new_core.processors.abc_data_processor import (
    DataProcessor,
    register_processor,
)
from climakitae.util.utils import read_csv_file


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
        self.warming_level_window = value.get("warming_level_window", 15)

        # Initialize instance variables
        self.name = "warming_level_simple"
        self.warming_level_times = read_csv_file(
            GWL_1981_2010_TIMEIDX_FILE, index_col="time", parse_dates=True
        )
        self.catalog = None
        self.value = {
            "warming_levels": self.warming_levels,
            "warming_level_window": self.warming_level_window,
        }

    def execute(
        self,
        result: Union[
            xr.Dataset, xr.DataArray, Iterable[Union[xr.Dataset, xr.DataArray]]
        ],
        context: Dict[str, Any],
    ) -> Union[xr.Dataset, xr.DataArray, Iterable[Union[xr.Dataset, xr.DataArray]]]:
        """
        Transform time-series data to warming level approach.

        1. find (from precomputed values) when a given warming level is reached by a
            simulation (GCM, run, scenario)
            a. some fancy handling of values between precomputed values happens?
        2. slice in a window of X years around that date
            a. if the slice has a start date earlier than the simulation data, splice
                historical onto the slice for the requested variable
        3. return the data

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
                    GWL_1981_2010_TIMEIDX_FILE, index_col="time", parse_dates=True
                )
            except (FileNotFoundError, pd.errors.ParserError) as e:
                raise RuntimeError(
                    f"Failed to load warming level times table: {e}"
                ) from e

        # extend the time domain of all ssp scenarios to 1980-2100
        ret = self.extend_time_domain(result)

        # first, extract the member IDs from the data
        member_ids = [v.attrs.get("variant_label", None) for k, v in ret.items()]

        # get center years for each key for each warming level
        center_years = self.get_center_years(member_ids, ret.keys())
        retkeys = list(ret.keys())
        for key in retkeys:
            if key not in center_years:
                del ret[key]

        for key, years in center_years.items():
            if not years:
                continue

            slices = []

            common_time_delta = None
            for year, wl in zip(years, self.warming_levels):
                start_year = pd.to_datetime(year).year - self.warming_level_window
                start_year = max(start_year, 1981)
                end_year = pd.to_datetime(year).year + self.warming_level_window
                end_year = min(end_year, 2100)

                da_slice = ret[key].sel(time=slice(f"{start_year}", f"{end_year}"))

                # Drop February 29th if it exists
                is_feb29 = (da_slice.time.dt.month == 2) & (da_slice.time.dt.day == 29)
                da_slice = da_slice.where(~is_feb29, drop=True)
                if common_time_delta is None:
                    # initialize by length starting at - len/2
                    # get length of the time dimension
                    length = da_slice.sizes["time"]
                    common_time_delta = range(-length // 2, length // 2)

                # Replace time dimension with time_delta
                da_slice = da_slice.swap_dims({"time": "time_delta"})
                da_slice = da_slice.drop_vars("time")
                da_slice = da_slice.assign_coords(time_delta=common_time_delta)

                # Expand dimensions to include warming_level as a new dimension
                da_slice = da_slice.expand_dims({"warming_level": [wl]})

                # Add simulation and centered_year coordinates
                da_slice = da_slice.assign_coords(
                    simulation=key,
                    centered_year=(["warming_level"], [pd.to_datetime(year).year]),
                )

                slices.append(da_slice)

            ret[key] = xr.concat(
                slices, dim="warming_level", join="outer", fill_value=np.nan
            )
        self.update_context(context)
        return ret

    def update_context(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Update the processing context with warming level information.

        Parameters
        ----------
        context : dict
            The processing context to update.

        Returns
        -------
        dict
            The updated processing context with warming level metadata.
        """
        context[_NEW_ATTRS_KEY][
            self.name
        ] = f"""Process '{self.name}' applied to the data. Transformation was done using the following settings: {self.value}."""

        return context

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

    def extend_time_domain(
        self, result: Dict[str, Union[xr.Dataset, xr.DataArray]]
    ) -> Union[xr.Dataset, xr.DataArray]:
        """
        Extend the time domain of the input data to cover 1980-2100.

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
        """
        ret = {}
        for key, data in result.items():
            if "ssp" not in key:
                continue  # Skip historical and reanalysis data

            hist_key = re.sub(r"ssp.{3}", "historical", key)
            if hist_key not in result:
                warnings.warn(
                    f"\n\nNo historical data found for {key} with key {hist_key}. "
                    f"\nHistorical data is required for warming level calculations."
                )
                continue

            ret[key] = xr.concat(
                [result[hist_key], data],
                dim="time",
            )
            ret[key].attrs.update(data.attrs)  # Preserve attributes

        return ret

    def get_center_years(
        self, member_ids: Iterable[str], keys: Iterable[str]
    ) -> Dict[str, list]:
        """
        Determine the year around which to center the warming level window for each
        simulation for each warming level.

        Parameters
        ----------
        member_ids : Iterable[str]
            List of member IDs corresponding to the keys.
        keys : Iterable[str]
            List of keys representing different simulations or scenarios.

        Returns
        -------
        Dict[str, list]
            A dictionary mapping each key to a list of center years for each warming level.

        Notes
        -----
        The center year is determined by finding the first occurrence of each
        warming level in the precomputed warming level times table.
        If no warming level data is found for a key, a warning is issued.
        If the warming level table does not contain data for a key, a warning is issued.
        The method assumes that the warming level times table is indexed by time
        and contains columns formatted as "key.join('_')", where the values are the
        warming levels and the index is the time dimension.
        """
        center_years = {}

        # load idx table
        # cols are key.join("_"), values are warming levels, index is time
        for key, member_id in zip(keys, member_ids):

            key_list = key.split(".")
            wl_table_key = f"{key_list[2]}_{member_id}_{key_list[3]}"
            if wl_table_key not in self.warming_level_times.columns:
                warnings.warn(
                    f"Warming level table does not contain data for {wl_table_key}. "
                    "Ensure the warming level times table is correctly configured."
                )
                continue
            if key not in center_years:
                center_years[key] = []
            # get the FIRST index value for this key
            for wl in self.warming_levels:
                mask = (self.warming_level_times[wl_table_key] >= wl).dropna()
                if mask.any():
                    center_years[key].append(
                        mask.idxmax()  # Get the first occurrence of the warming level
                    )
                else:
                    warnings.warn(
                        f"\n\nNo warming level data found for {wl_table_key} at {wl}C. "
                        f"\nPlease pick a warming level less than {self.warming_level_times[wl_table_key].max()}C."
                    )

        return center_years
