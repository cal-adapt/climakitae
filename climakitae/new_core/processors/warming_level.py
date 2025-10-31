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
from climakitae.core.paths import GWL_1850_1900_FILE, GWL_1981_2010_TIMEIDX_FILE
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
            GWL_1850_1900_FILE, index_col=[0, 1, 2], parse_dates=True
        )
        self.warming_level_times_idx = read_csv_file(
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

        The transformation process involves the following steps for each simulation:
        1. find the first year (from precomputed values) when a given warming level
            is reached by a simulation (GCM, run, scenario)
        2. slice in a window of `self.warming_level_window` years around that year
            a. if the slice has a start year earlier than the simulation data, splice
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

        # add member id as a suffix to the keys
        # and split the data by member_id
        ret = self.reformat_member_ids(result)

        # extend the time domain of all ssp scenarios to 1980-2100
        ret = self.extend_time_domain(ret)

        # first, extract the member IDs from the data
        member_ids = []
        for k in ret.keys():
            mem_id = k.split(".")[-1]  # Get the last part as member_id
            if mem_id[0] == "r":
                # If it starts with 'r', it's a member_id
                member_ids.append(mem_id)
            else:
                # If not, assume no member_id is present
                member_ids.append(None)

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

            for year, wl in zip(years, self.warming_levels):
                if year is None or pd.isna(year):
                    warnings.warn(
                        f"\n\nNo warming level data found for {key} at {wl}C. "
                        "\nSkipping this warming level."
                    )
                    continue
                start_year = year - self.warming_level_window
                start_year = max(start_year, 1981)
                end_year = year + self.warming_level_window - 1
                end_year = min(end_year, 2100)

                da_slice = ret[key].sel(time=slice(f"{start_year}", f"{end_year}"))

                # Drop February 29th if it exists
                is_feb29 = (da_slice.time.dt.month == 2) & (da_slice.time.dt.day == 29)
                da_slice = da_slice.where(~is_feb29, drop=True)

                # Create time_delta specific to this slice's length
                # This ensures each warming level has the correct time_delta length
                length = da_slice.sizes["time"]
                time_delta = range(-length // 2, length // 2)

                # Replace time dimension with time_delta
                da_slice = da_slice.swap_dims({"time": "time_delta"})
                da_slice = da_slice.drop_vars("time")
                da_slice = da_slice.assign_coords(time_delta=time_delta)

                # Expand dimensions to include warming_level as a new dimension
                da_slice = da_slice.expand_dims({"warming_level": [wl]})

                # Add simulation and centered_year coordinates
                da_slice = da_slice.assign_coords(
                    simulation=key,
                    centered_year=(["warming_level"], [year]),
                )

                slices.append(da_slice)

            if not slices:
                warnings.warn(
                    f"\n\nNo valid slices found for {key}. "
                    "Ensure the warming level times table is correctly configured."
                )
                del ret[key]  # Remove key if no valid slices found
                continue
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
        if _NEW_ATTRS_KEY not in context:
            context[_NEW_ATTRS_KEY] = {}

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

    def reformat_member_ids(
        self, result: Dict[str, Union[xr.Dataset, xr.DataArray]]
    ) -> Dict[str, Union[xr.Dataset, xr.DataArray]]:
        """
        Reformat member IDs in the input data.

        Parameters
        ----------
        result : Dict[str, Union[xr.Dataset, xr.DataArray]]
            A dictionary containing time-series data with keys representing different scenarios.

        Returns
        -------
        Dict[str, Union[xr.Dataset, xr.DataArray]]
            The reformatted time-series data.
        """
        ret = {}
        for key, data in result.items():
            if "member_id" in data.dims:
                # If member_id is present, reformat it
                member_ids = data.member_id.values
                for mem_id in member_ids:
                    # Create a new key with the member_id included
                    new_key = f"{key}.{mem_id}"
                    ret[new_key] = data.sel(member_id=mem_id).drop_vars("member_id")
                    ret[new_key].attrs.update(data.attrs)
            else:
                # If no member_id, keep the original key
                ret[key] = data
                warnings.warn(
                    f"\n\nNo member_id found in data for key {key}. "
                    "\nAssuming no member_id is present for this dataset."
                )
        return ret

    def extend_time_domain(
        self, result: Dict[str, Union[xr.Dataset, xr.DataArray]]
    ) -> Dict[str, Union[xr.Dataset, xr.DataArray]]:
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

            if "time" not in data.dims or "time" not in result[hist_key].dims:
                warnings.warn(
                    f"\n\nNo time dimension found in data for key {key} or {hist_key}. "
                    f"\nCannot extend time domain without time dimension."
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

        Center year can be np.nan if no warming level data is found.
        """
        center_years = {}

        # cols are key.join("_"), values are warming levels, index is time
        for key, member_id in zip(keys, member_ids):
            if member_id is None:
                continue  # Skip if member_id is None

            # build the key for the warming level table
            key_list = key.split(".")

            if key not in center_years:
                center_years[key] = []

            for wl in self.warming_levels:
                if str(wl) not in self.warming_level_times.columns:
                    # warming level is not an integer value, so we need to try to find
                    # the first year that the simulation crosses the given warming level
                    wl_table_key = f"{key_list[2]}_{member_id}_{key_list[3]}"
                    if wl_table_key not in self.warming_level_times_idx.columns:
                        warnings.warn(
                            f"\n\nWarming level table does not contain data for {wl_table_key}. "
                            "\nEnsure the warming level times table is correctly configured."
                        )
                        center_years[key].append(np.nan)
                        continue  # exit warming levels loop

                    # get the FIRST index value for this key
                    mask = (self.warming_level_times_idx[wl_table_key] >= wl).dropna()
                    if mask.any():
                        center_years[key].append(
                            mask.idxmax()  # Get the first occurrence of the warming level
                        )
                    else:
                        max_valid_wl = (
                            self.warming_level_times.loc[
                                (key_list[2], member_id, key_list[3])
                            ]
                            .dropna()
                            .index.max()
                        )
                        warnings.warn(
                            f"\n\nNo warming level data found for {wl_table_key} at {wl}C. "
                            f"\nPlease pick a warming level less than {max_valid_wl}C."
                        )
                        center_years[key].append(np.nan)
                else:
                    # this is a common warming level, so we can just get the year
                    tuple_key = (key_list[2], member_id, key_list[3])
                    center_time = pd.to_datetime(
                        self.warming_level_times.loc[tuple_key, str(wl)]
                    )
                    center_years[key].append(center_time.year)

        return center_years
