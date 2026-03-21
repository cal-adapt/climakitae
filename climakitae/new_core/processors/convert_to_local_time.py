"""
Convert time axis from UTC to local time.
"""

import logging
from typing import Any, Dict, Iterable, Union

import numpy as np
import pandas as pd
import xarray as xr
from timezonefinder import TimezoneFinder

from climakitae.core.constants import _NEW_ATTRS_KEY, UNSET
from climakitae.new_core.data_access.data_access import DataCatalog
from climakitae.new_core.processors.abc_data_processor import (
    DataProcessor,
    register_processor,
)

# Module logger
logger = logging.getLogger(__name__)


@register_processor(
    "convert_to_local_time",
    priority=70,
)
class ConvertToLocalTime(DataProcessor):
    """
    Convert time data from UTC to local time.

    Parameters
    ----------
    value : Union[str, list, dict[str, Any]]
        The configuration dictionary. Expected keys:
        - convert : str
            The value to subset the data by.
        - reindex_time_axis : str
            Default "no". If "yes", cleans the time axis by:
                - Removing duplicate hour due to Daylight Savings Time end in Fall
                - Filling skipped hour (with NaN) due to Daylight Savings Time start in Spring

    Methods
    -------
    _convert_to_local_time_hdp(obj: xr.Dataset | xr.DataArray) -> xr.Dataset | xr.DataArray
        Pull lat and lon from the HDP data, then call _find_timezone_and_convert to convert.
    _convert_to_local_time_gridded(obj: xr.Dataset | xr.DataArray) -> xr.Dataset | xr.DataArray
        Check time frequency and get area average lat and lon for time conversion.
    _contains_leap_days(obj: xr.Dataset | xr.DataArray) -> bool
        Return True if leap days exist in obj.
    _find_timezone_and_convert(obj: xr.Dataset | xr.DataArray, lat: float, lon: float) -> xr.Dataset | xr.DataArray
        Use the provided lat and lon to get the timezone, then convert the time data and update attributes.

    Notes
    -----
    By default, this process is set to "no" and data is returned in UTC time. For gridded data, the timezone will be
    selected using the central values of the data's longitude and latitude coordinates. Leap days will be preserved
    if present in the original data. If a timezone uses Daylight Savings Time, the returned time axis will not be
    continuous (there will be skipped timestamps and duplicated timestamps).

    The 'reindex_time_axis' option is provided for use cases which require a clean time axis (24-hour days with no
    duplicate timestamps). This may not be an appropriate option for many types of analysis as the underlying
    timeseries is edited at the beginning and end of DST (if it exists in a given time zone).
    """

    def __init__(self, value: Dict[str, Any]):
        """Initialize the processor.

        Parameters
        ----------
        value : str
            The state of the filter. If "yes", it converts time to local time.

        """
        self.valid_values = ["yes", "no"]
        self.value = value.get("convert", "no")
        self.reindex_time_axis = value.get("reindex_time_axis", "no")
        self.name = "convert_to_local_time"
        self.timezone = "None"
        self.catalog: Union[DataCatalog, object] = UNSET
        self.success = False
        logger.debug(
            "ConvertToLocalTime initialized with convert=%s, reindex_time_axis=%s",
            self.value,
            self.reindex_time_axis,
        )

    def execute(
        self,
        result: Union[
            xr.Dataset, xr.DataArray, Iterable[Union[xr.Dataset, xr.DataArray]]
        ],
        context: Dict[str, Any],
    ) -> Union[xr.Dataset, xr.DataArray, Iterable[Union[xr.Dataset, xr.DataArray]]]:
        """
        Run the time conversion operation on the data.

        Parameters
        ----------
        result : xr.Dataset | xr.DataArray | Iterable[xr.Dataset | xr.DataArray]
            The data to be sliced.

        context : dict
            The context for the processor.

        Returns
        -------
        Union[xr.Dataset, xr.DataArray, Iterable[xr.Dataset | xr.DataArray]]
            The converted data. This can be a single Dataset/DataArray or
            an iterable of them.
        """
        logger.debug(
            "ConvertToLocalTime.execute called with value=%s result_type=%s",
            self.value,
            type(result).__name__,
        )
        match self.value:
            case "yes":
                # Route to appropriate conversion method based on catalog
                # Check context for catalog type (more reliable than catalog object attribute)
                catalog_type = context.get("catalog", context.get("_catalog_key"))
                if catalog_type == "hdp":
                    func = self._convert_to_local_time_hdp
                else:
                    func = self._convert_to_local_time_gridded

                match result:
                    case dict():  # most likely case at top
                        subset_data = {}
                        for key, value in result.items():
                            subset_data[key] = func(value)
                        self.update_context(context)
                        if self.success:
                            msg = (
                                "Converted time to local time for %d data entries",
                                len(subset_data),
                            )
                        else:
                            msg = (
                                "Could not convert time to local time for all entries.",
                            )
                        logger.info(*msg)
                        return subset_data

                    case xr.DataArray() | xr.Dataset():
                        self.update_context(context)
                        data = func(result)
                        if self.success:
                            msg = "Converted time to local time."
                        else:
                            msg = (
                                "Could not convert time to local time for all entries."
                            )
                        logger.info(msg)
                        return data

                    case list() | tuple():
                        subset_data = []
                        for value in result:
                            subset_data.append(func(value))
                        # return as the same type as the input
                        self.update_context(context)
                        if self.success:
                            msg = (
                                "Converted time to local time for %d data entries",
                                len(subset_data),
                            )
                        else:
                            msg = (
                                "Could not convert time to local time for all entries.",
                            )
                        logger.info(*msg)
                        return type(result)(subset_data)
                    case _:
                        msg = f"Invalid data type for subsetting. Expected xr.Dataset, dict, list, or tuple but got {type(result)}."
                        logger.warning(msg)
            case "no":
                # Do nothing if processor value is "no" (default value).
                return result
            case _:
                raise ValueError(
                    f"Invalid value for {self.name} processor: {self.value}. "
                    f"Valid values are: {', '.join(self.valid_values)}."
                )

    def update_context(self, context: Dict[str, Any]):
        """
        Update the context with information about the clipping operation, to be stored
                in the "new_attrs" attribute.

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

        context[_NEW_ATTRS_KEY][
            self.name
        ] = f"""Process '{self.name}' applied to the data. Conversion was done using the following value: {self.timezone}."""

    def set_data_accessor(self, catalog: DataCatalog):
        # Placeholder for setting data accessor
        pass

    def _convert_to_local_time_gridded(
        self,
        obj: Union[xr.Dataset, xr.DataArray],
    ) -> xr.DataArray | xr.Dataset:
        """Convert time dimension from UTC to local time for gridded data.

        Parameters
        ----------
            obj : xr.DataArray | xr.Dataset
                Input data.

        Returns
        -------
            xr.DataArray | xr.Dataset
                Data with converted time coordinate.

        """

        # Only converting hourly data
        if not (frequency := obj.attrs.get("frequency", None)):
            # Make a guess at frequency
            timestep = pd.Timedelta(
                obj.time[1].item() - obj.time[0].item()
            ).total_seconds()
            match timestep:
                case 3600:
                    frequency = "1hr"
                case 86400:
                    frequency = "day"
                case _ if timestep > 86400:
                    frequency = "mon"

        # If timescale is not hourly, no need to convert
        if frequency != "1hr":
            logger.warning(
                f"This dataset's timescale {frequency} is not granular enough to covert to local time. Local timezone conversion requires hourly data."
            )
            return obj

        # Get latitude/longitude information

        # Finding central lat/lon coordinates
        if obj.lat.dims == ("station",):
            # Take coords of first station
            lat = obj.lat[0]
            lon = obj.lon[0]
        elif obj.lat.ndim < 2:
            lat = obj.lat.values.item()
            lon = obj.lon.values.item()
        else:
            lat = float((obj.lat[0, 0] + obj.lat[-1, 0]) / 2)
            lon = float((obj.lon[0, 0] + obj.lon[0, -1]) / 2)

        obj = self._find_timezone_and_convert(obj, lat, lon)

        return obj

    def _convert_to_local_time_hdp(
        self,
        obj: Union[xr.Dataset, xr.DataArray],
    ) -> xr.DataArray | xr.Dataset:
        """Convert time dimension from UTC to local time for station data.

        Parameters
        ----------
            obj : xr.DataArray | xr.Dataset
                Input data.

        Returns
        -------
            xr.DataArray | xr.Dataset
                Data with converted time coordinate.

        """

        # Use first lat/lon value since HDP data is not gridded
        # We assume station is unmoving, which isn't always true but likely will not make a difference here
        lat = obj.isel(time=0).lat.values[0].item()
        lon = obj.isel(time=0).lon.values[0].item()

        obj = self._find_timezone_and_convert(obj, lat, lon)
        return obj

    def _contains_leap_days(
        self,
        obj: Union[xr.Dataset, xr.DataArray],
    ):
        """Check if data has leap days.

        Parameters
        ----------
            obj : xr.DataArray | xr.Dataset
                Input data.

        Returns
        -------
            bool
        """
        # Create a boolean mask for leap days
        leap_days = obj.sel(
            time=(obj.time.dt.month == 2) & (obj.time.dt.day == 29)
        ).time

        # Dropping leap days from `obj`
        if len(leap_days) > 0:
            return True
        return False

    def _find_timezone_and_convert(
        self, obj: Union[xr.Dataset, xr.DataArray], lat: float, lon: float
    ) -> xr.DataArray | xr.Dataset:
        """Use lat and lon to determine correct timezone, then run conversion and
        update attributes.

        Parameters
        ----------
            obj : xr.DataArray | xr.Dataset
                Input data.
            lat : float
                Latitude
            lon : float
                Longitude

        Returns
        -------
            xr.DataArray | xr.Dataset
                Data with converted time coordinate.

        """
        # Check if data has leap days before converting time
        no_leap = not self._contains_leap_days(obj)

        # Find timezone for the coordinates
        tf = TimezoneFinder()
        local_tz = tf.timezone_at(lng=lon, lat=lat)

        # Change datetime objects to local time
        local_time = (
            pd.DatetimeIndex(obj.time)
            .tz_localize("UTC")
            .tz_convert(local_tz)
            .tz_localize(None)
            .astype("datetime64[ns]")
        )
        obj["time"] = local_time
        if no_leap:
            # Shift feb 29 afternoon hours to feb 28
            obj["time"] = xr.where(
                (obj.time.dt.month == 2) & (obj.time.dt.day == 29),
                pd.to_datetime(obj.time) - pd.DateOffset(days=1),
                obj.time,
            )
        logger.debug(f"Data converted to {local_tz} timezone.")
        self.timezone = local_tz

        if self.reindex_time_axis == "yes":
            # Drop duplicate timestamps due to daylight savings time start
            # in some timezones.
            times = pd.DatetimeIndex(obj.time.data)
            _, first_occurrence = np.unique(times, return_index=True)
            no_repeated_times = times[np.sort(first_occurrence)]
            obj_updated_times = obj.isel(time=np.sort(first_occurrence))

            # Find missing hour in spring due to daylight savings start
            diffs = no_repeated_times[1:] - no_repeated_times[:-1]
            gap_starts = no_repeated_times[:-1][diffs > pd.Timedelta("1h")]
            times_to_fill = [
                t + pd.Timedelta("1h") for t in gap_starts if t.month in (3, 4)
            ]

            # Expand time dim to include missing times
            if len(times_to_fill) > 0:
                # Now make sure time types match between new and original times
                if isinstance(obj_updated_times.time.data[0], np.datetime64):
                    # Need to make sure units are the same for datetime
                    time_units = np.datetime_data(obj_updated_times.time.data[0])[0]
                    times_to_fill = [
                        np.datetime64(t, time_units) for t in times_to_fill
                    ]
                else:
                    obj_time_type = type(obj_updated_times.time.data[0])
                    times_to_fill = [obj_time_type(t) for t in times_to_fill]
                # Create axis with missing times included
                new_time_axis = np.concat((no_repeated_times, times_to_fill))
                new_time_axis.sort()
                # Add new time axis to our object with NaN fill for missing times
                obj_updated_times = obj_updated_times.reindex(
                    time=new_time_axis, fill_value=np.nan
                ).sortby("time")

            obj = obj_updated_times
            logger.debug(f"Reindexed time axis after conversion to local time.")

        # Add timezone attribute to data
        match obj:
            case xr.DataArray():
                obj = obj.assign_attrs({"timezone": local_tz})
            case xr.Dataset():
                variables = list(obj.keys())
                for variable in variables:
                    obj[variable] = obj[variable].assign_attrs({"timezone": local_tz})
            case _:
                logger.warning(
                    f"Invalid data type {type(obj)}. Could not set timezone attribute."
                )
        # Update flag to print success message in execute() method.
        self.success = True
        return obj
