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
    priority=2,
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
        - repair_time_axis : str
            Default "no". If "yes", repairs the time axis by:
                - Removing duplicate timestamp due to daylight savings in Fall
                - Filling missing timestamp due to daylight savings end in Spring
                - Replacing any Feb 29 timestamps with Feb 28

    Methods
    -------
    _convert_to_local_time_station(obj: xr.Dataset | xr.DataArray) -> xr.Dataset | xr.DataArray
        Pull lat and lon from the station data, then call _find_timezone_and_convert to convert.
    _convert_to_local_time_gridded(obj: xr.Dataset | xr.DataArray) -> xr.Dataset | xr.DataArray
        Check time frequency and get area average lat and lon for time conversion.
    _find_timezone_and_convert(obj: xr.Dataset | xr.DataArray, lat: float, lon: float) -> xr.Dataset | xr.DataArray
        Use the provided lat and lon to get the timezone, then convert the time data and update attributes.

    Notes
    -----
    By default, this process is set to "no" and data is returned in UTC time.
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
        self.repair_time_axis = value.get("repair_time_axis", "no")
        self.name = "convert_to_local_time"
        self.timezone = "None"
        self.catalog: Union[DataCatalog, object] = UNSET
        self.success = False
        logger.debug(
            "ConvertToLocalTime initialized with convert=%s, repair_time_axis=%s",
            self.value,
            self.repair_time_axis,
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
                                "Could not convert time to local time for all entries."
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
                                "Could not convert time to local time for all entries."
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

        # Finding avg. lat/lon coordinates from all grid-cells
        lat = obj.lat.mean().compute().data.item()
        lon = obj.lon.mean().compute().data.item()

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

        lat = obj.lat.isel(time=0).compute().data.item()
        lon = obj.lon.isel(time=0).compute().data.item()

        obj = self._find_timezone_and_convert(obj, lat, lon)
        return obj

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
        # Find timezone for the coordinates
        tf = TimezoneFinder()
        local_tz = tf.timezone_at(lng=lon, lat=lat)

        # Change datetime objects to local time
        new_time = (
            pd.DatetimeIndex(obj.time)
            .tz_localize("UTC")
            .tz_convert(local_tz)
            .tz_localize(None)
            .astype("datetime64[ns]")
        )
        obj["time"] = new_time

        if self.repair_time_axis == "yes":
            # Drop duplicate timestamps due to daylight savings time start
            # in some timezones.
            obj = obj.drop_duplicates("time")

            # Find missing day in spring due to daylight savings end
            all_dates = pd.date_range(
                start=obj.time.min().data.item(),
                end=obj.time.max().data.item(),
                freq="1h",
            ).to_list()
            missing_times = set(all_dates).difference(
                [pd.Timestamp(t) for t in obj.time.data]
            )
            missing_times = [
                x for x in list(missing_times) if ((x.month == 3) | (x.month == 4))
            ]

            # Expand time dim
            new_times = np.array([np.datetime64(t) for t in missing_times])
            new_time_axis = np.concat((obj.time, new_times))
            new_time_axis.sort()
            obj_updated_times = obj.reindex(
                time=new_time_axis, fill_value=np.nan
            ).sortby("time")

            # Fill missing times with prior hour
            times_to_copy = [x - pd.Timedelta(hours=1) for x in missing_times]
            data_to_copy = obj.sel(time=times_to_copy)
            data_to_copy["time"] = missing_times
            obj_updated_times.loc[{"time": missing_times}] = data_to_copy
            obj = obj_updated_times

            # reset remaining feb 29 hours to feb 28
            obj["time"] = xr.where(
                (obj.time.dt.month == 2) & (obj.time.dt.day == 29),
                pd.to_datetime(obj.time) - pd.DateOffset(days=1),
                obj.time,
            )

        logger.debug(f"Data converted to {local_tz} timezone.")
        self.timezone = local_tz

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
