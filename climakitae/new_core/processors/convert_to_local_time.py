"""
Convert time axis from UTC to local time.
"""

import logging
from typing import Any, Dict, Iterable, Union

import xarray as xr

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
    Slice data based on time.

    Parameters
    ----------
    value : iterable of date-like or dict
        Either:

        - An iterable of two date-like objects specifying the start and end
          dates for the time slice, or
        - A dictionary with a required "dates" key and an optional
          "seasons" key.

        The "dates" value must be an iterable of two date-like objects.
        The "seasons" value, if provided, must be an iterable of seasons or a string of a single season.
        Allowed season inputs are: "DJF", "MAM", "JJA", "SON"

    Methods
    -------
    _convert_to_local_time_station(obj: xr.Dataset | xr.DataArray) -> xr.Dataset | xr.DataArray
        Description
    _convert_to_local_time_gridded(obj: xr.Dataset | xr.DataArray) -> xr.Dataset | xr.DataArray
        Description
    _find_timezone_and_convert(obj: xr.Dataset | xr.DataArray, lat: float, lon: float) -> xr.Dataset | xr.DataArray
    """


    def __init__(self, value: str = "no"):
        """Initialize the processor.

        Parameters
        ----------
        value : str
            The state of the filter. If "yes", it converts time to local time.

        """
        self.valid_values = ["yes", "no"]
        self.value = value.lower()
        self.name = "convert_to_local_time"
        self.timezone = "None"
        self.catalog: Union[DataCatalog, object] = UNSET
        self.needs_catalog = True
        logger.debug("ConvertToLocalTime initialized with value=%s", self.value)

    def execute(
        self,
        result: Union[
            xr.Dataset, xr.DataArray, Iterable[Union[xr.Dataset, xr.DataArray]]
        ],
        context: Dict[str, Any],
    ) -> Union[xr.Dataset, xr.DataArray, Iterable[Union[xr.Dataset, xr.DataArray]]]:
        """
        Run the time slicing operation on the data.

        Parameters
        ----------
        result : xr.Dataset | xr.DataArray | Iterable[xr.Dataset | xr.DataArray]
            The data to be sliced.

        context : dict
            The context for the processor. This is not used in this
            implementation but is included for consistency with the
            DataProcessor interface.

        Returns
        -------
        Union[xr.Dataset, xr.DataArray, Iterable[xr.Dataset | xr.DataArray]]
            The sliced data. This can be a single Dataset/DataArray or
            an iterable of them.
        """
        logger.debug(
            "ConvertToLocalTime.execute called with value=%s result_type=%s",
            self.value,
            type(result).__name__,
        )


        # Route to appropriate concat method based on catalog
        # Check context for catalog type (more reliable than catalog object attribute)
        catalog_type = context.get("catalog", context.get("_catalog_key"))
        if catalog_type == "hdp":
            func = self._convert_to_local_time_station
        else:
            func = self._convert_to_local_time_gridded
        
        match result:
            case dict():  # most likely case at top
                subset_data = {}
                for key, value in result.items():
                    subset_data[key] = func(value)
                self.update_context(context)
                logger.info(
                    "Converted time to local time for %d data entries",
                    len(subset_data),
                )
                return subset_data

            case xr.DataArray() | xr.Dataset():
                self.update_context(context)
                logger.info("Converted time to local time.")
                return func(result)

            case list() | tuple():
                subset_data = []
                for value in result:
                    subset_data.append(func(value))
                # return as the same type as the input
                self.update_context(context)
                logger.info(
                    "Converted time to local time for %d data entries",
                    len(subset_data),
                )
                return type(result)(subset_data)
            case _:
                msg = f"Invalid data type for subsetting. Expected xr.Dataset, dict, list, or tuple but got {type(result)}."
                logger.warning(msg)

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
        """Set the data catalog accessor for the processor.

        The processor requires access to station metadata through the DataCatalog.
        Station observational data is loaded directly from S3 zarr stores.

        Parameters
        ----------
        catalog : DataCatalog
            Data catalog for accessing station metadata.

        Returns
        -------
        None
        """
        self.catalog = catalog
    
    def _convert_to_local_time_gridded(
        self,
        obj: Union[xr.Dataset, xr.DataArray],
    ) -> xr.DataArray | xr.Dataset:
        """Convert time dimension from UTC to local time for the grid or station.
    
        Parameters
        ----------
            data : xr.DataArray | xr.Dataset
                Input data.
            grid_lon : float
                Mean longitude of dataset if no lat/lon coordinates
            grid_lat : float
                Mean latitude of dataset if no lat/lon coordinates
    
        Returns
        -------
            xr.DataArray | xr.Dataset
                Data with converted time coordinate.
    
        """
    
        # Only converting hourly data
        # TODO: Would new core loaded objects always have a frequency?
        if not (frequency := obj.attrs.get("frequency", None)):
            # Make a guess at frequency
            timestep = pd.Timedelta(
                obj.time[1].item() - obj.time[0].item()
            ).total_seconds()
            match timestep:
                case 3600:
                    frequency = "hourly"
                case 86400:
                    frequency = "daily"
                case _ if timestep > 86400:
                    frequency = "monthly"
    
        # If timescale is not hourly, no need to convert
        if frequency != "hourly":
            logger.warn(
                "This dataset's timescale is not granular enough to covert to local time. Local timezone conversion requires hourly data."
            )
            return obj
    
        # Get latitude/longitude information
    
        # if both lat and lon are set, can move on to timezone finding.
        if (lat is UNSET) or (lon is UNSET):
            try:
                # Finding avg. lat/lon coordinates from all grid-cells
                lat = obj.lat.mean().item()
                lon = obj.lon.mean().item()
            #TODO: logger, see how to throw errors properly
            except AttributeError:
                logger.error(
                    "Could not convert time because lat/lon coordinates not found in data. Please pass in data with 'lon' and 'lat' coordinates or set both 'lon' and 'lat' arguments."
                )
                return obj

        obj = _find_timezone_and_convert(obj,lat,lon)

        return obj

    def _convert_to_local_time_station(
        self,
        obj: Union[xr.Dataset, xr.DataArray],
    ) -> xr.DataArray | xr.Dataset:

        lat = station_data["LAT_Y"].values[0]
        lon = station_data["LON_X"].values[0]

        obj = _find_timezone_and_convert(obj,lat,lon)
        return obj

    def _find_timezone_and_convert(obj: Union[xr.Dataset, xr.DataArray],lat: float,lon: float) -> xr.DataArray | xr.Dataset:
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
    
        logger.info(f"Data converted to {local_tz} timezone.")
        self.timezone = local_tz
    
        # Add timezone attribute to data
        match obj:
            case xr.DataArray():
                obj = obj.assign_attrs({"timezone": local_tz})
            case xr.Dataset():
                variables = list(datobja.keys())
                for variable in variables:
                    obj[variable] = obj[variable].assign_attrs({"timezone": local_tz})
            # TODO: logger
            case _:
                logger.warn(f"Invalid data type {type(obj)}. Could not set timezone attribute.")
        return obj

