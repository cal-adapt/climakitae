"""
Drop leap days (February 29) from time series data.
"""

import logging
from typing import Any, Dict, Iterable, Union

import xarray as xr

from climakitae.core.constants import _NEW_ATTRS_KEY
from climakitae.new_core.data_access.data_access import DataCatalog
from climakitae.new_core.processors.abc_data_processor import (
    DataProcessor,
    register_processor,
)

# Module logger
logger = logging.getLogger(__name__)


@register_processor(
    "drop_leap_days",
    priority=1,
)
class DropLeapDays(DataProcessor):
    """
    Drop leap days (February 29) from time series data.

    This processor removes all February 29 timestamps from the data,
    which is useful for creating 365-day calendars or for compatibility
    with models that don't support leap years.

    Parameters
    ----------
    value : str
        If "yes", drop leap days. If "no", pass through data unchanged.

    Methods
    -------
    execute(result, context)
        Run the processor on the provided data.
    update_context(context)
        Update the context with information about the transformation.
    set_data_accessor(catalog)
        Set the data accessor for the processor (not used).
    """

    def __init__(self, value: str = "yes"):
        """
        Initialize the DropLeapDays processor.

        Parameters
        ----------
        value : str, optional
            If "yes", drop leap days. If "no", pass through data unchanged.
            Default is "yes".
        """
        self.valid_values = ["yes", "no"]
        self.value = value.lower()
        self.name = "drop_leap_days"
        logger.debug("DropLeapDays initialized with value=%s", self.value)

    def execute(
        self,
        result: Union[
            xr.Dataset, xr.DataArray, Iterable[Union[xr.Dataset, xr.DataArray]]
        ],
        context: Dict[str, Any],
    ) -> Union[xr.Dataset, xr.DataArray, Iterable[Union[xr.Dataset, xr.DataArray]]]:
        """
        Run the leap day dropping operation on the data.

        Parameters
        ----------
        result : xr.Dataset | xr.DataArray | Iterable[xr.Dataset | xr.DataArray]
            The data to be processed.

        context : dict
            The context for the processor.

        Returns
        -------
        Union[xr.Dataset, xr.DataArray, Iterable[xr.Dataset | xr.DataArray]]
            The processed data with leap days removed. This can be a single
            Dataset/DataArray or an iterable of them.

        Raises
        ------
        ValueError
            If the value is not one of the valid values.
        """
        logger.debug(
            "DropLeapDays.execute called with value=%s result_type=%s",
            self.value,
            type(result).__name__,
        )

        match self.value:
            case "yes":
                return self._process_data(result, context)
            case "no":
                # Pass through unchanged
                return result
            case _:
                logger.error(
                    "Invalid value for DropLeapDays processor, must be 'yes' or 'no'"
                )
                raise ValueError(
                    f"Invalid value for {self.name} processor: {self.value}. "
                    f"Valid values are: {', '.join(self.valid_values)}."
                )

    def _process_data(
        self,
        result: Union[
            xr.Dataset, xr.DataArray, Iterable[Union[xr.Dataset, xr.DataArray]]
        ],
        context: Dict[str, Any],
    ) -> Union[xr.Dataset, xr.DataArray, Iterable[Union[xr.Dataset, xr.DataArray]]]:
        """Process data by removing leap days."""
        match result:
            case dict():
                processed_data = {}
                for key, value in result.items():
                    processed_data[key] = self._drop_leap_days(value)
                self.update_context(context)
                logger.info(
                    "Dropped leap days from %d data entries",
                    len(processed_data),
                )
                return processed_data

            case xr.DataArray() | xr.Dataset():
                self.update_context(context)
                logger.info("Dropped leap days from data")
                return self._drop_leap_days(result)

            case list() | tuple():
                processed_data = []
                for value in result:
                    processed_data.append(self._drop_leap_days(value))
                self.update_context(context)
                logger.info(
                    "Dropped leap days from %d data entries",
                    len(processed_data),
                )
                return type(result)(processed_data)

            case _:
                msg = f"Invalid data type for DropLeapDays. Expected xr.Dataset, dict, list, or tuple but got {type(result)}."
                logger.warning(msg)
                return result

    def update_context(self, context: Dict[str, Any]):
        """
        Update the context with information about the leap day dropping operation.

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
        ] = "Leap days (February 29) have been removed from the data."

    def set_data_accessor(self, catalog: DataCatalog):
        """Set the data accessor for the processor (not used)."""
        pass

    def _drop_leap_days(
        self,
        obj: Union[xr.Dataset, xr.DataArray],
    ) -> Union[xr.Dataset, xr.DataArray]:
        """
        Drop February 29 from the data.

        Parameters
        ----------
        obj : xr.Dataset | xr.DataArray
            The data to process.

        Returns
        -------
        xr.Dataset | xr.DataArray
            The data with leap days removed.
        """
        if "time" not in obj.dims:
            logger.debug("No time dimension found, returning data unchanged")
            return obj

        import pdb

        pdb.set_trace()
        # Create a boolean mask for leap days
        leap_days = obj.sel(
            time=(obj.time.dt.month == 2) & (obj.time.dt.day == 29)
        ).time

        # Dropping leap days from `obj`
        return obj.drop_sel(time=leap_days)

        import pdb

        pdb.set_trace()
        return obj.drop_sel(time=is_leap_day)
        # is_not_leap_day = ~((obj.time.dt.month == 2) & (obj.time.dt.day == 29))
        # return obj.sel(time=is_not_leap_day)
