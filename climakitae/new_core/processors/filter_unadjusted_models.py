"""Filter Unadjusted Models Processor"""

import warnings
from typing import Any, Dict, Iterable, Union

import xarray as xr

from climakitae.core.constants import _NEW_ATTRS_KEY, NON_WRF_BA_MODELS, UNSET
from climakitae.new_core.data_access.data_access import DataCatalog
from climakitae.new_core.processors.abc_data_processor import (
    DataProcessor,
    register_processor,
)


@register_processor("filter_unadjusted_models", priority=0)
class FilterUnAdjustedModels(DataProcessor):
    """Processor to filter out models that do not have a-priori bias adjustment.

    Parameters
    ----------
    value : tuple(date-like, date-like)
        The value to subset the data by. This should be a tuple of two
        date-like values.

    Methods
    -------
    execute(result, context)
        Run the processor on the given result and context.
    update_context(context)
        Update the context with information about the transformation.
    set_data_accessor(catalog)
        Set the data accessor for the processor.
    _contains_unadjusted_models(result) -> bool
        Check if the result contains any unadjusted models.
    _remove_unadjusted_models(result) -> Union[xr.Dataset, xr.DataArray, Iterable[xr.Dataset, xr.DataArray], None]
        Remove unadjusted models from the result.

    Notes
    -----
    This processor filters out models that do not have a-priori bias adjustment.
    It is added to the processor chain by default when using the `ClimateData` class.
    If you want to include these models, you manually add the processor to your query
    and set the value to "no".

    """

    def __init__(self, value: str = "yes"):
        """Initialize the processor.

        Parameters
        ----------
        value : str
            The state of the filter. If "yes", it filters out unadjusted models.

        """
        self.valid_values = ["yes", "no"]
        self.value = value.lower()
        self.name = "filter_unadjusted_models"

    def execute(
        self,
        result: Union[
            xr.Dataset, xr.DataArray, Iterable[Union[xr.Dataset, xr.DataArray]]
        ],
        context: Dict[str, Any],
    ) -> Union[xr.Dataset, xr.DataArray, Iterable[Union[xr.Dataset, xr.DataArray]]]:
        """Run the processor

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

        Raises
        ------
        ValueError
            If the value is not one of the valid values.

        """
        match self.value:
            case "yes":
                # check if there are any unadjusted models in the dataset
                if self._contains_unadjusted_models(result):
                    warnings.warn(
                        f"\n\nYour query selected models that do not have a-priori bias adjustment. "
                        f"\nThese models have been removed from the returned query."
                        f"\nTo include them, please add the following processor to your query: "
                        f"\nClimateData().processes('{self.name}': 'no')",
                        stacklevel=999,
                    )
                    return self._remove_unadjusted_models(result)

                # If no unadjusted models are found, return the result as is
                return result
            case "no":
                # check if there are any biased models in the dataset
                if self._contains_unadjusted_models(result):
                    warnings.warn(
                        "\n\nYour query selected models that do not have a-priori bias adjustment. "
                        "\nThese models HAVE NOT been removed from the returned query."
                        "\nProceed with caution as these models may not be suitable for your analysis.\n\n",
                        stacklevel=999,
                    )

                return result
            case _:
                raise ValueError(
                    f"Invalid value for {self.name} processor: {self.value}. "
                    f"Valid values are: {', '.join(self.valid_values)}."
                )

    def _contains_unadjusted_models(
        self,
        result: Union[
            xr.Dataset, xr.DataArray, Iterable[Union[xr.Dataset, xr.DataArray]]
        ],
    ) -> bool:
        """Check if the result contains any unadjusted models.

        Parameters
        ----------
        result : Union[xr.Dataset, xr.DataArray, Iterable[xr.Dataset, xr.DataArray]]
            The data to check for unadjusted models.

        Returns
        -------
        bool
            True if the result contains unadjusted models, False otherwise.

        Raises
        ------
        TypeError
            If the result is not of type xr.Dataset, xr.DataArray, or Iterable.

        """
        match result:
            case xr.Dataset() | xr.DataArray():
                activity_id = result.attrs.get("intake_esm_attrs:activity_id", UNSET)
                source_id = result.attrs.get("intake_esm_attrs:source_id", UNSET)
                member_id = result.attrs.get("intake_esm_attrs:member_id", UNSET)
                model_id = f"{activity_id}_{source_id}_{member_id}"
                return model_id in NON_WRF_BA_MODELS
            case dict():
                return any(
                    self._contains_unadjusted_models(item) for item in result.values()
                )
            case list() | tuple():
                return any(self._contains_unadjusted_models(item) for item in result)
            case _:
                raise TypeError(
                    f"Unsupported type for result: {type(result)}. "
                    "Expected xr.Dataset, xr.DataArray, or Iterable."
                )

    def _remove_unadjusted_models(
        self,
        result: Union[
            xr.Dataset, xr.DataArray, Iterable[Union[xr.Dataset, xr.DataArray]]
        ],
    ) -> Union[
        xr.Dataset, xr.DataArray, Iterable[Union[xr.Dataset, xr.DataArray]], None
    ]:
        """Remove unadjusted models from the result.

        Parameters
        ----------
        result : Union[xr.Dataset, xr.DataArray, Iterable[xr.Dataset, xr.DataArray]]
            The data to filter.

        Returns
        -------
        Union[xr.Dataset, xr.DataArray, Iterable[xr.Dataset, xr.DataArray]] | None
            The filtered data without unadjusted models or None if all models are unadjusted.

        Raises
        ------
        TypeError
            If the result is not of type xr.Dataset, xr.DataArray, or Iterable.

        """
        match result:
            case xr.Dataset() | xr.DataArray():
                return result if not self._contains_unadjusted_models(result) else None
            case dict():
                return {
                    key: value
                    for key, value in result.items()
                    if not self._contains_unadjusted_models(value)
                }
            case list() | tuple():
                return type(result)(
                    [
                        value
                        for value in result
                        if not self._contains_unadjusted_models(value)
                    ]
                )
            case _:
                raise TypeError(
                    f"Unsupported type for result: {type(result)}. "
                    "Expected xr.Dataset, xr.DataArray, or Iterable."
                )

    def update_context(self, context: Dict[str, Any]):
        """Update the context with information about the transformation.

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
        ] = f"""Process '{self.name}' applied to the data. Transformation was done using the following value: {self.value}."""

    def set_data_accessor(self, catalog: DataCatalog):
        # Placeholder for setting data accessor
        pass
