from typing import Any, Dict

import xarray as xr

from climakitae.core.constants import UNSET
from climakitae.new_core.data_access import DataCatalog
from climakitae.new_core.processors.abc_data_processor import _PROCESSOR_REGISTRY
from climakitae.new_core.param_validation.abc_param_validation import ParameterValidator


class Dataset:
    """Base class for all dataset types providing a common interface."""

    def __init__(self):
        self.data_access = UNSET
        self.parameter_validator = UNSET
        self.processing_pipeline = UNSET  # list of processing steps

    def execute(self, parameters: Dict[str, Any] = UNSET) -> xr.Dataset:
        """
        Execute the dataset processing pipeline.

        Parameters
        ----------
        parameters : Dict[str, Any], optional
            Parameters to pass to the processing pipeline

        Returns
        -------
        xr.Dataset
            Result of the processing pipeline
        """
        # Initialize context with parameters
        context = parameters.copy() if parameters is not UNSET else {}

        # Validate parameters if validator is set
        valid_query = UNSET
        if self.parameter_validator is not UNSET:
            if not isinstance(self.parameter_validator, ParameterValidator):
                raise TypeError(
                    "Parameter validator must be an instance of ParameterValidator."
                )
            valid_query = self.parameter_validator.is_valid_query(context)
            if valid_query is None:
                return xr.Dataset()  # return empty dataset if validation fails

        # Check if data access is properly configured
        if self.data_access is UNSET:
            raise ValueError("Data accessor is not configured.")

        # Initialize the processing result - will be updated through pipeline steps
        current_result = None

        # Check if we have a processing pipeline
        if self.processing_pipeline is UNSET or not self.processing_pipeline:
            # If no pipeline is defined, just return the raw data from data_access
            return self.data_access.get_data(valid_query)

        # Execute each step in the pipeline in sequence
        try:
            for step in self.processing_pipeline:
                # Some steps might need access to the data_access component
                if hasattr(step, "set_data_accessor") and callable(
                    getattr(step, "set_data_accessor")
                ):
                    step.set_data_accessor(self.data_access)

                # Execute the current step
                current_result = step.execute(current_result, context)

                # Allow the step to update the context for subsequent steps
                if hasattr(step, "update_context") and callable(
                    getattr(step, "update_context")
                ):
                    step.update_context(context, current_result)

            # Ensure the final result is an xarray Dataset
            if not isinstance(current_result, xr.Dataset):
                raise TypeError(
                    f"Pipeline result must be an xr.Dataset, got {type(current_result)}"
                )

            return current_result

        except Exception as e:
            # Consider implementing proper error handling/logging here
            raise RuntimeError(f"Error in processing pipeline: {str(e)}") from e

    def with_param_validator(
        self, parameter_validator: ParameterValidator
    ) -> "Dataset":
        """
        Set a new parameter validator.

        Parameters
        ----------
        parameter_validator : ParameterValidator
            Parameter validator to set for the dataset.

        Returns
        -------
        Dataset
            The current instance of Dataset allowing method chaining.

        Raises
        ------
        TypeError
            If the parameter validator is not an instance of ParameterValidator.
        """
        if not isinstance(parameter_validator, ParameterValidator):
            raise TypeError(
                "Parameter validator must be an instance of ParameterValidator."
            )
        self.parameter_validator = parameter_validator
        return self

    def with_catalog(self, catalog: DataCatalog) -> "Dataset":
        """
        Set a new data catalog.

        Parameters
        ----------
        catalog : DataCatalog
            Data catalog to set for the dataset.

        Returns
        -------
        Dataset
            The current instance of Dataset allowing method chaining.

        Raises
        ------
        TypeError
            If the catalog is not an instance of DataCatalog.
        AttributeError
            If the catalog does not have a 'get_data' method.
        TypeError
            If the 'get_data' method is not callable.
        """
        if not isinstance(catalog, DataCatalog):
            raise TypeError("Data catalog must be an instance of DataCatalog.")
        if not hasattr(catalog, "get_data"):
            raise AttributeError(
                "Data catalog must have a 'get_data' method to retrieve data."
            )
        if not callable(getattr(catalog, "get_data")):
            raise TypeError("'get_data' method in data catalog must be callable.")
        self.data_access = catalog
        return self

    def with_processing_step(self, step_key: str, value: Any) -> "Dataset":
        """
        Add a new processing step to the pipeline.

        Parameters
        ----------
        step : DataProcessor
            Processing step to add to the pipeline. Must have 'execute' and 'update_context' methods.
        """
        step = _PROCESSOR_REGISTRY.get(step_key, (object, object))[0](value) 
        if not hasattr(step, "execute") or not callable(getattr(step, "execute")):
            raise TypeError("Processing step must have an 'execute' method.")
        if not hasattr(step, "update_context") or not callable(
            getattr(step, "update_context")
        ):
            raise TypeError("Processing step must have an 'update_context' method.")
        if not hasattr(step, "set_data_accessor") or not callable(
            getattr(step, "set_data_accessor")
        ):
            raise TypeError("Processing step must have a 'set_data_accessor' method.")
        if self.processing_pipeline is UNSET:
            self.processing_pipeline = []
        self.processing_pipeline.append(step)
        return self
