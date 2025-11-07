"""Dataset Processing Pipeline Module

This module provides the core Dataset class that implements a flexible, pipeline-based
approach for climate data processing. The Dataset class serves as a central orchestrator
that coordinates data access, parameter validation, and a series of processing steps.

Classes
-------
Dataset
    A pipeline-based data processing class that supports method chaining for building
    complex data workflows.

Key Features
------------
- **Pipeline Architecture**: Execute sequential processing steps on climate data
- **Method Chaining**: Fluent interface for building complex data workflows
- **Parameter Validation**: Integrated validation system for query parameters
- **Data Access Integration**: Pluggable data catalog system for various data sources
- **Error Handling**: Comprehensive error handling with meaningful error messages

Usage Example
-------------
```python
from climakitae.new_core.dataset import Dataset
from climakitae.new_core.data_access import DataCatalog
from climakitae.new_core.param_validation import ParameterValidator
from climakitae.new_core.processors import TimeSliceProcessor, ClipProcessor

# Create a dataset processing pipeline
dataset = (Dataset()
    .with_catalog(my_data_catalog)
    .with_param_validator(my_validator)
    .with_processing_step(TimeSliceProcessor("2010-01-01", "2020-12-31"))
    .with_processing_step(ClipProcessor(bounds=((32, 42), (-125, -115))))
)

# Execute the pipeline
result = dataset.execute({"variable": "temperature", "grid_label": "d03"})
```

Pipeline Processing
-------------------
The Dataset class executes processing in the following order:

1. **Parameter Validation**: Validates input parameters using the configured validator
2. **Data Access**: Retrieves raw data using the configured data catalog
3. **Processing Steps**: Applies each processing step in sequence
4. **Result Return**: Returns the final processed xarray.Dataset

Each processing step receives the output of the previous step, allowing for complex
data transformations and filtering operations.

Error Handling
--------------
The class provides comprehensive error handling:

- TypeError: Raised for incorrect component types (validators, catalogs, processors)
- ValueError: Raised for missing required components
- AttributeError: Raised for components missing required methods
- RuntimeError: Raised for pipeline execution failures

Notes
-----
- Processing steps are executed in the order they are added to the pipeline
- The context dictionary is passed through all processing steps and may be modified
- Steps that require data access can set `needs_catalog = True` to receive the data accessor
- Validation failures return an empty xarray.Dataset rather than raising exceptions

"""

import traceback
import warnings
from typing import Any, Dict

import xarray as xr

from climakitae.core.constants import UNSET
from climakitae.new_core.data_access.data_access import DataCatalog
from climakitae.new_core.param_validation.abc_param_validation import ParameterValidator
from climakitae.new_core.processors.abc_data_processor import (
    _PROCESSOR_REGISTRY,
    DataProcessor,
)


class Dataset:
    """A pipeline-based data processing class for climate data workflows.

    The Dataset class serves as a central orchestrator that coordinates data access,
    parameter validation, and sequential processing steps. It implements a fluent
    interface pattern allowing method chaining for building complex data workflows.

    Parameters
    ----------
    None

    Attributes
    ----------
    data_access : DataCatalog or UNSET
        The data catalog instance used for retrieving raw data from various sources.
    parameter_validator : ParameterValidator or UNSET
        The parameter validator instance used for validating query parameters.
    processing_pipeline : list of DataProcessor or UNSET
        A list of processing steps to be executed sequentially on the data.

    Methods
    -------
    execute(parameters=UNSET)
        Execute the complete data processing pipeline and return the result.
    with_param_validator(parameter_validator)
        Set the parameter validator for the dataset (method chaining).
    with_catalog(catalog)
        Set the data catalog for the dataset (method chaining).
    with_processing_step(step)
        Add a processing step to the pipeline (method chaining).

    Raises
    ------
    TypeError
        If provided components don't match expected types or lack required methods.
    ValueError
        If required components are missing during execution.
    RuntimeError
        If the processing pipeline encounters execution errors.

    Notes
    -----
    - Processing steps are executed in the order they are added to the pipeline
    - The context dictionary is passed through all processing steps and may be modified
    - Steps that require data access can set `needs_catalog = True` to receive the data accessor
    - Validation failures return an empty xarray.Dataset rather than raising exceptions
    - All components (validator, catalog, processors) must implement their respective interfaces

    See Also
    --------
    DataCatalog : Interface for data access components
    ParameterValidator : Interface for parameter validation components
    DataProcessor : Interface for data processing components

    """

    def __init__(self):
        """Initialize the Dataset class.

        Attributes
        ----------
        data_access : DataCatalog or UNSET
            The data catalog instance used for retrieving raw data from various sources.
        parameter_validator : ParameterValidator or UNSET
            The parameter validator instance used for validating query parameters.
        processing_pipeline : list of DataProcessor or UNSET
            A list of processing steps to be executed sequentially on the data.

        """
        self.data_access = UNSET
        self.parameter_validator = UNSET
        self.processing_pipeline = UNSET  # list of processing steps

    def execute(self, parameters: Dict[str, Any] = UNSET) -> xr.Dataset:
        """Execute the dataset processing pipeline.

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
        current_result = self.data_access.get_data(valid_query)

        # Check if we have a processing pipeline
        if self.processing_pipeline is UNSET or not self.processing_pipeline:
            # If no pipeline is defined, just return the raw data from data_access
            return current_result

        # Execute each step in the pipeline in sequence
        try:
            for step in self.processing_pipeline:
                # Some steps might need access to the data_access component
                # steps that need it should define and set `needs_catalog = True`
                # in their __init__ method
                if getattr(step, "needs_catalog", False):
                    step.set_data_accessor(self.data_access)

                # Execute the current step
                # context is updated in place by the step
                current_result = step.execute(current_result, context)
                if current_result is None:
                    warnings.warn(
                        f"\n\nProcessing step {step.name} returned None. "
                        "\nEnsure that the step is implemented correctly.",
                        UserWarning,
                        stacklevel=999,
                    )

            return current_result

        except Exception as e:
            # Consider implementing proper error handling/logging here
            # Get detailed traceback information
            tb_info = traceback.format_exc()
            # Log the traceback for debugging
            print(f"Exception traceback:\n{tb_info}")
            raise RuntimeError(f"Error in processing pipeline: {str(e)}") from e

    def with_param_validator(
        self, parameter_validator: ParameterValidator
    ) -> "Dataset":
        """Set a new parameter validator.

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
        if not parameter_validator:
            warnings.warn(
                "No parameter validator provided. This may lead to unvalidated queries.",
                stacklevel=999,
            )
        if not isinstance(parameter_validator, ParameterValidator):
            raise TypeError(
                "Parameter validator must be an instance of ParameterValidator."
            )
        self.parameter_validator = parameter_validator
        return self

    def with_catalog(self, catalog: DataCatalog) -> "Dataset":
        """Set a new data catalog.

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

    def with_processing_step(self, step: DataProcessor) -> "Dataset":
        """Add a new processing step to the pipeline.

        Parameters
        ----------
        step : DataProcessor
            Processing step to add to the pipeline. Must have 'execute' and 'update_context' methods.

        Returns
        -------
        Dataset
            The current instance of Dataset allowing method chaining.

        Raises
        ------
        TypeError
            If the step is not an instance of DataProcessor.
        AttributeError
            If the step does not have 'execute', 'update_context', or 'set_data_accessor' methods.
        TypeError
            If the step is not callable.

        """
        if not hasattr(step, "execute") or not callable(getattr(step, "execute")):
            raise TypeError("Processing step must have an 'execute' method.")

        if not hasattr(step, "update_context") or not callable(
            getattr(step, "update_context")
        ):
            raise AttributeError(
                "Processing step must have an 'update_context' method."
            )

        if not hasattr(step, "set_data_accessor") or not callable(
            getattr(step, "set_data_accessor")
        ):
            raise TypeError("Processing step must have a 'set_data_accessor' method.")

        if self.processing_pipeline is UNSET:
            self.processing_pipeline = []

        self.processing_pipeline.append(step)
        return self
