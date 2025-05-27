"""
DatasetFactory Module

This module provides a factory class for creating climate data processing components
and complete datasets with appropriate validation and processing pipelines. It serves
as the central orchestrator for constructing validators, processors, and data access
objects based on data type, analytical approach, and user requirements.

The factory pattern implemented here simplifies the instantiation of complex component
combinations while maintaining flexibility for different climate data scenarios
including gridded versus station-based observations, time-based versus warming-level
analysis approaches, and different data catalogs and processing requirements.

Key Features
------------
- Dynamic component registration and discovery
- Automatic processing pipeline construction
- Catalog-based data source management
- Extensible validator and processor registries

See Also
--------
climakitae.new_core.dataset.Dataset : Dataset container class
climakitae.new_core.data_access.DataCatalog : Data catalog management
climakitae.new_core.param_validation.abc_param_validator : Parameter validation framework
climakitae.new_core.processors.abc_data_processor : Data processing framework

Notes
-----
This module follows the factory design pattern to encapsulate the complex logic
of creating appropriate combinations of data access, validation, and processing
components based on user queries from the ClimateData UI.
"""

from __future__ import annotations

import warnings
from typing import Any, Dict, List, Optional, Type

import pandas as pd
import xarray as xr

from climakitae.core.constants import _NEW_ATTRS_KEY, UNSET
from climakitae.new_core.data_access import DataCatalog
from climakitae.new_core.dataset import Dataset
from climakitae.new_core.param_validation.abc_param_validation import (
    _VALIDATOR_REGISTRY,
    ParameterValidator,
)
from climakitae.new_core.processors.abc_data_processor import (
    _PROCESSOR_REGISTRY,
    DataProcessor,
)

PROC_KEY = "processes"


class DatasetFactory:
    """
    Factory for creating Dataset objects with appropriate catalogs, validators, and processors.

    This factory translates UI queries from the ClimateData interface into fully
    configured Dataset objects with the correct combination of data catalogs for
    accessing climate data, parameter validators for query validation, and processing
    steps for data transformation.

    The factory uses registries to maintain extensible collections of components and
    automatically determines the appropriate combination based on query parameters.

    Parameters
    ----------
    catalog_path : str, optional
        Path to the catalog configuration CSV file. Default is
        'climakitae/data/catalogs.csv'.

    Attributes
    ----------
    catalog_path : str
        Path to the catalog configuration CSV file.
    _catalog : dict
        Dictionary mapping catalog keys to DataCatalog instances.
    _catalog_df : pandas.DataFrame
        DataFrame containing catalog metadata loaded from CSV.
    _validator_registry : dict
        Registry mapping validator keys to ParameterValidator classes.
    _processing_step_registry : dict
        Registry mapping processing step names to DataProcessor classes.

    Methods
    -------
    register_catalog(key, catalog)
        Register a data catalog with the factory.
    register_validator(key, validator_class)
        Register a parameter validator with the factory.
    register_processing_step(step_type, step_class)
        Register a processing step with the factory.
    create_validator(val_reg_key)
        Create a parameter validator based on registry key.
    create_dataset(ui_query)
        Create a Dataset based on a UI query from ClimateData.
    get_catalog_options(key, query=None)
        Get available options for a specific catalog.
    get_validators()
        Get a list of available validators.
    get_processors()
        Get a list of available processors.

    Examples
    --------
    Creating a basic dataset:

    >>> factory = DatasetFactory()
    >>> query = {'data_type': 'gridded', 'variable': 'precipitation'}
    >>> dataset = factory.create_dataset(query)

    Registering custom components:

    >>> factory = DatasetFactory()
    >>> factory.register_validator('custom_type', CustomValidator)
    >>> factory.register_processing_step('custom_process', CustomProcessor)

    Notes
    -----
    The factory automatically handles the selection of appropriate processing
    steps based on the query parameters. Some processing steps are mandatory
    and will be added automatically even if not explicitly requested.

    See Also
    --------
    Dataset : The main dataset container class
    DataCatalog : Data access abstraction
    ParameterValidator : Base class for parameter validation
    DataProcessor : Base class for data processing steps
    """

    def __init__(self):
        """
        Initialize the DatasetFactory.

        Parameters
        ----------
        catalog_path : str, optional
            Path to the catalog configuration CSV file. If None, uses the
            default path 'climakitae/data/catalogs.csv'.

        Raises
        ------
        FileNotFoundError
            If the catalog file cannot be found at the specified path.
        RuntimeError
            If the catalog file cannot be loaded or parsed.
        """
        self._catalog = None
        self.catalog_path = (
            "climakitae/data/catalogs.csv"  # ! Move to paths or constants
        )
        self._catalog_df = pd.read_csv(self.catalog_path)
        self._validator_registry = _VALIDATOR_REGISTRY
        self._processing_step_registry = _PROCESSOR_REGISTRY

    def create_dataset(self, ui_query: Dict[str, Any]) -> Dataset:
        """
        Create a Dataset based on a UI query from ClimateData.

        This method orchestrates the creation of a complete Dataset by:
        1. Determining the appropriate catalog based on query parameters
        2. Creating and configuring the parameter validator
        3. Adding the necessary processing steps in the correct order

        Parameters
        ----------
        ui_query : dict
            Query dictionary from ClimateData UI containing at minimum:
            - 'data_type' : str, type of climate data
            - Additional keys depend on the specific data type and analysis

        Returns
        -------
        Dataset
            Properly configured Dataset instance ready for data retrieval
            and processing.

        Raises
        ------
        ValueError
            If required query parameters are missing, invalid, or if no
            appropriate catalog can be determined.
        RuntimeError
            If dataset creation fails due to internal errors.

        Notes
        -----
        The method automatically adds mandatory processing steps such as
        concatenation and attribute updates even if not specified in the query.

        Processing steps are applied in priority order, with preprocessing
        steps (like bias correction) applied before postprocessing steps.

        See Also
        --------
        Dataset : The returned dataset class
        create_validator : Method for creating parameter validators
        """
        dataset = Dataset()

        # Create and configure parameter validator
        catalog_key = self._get_catalog_key_from_query(ui_query)
        self._catalog = DataCatalog()
        self._catalog.set_catalog_key(catalog_key)
        dataset.with_param_validator(self.create_validator(catalog_key))

        # Configure the appropriate catalog based on query parameters
        dataset.with_catalog(self._catalog)
        # Add processing steps based on query parameters
        if _NEW_ATTRS_KEY not in ui_query:
            ui_query[_NEW_ATTRS_KEY] = {}
        for proc in self._get_list_of_processing_steps(ui_query):
            dataset.with_processing_step(proc)

        return dataset

    def _get_list_of_processing_steps(
        self, query: Dict[str, Any]
    ) -> List[tuple[str, Any]]:
        """
        Get a list of processing steps based on query parameters.

        This method determines the complete set of processing steps required
        for a query by examining explicit user requests, implicit requirements
        based on query parameters, and mandatory system processing steps.

        Parameters
        ----------
        query : dict
            Query dictionary from ClimateData UI. This dictionary may be
            modified in place to add processing metadata.

        Returns
        -------
        list of tuple
            List of tuples containing (processing_step_key, parameters)
            ordered by processing priority.

        Warnings
        --------
        UserWarning
            If a requested processing step is not found in the registry.

        Notes
        -----
        Processing step priority determines execution order:
        - Priority 0-10: Preprocessing (bias correction, warming level)
        - Priority 11-20: Core processing (variable calculations)
        - Priority 21-30: Postprocessing (concatenation, attribute updates)

        The method modifies the input query dictionary by adding a
        '_new_attributes' key containing metadata about applied processing steps.

        See Also
        --------
        _PROCESSOR_REGISTRY : Global registry of available processors
        """
        processing_steps = []

        if query[PROC_KEY] is UNSET:
            # create empty processing step key
            query[PROC_KEY] = {}

        for key, value in query[PROC_KEY].items():
            if key not in self._processing_step_registry:
                warnings.warn(
                    f"Processing step '{key}' not found in registry. Skipping."
                )
                continue

            processor_class, _ = self._processing_step_registry[
                key
            ]  # get the class and priority
            processing_steps.append(processor_class(value))

            # modify query in place
            query[_NEW_ATTRS_KEY][key] = value

        # Mandatory processing steps
        if "filter_unbiased_models" not in query[PROC_KEY]:
            # remove unbiased models
            processing_steps.append(
                self._processing_step_registry["filter_unbiased_models"][0]()
            )
            query[_NEW_ATTRS_KEY]["filter_unbiased_models"] = "yes"

        if "concat" not in query[PROC_KEY]:
            processing_steps.append(self._processing_step_registry["concat"][0]())
            query[_NEW_ATTRS_KEY]["concat"] = "sim"

        processing_steps.append(
            self._processing_step_registry["update_attributes"][0]()
        )
        return processing_steps

    def register_catalog(self, key: str, catalog: DataCatalog):
        """
        Register a data catalog with the factory.

        Parameters
        ----------
        key : str
            Identifier for the catalog. Should correspond to data_type,
            installation, or other distinguishing characteristics.
        catalog : DataCatalog
            Catalog implementation to register for the given key.

        Raises
        ------
        TypeError
            If catalog is not an instance of DataCatalog.
        ValueError
            If key is empty or None.

        Examples
        --------
        >>> factory = DatasetFactory()
        >>> custom_catalog = DataCatalog()
        >>> factory.register_catalog('wind_data', custom_catalog)

        See Also
        --------
        DataCatalog : Base catalog class
        """
        self._catalog[key] = catalog

    def register_validator(self, key: str, validator_class: Type[ParameterValidator]):
        """
        Register a parameter validator with the factory.

        Parameters
        ----------
        key : str
            Identifier for the validator (approach, data_type combination)
        validator_class : Type[ParameterValidator]
            Validator class to register
        """
        self._validator_registry[key] = validator_class

    def register_processing_step(self, step_type: str, step_class):
        """
        Register a processing step with the factory.

        Parameters
        ----------
        step_type : str
            Identifier for the processing step
        step_class : class
            Processing step class to register
        """
        self._processing_step_registry[step_type] = step_class

    def create_validator(self, val_reg_key: str) -> ParameterValidator:
        """
        Create a parameter validator based on data_type and approach.

        Parameters
        ----------
        val_reg_key : str
            Key for the validator (data_type_approach)

        Returns
        -------
        ParameterValidator
            An appropriate parameter validator

        Raises
        ------
        ValueError
            If no validator is registered for the given combination
        """
        if val_reg_key in self._validator_registry:
            return self._validator_registry[val_reg_key](self._catalog)

        raise ValueError(
            f"\n\nNo validator registered for catalog = '{val_reg_key}'"
            f"\nPlease check the input query or register a new validator.\n\n"
        )

    def _get_catalog_key_from_query(self, query: Dict[str, Any]) -> str:
        """
        Get the appropriate catalog for the query.

        Parameters
        ----------
        query : Dict[str, Any]
            Query dictionary from ClimateData UI

        Returns
        -------
        str
            Key for the catalog to use (e.g., "data", "renewables")

        """
        # search catalog for matching datasets
        catalog_key = None
        if (catalog_key := query["catalog"]) is not UNSET:
            return catalog_key

        # otherwise, do a quick lookup in the dataframe
        # to find the catalog key
        valid_keys = [
            query[key]
            for key in self._catalog_df.columns
            if key in query and query[key] != UNSET
        ]
        subset = self._catalog_df[
            self._catalog_df.isin(valid_keys).any(axis=1)
        ]  # filter rows with matching keys
        match len(subset):
            case 0:
                warnings.warn("No matching catalogs found initially.")
            case 1:
                return subset.iloc[0]["catalog"]
            case _:
                warnings.warn(
                    "Multiple matching datasets found. Please refine your query."
                )

        return None

    def get_catalog_options(
        self, key: str, query: dict[str, Any] | object = UNSET
    ) -> List[str]:
        """
        Get available options for a specific catalog.

        Parameters
        ----------
        key : str
            Key of the catalog to query.

        Returns
        -------
        List[str]
            List of available options for the specified catalog.
        """
        if key not in self._catalog_df.columns:
            raise ValueError(f"Catalog key '{key}' not found.")
        filtered_df = self._catalog_df.copy()
        if query is not UNSET:
            # Filter the catalog DataFrame based on the query
            for k, v in query.items():
                if k in filtered_df.columns:
                    filtered_df = filtered_df[filtered_df[k] == v]
        return sorted(list(filtered_df[key].dropna().unique()))

    def get_validators(self) -> List[str]:
        """
        Get a list of available validators.

        Returns
        -------
        List[str]
            List of available validators.
        """
        return sorted(list(self._validator_registry.keys()))

    def get_processors(self) -> List[str]:
        """
        Get a list of available processors.

        Returns
        -------
        List[str]
            List of available processors.
        """
        return sorted(list(self._processing_step_registry.keys()))
