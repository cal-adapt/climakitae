"""
DatasetFactory Module.

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

import logging
from typing import Any, Dict, List, Optional, Type

from climakitae.core.constants import _NEW_ATTRS_KEY, PROC_KEY, UNSET
from climakitae.new_core.data_access.data_access import DataCatalog
from climakitae.new_core.dataset import Dataset
from climakitae.new_core.param_validation.abc_param_validation import (
    _CATALOG_VALIDATOR_REGISTRY,
    ParameterValidator,
)
from climakitae.new_core.param_validation.param_validation_tools import (
    _get_closest_options,
)
from climakitae.new_core.processors.abc_data_processor import _PROCESSOR_REGISTRY

# Module logger
logger = logging.getLogger(__name__)


class DatasetFactory:
    """Factory for creating Dataset objects with appropriate catalogs, validators, and processors.

    This factory translates UI queries from the ClimateData interface into fully
    configured Dataset objects with the correct combination of data catalogs for
    accessing climate data, parameter validators for query validation, and processing
    steps for data transformation.

    The factory uses registries to maintain extensible collections of components and
    automatically determines the appropriate combination based on query parameters.

    Attributes
    ----------
    _catalog : DataCatalog or None
        Reference to the DataCatalog singleton instance.
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
        """Initialize the DatasetFactory.

        Sets up the factory with access to the data catalog singleton,
        validator registry, and processor registry for creating fully
        configured Dataset objects.

        """
        logger.debug("Initializing DatasetFactory")
        self._catalog = DataCatalog()
        self._catalog_df = self._catalog.catalog_df
        self._validator_registry = _CATALOG_VALIDATOR_REGISTRY
        self._processing_step_registry = _PROCESSOR_REGISTRY
        logger.info(
            "DatasetFactory initialized with %d validators and %d processors",
            len(self._validator_registry),
            len(self._processing_step_registry),
        )

    def create_dataset(self, ui_query: Dict[str, Any]) -> Dataset:
        """Create a Dataset based on a UI query from ClimateData.

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
        logger.debug("Creating dataset from query: %s", ui_query)
        dataset = Dataset()

        # Create and configure parameter validator
        catalog_key = self._get_catalog_key_from_query(ui_query)
        logger.info("Determined catalog key: %s", catalog_key)

        # Store catalog_key in query for thread-safe access during execution
        # This avoids storing mutable state on the singleton DataCatalog
        ui_query["_catalog_key"] = catalog_key

        self._catalog = DataCatalog()
        # Resolve the key to validate it (but don't store it on the singleton)
        resolved_key = self._catalog.resolve_catalog_key(catalog_key)
        if resolved_key is None:
            raise ValueError(f"Invalid catalog key: {catalog_key}")
        ui_query["_catalog_key"] = resolved_key

        logger.debug("Creating validator for catalog: %s", resolved_key)
        dataset.with_param_validator(self.create_validator(resolved_key))

        # Configure the appropriate catalog based on query parameters
        logger.debug("Configuring dataset with data catalog")
        dataset.with_catalog(self._catalog)

        # Add processing steps based on query parameters
        if _NEW_ATTRS_KEY not in ui_query:
            ui_query[_NEW_ATTRS_KEY] = {}

        logger.debug("Determining processing steps for query")
        proc_steps = self._get_list_of_processing_steps(ui_query)
        logger.info("Adding %d processing steps to dataset", len(proc_steps))
        for proc in proc_steps:
            logger.debug(
                "Adding processing step: %s",
                proc[0] if isinstance(proc, tuple) else proc,
            )
            dataset.with_processing_step(proc)

        logger.info("Dataset created successfully")
        return dataset

    def _get_list_of_processing_steps(
        self, query: Dict[str, Any]
    ) -> List[tuple[str, Any]]:
        """Get a list of processing steps based on query parameters.

        This method determines the complete set of processing steps required
        for a query by examining explicit user requests, implicit requirements
        based on query parameters, and mandatory system processing steps.

        Default processors are obtained from the catalog's validator, which
        knows catalog-specific requirements (e.g., HDP uses station_id for
        concatenation, climate model data uses filter_unadjusted_models).

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
        priorities = []

        if query[PROC_KEY] is UNSET:
            # create empty processing step key
            query[PROC_KEY] = {}

        # Get default processors from validator
        catalog_key = query.get("_catalog_key")
        if catalog_key:
            try:
                validator = self.create_validator(catalog_key)
                default_processors = validator.get_default_processors(query)

                # Apply defaults for any processors not explicitly set by user
                for proc_name, default_value in default_processors.items():
                    if proc_name not in query[PROC_KEY]:
                        query[PROC_KEY][proc_name] = default_value
            except Exception as e:
                logger.warning(f"Could not get default processors: {e}")

        # Process all processors in query[PROC_KEY]
        for key, value in query[PROC_KEY].items():
            if key not in self._processing_step_registry:
                logger.warning(
                    "Processing step '%s' not found in registry. Skipping.",
                    key,
                )
                continue

            # Unpack processor info (class, priority)
            registry_entry = self._processing_step_registry[key]
            processor_class, priority = registry_entry[0], registry_entry[1]

            index = len(processing_steps)
            if not priorities:
                priorities.append(priority)
                index = 0
            elif priority not in priorities:
                # insert the new step in the correct order
                # lowest priority first
                for i, p in enumerate(priorities):
                    if priority < p:
                        index = i
                        break
                priorities.insert(index, priority)
            else:
                # if the priority already exists, we append after the last occurrence
                indices = [i for i, p in enumerate(priorities) if p == priority]
                index = indices[-1] + 1
            processing_steps.insert(index, processor_class(value))

            # modify query in place
            query[_NEW_ATTRS_KEY][key] = value

        return processing_steps

    def register_catalog(self, key: str, catalog_url: str):
        """Register a data catalog with the factory.

        Parameters
        ----------
        key : str
            Identifier for the catalog. Should correspond to data_type,
            installation, or other distinguishing characteristics.
        catalog_url : str
            URL or path to the catalog to register for the given key.

        Raises
        ------
        ValueError
            If key is empty or None.

        Examples
        --------
        >>> factory = DatasetFactory()
        >>> factory.register_catalog('wind_data', 's3://bucket/catalog.csv')

        See Also
        --------
        DataCatalog : Base catalog class

        """
        if not key:
            raise ValueError("Catalog key cannot be empty or None.")
        DataCatalog().set_catalog(key, catalog_url)

    def register_validator(self, key: str, validator_class: Type[ParameterValidator]):
        """Register a parameter validator with the factory.

        Parameters
        ----------
        key : str
            Identifier for the validator (approach, data_type combination)
        validator_class : Type[ParameterValidator]
            Validator class to register

        """
        self._validator_registry[key] = validator_class

    def register_processing_step(self, step_type: str, step_class):
        """Register a processing step with the factory.

        Parameters
        ----------
        step_type : str
            Identifier for the processing step
        step_class : class
            Processing step class to register

        """
        self._processing_step_registry[step_type] = step_class

    def create_validator(self, val_reg_key: str) -> Optional[ParameterValidator]:
        """Create a parameter validator based on data_type and approach.

        Parameters
        ----------
        val_reg_key : str
            Key for the validator (data_type_approach)

        Returns
        -------
        ParameterValidator or None
            An appropriate parameter validator, or None if not found.

        """
        if val_reg_key in self._validator_registry:
            return self._validator_registry[val_reg_key](self._catalog)

        # check for typo or close matches
        closest = _get_closest_options(val_reg_key, self._validator_registry.keys())

        match len(closest):
            case 0:
                logger.warning(
                    "No validator registered for '%s'. Available options: %s",
                    val_reg_key,
                    list(self._validator_registry.keys()),
                )
                return None
            case 1:
                logger.warning(
                    "Using closest match '%s' for validator '%s'.",
                    closest[0],
                    val_reg_key,
                )
                return self._validator_registry[closest[0]](self._catalog)
            case _:
                logger.warning(
                    "Multiple closest matches found for '%s': %s. "
                    "Please specify a more precise key.",
                    val_reg_key,
                    closest,
                )
                return None

    def _get_catalog_key_from_query(self, query: Dict[str, Any]) -> Optional[str]:
        """Get the appropriate catalog for the query.

        Parameters
        ----------
        query : Dict[str, Any]
            Query dictionary from ClimateData UI

        Returns
        -------
        str or None
            Key for the catalog to use (e.g., "cadcat", "renewable energy generation", "hdp"),
            or None if no matching catalog is found.

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
                logger.warning("No matching catalogs found initially.")
            case 1:
                return subset.iloc[0]["catalog"]
            case _:
                logger.warning(
                    "Multiple matching datasets found. Please refine your query."
                )

        return None

    def get_catalog_options(
        self, key: str, query: Optional[Dict[str, Any]] = None
    ) -> List[str]:
        """Get available options for a specific catalog.

        Parameters
        ----------
        key : str
            Key of the catalog to query.
        query : dict, optional
            A dictionary to filter the catalog options. The keys of the
            dictionary should correspond to columns in the catalog, and the
            values are the values to filter by.

        Returns
        -------
        List[str]
            List of available options for the specified catalog.

        """
        if key not in self._catalog_df.columns:
            raise ValueError(f"Catalog key '{key}' not found.")
        filtered_df = self._catalog_df.copy()
        if query is not None:
            # Filter the catalog DataFrame based on the query
            for k, v in query.items():
                if k in filtered_df.columns:
                    if isinstance(v, (list, tuple)):
                        if len(v) == 0:
                            # Empty list - no filtering needed for this key
                            continue
                        elif len(v) == 1:
                            # Single element - use exact or partial match
                            filtered_df = filtered_df[
                                filtered_df[k].str.contains(
                                    str(v[0]), case=False, na=False
                                )
                            ]
                        else:
                            # Multiple elements - match any of them (partial match)
                            pattern = "|".join([str(item) for item in v])
                            filtered_df = filtered_df[
                                filtered_df[k].str.contains(
                                    pattern, case=False, na=False
                                )
                            ]
                    else:
                        # Single value - do not use partial match
                        filtered_df = filtered_df[
                            filtered_df[k].str.lower() == str(v).lower()
                        ]
        return sorted(list(filtered_df[key].dropna().unique()))

    def get_validators(self) -> List[str]:
        """Get a list of available validators.

        Returns
        -------
        List[str]
            List of available validators.

        """
        return sorted(list(self._validator_registry.keys()))

    def get_valid_processors(self, catalog_key: str) -> List[str]:
        """Get a list of valid processors for a specific catalog.

        Parameters
        ----------
        catalog_key : str
            The catalog key to filter processors by (required).

        Returns
        -------
        List[str]
            List of processors valid for the specified catalog.

        """
        all_processors = sorted(list(self._processing_step_registry.keys()))

        # Get the validator for this catalog to determine invalid processors
        validator = self.create_validator(catalog_key)
        if validator and hasattr(validator, "invalid_processors"):
            invalid_processors = validator.invalid_processors
            return [p for p in all_processors if p not in invalid_processors]

        return all_processors

    def get_stations(self) -> List[str]:
        """Get a list of available station datasets.

        Returns
        -------
        List[str]
            List of available station datasets.

        """
        return DataCatalog()["stations"]["station"].unique().tolist()

    def get_boundaries(self, boundary_type: str) -> List[str]:
        """Get a list of available boundary datasets.

        Parameters
        ----------
        boundary_type : str
            The type of boundary datasets to retrieve. If the type is not found
            in the cache, returns all available boundary types.

        Returns
        -------
        List[str]
            List of available boundary datasets for the specified type, or
            all available boundary types if the specified type is not found.

        """
        if boundary_type not in DataCatalog().boundaries._lookup_cache:
            return list(DataCatalog().boundaries._lookup_cache.keys())
        else:
            return list(DataCatalog().boundaries._lookup_cache[boundary_type].keys())

    def reset(self):
        """Reset the factory state, clearing all registered catalogs, validators, and processors.

        This method is useful for reinitializing the factory without creating a new instance.

        """
        self._validator_registry = _CATALOG_VALIDATOR_REGISTRY
        self._processing_step_registry = _PROCESSOR_REGISTRY
        DataCatalog().reset()
