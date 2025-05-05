"""
DatasetFactory Module

This module provides a factory class for creating climate data processing components
and complete datasets. It serves as a central point for constructing validators,
processors, and data access objects appropriate for different data types and
analytical approaches.

The factory pattern implemented here simplifies the instantiation of the correct
combination of components based on whether the data is gridded climate data or
station-based observations, and whether the analysis follows a time-based or
warming-level approach.

Classes:
    DatasetFactory: Factory for creating datasets and associated components.

Dependencies:
    - climakitae.core.constants
    - climakitae.core.data_interface
    - climakitae.new_core.data_access
    - climakitae.new_core.data_processor
    - climakitae.new_core.dataset
    - climakitae.new_core.param_validation
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Type

import xarray as xr

from climakitae.core.constants import UNSET
from climakitae.new_core.data_access import DataCatalog
from climakitae.new_core.data_processor import _PROCESSOR_REGISTRY, DataProcessor
from climakitae.new_core.dataset import Dataset
from climakitae.new_core.param_validation import _VALIDATOR_REGISTRY, ParameterValidator


class DatasetFactory:
    """
    Factory for creating Dataset objects based on queries from the ClimateData UI.

    This factory translates UI queries into configured Dataset objects with
    appropriate catalog settings, validators, and processing steps.
    """

    def __init__(self):
        """Initialize the factory with registries for catalogs, validators and processing steps."""
        self._catalog = DataCatalog()
        self._validator_registry = {}
        self._processing_step_registry = {}

        # Register default components
        self._register_defaults()

    def _register_defaults(self):
        """
        Register default components for the factory.
        This includes default validators and processing steps.

        Note
        ----
        The default catalogs are already registered in the DataCatalog class.
        The specific catalog to use will be determined by the query
        and registered in the _catalog_registry.
        """

        # Register default validators
        for key, validator_class in _VALIDATOR_REGISTRY.items():
            self.register_validator(key, validator_class)

        # Register default processors
        for key, processor_class in _PROCESSOR_REGISTRY.items():
            self.register_processing_step(key, processor_class)

    def register_catalog(self, key: str, catalog: DataCatalog):
        """
        Register a data catalog with the factory.

        Parameters
        ----------
        key : str
            Identifier for the catalog (data_type, installation, etc.)
        catalog : DataCatalog
            Catalog implementation to register
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

    def create_validator(self, data_type: str, approach: str) -> ParameterValidator:
        """
        Create a parameter validator based on data_type and approach.

        Parameters
        ----------
        data_type : str
            Type of data (e.g., "Gridded", "Stations")
        approach : str
            Approach to use (e.g., "Time", "Warming Level")

        Returns
        -------
        ParameterValidator
            An appropriate parameter validator

        Raises
        ------
        ValueError
            If no validator is registered for the given combination
        """
        key = f"{data_type}_{approach}"
        if key in self._validator_registry:
            return self._validator_registry[key]()

        raise ValueError(
            f"No validator registered for {data_type} data with {approach} approach"
        )

    def create_dataset(self, ui_query: Dict[str, Any]) -> Dataset:
        """
        Create a Dataset based on a UI query from ClimateData.

        Parameters
        ----------
        ui_query : Dict[str, Any]
            Query dictionary from ClimateData UI

        Returns
        -------
        Dataset
            Properly configured Dataset instance

        Raises
        ------
        ValueError
            If required query parameters are missing or invalid
        """
        dataset = Dataset()

        # Configure the appropriate catalog based on query parameters
        dataset.with_catalog(self._get_catalog_for_query(ui_query))

        # Create and configure parameter validator
        validator = self._create_validator_for_query(ui_query)
        dataset.with_param_validator(validator)

        # Add processing steps based on query parameters
        self._add_processing_steps(dataset, ui_query)

        return dataset

    def _get_catalog_for_query(self, query: Dict[str, Any]) -> DataCatalog:
        """
        Get the appropriate catalog for the query.

        Parameters
        ----------
        query : Dict[str, Any]
            Query dictionary from ClimateData UI

        Returns
        -------
        DataCatalog
            Selected catalog for the query

        Raises
        ------
        ValueError
            If no catalog matches the query parameters
        """
        # Select catalog based on data_type, installation, etc.
        if any(
            value is not UNSET
            for key, value in query.items()
            if key
            in [
                "installation",
                "activity_id",
                "institution_id",
                "source_id",
                "experiment_id",
            ]
        ):
            catalog_key = "renewables"
        else:
            catalog_key = "data"

        return self._catalog.set_catalog_key(catalog_key)

    def _create_validator_for_query(self, query: Dict[str, Any]) -> ParameterValidator:
        """
        Create parameter validator based on query parameters.

        Parameters
        ----------
        query : Dict[str, Any]
            Query dictionary from ClimateData UI

        Returns
        -------
        ParameterValidator
            Configured parameter validator
        """
        if any(
            value is not UNSET
            for key, value in query.items()
            if key
            in [
                "installation",
                "activity_id",
                "institution_id",
                "source_id",
                "experiment_id",
            ]
        ):
            validator_key = "renewables"
        else:
            validator_key = "default"

        if validator_key in self._validator_registry:
            validator_class = self._validator_registry[validator_key]
            validator = validator_class(self._catalog)
            # Configure validator with additional settings if needed
            return validator

        # Fallback to a default validator
        return None

    def _add_processing_steps(self, dataset: Dataset, query: Dict[str, Any]):
        # TODO: use query and single/multiple dispatch to create processing steps
        """
        Add processing steps to dataset based on query parameters.

        Parameters
        ----------
        dataset : Dataset
            Dataset to configure with processing steps
        query : Dict[str, Any]
            Query dictionary from ClimateData UI
        """
        # Add spatial subsetting if area_subset is specified
        # if (
        #     query.get("area_subset") == "region"
        #     and query.get("latitude", (UNSET, UNSET))[0] is not UNSET
        # ):
        #     spatial_step = self._create_spatial_subset_step(query)
        #     dataset.with_processing_step(spatial_step)

        # # Add time slicing if time_slice is specified
        # if query.get("time_slice", (UNSET, UNSET))[0] is not UNSET:
        #     time_step = self._create_time_subset_step(query)
        #     dataset.with_processing_step(time_step)

        # # Add variable selection
        # if query.get("variable") is not UNSET:
        #     variable_step = self._create_variable_selection_step(query)
        #     dataset.with_processing_step(variable_step)

        # # Add unit conversion if units are specified
        # if query.get("units") is not UNSET:
        #     unit_step = self._create_unit_conversion_step(query)
        #     dataset.with_processing_step(unit_step)

        # # Add spatial averaging if requested
        # if query.get("area_average") == "yes":
        #     avg_step = self._create_spatial_average_step()
        #     dataset.with_processing_step(avg_step)
