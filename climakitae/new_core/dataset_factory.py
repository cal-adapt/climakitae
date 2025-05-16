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
    Factory for creating Dataset objects based on queries from the ClimateData UI.

    This factory translates UI queries into configured Dataset objects with
    appropriate catalog settings, validators, and processing steps.
    """

    def __init__(self):
        """Initialize the factory with registries for catalogs, validators and processing steps."""
        self._catalog = None
        self.catalog_path = (
            "climakitae/data/catalogs.csv"  # ! Move to paths or constants
        )
        self._catalog_df = pd.read_csv(self.catalog_path)
        self._validator_registry = _VALIDATOR_REGISTRY
        self._processing_step_registry = _PROCESSOR_REGISTRY

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

        raise ValueError(f"No validator registered for {val_reg_key}")

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

        This method checks the query for explicitly defined processing steps requested
        by the user, checks the query for keys that implicitly require processing steps,
        like station data, wind data, etc., adds processing steps that the user has no
        control over like adding new attributes, and finally returns a list of tuples
        containing the processing step key and value in order of priority.

        Parameters
        ----------
        query : Dict[str, Any]
            Query dictionary from ClimateData UI

        Returns
        -------
        List[tuple[str, Any]]
            List of tuples containing processing step key and value

        Notes
        -----
        - The order of processing steps CAN be important. For example, adding new
        attributes always happens last, and pre-processing steps like global warming
        levels and bias corrections always happen first. This is implemented via the
        priority key in the processing step registry.
        """
        processing_steps = []

        if query[PROC_KEY] is UNSET:
            return processing_steps

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

        processing_steps.append(
            self._processing_step_registry["update_attributes"][0]()
        )
        return processing_steps

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

    def get_catalog_options(self, key: str) -> List[str]:
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
        return sorted(list(self._catalog_df[key].dropna().unique()))

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
