"""Parameter validation module for climakitae.

This module provides a comprehensive framework for validating query parameters
used throughout the climakitae package. It includes:

- Abstract base class for parameter validation (`ParameterValidator`)
- Registry system for catalog and processor validators
- Validation logic for dataset queries and processing parameters
- Helper functions for finding closest matching options when validation fails

The validation system operates on two levels:
1. **Catalog validation**: Ensures query parameters match available datasets
2. **Processor validation**: Validates processing parameters for data transformations

Classes
-------
ParameterValidator
    Abstract base class defining the parameter validation interface.
    Subclasses must implement `is_valid_query()` method.

Functions
---------
register_catalog_validator
    Decorator for registering catalog validator classes.
register_processor_validator
    Decorator for registering processor validator classes.

Module Variables
----------------
_CATALOG_VALIDATOR_REGISTRY : dict
    Registry mapping validator names to catalog validator classes.
_PROCESSOR_VALIDATOR_REGISTRY : dict
    Registry mapping validator names to processor validator classes.

Examples
--------
>>> @register_catalog_validator("my_catalog")
... class MyCatalogValidator(ParameterValidator):
...     def is_valid_query(self, query):
...         # Implementation here
...         pass

"""

import warnings
from abc import ABC, abstractmethod
from typing import Any, Dict

from climakitae.core.constants import PROC_KEY, UNSET
from climakitae.new_core.data_access.data_access import DataCatalog
from climakitae.new_core.param_validation.param_validation_tools import (
    _get_closest_options,
    _validate_experimental_id_param,
)

_CATALOG_VALIDATOR_REGISTRY = {}
_PROCESSOR_VALIDATOR_REGISTRY = {}


def register_catalog_validator(name: str):
    """Decorator to register a catalog validator class in the global registry.

    This decorator allows validator classes to be registered for use with
    specific catalog types. Registered validators can be retrieved and
    instantiated by name from the global registry.

    Parameters
    ----------
    name : str
        Unique name to register the validator under. This name will be used
        to look up the validator class in the registry.

    Returns
    -------
    function
        Decorator function that registers the class and returns it unchanged.

    Examples
    --------
    >>> @register_catalog_validator("my_catalog")
    ... class MyCatalogValidator(ParameterValidator):
    ...     def is_valid_query(self, query):
    ...         return self._is_valid_query(query)

    >>> # Later retrieval:
    >>> validator_class = _CATALOG_VALIDATOR_REGISTRY["my_catalog"]
    >>> validator = validator_class()

    Notes
    -----
    The registered class is stored in the module-level
    `_CATALOG_VALIDATOR_REGISTRY` dictionary.

    """

    def decorator(cls):
        _CATALOG_VALIDATOR_REGISTRY[name] = cls
        return cls

    return decorator


def register_processor_validator(name: str):
    """Decorator to register a processor validator function in the global registry.

    This decorator allows processor validation functions to be registered for
    use with specific processing parameters. Registered validators can be
    retrieved and called by name from the global registry.

    Parameters
    ----------
    name : str
        Unique name to register the processor validator under. This should
        match the processor parameter name that the validator handles.

    Returns
    -------
    function
        Decorator function that registers the validator function and returns it unchanged.

    Examples
    --------
    >>> @register_processor_validator("spatial_subset")
    ... def validate_spatial_subset(value, query=None):
    ...     # Validation logic for spatial_subset processor
    ...     return isinstance(value, dict) and 'bounds' in value

    >>> # Later retrieval and use:
    >>> validator_func = _PROCESSOR_VALIDATOR_REGISTRY["spatial_subset"]
    >>> is_valid = validator_func(subset_params, query=user_query)

    Notes
    -----
    - The registered function is stored in the module-level
      `_PROCESSOR_VALIDATOR_REGISTRY` dictionary
    - Processor validators should accept `value` and optional `query` parameters
    - Validators may modify the query in-place for parameter normalization

    """

    def decorator(cls):
        _PROCESSOR_VALIDATOR_REGISTRY[name] = cls
        return cls

    return decorator


class ParameterValidator(ABC):
    """Abstract base class for parameter validation in climakitae.

    This class provides a framework for validating user queries containing
    dataset selection parameters and processing parameters. It handles:

    - Catalog parameter validation (dataset selection)
    - Processor parameter validation (data transformations)
    - Error handling and user-friendly suggestions
    - Parameter conversion and normalization

    The validation process includes:
    1. Converting user input to catalog keys
    2. Searching for matching datasets in the catalog
    3. Providing suggestions for invalid parameters
    4. Validating processing parameters

    Attributes
    ----------
    catalog_path : str
        Path to the catalog CSV file.
    catalog : object
        Data catalog instance for dataset searching.
    all_catalog_keys : dict
        Dictionary of catalog keys populated from user query.
    catalog_df : pandas.DataFrame
        DataFrame containing catalog information.

    Methods
    -------
    is_valid_query(query)
        Abstract method to validate query parameters. Must be implemented by subclasses.
    populate_catalog_keys(query)
        Populate catalog keys from user query.
    load_catalog_df()
        Load the catalog DataFrame.

    Notes
    -----
    Subclasses must implement the `is_valid_query` method to define
    specific validation logic for their use case.

    Examples
    --------
    >>> class MyValidator(ParameterValidator):
    ...     def is_valid_query(self, query):
    ...         # Custom validation logic
    ...         return self._is_valid_query(query)

    """

    def __init__(self):
        """Initialize the ParameterValidator.

        Sets up the validator with default catalog path and initializes
        catalog-related attributes. Loads the catalog DataFrame upon instantiation.

        Attributes initialized:
        - catalog_path: Path to the catalog CSV file
        - catalog: Set to UNSET initially, populated by subclasses
        - all_catalog_keys: Set to UNSET initially, populated during validation
        - catalog_df: Loaded from DataCatalog

        """
        self.catalog_path = "climakitae/data/catalogs.csv"
        self.catalog = UNSET
        self.all_catalog_keys = UNSET
        self.load_catalog_df()

    @abstractmethod
    def is_valid_query(self, query: Dict[str, Any]) -> Dict[str, Any] | None:
        """Validate the query parameters (abstract method).

        This method must be implemented by subclasses to define specific
        validation logic for their use case. It should validate both
        catalog parameters (for dataset selection) and processing parameters.

        Parameters
        ----------
        query : Dict[str, Any]
            Query parameters to validate. Expected to contain:
            - Dataset selection parameters (e.g., variable, experiment_id, etc.)
            - Processing parameters under the 'processes' key
            - Any other relevant validation parameters

        Returns
        -------
        Dict[str, Any] | None
            Validated and processed query parameters if valid, None if invalid.
            When returning a dictionary, it should contain the cleaned and
            validated parameters ready for dataset retrieval.

        Notes
        -----
        Implementations typically call `_is_valid_query()` to leverage the
        common validation logic provided by the base class.

        Examples
        --------
        >>> def is_valid_query(self, query):
        ...     # Custom pre-processing
        ...     processed_query = self.preprocess_query(query)
        ...     # Use base class validation
        ...     return self._is_valid_query(processed_query)

        """

    def _is_valid_query(self, query: Dict[str, Any]) -> Dict[str, Any] | None:
        """Internal method to validate query parameters and provide user feedback.

        This method performs the core validation logic:
        1. Converts user query to catalog keys
        2. Searches for matching datasets in the catalog
        3. Provides detailed error messages and suggestions for invalid parameters
        4. Validates processing parameters

        Parameters
        ----------
        query : Dict[str, Any]
            Query parameters to validate, including dataset selection
            and processing parameters.

        Returns
        -------
        Dict[str, Any] | None
            Dictionary of validated catalog keys if successful, None if validation
            fails. When None is returned, appropriate warning messages are printed
            to guide the user.

        Raises
        ------
        ValueError
            When catalog search fails due to invalid query structure.

        Notes
        -----
        This method provides detailed user feedback including:
        - Number of matching datasets found
        - Suggestions for typos or close matches
        - Identification of conflicting parameters
        - Guidance on available options

        The method handles special cases like:
        - experiment_id parameters (which can be lists)
        - Parameter conflicts between dataset attributes
        - Missing or invalid catalog keys

        """
        # convert user input to keys
        self.populate_catalog_keys(query)
        # check if the catalog keys can be found

        try:
            subset = self.catalog.search(**self.all_catalog_keys)
        except ValueError as e:
            # no datasets found or invalid query
            warnings.warn(
                f"Query did not match any datasets: {e}\n\nSearching for close matches...",
                UserWarning,
                stacklevel=999,
            )

        if len(subset) != 0:
            print(f"Found {len(subset)} datasets matching your query.")
            print("Checking processes ...")
            return self.all_catalog_keys if self._has_valid_processes(query) else None

        # dataset not found
        # find closest match to each provided key
        print("Checking for valid options...")
        df = self.catalog.df.copy()
        last_key = None
        for key, value in self.all_catalog_keys.items():
            # don't check unset values
            if value is UNSET:
                continue

            # check if the key is in the catalog
            if key not in df.columns:
                warnings.warn(
                    f"Key {key} not found in catalog. Did you specify the correct catalog?",
                    stacklevel=999,
                )
                continue  # skip to the next key

            if key == "experiment_id":
                # special case for experiment_id, since it can be a list of values
                if not _validate_experimental_id_param(
                    value, df[key].unique().tolist()
                ):
                    warnings.warn(
                        f"Experiment ID {value} is not valid. "
                        "Please check the available options using `show_experiment_id_options()`.",
                        stacklevel=999,
                    )
                else:
                    self.all_catalog_keys[key] = value
                continue  # skip to the next key

            # subset the dataframe to the key
            remaining_key_values = df[key].unique()
            df = df[df[key] == value]
            if df.empty:
                # this means no datasets were found for this key.
                # this can happen for two reasons:
                # 1. the user made a typo in the key or value
                # 2. the requested dataset does not exist in the catalog

                # start by checking if the value is in the catalog
                if value not in self.catalog.df[key].unique():
                    # the value is not in the catalog, check for closest options
                    print(f"Could not find any datasets with {key} = {value}.")
                    closest_options = _get_closest_options(
                        value, self.catalog.df[key].unique()
                    )
                    if closest_options is not None:
                        # probably a typo in the value
                        warnings.warn(
                            f"\n\nDid you mean one of these options for {key}: {closest_options}?",
                            stacklevel=999,
                        )
                    else:
                        # no close matches found
                        warnings.warn(
                            f"\n\nNo close matches found for {key} = {value}. "
                            "\nBased on your query, the available options for this key are: "
                            f"{remaining_key_values}.",
                            stacklevel=999,
                        )
                else:
                    # the value is in the catalog, but no datasets were found
                    warnings.warn(
                        f"\n\nNo datasets found for {key} = {value}. "
                        f"\n\nMost likely, this is because the dataset you requested "
                        f"\n    does not exist in the catalog. "
                        f"\nThis most often happens when searching for conflicting time"
                        f"\n    or spatial resolutions"
                        f"\n\nIn this case, it appears that there is a conflict between {key} and {last_key}. "
                        f"\nYour options for {key} are: {remaining_key_values}. "
                        f"\nThis is constrained by the earlier key {last_key} = {self.all_catalog_keys[last_key]}. "
                        f"\nPlease check your query and try again. "
                        f"\n\nTo explore available options, "
                        f"\n please use the `show_*_options()` methods.",
                        stacklevel=999,
                    )
                break
            # else:
            # print(f"Found {len(df)} datasets with {key} = {value}.")
            last_key = key
            # check if the value is in the catalog
        if not df.empty:
            print(f"Found up to {len(df)} datasets matching your query.")
            print("Checking processes ...")
            return self.all_catalog_keys if self._has_valid_processes(query) else None
        return None

    def _has_valid_processes(self, query: Dict[str, Any]) -> bool:
        """Validate processor parameters using registered processor validators.

        This method loops through processing parameters in the query and validates
        each one using registered processor validators. If a processor is not
        registered, a warning is issued but validation continues.

        Parameters
        ----------
        query : Dict[str, Any]
            Query containing processing parameters under the PROC_KEY.
            The query may be modified in-place by processor validators.

        Returns
        -------
        bool
            True if all processor parameters are valid or no processors are specified,
            False if any processor validation fails.

        Notes
        -----
        - Processor validators are called with both the parameter value and the full query
        - Validators may modify the query in-place for parameter normalization
        - Unregistered processors generate warnings but don't fail validation
        - Missing processor registrations indicate incomplete validation coverage

        See Also
        --------
        register_processor_validator : Decorator for registering processor validators

        """

        # loop through keys in query['processes']
        # and check if they are in the processor validator registry
        # if they are, call the processor validator
        # otherwise warn the user that the processor input has not been validated
        for key, value in query.get(PROC_KEY, {}).items():
            if key in _PROCESSOR_VALIDATOR_REGISTRY:
                valid_value_for_processor = _PROCESSOR_VALIDATOR_REGISTRY[key](
                    value, query=query
                )  #! this call is allowed to modify the query in place
                if not valid_value_for_processor:
                    warnings.warn(
                        f"\n\nProcessor {key} with value {value} is not valid. "
                        "\nPlease check the processor documentation for valid options.",
                        stacklevel=999,
                    )
                    return False
            else:
                warnings.warn(
                    f"\n\nProcessor {key} is not registered. "
                    "\nThis processor input has not been validated.",
                    stacklevel=999,
                )

        return True

    def populate_catalog_keys(self, query: Dict[str, Any]) -> None:
        """Populate catalog keys from user query, filtering out unset values.

        This method extracts relevant catalog parameters from the user query
        and stores them in `self.all_catalog_keys`. Only parameters that are
        actually set (not UNSET) are retained.

        Parameters
        ----------
        query : Dict[str, Any]
            User query containing potential catalog parameters.

        Returns
        -------
        None
            Updates `self.all_catalog_keys` in-place.

        Notes
        -----
        This method assumes `self.all_catalog_keys` already contains the expected
        catalog parameter names (typically initialized by subclasses). The method:
        1. Maps query values to catalog keys
        2. Removes any UNSET values
        3. Stores the result in `self.all_catalog_keys`

        Side Effects
        ------------
        Modifies `self.all_catalog_keys` attribute.

        """
        # populate catalog keys with the values from the query
        for key in self.all_catalog_keys.keys():
            self.all_catalog_keys[key] = query.get(key, UNSET)

        # remove any unset values
        self.all_catalog_keys = {
            k: v for k, v in self.all_catalog_keys.items() if v is not UNSET
        }

    def load_catalog_df(self):
        """Load the data catalog DataFrame and assign to instance attribute.

        Creates a DataCatalog instance and extracts its catalog DataFrame
        for use in parameter validation. The DataFrame contains metadata
        about available datasets.

        Returns
        -------
        None
            Sets `self.catalog_df` attribute.

        Notes
        -----
        This method is called during initialization and provides access to
        the catalog data needed for parameter validation and suggestion generation.

        Side Effects
        ------------
        Sets `self.catalog_df` attribute with the loaded catalog DataFrame.

        """
        self.catalog_df = DataCatalog().catalog_df
