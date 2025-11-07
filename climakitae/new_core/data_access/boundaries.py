"""Lazy-loading boundaries module for ClimakitAE.

This module provides efficient access to geospatial boundary data for climate
data subsetting and analysis. The Boundaries class implements lazy loading
to minimize memory usage and improve startup performance by only loading
datasets when they are first accessed.

The module supports various types of geographical boundaries including:
- US western states
- California counties
- California watersheds (HUC8 level)
- California electric utilities (IOUs and POUs)
- California electricity demand forecast zones
- California electric balancing authority areas

All boundary data is sourced from S3-stored parquet files accessed through
intake catalogs, providing fast and efficient data retrieval.

Classes
-------
Boundaries
    Lazy-loading class for managing geospatial polygon data from S3 stored
    parquet catalogs. Provides cached lookup dictionaries and memory-efficient
    access to boundary datasets for geographic subsetting operations.

Examples
--------
>>> import intake
>>> catalog = intake.open_catalog('boundaries.yaml')
>>> boundaries = Boundaries(catalog)
>>>
>>> # Get all boundary options for UI population
>>> boundary_options = boundaries.boundary_dict()
>>>
>>> # Access specific boundary data (loaded lazily)
>>> ca_counties = boundaries._ca_counties
>>>
>>> # Preload all data for performance-critical scenarios
>>> boundaries.preload_all()

"""

import warnings
from typing import Dict, Optional, Union

import intake
import pandas as pd

from climakitae.core.constants import (
    CALISO_AREA_THRESHOLD,
    PRIORITY_UTILITIES,
    WESTERN_STATES_LIST,
)


class Boundaries:
    """Lazy-loading geospatial polygon data manager for ClimakitAE.

    This class provides efficient access to various boundary datasets stored
    in S3 parquet catalogs. Data is loaded only when first accessed, improving
    memory usage and initialization performance. All lookup dictionaries are
    cached to avoid recomputation.

    The class supports geographic subsetting for climate data analysis by providing
    access to various administrative and utility boundaries in California and the
    western United States. All data access is optimized for memory efficiency
    through lazy loading and intelligent caching.

    Parameters
    ----------
    boundary_catalog : intake.catalog.Catalog
        Intake catalog instance for accessing boundary parquet files from S3

    Attributes
    ----------
    _cat : intake.catalog.Catalog
        Reference to the boundary catalog instance used for data access

    Properties
    ----------
    _us_states : pd.DataFrame
        US western states with names, abbreviations, and geometries (lazy-loaded)
    _ca_counties : pd.DataFrame
        California counties with names and geometries, sorted alphabetically (lazy-loaded)
    _ca_watersheds : pd.DataFrame
        California HUC8 watersheds with names and geometries, sorted alphabetically (lazy-loaded)
    _ca_utilities : pd.DataFrame
        California electric utilities (IOUs and POUs) with names and geometries (lazy-loaded)
    _ca_forecast_zones : pd.DataFrame
        California electricity demand forecast zones with processed names (lazy-loaded)
    _ca_electric_balancing_areas : pd.DataFrame
        Electric balancing authority areas with filtered geometries (lazy-loaded)

    Methods
    -------
    boundary_dict() -> Dict[str, Dict[str, int]]
        Return dictionary of all boundary lookup dictionaries for UI population
    preload_all() -> None
        Preload all boundary data for performance-critical scenarios
    clear_cache() -> None
        Clear all cached data and lookup dictionaries to free memory
    validate_catalog() -> None
        Validate that required catalog entries exist and are accessible
    get_memory_usage() -> Dict[str, Union[int, str]]
        Get detailed memory usage information for loaded boundary datasets
    load() -> None
        Deprecated method for backward compatibility - use preload_all() instead

    Examples
    --------
    Basic usage with lazy loading:

    >>> import intake
    >>> catalog = intake.open_catalog('boundaries.yaml')
    >>> boundaries = Boundaries(catalog)
    >>>
    >>> # Data loads automatically when accessed
    >>> counties = boundaries._ca_counties
    >>> watersheds = boundaries._ca_watersheds

    Getting boundary options for UI components:

    >>> boundary_options = boundaries.boundary_dict()
    >>> state_options = boundary_options['states']
    >>> county_options = boundary_options['CA counties']

    Performance optimization:

    >>> # Preload all data if you know you'll need it
    >>> boundaries.preload_all()
    >>>
    >>> # Check memory usage
    >>> usage = boundaries.get_memory_usage()
    >>> print(f"Total memory: {usage['total_human']}")

    Memory management:

    >>> # Clear cache to free memory
    >>> boundaries.clear_cache()
    >>>
    >>> # Data will be reloaded on next access
    >>> counties = boundaries._ca_counties

    Notes
    -----
    - All boundary data is cached after first access for performance
    - The class automatically validates catalog structure on initialization
    - Processing includes sorting, filtering, and name standardization
    - Memory usage can be monitored and managed through provided methods
    - Western states are ordered according to WESTERN_STATES_LIST constant
    - Utilities are ordered with priority utilities first, then alphabetically

    """

    def __init__(self, boundary_catalog: intake.catalog.Catalog):
        """Initialize the Boundaries class with a boundary catalog.

        Sets up the lazy-loading infrastructure and validates the catalog
        structure to ensure all required boundary datasets are available.
        No data is loaded during initialization - it's loaded on first access.

        Parameters
        ----------
        boundary_catalog : intake.catalog.Catalog
            Intake catalog instance for accessing boundary parquet files.
            Must contain entries for: 'states', 'counties', 'huc8',
            'utilities', 'dfz', and 'eba'.

        Raises
        ------
        ValueError
            If the catalog is missing required entries

        Examples
        --------
        >>> import intake
        >>> catalog = intake.open_catalog('s3://bucket/boundaries.yaml')
        >>> boundaries = Boundaries(catalog)

        """
        self._cat = boundary_catalog

        # Private storage for lazy-loaded DataFrames
        self.__us_states: Optional[pd.DataFrame] = None
        self.__ca_counties: Optional[pd.DataFrame] = None
        self.__ca_watersheds: Optional[pd.DataFrame] = None
        self.__ca_utilities: Optional[pd.DataFrame] = None
        self.__ca_forecast_zones: Optional[pd.DataFrame] = None
        self.__ca_electric_balancing_areas: Optional[pd.DataFrame] = None

        # Cache for lookup dictionaries
        self._lookup_cache: Dict[str, Dict[str, int]] = {}

        # Validate catalog on initialization
        self.validate_catalog()

    def validate_catalog(self) -> None:
        """Validate that required catalog entries exist and are accessible.

        Checks for the presence of all required boundary datasets in the
        catalog. This ensures that the boundary data can be loaded when
        requested by the user.

        Raises
        ------
        ValueError
            If any required catalog entries are missing. The error message
            will list all missing entries.

        Notes
        -----
        Required catalog entries:
        - 'states': US state boundaries
        - 'counties': California county boundaries
        - 'huc8': California watershed boundaries (HUC8 level)
        - 'utilities': California electric utility boundaries
        - 'dfz': California demand forecast zones
        - 'eba': Electric balancing authority areas
        """
        required_entries = ["states", "counties", "huc8", "utilities", "dfz", "eba"]
        missing = [entry for entry in required_entries if not hasattr(self._cat, entry)]
        if missing:
            raise ValueError(f"Missing required catalog entries: {missing}")

    @property
    def _us_states(self) -> pd.DataFrame:
        """Lazy-loaded US states data."""
        if self.__us_states is None:
            try:
                self.__us_states = self._process_us_states(self._cat.states.read())
            except Exception as e:
                raise RuntimeError(f"Failed to load US states data: {e}") from e
        return self.__us_states

    @_us_states.setter
    def _us_states(self, value: pd.DataFrame) -> None:
        self.__us_states = value

    @property
    def _ca_counties(self) -> pd.DataFrame:
        """Lazy-loaded California counties data."""
        if self.__ca_counties is None:
            try:
                self.__ca_counties = self._process_ca_counties(
                    self._cat.counties.read()
                )
            except Exception as e:
                raise RuntimeError(f"Failed to load CA counties data: {e}") from e
        return self.__ca_counties

    @_ca_counties.setter
    def _ca_counties(self, value: pd.DataFrame) -> None:
        self.__ca_counties = value

    @property
    def _ca_watersheds(self) -> pd.DataFrame:
        """Lazy-loaded California watersheds data."""
        if self.__ca_watersheds is None:
            try:
                self.__ca_watersheds = self._process_ca_watersheds(
                    self._cat.huc8.read()
                )
            except Exception as e:
                raise RuntimeError(f"Failed to load CA watersheds data: {e}") from e
        return self.__ca_watersheds

    @_ca_watersheds.setter
    def _ca_watersheds(self, value: pd.DataFrame) -> None:
        self.__ca_watersheds = value

    @property
    def _ca_utilities(self) -> pd.DataFrame:
        """Lazy-loaded California utilities data."""
        if self.__ca_utilities is None:
            try:
                self.__ca_utilities = self._process_ca_utilities(
                    self._cat.utilities.read()
                )
            except Exception as e:
                raise RuntimeError(f"Failed to load CA utilities data: {e}") from e
        return self.__ca_utilities

    @_ca_utilities.setter
    def _ca_utilities(self, value: pd.DataFrame) -> None:
        self.__ca_utilities = value

    @property
    def _ca_forecast_zones(self) -> pd.DataFrame:
        """Lazy-loaded California forecast zones data."""
        if self.__ca_forecast_zones is None:
            try:
                self.__ca_forecast_zones = self._process_ca_forecast_zones(
                    self._cat.dfz.read()
                )
            except Exception as e:
                raise RuntimeError(f"Failed to load CA forecast zones data: {e}") from e
        return self.__ca_forecast_zones

    @_ca_forecast_zones.setter
    def _ca_forecast_zones(self, value: pd.DataFrame) -> None:
        self.__ca_forecast_zones = value

    @property
    def _ca_electric_balancing_areas(self) -> pd.DataFrame:
        """Lazy-loaded California electric balancing areas data."""
        if self.__ca_electric_balancing_areas is None:
            try:
                self.__ca_electric_balancing_areas = (
                    self._process_ca_electric_balancing_areas(self._cat.eba.read())
                )
            except Exception as e:
                raise RuntimeError(
                    f"Failed to load CA electric balancing areas data: {e}"
                ) from e
        return self.__ca_electric_balancing_areas

    @_ca_electric_balancing_areas.setter
    def _ca_electric_balancing_areas(self, value: pd.DataFrame) -> None:
        self.__ca_electric_balancing_areas = value

    def _process_us_states(self, df: pd.DataFrame) -> pd.DataFrame:
        """Process raw US states data.

        Parameters
        ----------
        df : pd.DataFrame
            Raw US states DataFrame

        Returns
        -------
        pd.DataFrame
            Processed US states DataFrame

        """
        return df  # No processing needed currently

    def _process_ca_counties(self, df: pd.DataFrame) -> pd.DataFrame:
        """Process raw CA counties data.

        Parameters
        ----------
        df : pd.DataFrame
            Raw CA counties DataFrame

        Returns
        -------
        pd.DataFrame
            Processed CA counties DataFrame sorted by name

        """
        return df.sort_values("NAME")

    def _process_ca_watersheds(self, df: pd.DataFrame) -> pd.DataFrame:
        """Process raw CA watersheds data.

        Parameters
        ----------
        df : pd.DataFrame
            Raw CA watersheds DataFrame

        Returns
        -------
        pd.DataFrame
            Processed CA watersheds DataFrame sorted by name

        """
        return df.sort_values("Name")

    def _process_ca_utilities(self, df: pd.DataFrame) -> pd.DataFrame:
        """Process raw CA utilities data.

        Parameters
        ----------
        df : pd.DataFrame
            Raw CA utilities DataFrame

        Returns
        -------
        pd.DataFrame
            Processed CA utilities DataFrame

        """
        return df  # No processing needed currently

    def _process_ca_forecast_zones(self, df: pd.DataFrame) -> pd.DataFrame:
        """Process CA forecast zones data - replace 'Other' with county names.

        Parameters
        ----------
        df : pd.DataFrame
            Raw CA forecast zones DataFrame

        Returns
        -------
        pd.DataFrame
            Processed CA forecast zones DataFrame with 'Other' names replaced

        """
        df = df.copy()
        df.loc[df["FZ_Name"] == "Other", "FZ_Name"] = df["FZ_Def"]
        return df

    def _process_ca_electric_balancing_areas(self, df: pd.DataFrame) -> pd.DataFrame:
        """Process CA electric balancing areas data - remove tiny CALISO polygon.

        The CALISO polygon has two options where one is super tiny with negligible area.
        This removes the tiny polygon and keeps only the large one.

        Parameters
        ----------
        df : pd.DataFrame
            Raw CA electric balancing areas DataFrame

        Returns
        -------
        pd.DataFrame
            Processed DataFrame with tiny CALISO polygon removed

        """
        tiny_caliso = df.loc[
            (df["NAME"] == "CALISO") & (df["SHAPE_Area"] < CALISO_AREA_THRESHOLD)
        ].index
        return df.drop(tiny_caliso)

    def _get_us_states(self) -> Dict[str, int]:
        """Get cached lookup dictionary for western US states.

        Returns
        -------
        Dict[str, int]
            Dictionary mapping state abbreviations to DataFrame indices

        """
        if "us_states" not in self._lookup_cache:
            self._lookup_cache["us_states"] = self._build_us_states_lookup()
        return self._lookup_cache["us_states"]

    def _build_us_states_lookup(self) -> Dict[str, int]:
        """Build lookup dictionary for western US states with custom ordering.

        Returns
        -------
        Dict[str, int]
            Dictionary mapping state abbreviations to DataFrame indices

        """
        us_states_subset = self._us_states.query("abbrevs in @WESTERN_STATES_LIST")[
            ["abbrevs"]
        ]
        us_states_subset["abbrevs"] = pd.Categorical(
            us_states_subset["abbrevs"], categories=WESTERN_STATES_LIST
        )
        us_states_subset.sort_values(by="abbrevs", inplace=True)
        return dict(zip(us_states_subset.abbrevs, us_states_subset.index))

    def _get_ca_counties(self) -> Dict[str, int]:
        """Get cached lookup dictionary for California counties.

        Returns
        -------
        Dict[str, int]
            Dictionary mapping county names to DataFrame indices

        """
        if "ca_counties" not in self._lookup_cache:
            self._lookup_cache["ca_counties"] = pd.Series(
                self._ca_counties.index, index=self._ca_counties["NAME"]
            ).to_dict()
        return self._lookup_cache["ca_counties"]

    def _get_ca_watersheds(self) -> Dict[str, int]:
        """Get cached lookup dictionary for California watersheds.

        Returns
        -------
        Dict[str, int]
            Dictionary mapping watershed names to DataFrame indices

        """
        if "ca_watersheds" not in self._lookup_cache:
            self._lookup_cache["ca_watersheds"] = pd.Series(
                self._ca_watersheds.index, index=self._ca_watersheds["Name"]
            ).to_dict()
        return self._lookup_cache["ca_watersheds"]

    def _get_forecast_zones(self) -> Dict[str, int]:
        """Get cached lookup dictionary for CA electricity demand forecast zones.

        Returns
        -------
        Dict[str, int]
            Dictionary mapping forecast zone names to DataFrame indices

        """
        if "forecast_zones" not in self._lookup_cache:
            self._lookup_cache["forecast_zones"] = pd.Series(
                self._ca_forecast_zones.index, index=self._ca_forecast_zones["FZ_Name"]
            ).to_dict()
        return self._lookup_cache["forecast_zones"]

    def _get_ious_pous(self) -> Dict[str, int]:
        """Get cached lookup dictionary for CA electric load serving entities (IOUs & POUs).

        Returns prioritized utilities first, then remaining utilities alphabetically.

        Returns
        -------
        Dict[str, int]
            Dictionary mapping utility names to DataFrame indices

        """
        if "ious_pous" not in self._lookup_cache:
            self._lookup_cache["ious_pous"] = self._build_ious_pous_lookup()
        return self._lookup_cache["ious_pous"]

    def _build_ious_pous_lookup(self) -> Dict[str, int]:
        """Build lookup dictionary for CA electric utilities with custom ordering.

        Returns
        -------
        Dict[str, int]
            Dictionary mapping utility names to DataFrame indices

        """
        other_utilities = [
            utility
            for utility in self._ca_utilities["Utility"]
            if utility not in PRIORITY_UTILITIES
        ]
        other_utilities = sorted(other_utilities)  # Alphabetical order
        ordered_list = PRIORITY_UTILITIES + other_utilities

        subset = self._ca_utilities.query("Utility in @ordered_list")[["Utility"]]
        subset["Utility"] = pd.Categorical(subset["Utility"], categories=ordered_list)
        subset.sort_values(by="Utility", inplace=True)
        return dict(zip(subset["Utility"], subset.index))

    def _get_electric_balancing_areas(self) -> Dict[str, int]:
        """Get cached lookup dictionary for CA electric balancing authority areas.

        Returns
        -------
        Dict[str, int]
            Dictionary mapping balancing area names to DataFrame indices

        """
        if "electric_balancing_areas" not in self._lookup_cache:
            self._lookup_cache["electric_balancing_areas"] = pd.Series(
                self._ca_electric_balancing_areas.index,
                index=self._ca_electric_balancing_areas["NAME"],
            ).to_dict()
        return self._lookup_cache["electric_balancing_areas"]

    def boundary_dict(self) -> Dict[str, Dict[str, int]]:
        """Return dictionary of all boundary lookup dictionaries for UI population.

        Creates a comprehensive dictionary of all available boundary datasets
        with their corresponding lookup dictionaries. This is primarily used
        to populate user interface components that allow boundary selection
        for geographic subsetting of climate data.

        The returned dictionary maps boundary category names to lookup dictionaries
        that map specific boundary names to their DataFrame indices. This enables
        efficient boundary selection and data subsetting operations.

        Returns
        -------
        Dict[str, Dict[str, int]]
            Nested dictionary structure:
            - Outer keys: boundary category names (e.g., 'states', 'CA counties')
            - Inner dictionaries: map boundary names to DataFrame indices

            Available categories:
            - 'none': No geographic subsetting
            - 'lat/lon': Custom coordinate-based selection
            - 'states': Western US states
            - 'CA counties': California counties (alphabetical)
            - 'CA watersheds': California HUC8 watersheds (alphabetical)
            - 'CA Electric Load Serving Entities (IOU & POU)': Electric utilities
            - 'CA Electricity Demand Forecast Zones': Forecast zones
            - 'CA Electric Balancing Authority Areas': Balancing areas

        Examples
        --------
        >>> boundaries = Boundaries(catalog)
        >>> boundary_options = boundaries.boundary_dict()
        >>>
        >>> # Get available states
        >>> states = boundary_options['states']
        >>> print(states.keys())  # ['CA', 'OR', 'WA', ...]
        >>>
        >>> # Get available counties
        >>> counties = boundary_options['CA counties']
        >>> alameda_idx = counties['Alameda']
        >>>
        >>> # Use in UI dropdown population
        >>> for category, options in boundary_options.items():
        >>>     populate_dropdown(category, options.keys())

        Notes
        -----
        - Lookup dictionaries are cached for performance
        - Western states follow ordering in WESTERN_STATES_LIST
        - Utilities are ordered with priority utilities first
        - All other boundaries are sorted alphabetically

        """
        return {
            "none": {"entire domain": 0},
            "lat/lon": {"coordinate selection": 0},
            "states": self._get_us_states(),
            "CA counties": self._get_ca_counties(),
            "CA watersheds": self._get_ca_watersheds(),
            "CA Electric Load Serving Entities (IOU & POU)": self._get_ious_pous(),
            "CA Electricity Demand Forecast Zones": self._get_forecast_zones(),
            "CA Electric Balancing Authority Areas": self._get_electric_balancing_areas(),
        }

    def load(self) -> None:
        """Preload all boundary data (deprecated - data loads automatically when accessed).

        This method is kept for backward compatibility. Data now loads automatically
        when first accessed through the property system.

        Deprecated
        ----------
        This method is deprecated as of version X.X.X. Use preload_all() instead
        for explicit preloading, or simply access data normally for automatic
        lazy loading.

        """
        warnings.warn(
            "The load() method is deprecated. Data now loads automatically when accessed. "
            "Use preload_all() for explicit preloading.",
            DeprecationWarning,
            stacklevel=999,
        )
        self.preload_all()

    def preload_all(self) -> None:
        """Preload all boundary data for performance-critical scenarios.

        Forces immediate loading of all boundary datasets and builds all
        lookup caches. This eliminates lazy loading delays for subsequent
        data access operations, making it ideal for performance-critical
        scenarios or when you know all boundary data will be needed.

        The method loads all six boundary datasets:
        - US western states
        - California counties
        - California watersheds
        - California utilities
        - California forecast zones
        - California electric balancing areas

        And builds all corresponding lookup dictionaries for fast boundary
        selection operations.

        Examples
        --------
        >>> boundaries = Boundaries(catalog)
        >>>
        >>> # Preload for performance-critical batch processing
        >>> boundaries.preload_all()
        >>>
        >>> # All subsequent access is now immediate
        >>> for county in boundaries._ca_counties.itertuples():
        >>>     process_county_data(county)

        Notes
        -----
        - Increases initial memory usage but eliminates loading delays
        - Useful for batch processing or repeated boundary access
        - Data remains cached until clear_cache() is called
        - Memory usage can be monitored with get_memory_usage()

        """
        # Force loading of all properties
        _ = (
            self._us_states,
            self._ca_counties,
            self._ca_watersheds,
            self._ca_utilities,
            self._ca_forecast_zones,
            self._ca_electric_balancing_areas,
        )

        # Build all lookup caches
        _ = (
            self._get_us_states(),
            self._get_ca_counties(),
            self._get_ca_watersheds(),
            self._get_forecast_zones(),
            self._get_ious_pous(),
            self._get_electric_balancing_areas(),
        )

    def clear_cache(self) -> None:
        """Clear all cached data and lookup dictionaries to free memory.

        Removes all loaded boundary DataFrames and lookup dictionaries from
        memory, returning the Boundaries instance to its initial state. Data
        will be reloaded on next access through the lazy loading mechanism.

        This is useful for:
        - Memory management in long-running applications
        - Forcing fresh data loads after catalog updates
        - Resetting state during testing or debugging

        Examples
        --------
        >>> boundaries = Boundaries(catalog)
        >>> boundaries.preload_all()
        >>> usage_before = boundaries.get_memory_usage()
        >>> print(f"Memory before: {usage_before['total_human']}")
        >>>
        >>> boundaries.clear_cache()
        >>> usage_after = boundaries.get_memory_usage()
        >>> print(f"Memory after: {usage_after['total_human']}")  # Much lower
        >>>
        >>> # Data loads again on next access
        >>> counties = boundaries._ca_counties  # Triggers reload

        Notes
        -----
        - All subsequent data access will trigger fresh loads from catalog
        - Lookup dictionaries will be rebuilt as needed
        - Does not affect the underlying catalog or data sources
        - Memory savings are immediate and substantial for loaded datasets

        """
        # Clear raw DataFrames
        self.__us_states = None
        self.__ca_counties = None
        self.__ca_watersheds = None
        self.__ca_utilities = None
        self.__ca_forecast_zones = None
        self.__ca_electric_balancing_areas = None

        # Clear lookup cache
        self._lookup_cache.clear()

    def get_memory_usage(self) -> Dict[str, Union[int, str]]:
        """Get detailed memory usage information for loaded boundary datasets.

        Analyzes memory consumption of all loaded boundary DataFrames and
        provides both detailed per-dataset usage and summary statistics.
        Useful for memory monitoring and optimization decisions.

        Returns
        -------
        Dict[str, Union[int, str]]
            Comprehensive memory usage information:

            Per-dataset usage (bytes):
            - 'us_states': Memory used by US states DataFrame (0 if not loaded)
            - 'ca_counties': Memory used by CA counties DataFrame (0 if not loaded)
            - 'ca_watersheds': Memory used by CA watersheds DataFrame (0 if not loaded)
            - 'ca_utilities': Memory used by CA utilities DataFrame (0 if not loaded)
            - 'ca_forecast_zones': Memory used by forecast zones DataFrame (0 if not loaded)
            - 'ca_electric_balancing_areas': Memory used by balancing areas DataFrame (0 if not loaded)

            Summary statistics:
            - 'total_bytes': Total memory usage in bytes
            - 'total_human': Human-readable total memory usage (e.g., '15.2 MB')
            - 'loaded_datasets': Count of currently loaded datasets
            - 'cached_lookups': Count of cached lookup dictionaries

        Examples
        --------
        >>> boundaries = Boundaries(catalog)
        >>> boundaries.preload_all()
        >>> usage = boundaries.get_memory_usage()
        >>>
        >>> print(f"Total memory: {usage['total_human']}")
        >>> print(f"Loaded datasets: {usage['loaded_datasets']}/6")
        >>> print(f"Largest dataset: {max(usage['us_states'], usage['ca_counties'])}")
        >>>
        >>> # Check if specific dataset is loaded
        >>> if usage['ca_counties'] > 0:
        >>>     print("Counties data is loaded")

        >>> # Monitor memory before/after operations
        >>> usage_before = boundaries.get_memory_usage()
        >>> boundaries.clear_cache()
        >>> usage_after = boundaries.get_memory_usage()
        >>> saved = usage_before['total_bytes'] - usage_after['total_bytes']
        >>> print(f"Memory freed: {boundaries._format_bytes(saved)}")

        Notes
        -----
        - Memory usage includes deep analysis of DataFrame contents
        - Unloaded datasets report 0 bytes usage
        - Lookup dictionary cache usage is counted separately
        - Total includes all loaded DataFrames but not lookup dictionaries

        """
        usage = {}
        total_bytes = 0

        datasets = {
            "us_states": self.__us_states,
            "ca_counties": self.__ca_counties,
            "ca_watersheds": self.__ca_watersheds,
            "ca_utilities": self.__ca_utilities,
            "ca_forecast_zones": self.__ca_forecast_zones,
            "ca_electric_balancing_areas": self.__ca_electric_balancing_areas,
        }

        for name, df in datasets.items():
            if df is not None:
                memory_bytes = df.memory_usage(deep=True).sum()
                usage[name] = memory_bytes
                total_bytes += memory_bytes
            else:
                usage[name] = 0

        usage["total_bytes"] = total_bytes
        usage["total_human"] = self._format_bytes(total_bytes)
        usage["loaded_datasets"] = len(
            [df for df in datasets.values() if df is not None]
        )
        usage["cached_lookups"] = len(self._lookup_cache)

        return usage

    @staticmethod
    def _format_bytes(bytes_value: int | float) -> str:
        """Convert bytes to human-readable format with appropriate units.

        Parameters
        ----------
        bytes_value : int | float
            Memory size in bytes to format

        Returns
        -------
        str
            Human-readable string with appropriate unit (B, KB, MB, GB, TB)

        Examples
        --------
        >>> Boundaries._format_bytes(1024)
        '1.0 KB'
        >>> Boundaries._format_bytes(1048576)
        '1.0 MB'
        >>> Boundaries._format_bytes(1536)
        '1.5 KB'

        """
        for unit in ["B", "KB", "MB", "GB"]:
            if bytes_value < 1024.0:
                return f"{bytes_value:.1f} {unit}"
            bytes_value /= 1024.0
        return f"{bytes_value:.1f} TB"
