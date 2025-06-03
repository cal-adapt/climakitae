"""
Lazy-loading boundaries module for ClimakitAE.

This module provides a Boundaries class for managing geospatial boundary data
with lazy loading capabilities. The class automatically loads boundary data
only when first accessed, improving memory efficiency and startup performance.

Classes
-------
Boundaries
    Lazy-loading class for managing geospatial polygon data from S3 stored
    parquet catalogs. Used to access boundaries for subsetting data by state,
    county, watershed, and utility service areas.
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
    """
    Lazy-loading geospatial polygon data manager for ClimakitAE.

    This class provides efficient access to various boundary datasets stored
    in S3 parquet catalogs. Data is loaded only when first accessed, improving
    memory usage and initialization performance. All lookup dictionaries are
    cached to avoid recomputation.

    Attributes
    ----------
    _cat : intake.catalog.Catalog
        Parquet boundary catalog instance

    Properties
    ----------
    _us_states : pd.DataFrame
        Table of US state names and geometries (lazy-loaded)
    _ca_counties : pd.DataFrame
        Table of California county names and geometries (lazy-loaded)
    _ca_watersheds : pd.DataFrame
        Table of California watershed names and geometries (lazy-loaded)
    _ca_utilities : pd.DataFrame
        Table of California IOUs and POUs, names and geometries (lazy-loaded)
    _ca_forecast_zones : pd.DataFrame
        Table of California Demand Forecast Zones (lazy-loaded)
    _ca_electric_balancing_areas : pd.DataFrame
        Table of Electric Balancing Authority Areas (lazy-loaded)

    Methods
    -------
    boundary_dict() -> Dict[str, Dict[str, int]]
        Return dictionary of all boundary lookup dictionaries for UI population
    preload_all() -> None
        Preload all boundary data for performance-critical scenarios
    clear_cache() -> None
        Clear all cached data and lookup dictionaries
    validate_catalog() -> None
        Validate that required catalog entries exist
    """

    def __init__(self, boundary_catalog: intake.catalog.Catalog):
        """
        Initialize the Boundaries class with a boundary catalog.

        Parameters
        ----------
        boundary_catalog : intake.catalog.Catalog
            Intake catalog instance for accessing boundary parquet files
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
        """
        Validate that required catalog entries exist.

        Raises
        ------
        ValueError
            If required catalog entries are missing
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
        """
        Process raw US states data.

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
        """
        Process raw CA counties data.

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
        """
        Process raw CA watersheds data.

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
        """
        Process raw CA utilities data.

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
        """
        Process CA forecast zones data - replace 'Other' with county names.

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
        """
        Process CA electric balancing areas data - remove tiny CALISO polygon.

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
        """
        Get cached lookup dictionary for western US states.

        Returns
        -------
        Dict[str, int]
            Dictionary mapping state abbreviations to DataFrame indices
        """
        if "us_states" not in self._lookup_cache:
            self._lookup_cache["us_states"] = self._build_us_states_lookup()
        return self._lookup_cache["us_states"]

    def _build_us_states_lookup(self) -> Dict[str, int]:
        """
        Build lookup dictionary for western US states with custom ordering.

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
        """
        Get cached lookup dictionary for California counties.

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
        """
        Get cached lookup dictionary for California watersheds.

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
        """
        Get cached lookup dictionary for CA electricity demand forecast zones.

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
        """
        Get cached lookup dictionary for CA electric load serving entities (IOUs & POUs).

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
        """
        Build lookup dictionary for CA electric utilities with custom ordering.

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
        """
        Get cached lookup dictionary for CA electric balancing authority areas.

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
        """
        Return dictionary of all boundary lookup dictionaries for UI population.

        This returns a dictionary of lookup dictionaries for each set of
        geoparquet files that the user might be choosing from. It is used to
        populate the `DataParameters` cached_area dynamically as the category
        in the area_subset parameter changes.

        Returns
        -------
        Dict[str, Dict[str, int]]
            Dictionary containing lookup dictionaries for each boundary type
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
        """
        Preload all boundary data (deprecated - data loads automatically when accessed).

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
            stacklevel=2,
        )
        self.preload_all()

    def preload_all(self) -> None:
        """
        Preload all boundary data for performance-critical scenarios.

        This method forces loading of all boundary datasets and builds all
        lookup caches. Useful when you know you'll need all data and want
        to avoid lazy loading delays.
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
        """
        Clear all cached data and lookup dictionaries.

        This method clears both the raw DataFrames and the lookup dictionary
        cache, forcing fresh data loads on next access. Useful for memory
        management or when catalog data might have changed.
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
        """
        Get memory usage information for loaded boundary data.

        Returns
        -------
        Dict[str, Union[int, str]]
            Dictionary with memory usage information in bytes for each dataset,
            plus total usage and human-readable format
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
        """Convert bytes to human readable format."""
        for unit in ["B", "KB", "MB", "GB"]:
            if bytes_value < 1024.0:
                return f"{bytes_value:.1f} {unit}"
            bytes_value /= 1024.0
        return f"{bytes_value:.1f} TB"
