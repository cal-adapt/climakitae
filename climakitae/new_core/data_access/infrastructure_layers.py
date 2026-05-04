"""Lazy-loading infrastructure layers module for ClimakitAE.

This module provides efficient access to California power plant and grid
infrastructure data stored as GeoParquet files. The InfrastructureLayers
class implements lazy loading to minimize memory usage and improve startup
performance by only loading datasets when they are first accessed.

The module supports access to the following infrastructure layer types:
- EIA-860M operating power plants (point data)
- Global Energy Monitor (GEM) tracked power projects (point data)
- HIFLD electric power transmission lines (line data)
- HIFLD electric substations (point data)
- OpenStreetMap power infrastructure (mixed geometry)

All data is California-scoped, EPSG:4326, sourced from public-domain or
open-license datasets. See ``scripts/build_infrastructure_parquets.py``
for the ETL pipeline that produces the source parquet files.

Classes
-------
InfrastructureLayers
    Lazy-loading class for managing CA power plant and grid infrastructure
    data from GeoParquet files. Provides lookup dictionaries and
    memory-efficient access to infrastructure datasets.

Examples
--------
>>> urls = {
...     "eia_plants": "s3://bucket/eia860m_ca_plants.parquet",
...     "transmission": "s3://bucket/hifld_ca_transmission.parquet",
... }
>>> infra = InfrastructureLayers(urls)
>>>
>>> # Data loads automatically when accessed
>>> plants = infra.eia_plants
>>>
>>> # Filter by fuel type
>>> by_fuel = infra._get_plants_by_fuel_type()
>>> solar_indices = by_fuel.get("Solar", [])
>>>
>>> # Preload all layers
>>> infra.preload_all()

"""

import io
import logging
import warnings
from typing import Dict, List, Optional, Union

import geopandas as gpd
import pandas as pd
import requests

# Module logger
logger = logging.getLogger(__name__)

# Layer keys — used in layer_urls dict and as canonical names
LAYER_KEY_EIA_PLANTS = "eia_plants"
LAYER_KEY_GEM_PLANTS = "gem_plants"
LAYER_KEY_TRANSMISSION = "transmission"
LAYER_KEY_SUBSTATIONS = "substations"
LAYER_KEY_OSM_POWER = "osm_power"

ALL_LAYER_KEYS = [
    LAYER_KEY_EIA_PLANTS,
    LAYER_KEY_GEM_PLANTS,
    LAYER_KEY_TRANSMISSION,
    LAYER_KEY_SUBSTATIONS,
    LAYER_KEY_OSM_POWER,
]


class InfrastructureLayers:
    """Lazy-loading CA power plant and grid infrastructure data manager.

    Provides efficient access to California power plant and grid infrastructure
    datasets stored as GeoParquet files. Data is loaded only when first
    accessed, improving memory usage and initialization performance. All lookup
    dictionaries are cached to avoid recomputation.

    The class mirrors the architectural pattern of ``Boundaries`` but is
    designed for point-like and line infrastructure data rather than polygon
    boundaries. It is intended for spatial analysis (nearest plant, plants
    within boundary), UI population (plant name lookups), and filtering by
    fuel type or operator.

    Parameters
    ----------
    layer_urls : Dict[str, str]
        Dictionary mapping layer keys to GeoParquet file URLs or local paths.
        Accepted keys: ``"eia_plants"``, ``"gem_plants"``, ``"transmission"``,
        ``"substations"``, ``"osm_power"``. Missing keys are allowed — the
        corresponding property will raise ``RuntimeError`` if accessed.

    Attributes
    ----------
    _urls : Dict[str, str]
        Reference to the layer URLs dictionary.

    Properties
    ----------
    eia_plants : gpd.GeoDataFrame
        EIA-860M operating CA power plants (point geometry, lazy-loaded).
    gem_plants : gpd.GeoDataFrame
        Global Energy Monitor tracked CA power projects (point, lazy-loaded).
    transmission : gpd.GeoDataFrame
        HIFLD CA electric power transmission lines (line geometry, lazy-loaded).
    substations : gpd.GeoDataFrame
        HIFLD CA electric substations (point geometry, lazy-loaded).
    osm_power : gpd.GeoDataFrame
        OpenStreetMap CA power infrastructure (mixed geometry, lazy-loaded).

    Methods
    -------
    layers_dict() -> Dict[str, gpd.GeoDataFrame]
        Return all available layers as a dictionary of GeoDataFrames.
    available_layers() -> List[str]
        Return list of layer keys with configured URLs.
    preload_all() -> None
        Preload all configured layers into memory.
    clear_cache() -> None
        Clear all cached data to free memory.
    get_memory_usage() -> Dict[str, Union[int, str]]
        Get memory usage information for loaded layers.
    validate_urls() -> None
        Warn on missing or unrecognized URL keys.

    Examples
    --------
    Basic usage with lazy loading:

    >>> urls = {"eia_plants": "s3://cadcat/infrastructure/eia860m_ca_plants.parquet"}
    >>> infra = InfrastructureLayers(urls)
    >>> plants = infra.eia_plants           # loads on first access
    >>> plants2 = infra.eia_plants          # returned from cache

    Getting lookup dictionaries:

    >>> by_fuel = infra._get_plants_by_fuel_type()
    >>> solar_rows = infra.eia_plants.loc[by_fuel["Solar"]]

    All layers at once:

    >>> all_layers = infra.layers_dict()
    >>> for name, gdf in all_layers.items():
    ...     print(name, len(gdf))

    Notes
    -----
    - All data is California-scoped, EPSG:4326.
    - Lookup caches combine EIA and GEM plant data for ``_get_plants_by_name``,
      ``_get_plants_by_fuel_type``, and ``_get_plants_by_operator``.
    - ``layers_dict()`` returns the already-lazy-loaded backing GeoDataFrames,
      not copies — no extra memory cost until each layer is first accessed.
    - Source parquet files are produced by
      ``scripts/build_infrastructure_parquets.py``.

    """

    def __init__(self, layer_urls: Dict[str, str]) -> None:
        """Initialize InfrastructureLayers with a mapping of layer URLs.

        Sets up the lazy-loading infrastructure. No data is loaded during
        initialization — data loads on first property access. Runs
        ``validate_urls()`` to warn on missing or unrecognized keys.

        Parameters
        ----------
        layer_urls : Dict[str, str]
            Dictionary mapping layer keys to GeoParquet URLs or local paths.
            Keys: ``"eia_plants"``, ``"gem_plants"``, ``"transmission"``,
            ``"substations"``, ``"osm_power"``. Partial dicts are accepted;
            missing layers raise ``RuntimeError`` only when accessed.

        Examples
        --------
        >>> urls = {
        ...     "eia_plants": "s3://cadcat/infrastructure/eia860m_ca_plants.parquet",
        ...     "gem_plants": "s3://cadcat/infrastructure/gem_ca_plants.parquet",
        ... }
        >>> infra = InfrastructureLayers(urls)

        """
        self._urls = layer_urls

        # Private backing stores for lazy-loaded GeoDataFrames
        self.__eia_plants: Optional[gpd.GeoDataFrame] = None
        self.__gem_plants: Optional[gpd.GeoDataFrame] = None
        self.__transmission: Optional[gpd.GeoDataFrame] = None
        self.__substations: Optional[gpd.GeoDataFrame] = None
        self.__osm_power: Optional[gpd.GeoDataFrame] = None

        # Cache for lookup dictionaries
        self._lookup_cache: Dict[str, object] = {}

        # Validate URL keys on initialization
        self.validate_urls()

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------

    def validate_urls(self) -> None:
        """Warn on missing or unrecognized layer URL keys.

        Checks the provided ``layer_urls`` against the known set of layer
        keys. Issues a warning for each unknown key (possible typo) and
        logs an info-level message for each expected key that is absent
        (that layer will raise ``RuntimeError`` if accessed).

        This method does not raise — missing layers are allowed for partial
        deployments where only a subset of parquet files have been uploaded.

        Examples
        --------
        >>> infra = InfrastructureLayers({"eia_plants": "s3://bucket/eia.parquet"})
        >>> infra.validate_urls()  # logs info for gem_plants, transmission, etc.

        """
        for key in self._urls:
            if key not in ALL_LAYER_KEYS:
                warnings.warn(
                    f"Unknown layer key '{key}' in layer_urls. "
                    f"Expected one of: {ALL_LAYER_KEYS}",
                    UserWarning,
                    stacklevel=2,
                )

        for key in ALL_LAYER_KEYS:
            if key not in self._urls:
                logger.info(
                    "Layer '%s' has no URL configured and will be unavailable.", key
                )

    # ------------------------------------------------------------------
    # Private load helper
    # ------------------------------------------------------------------

    def _load_layer(self, key: str) -> gpd.GeoDataFrame:
        """Load a single layer from its configured URL.

        Parameters
        ----------
        key : str
            The layer key (one of ``ALL_LAYER_KEYS``).

        Returns
        -------
        gpd.GeoDataFrame
            The loaded GeoDataFrame.

        Raises
        ------
        RuntimeError
            If ``key`` is not present in ``layer_urls``, or if loading fails.

        """
        if key not in self._urls:
            raise RuntimeError(
                f"Layer '{key}' has no URL configured. "
                "Pass it in layer_urls when constructing InfrastructureLayers."
            )
        url = self._urls[key]
        try:
            logger.debug("Loading infrastructure layer '%s' from %s", key, url)
            if url.startswith("https://") or url.startswith("http://"):
                resp = requests.get(url, timeout=120)
                resp.raise_for_status()
                return gpd.read_parquet(io.BytesIO(resp.content))
            return gpd.read_parquet(url)
        except Exception as e:
            raise RuntimeError(
                f"Failed to load infrastructure layer '{key}' from {url}: {e}"
            ) from e

    # ------------------------------------------------------------------
    # Lazy properties — eia_plants
    # ------------------------------------------------------------------

    @property
    def eia_plants(self) -> gpd.GeoDataFrame:
        """Lazy-loaded EIA-860M CA operating power plants.

        Returns
        -------
        gpd.GeoDataFrame
            Point GeoDataFrame with columns: plant_id, name, fuel_type,
            capacity_mw, status, operator, county, source, geometry.

        Raises
        ------
        RuntimeError
            If the layer URL is not configured or loading fails.

        """
        if self.__eia_plants is None:
            self.__eia_plants = self._process_eia_plants(
                self._load_layer(LAYER_KEY_EIA_PLANTS)
            )
        return self.__eia_plants

    @eia_plants.setter
    def eia_plants(self, value: gpd.GeoDataFrame) -> None:
        """Set eia_plants backing store (used in tests and preload_all)."""
        self.__eia_plants = value

    # ------------------------------------------------------------------
    # Lazy properties — gem_plants
    # ------------------------------------------------------------------

    @property
    def gem_plants(self) -> gpd.GeoDataFrame:
        """Lazy-loaded Global Energy Monitor CA power projects.

        Returns
        -------
        gpd.GeoDataFrame
            Point GeoDataFrame with columns: plant_id, name, fuel_type,
            capacity_mw, status, operator, county, source, geometry.

        Raises
        ------
        RuntimeError
            If the layer URL is not configured or loading fails.

        """
        if self.__gem_plants is None:
            self.__gem_plants = self._process_gem_plants(
                self._load_layer(LAYER_KEY_GEM_PLANTS)
            )
        return self.__gem_plants

    @gem_plants.setter
    def gem_plants(self, value: gpd.GeoDataFrame) -> None:
        """Set gem_plants backing store (used in tests and preload_all)."""
        self.__gem_plants = value

    # ------------------------------------------------------------------
    # Lazy properties — transmission
    # ------------------------------------------------------------------

    @property
    def transmission(self) -> gpd.GeoDataFrame:
        """Lazy-loaded HIFLD CA electric power transmission lines.

        Returns
        -------
        gpd.GeoDataFrame
            LineString GeoDataFrame with columns: line_id, voltage_kv,
            status, owner, source, geometry.

        Raises
        ------
        RuntimeError
            If the layer URL is not configured or loading fails.

        """
        if self.__transmission is None:
            self.__transmission = self._process_transmission(
                self._load_layer(LAYER_KEY_TRANSMISSION)
            )
        return self.__transmission

    @transmission.setter
    def transmission(self, value: gpd.GeoDataFrame) -> None:
        """Set transmission backing store (used in tests and preload_all)."""
        self.__transmission = value

    # ------------------------------------------------------------------
    # Lazy properties — substations
    # ------------------------------------------------------------------

    @property
    def substations(self) -> gpd.GeoDataFrame:
        """Lazy-loaded HIFLD CA electric substations.

        Returns
        -------
        gpd.GeoDataFrame
            Point GeoDataFrame with columns: substation_id, name, voltage_kv,
            status, owner, source, geometry.

        Raises
        ------
        RuntimeError
            If the layer URL is not configured or loading fails.

        """
        if self.__substations is None:
            self.__substations = self._process_substations(
                self._load_layer(LAYER_KEY_SUBSTATIONS)
            )
        return self.__substations

    @substations.setter
    def substations(self, value: gpd.GeoDataFrame) -> None:
        """Set substations backing store (used in tests and preload_all)."""
        self.__substations = value

    # ------------------------------------------------------------------
    # Lazy properties — osm_power
    # ------------------------------------------------------------------

    @property
    def osm_power(self) -> gpd.GeoDataFrame:
        """Lazy-loaded OpenStreetMap CA power infrastructure.

        Returns
        -------
        gpd.GeoDataFrame
            Mixed-geometry GeoDataFrame with columns: osm_id, name,
            power_type, fuel_type, capacity_mw, voltage_kv, operator,
            source, geometry.

        Raises
        ------
        RuntimeError
            If the layer URL is not configured or loading fails.

        """
        if self.__osm_power is None:
            self.__osm_power = self._process_osm_power(
                self._load_layer(LAYER_KEY_OSM_POWER)
            )
        return self.__osm_power

    @osm_power.setter
    def osm_power(self, value: gpd.GeoDataFrame) -> None:
        """Set osm_power backing store (used in tests and preload_all)."""
        self.__osm_power = value

    # ------------------------------------------------------------------
    # Process methods
    # ------------------------------------------------------------------

    def _process_eia_plants(self, df: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
        """Process raw EIA-860M plants data.

        Sorts by plant name for consistent ordering.

        Parameters
        ----------
        df : gpd.GeoDataFrame
            Raw EIA-860M plants GeoDataFrame.

        Returns
        -------
        gpd.GeoDataFrame
            Processed GeoDataFrame sorted by name.

        """
        if "name" in df.columns:
            return df.sort_values("name").reset_index(drop=True)
        return df

    def _process_gem_plants(self, df: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
        """Process raw GEM plants data.

        Sorts by plant name for consistent ordering.

        Parameters
        ----------
        df : gpd.GeoDataFrame
            Raw GEM plants GeoDataFrame.

        Returns
        -------
        gpd.GeoDataFrame
            Processed GeoDataFrame sorted by name.

        """
        if "name" in df.columns:
            return df.sort_values("name").reset_index(drop=True)
        return df

    def _process_transmission(self, df: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
        """Process raw HIFLD transmission lines data.

        Sorts by voltage descending so highest-voltage lines appear first.

        Parameters
        ----------
        df : gpd.GeoDataFrame
            Raw HIFLD transmission GeoDataFrame.

        Returns
        -------
        gpd.GeoDataFrame
            Processed GeoDataFrame sorted by voltage_kv descending.

        """
        if "voltage_kv" in df.columns:
            return df.sort_values("voltage_kv", ascending=False).reset_index(drop=True)
        return df

    def _process_substations(self, df: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
        """Process raw HIFLD substations data.

        Sorts by substation name for consistent ordering.

        Parameters
        ----------
        df : gpd.GeoDataFrame
            Raw HIFLD substations GeoDataFrame.

        Returns
        -------
        gpd.GeoDataFrame
            Processed GeoDataFrame sorted by name.

        """
        if "name" in df.columns:
            return df.sort_values("name").reset_index(drop=True)
        return df

    def _process_osm_power(self, df: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
        """Process raw OSM power infrastructure data.

        No processing currently applied — pass-through for future use.

        Parameters
        ----------
        df : gpd.GeoDataFrame
            Raw OSM power GeoDataFrame.

        Returns
        -------
        gpd.GeoDataFrame
            Unmodified input GeoDataFrame.

        """
        return df

    # ------------------------------------------------------------------
    # Lookup methods
    # ------------------------------------------------------------------

    def _get_plants_by_name(self) -> Dict[str, int]:
        """Get cached lookup dictionary mapping plant names to row indices.

        Combines EIA-860M and GEM plants. EIA indices are prefixed with
        ``"eia:"`` and GEM indices with ``"gem:"`` to avoid key collisions.
        Format: ``{"eia:Plant Name": row_index, "gem:Plant Name": row_index}``.

        Returns
        -------
        Dict[str, int]
            Dictionary mapping ``"source:name"`` strings to integer row
            indices in the respective GeoDataFrame.

        Examples
        --------
        >>> lookup = infra._get_plants_by_name()
        >>> eia_idx = lookup.get("eia:Geysers Power Plant")
        >>> if eia_idx is not None:
        ...     row = infra.eia_plants.iloc[eia_idx]

        """
        if "plants_by_name" not in self._lookup_cache:
            result: Dict[str, int] = {}
            if LAYER_KEY_EIA_PLANTS in self._urls:
                for idx, name in enumerate(self.eia_plants.get("name", pd.Series())):
                    if pd.notna(name):
                        result[f"eia:{name}"] = idx
            if LAYER_KEY_GEM_PLANTS in self._urls:
                for idx, name in enumerate(self.gem_plants.get("name", pd.Series())):
                    if pd.notna(name):
                        result[f"gem:{name}"] = idx
            self._lookup_cache["plants_by_name"] = result
        return self._lookup_cache["plants_by_name"]  # type: ignore[return-value]

    def _get_plants_by_fuel_type(self) -> Dict[str, List[int]]:
        """Get cached lookup mapping fuel type to lists of EIA plant row indices.

        Uses EIA-860M plants only (standardized fuel type column).

        Returns
        -------
        Dict[str, List[int]]
            Dictionary mapping standardized fuel type strings to lists of
            integer row indices in ``eia_plants``.

        Examples
        --------
        >>> by_fuel = infra._get_plants_by_fuel_type()
        >>> solar_plants = infra.eia_plants.iloc[by_fuel.get("Solar", [])]

        """
        if "plants_by_fuel_type" not in self._lookup_cache:
            result: Dict[str, List[int]] = {}
            if LAYER_KEY_EIA_PLANTS in self._urls:
                fuel_col = self.eia_plants.get("fuel_type", pd.Series())
                for idx, fuel in enumerate(fuel_col):
                    if pd.notna(fuel):
                        result.setdefault(str(fuel), []).append(idx)
            self._lookup_cache["plants_by_fuel_type"] = result
        return self._lookup_cache["plants_by_fuel_type"]  # type: ignore[return-value]

    def _get_plants_by_operator(self) -> Dict[str, List[int]]:
        """Get cached lookup mapping operator name to lists of EIA plant row indices.

        Uses EIA-860M plants only.

        Returns
        -------
        Dict[str, List[int]]
            Dictionary mapping operator name strings to lists of integer row
            indices in ``eia_plants``.

        Examples
        --------
        >>> by_op = infra._get_plants_by_operator()
        >>> ladwp_plants = infra.eia_plants.iloc[by_op.get("Los Angeles DWP", [])]

        """
        if "plants_by_operator" not in self._lookup_cache:
            result: Dict[str, List[int]] = {}
            if LAYER_KEY_EIA_PLANTS in self._urls:
                op_col = self.eia_plants.get("operator", pd.Series())
                for idx, op in enumerate(op_col):
                    if pd.notna(op):
                        result.setdefault(str(op), []).append(idx)
            self._lookup_cache["plants_by_operator"] = result
        return self._lookup_cache["plants_by_operator"]  # type: ignore[return-value]

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def available_layers(self) -> List[str]:
        """Return list of layer keys that have configured URLs.

        Returns
        -------
        List[str]
            Layer keys present in the ``layer_urls`` dict passed at
            construction time.

        Examples
        --------
        >>> infra.available_layers()
        ['eia_plants', 'gem_plants', 'transmission']

        """
        return [key for key in ALL_LAYER_KEYS if key in self._urls]

    def layers_dict(self) -> Dict[str, gpd.GeoDataFrame]:
        """Return all available layers as a dictionary of GeoDataFrames.

        Loads each configured layer lazily on first call. Returns the
        same backing GeoDataFrame objects (no copies), so no additional
        memory cost for layers already loaded.

        Returns
        -------
        Dict[str, gpd.GeoDataFrame]
            Dictionary mapping layer keys to GeoDataFrames for all
            layers with configured URLs. Keys are a subset of:
            ``"eia_plants"``, ``"gem_plants"``, ``"transmission"``,
            ``"substations"``, ``"osm_power"``.

        Examples
        --------
        >>> all_layers = infra.layers_dict()
        >>> for name, gdf in all_layers.items():
        ...     print(name, len(gdf), gdf.crs)

        """
        _property_map = {
            LAYER_KEY_EIA_PLANTS: lambda: self.eia_plants,
            LAYER_KEY_GEM_PLANTS: lambda: self.gem_plants,
            LAYER_KEY_TRANSMISSION: lambda: self.transmission,
            LAYER_KEY_SUBSTATIONS: lambda: self.substations,
            LAYER_KEY_OSM_POWER: lambda: self.osm_power,
        }
        result = {}
        for key in self.available_layers():
            try:
                result[key] = _property_map[key]()
            except RuntimeError as e:
                logger.warning("Skipping layer '%s' in layers_dict: %s", key, e)
        return result

    def preload_all(self) -> None:
        """Preload all configured infrastructure layers into memory.

        Forces immediate loading of all layers with configured URLs and
        builds all lookup caches. Useful when you know all layers will be
        needed and want to avoid per-access latency.

        Layers without configured URLs are silently skipped.

        Examples
        --------
        >>> infra.preload_all()
        >>> # All subsequent property accesses return immediately from cache

        """
        _loaders = [
            (LAYER_KEY_EIA_PLANTS, lambda: self.eia_plants),
            (LAYER_KEY_GEM_PLANTS, lambda: self.gem_plants),
            (LAYER_KEY_TRANSMISSION, lambda: self.transmission),
            (LAYER_KEY_SUBSTATIONS, lambda: self.substations),
            (LAYER_KEY_OSM_POWER, lambda: self.osm_power),
        ]
        for key, loader in _loaders:
            if key in self._urls:
                try:
                    loader()
                    logger.debug("Preloaded infrastructure layer '%s'", key)
                except RuntimeError as e:
                    logger.warning("Failed to preload layer '%s': %s", key, e)

        # Build lookup caches while we're at it
        if LAYER_KEY_EIA_PLANTS in self._urls or LAYER_KEY_GEM_PLANTS in self._urls:
            self._get_plants_by_name()
            if LAYER_KEY_EIA_PLANTS in self._urls:
                self._get_plants_by_fuel_type()
                self._get_plants_by_operator()

    def clear_cache(self) -> None:
        """Clear all cached layer data and lookup dictionaries to free memory.

        Resets all backing stores and the lookup cache. Data will be
        reloaded from the source URLs on next access.

        Examples
        --------
        >>> infra.clear_cache()
        >>> # Next access reloads from URL
        >>> plants = infra.eia_plants

        """
        self.__eia_plants = None
        self.__gem_plants = None
        self.__transmission = None
        self.__substations = None
        self.__osm_power = None
        self._lookup_cache.clear()
        logger.debug("InfrastructureLayers cache cleared.")

    def get_memory_usage(self) -> Dict[str, Union[int, str]]:
        """Get memory usage information for all loaded infrastructure layers.

        Returns a dictionary with per-layer memory estimates and a total.
        Only counts layers that are currently loaded in memory.

        Returns
        -------
        Dict[str, Union[int, str]]
            Dictionary with keys:
            - ``"eia_plants_bytes"`` / ``"gem_plants_bytes"`` / etc.: int bytes
            - ``"total_bytes"``: int, sum of all loaded layer sizes
            - ``"total_human"``: str, human-readable total (e.g. "12.4 MB")

        Examples
        --------
        >>> usage = infra.get_memory_usage()
        >>> print(f"Total memory: {usage['total_human']}")

        """

        def _gdf_bytes(gdf: Optional[gpd.GeoDataFrame]) -> int:
            if gdf is None:
                return 0
            try:
                return int(gdf.memory_usage(deep=True).sum())
            except Exception:
                return 0

        def _human(n: int) -> str:
            for unit in ("B", "KB", "MB", "GB"):
                if n < 1024:
                    return f"{n:.1f} {unit}"
                n //= 1024
            return f"{n:.1f} TB"

        usage = {
            f"{LAYER_KEY_EIA_PLANTS}_bytes": _gdf_bytes(self.__eia_plants),
            f"{LAYER_KEY_GEM_PLANTS}_bytes": _gdf_bytes(self.__gem_plants),
            f"{LAYER_KEY_TRANSMISSION}_bytes": _gdf_bytes(self.__transmission),
            f"{LAYER_KEY_SUBSTATIONS}_bytes": _gdf_bytes(self.__substations),
            f"{LAYER_KEY_OSM_POWER}_bytes": _gdf_bytes(self.__osm_power),
        }
        total = sum(v for v in usage.values() if isinstance(v, int))
        usage["total_bytes"] = total
        usage["total_human"] = _human(total)
        return usage
