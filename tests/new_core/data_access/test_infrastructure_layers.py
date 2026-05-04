"""Unit tests for the new_core InfrastructureLayers module.

Tests cover initialization, validation, lazy loading, lookup caches,
public API, and memory management. External I/O (gpd.read_parquet) is
always mocked so tests run without network access.

"""

import warnings
from typing import Dict
from unittest.mock import MagicMock, patch

import geopandas as gpd
import numpy as np
import pandas as pd
import pytest
from shapely.geometry import LineString, Point

from climakitae.new_core.data_access.infrastructure_layers import (
    ALL_LAYER_KEYS,
    LAYER_KEY_EIA_PLANTS,
    LAYER_KEY_GEM_PLANTS,
    LAYER_KEY_OSM_POWER,
    LAYER_KEY_SUBSTATIONS,
    LAYER_KEY_TRANSMISSION,
    InfrastructureLayers,
)


# ---------------------------------------------------------------------------
# Helpers — sample GeoDataFrames
# ---------------------------------------------------------------------------


def _make_plants_gdf(n: int = 3, source: str = "EIA-860M") -> gpd.GeoDataFrame:
    """Return a minimal plants GeoDataFrame with the expected schema."""
    names = [f"Plant {i}" for i in range(n)]
    fuels = ["Solar", "Wind", "NaturalGas"][:n]
    operators = ["PG&E", "SCE", "LADWP"][:n]
    geom = [Point(-120 + i, 36 + i) for i in range(n)]
    return gpd.GeoDataFrame(
        {
            "plant_id": [str(i) for i in range(n)],
            "name": names,
            "fuel_type": fuels,
            "capacity_mw": [100.0 * (i + 1) for i in range(n)],
            "status": ["Operating"] * n,
            "operator": operators,
            "county": ["Fresno", "Kern", "LA"][:n],
            "source": [source] * n,
        },
        geometry=geom,
        crs="EPSG:4326",
    )


def _make_transmission_gdf(n: int = 2) -> gpd.GeoDataFrame:
    """Return a minimal transmission lines GeoDataFrame."""
    geom = [LineString([(-120, 35), (-118, 36)]) for _ in range(n)]
    return gpd.GeoDataFrame(
        {
            "line_id": [str(i) for i in range(n)],
            "voltage_kv": [500.0, 230.0][:n],
            "status": ["In Service"] * n,
            "owner": ["WAPA", "SCE"][:n],
            "source": ["HIFLD"] * n,
        },
        geometry=geom,
        crs="EPSG:4326",
    )


def _make_substations_gdf(n: int = 2) -> gpd.GeoDataFrame:
    """Return a minimal substations GeoDataFrame."""
    geom = [Point(-120 + i, 36) for i in range(n)]
    return gpd.GeoDataFrame(
        {
            "substation_id": [str(i) for i in range(n)],
            "name": [f"Sub {i}" for i in range(n)],
            "voltage_kv": [500.0, 230.0][:n],
            "status": ["In Service"] * n,
            "owner": ["WAPA", "SCE"][:n],
            "source": ["HIFLD"] * n,
        },
        geometry=geom,
        crs="EPSG:4326",
    )


def _make_osm_gdf(n: int = 2) -> gpd.GeoDataFrame:
    """Return a minimal OSM power GeoDataFrame."""
    geom = [Point(-119 + i, 37) for i in range(n)]
    return gpd.GeoDataFrame(
        {
            "osm_id": [str(i) for i in range(n)],
            "name": [f"OSM {i}" for i in range(n)],
            "power_type": ["plant", "substation"][:n],
            "fuel_type": ["Solar", ""] [:n],
            "capacity_mw": [50.0, float("nan")][:n],
            "voltage_kv": [float("nan"), 230.0][:n],
            "operator": ["PG&E", "SCE"][:n],
            "source": ["OSM"] * n,
        },
        geometry=geom,
        crs="EPSG:4326",
    )


def _all_urls() -> Dict[str, str]:
    """Return a dict of all 5 layer URLs (fake paths)."""
    return {
        "eia_plants": "s3://cadcat/eia.parquet",
        "gem_plants": "s3://cadcat/gem.parquet",
        "transmission": "s3://cadcat/tx.parquet",
        "substations": "s3://cadcat/sub.parquet",
        "osm_power": "s3://cadcat/osm.parquet",
    }


# ---------------------------------------------------------------------------
# TestInfrastructureLayersInit
# ---------------------------------------------------------------------------


class TestInfrastructureLayersInit:
    """Tests for InfrastructureLayers.__init__ and validate_urls."""

    def test_init_stores_urls(self):
        """Test that layer_urls are stored on _urls."""
        urls = _all_urls()
        infra = InfrastructureLayers(urls)
        assert infra._urls == urls

    def test_init_backing_stores_are_none(self):
        """Test all backing stores are None immediately after init (lazy)."""
        infra = InfrastructureLayers(_all_urls())
        # Access private backing stores via name-mangled attribute
        assert infra._InfrastructureLayers__eia_plants is None
        assert infra._InfrastructureLayers__gem_plants is None
        assert infra._InfrastructureLayers__transmission is None
        assert infra._InfrastructureLayers__substations is None
        assert infra._InfrastructureLayers__osm_power is None

    def test_init_lookup_cache_empty(self):
        """Test _lookup_cache is empty after init."""
        infra = InfrastructureLayers(_all_urls())
        assert infra._lookup_cache == {}

    def test_validate_urls_warns_on_unknown_key(self):
        """Test that an unknown key in layer_urls triggers a UserWarning."""
        urls = {"eia_plants": "s3://cadcat/eia.parquet", "bogus_key": "s3://cadcat/x"}
        with pytest.warns(UserWarning, match="Unknown layer key 'bogus_key'"):
            InfrastructureLayers(urls)

    def test_validate_urls_no_warning_for_valid_keys(self):
        """Test no UserWarning when all keys are valid."""
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            InfrastructureLayers(_all_urls())
        user_warnings = [x for x in w if issubclass(x.category, UserWarning)]
        assert len(user_warnings) == 0

    def test_init_partial_urls_allowed(self):
        """Test init succeeds with only a subset of layer keys."""
        # No exception should be raised
        infra = InfrastructureLayers({"eia_plants": "s3://cadcat/eia.parquet"})
        assert infra.available_layers() == ["eia_plants"]


# ---------------------------------------------------------------------------
# TestInfrastructureLayersLazyLoad
# ---------------------------------------------------------------------------


class TestInfrastructureLayersLazyLoad:
    """Tests for lazy-loading behavior of each property."""

    def setup_method(self):
        """Set up a standard InfrastructureLayers instance for each test."""
        self.urls = _all_urls()
        self.infra = InfrastructureLayers(self.urls)

    def test_eia_plants_loads_on_first_access(self):
        """Test eia_plants property triggers gpd.read_parquet on first access."""
        sample = _make_plants_gdf()
        with patch(
            "climakitae.new_core.data_access.infrastructure_layers.gpd.read_parquet",
            return_value=sample,
        ) as mock_read:
            result = self.infra.eia_plants
            mock_read.assert_called_once_with(self.urls["eia_plants"])
            assert result is not None
            assert len(result) == 3

    def test_eia_plants_cached_after_first_access(self):
        """Test eia_plants returns cached value on second access (no double load)."""
        sample = _make_plants_gdf()
        with patch(
            "climakitae.new_core.data_access.infrastructure_layers.gpd.read_parquet",
            return_value=sample,
        ) as mock_read:
            _ = self.infra.eia_plants
            _ = self.infra.eia_plants
            assert mock_read.call_count == 1  # loaded only once

    def test_transmission_loads_on_first_access(self):
        """Test transmission property triggers gpd.read_parquet on first access."""
        sample = _make_transmission_gdf()
        with patch(
            "climakitae.new_core.data_access.infrastructure_layers.gpd.read_parquet",
            return_value=sample,
        ):
            result = self.infra.transmission
            assert result is not None
            assert len(result) == 2

    def test_substations_loads_on_first_access(self):
        """Test substations property loads correctly."""
        sample = _make_substations_gdf()
        with patch(
            "climakitae.new_core.data_access.infrastructure_layers.gpd.read_parquet",
            return_value=sample,
        ):
            result = self.infra.substations
            assert len(result) == 2

    def test_gem_plants_loads_on_first_access(self):
        """Test gem_plants property loads correctly."""
        sample = _make_plants_gdf(source="GEM")
        with patch(
            "climakitae.new_core.data_access.infrastructure_layers.gpd.read_parquet",
            return_value=sample,
        ):
            result = self.infra.gem_plants
            assert len(result) == 3

    def test_osm_power_loads_on_first_access(self):
        """Test osm_power property loads correctly."""
        sample = _make_osm_gdf()
        with patch(
            "climakitae.new_core.data_access.infrastructure_layers.gpd.read_parquet",
            return_value=sample,
        ):
            result = self.infra.osm_power
            assert len(result) == 2

    def test_missing_url_raises_runtime_error(self):
        """Test accessing a layer with no URL raises RuntimeError."""
        infra = InfrastructureLayers({})  # no URLs configured
        with pytest.raises(RuntimeError, match="has no URL configured"):
            _ = infra.eia_plants

    def test_read_parquet_failure_raises_runtime_error(self):
        """Test that gpd.read_parquet failure is wrapped in RuntimeError."""
        with patch(
            "climakitae.new_core.data_access.infrastructure_layers.gpd.read_parquet",
            side_effect=FileNotFoundError("not found"),
        ):
            with pytest.raises(RuntimeError, match="Failed to load infrastructure layer"):
                _ = self.infra.eia_plants


# ---------------------------------------------------------------------------
# TestInfrastructureLayersProcessMethods
# ---------------------------------------------------------------------------


class TestInfrastructureLayersProcessMethods:
    """Tests for _process_* methods that run at load time."""

    def setup_method(self):
        """Set up a plain InfrastructureLayers instance."""
        self.infra = InfrastructureLayers(_all_urls())

    def test_process_eia_plants_sorts_by_name(self):
        """Test _process_eia_plants returns GDF sorted by name."""
        unsorted = gpd.GeoDataFrame(
            {"name": ["Zephyr", "Alpha", "Mira"], "fuel_type": ["Wind", "Solar", "NG"]},
            geometry=[Point(0, 0)] * 3,
        )
        result = self.infra._process_eia_plants(unsorted)
        assert list(result["name"]) == ["Alpha", "Mira", "Zephyr"]

    def test_process_gem_plants_sorts_by_name(self):
        """Test _process_gem_plants returns GDF sorted by name."""
        unsorted = gpd.GeoDataFrame(
            {"name": ["Zeta", "Beta"]},
            geometry=[Point(0, 0)] * 2,
        )
        result = self.infra._process_gem_plants(unsorted)
        assert list(result["name"]) == ["Beta", "Zeta"]

    def test_process_transmission_sorts_by_voltage_descending(self):
        """Test _process_transmission returns GDF sorted voltage_kv desc."""
        unsorted = gpd.GeoDataFrame(
            {"voltage_kv": [115.0, 500.0, 230.0]},
            geometry=[LineString([(0, 0), (1, 1)])] * 3,
        )
        result = self.infra._process_transmission(unsorted)
        assert list(result["voltage_kv"]) == [500.0, 230.0, 115.0]

    def test_process_substations_sorts_by_name(self):
        """Test _process_substations returns GDF sorted by name."""
        unsorted = gpd.GeoDataFrame(
            {"name": ["Substation Z", "Substation A"]},
            geometry=[Point(0, 0)] * 2,
        )
        result = self.infra._process_substations(unsorted)
        assert list(result["name"]) == ["Substation A", "Substation Z"]

    def test_process_osm_power_returns_passthrough(self):
        """Test _process_osm_power returns the input unchanged."""
        gdf = _make_osm_gdf()
        result = self.infra._process_osm_power(gdf)
        assert result is gdf  # exact same object

    def test_process_eia_plants_no_name_column_passes_through(self):
        """Test _process_eia_plants handles GDF with no 'name' column."""
        gdf = gpd.GeoDataFrame(
            {"fuel_type": ["Solar"]},
            geometry=[Point(0, 0)],
        )
        result = self.infra._process_eia_plants(gdf)
        assert len(result) == 1  # no error raised


# ---------------------------------------------------------------------------
# TestInfrastructureLayersLookups
# ---------------------------------------------------------------------------


class TestInfrastructureLayersLookups:
    """Tests for lookup dictionary methods."""

    def setup_method(self):
        """Set up InfrastructureLayers with mocked EIA + GEM data."""
        self.infra = InfrastructureLayers(_all_urls())
        # Pre-inject loaded data to avoid gpd.read_parquet calls
        self.eia_gdf = _make_plants_gdf(n=3, source="EIA-860M")
        self.gem_gdf = _make_plants_gdf(n=2, source="GEM")
        self.infra._InfrastructureLayers__eia_plants = (
            self.infra._process_eia_plants(self.eia_gdf)
        )
        self.infra._InfrastructureLayers__gem_plants = (
            self.infra._process_gem_plants(self.gem_gdf)
        )

    def test_get_plants_by_name_returns_dict(self):
        """Test _get_plants_by_name returns a non-empty dict."""
        result = self.infra._get_plants_by_name()
        assert isinstance(result, dict)
        assert len(result) > 0

    def test_get_plants_by_name_eia_prefix(self):
        """Test EIA plant names are prefixed with 'eia:'."""
        result = self.infra._get_plants_by_name()
        eia_keys = [k for k in result if k.startswith("eia:")]
        assert len(eia_keys) == 3  # 3 EIA plants

    def test_get_plants_by_name_gem_prefix(self):
        """Test GEM plant names are prefixed with 'gem:'."""
        result = self.infra._get_plants_by_name()
        gem_keys = [k for k in result if k.startswith("gem:")]
        assert len(gem_keys) == 2  # 2 GEM plants

    def test_get_plants_by_name_is_cached(self):
        """Test second call to _get_plants_by_name returns same dict object."""
        r1 = self.infra._get_plants_by_name()
        r2 = self.infra._get_plants_by_name()
        assert r1 is r2  # same dict object from cache

    def test_get_plants_by_fuel_type_returns_dict(self):
        """Test _get_plants_by_fuel_type returns dict with list values."""
        result = self.infra._get_plants_by_fuel_type()
        assert isinstance(result, dict)
        for v in result.values():
            assert isinstance(v, list)

    def test_get_plants_by_fuel_type_groups_correctly(self):
        """Test _get_plants_by_fuel_type groups EIA plants by fuel type."""
        result = self.infra._get_plants_by_fuel_type()
        # EIA data has Solar, Wind, NaturalGas (one each)
        assert "Solar" in result
        assert len(result["Solar"]) == 1

    def test_get_plants_by_fuel_type_is_cached(self):
        """Test _get_plants_by_fuel_type result is cached."""
        r1 = self.infra._get_plants_by_fuel_type()
        r2 = self.infra._get_plants_by_fuel_type()
        assert r1 is r2

    def test_get_plants_by_operator_returns_dict(self):
        """Test _get_plants_by_operator returns dict with list values."""
        result = self.infra._get_plants_by_operator()
        assert isinstance(result, dict)
        for v in result.values():
            assert isinstance(v, list)

    def test_get_plants_by_operator_groups_correctly(self):
        """Test _get_plants_by_operator groups plants by operator name."""
        result = self.infra._get_plants_by_operator()
        assert "PG&E" in result
        assert len(result["PG&E"]) == 1

    def test_get_plants_by_operator_is_cached(self):
        """Test _get_plants_by_operator result is cached."""
        r1 = self.infra._get_plants_by_operator()
        r2 = self.infra._get_plants_by_operator()
        assert r1 is r2


# ---------------------------------------------------------------------------
# TestInfrastructureLayersPublicAPI
# ---------------------------------------------------------------------------


class TestInfrastructureLayersPublicAPI:
    """Tests for public API methods: layers_dict, available_layers, etc."""

    def setup_method(self):
        """Set up InfrastructureLayers with all 5 URLs."""
        self.infra = InfrastructureLayers(_all_urls())

    def test_available_layers_all_keys(self):
        """Test available_layers returns all 5 keys when all URLs configured."""
        result = self.infra.available_layers()
        assert set(result) == set(ALL_LAYER_KEYS)

    def test_available_layers_partial(self):
        """Test available_layers returns only configured keys."""
        infra = InfrastructureLayers({"eia_plants": "s3://x/y.parquet"})
        assert infra.available_layers() == ["eia_plants"]

    def test_available_layers_empty(self):
        """Test available_layers returns empty list when no URLs given."""
        infra = InfrastructureLayers({})
        assert infra.available_layers() == []

    def test_layers_dict_returns_gdfs(self):
        """Test layers_dict returns a dict of GeoDataFrames."""
        eia_gdf = _make_plants_gdf()
        gem_gdf = _make_plants_gdf(source="GEM")
        tx_gdf = _make_transmission_gdf()
        sub_gdf = _make_substations_gdf()
        osm_gdf = _make_osm_gdf()

        gdfs = {
            "eia_plants": eia_gdf,
            "gem_plants": gem_gdf,
            "transmission": tx_gdf,
            "substations": sub_gdf,
            "osm_power": osm_gdf,
        }

        def _side_effect(url):
            for key, val in self.infra._urls.items():
                if url == val:
                    return gdfs[key]
            raise FileNotFoundError(url)

        with patch(
            "climakitae.new_core.data_access.infrastructure_layers.gpd.read_parquet",
            side_effect=_side_effect,
        ):
            result = self.infra.layers_dict()

        assert set(result.keys()) == set(ALL_LAYER_KEYS)
        for gdf in result.values():
            assert isinstance(gdf, gpd.GeoDataFrame)

    def test_layers_dict_skips_failed_layers(self):
        """Test layers_dict skips layers that fail to load (logs warning)."""
        with patch(
            "climakitae.new_core.data_access.infrastructure_layers.gpd.read_parquet",
            side_effect=FileNotFoundError("not found"),
        ):
            result = self.infra.layers_dict()
        assert result == {}  # all failed, all skipped

    def test_clear_cache_resets_backing_stores(self):
        """Test clear_cache sets all backing stores back to None."""
        # Manually set some data
        self.infra._InfrastructureLayers__eia_plants = _make_plants_gdf()
        self.infra._lookup_cache["some_key"] = {"test": 0}

        self.infra.clear_cache()

        assert self.infra._InfrastructureLayers__eia_plants is None
        assert self.infra._InfrastructureLayers__gem_plants is None
        assert self.infra._InfrastructureLayers__transmission is None
        assert self.infra._InfrastructureLayers__substations is None
        assert self.infra._InfrastructureLayers__osm_power is None
        assert self.infra._lookup_cache == {}

    def test_get_memory_usage_returns_dict(self):
        """Test get_memory_usage returns expected dict structure."""
        usage = self.infra.get_memory_usage()
        assert "total_bytes" in usage
        assert "total_human" in usage
        assert isinstance(usage["total_bytes"], int)
        assert isinstance(usage["total_human"], str)

    def test_get_memory_usage_zero_when_nothing_loaded(self):
        """Test memory usage is 0 before any layers are loaded."""
        usage = self.infra.get_memory_usage()
        assert usage["total_bytes"] == 0

    def test_get_memory_usage_increases_after_loading(self):
        """Test memory usage increases after a layer is loaded."""
        self.infra._InfrastructureLayers__eia_plants = _make_plants_gdf(n=3)
        usage = self.infra.get_memory_usage()
        assert usage["total_bytes"] > 0
        assert usage[f"{LAYER_KEY_EIA_PLANTS}_bytes"] > 0

    def test_preload_all_loads_all_configured_layers(self):
        """Test preload_all calls gpd.read_parquet for each configured layer."""
        gdfs = {
            "eia_plants": _make_plants_gdf(),
            "gem_plants": _make_plants_gdf(source="GEM"),
            "transmission": _make_transmission_gdf(),
            "substations": _make_substations_gdf(),
            "osm_power": _make_osm_gdf(),
        }

        def _side_effect(url):
            for key, val in self.infra._urls.items():
                if url == val:
                    return gdfs[key]
            raise FileNotFoundError(url)

        with patch(
            "climakitae.new_core.data_access.infrastructure_layers.gpd.read_parquet",
            side_effect=_side_effect,
        ) as mock_read:
            self.infra.preload_all()
            assert mock_read.call_count == 5

    def test_preload_all_populates_lookup_caches(self):
        """Test preload_all also builds lookup caches."""
        gdfs = {
            "eia_plants": _make_plants_gdf(),
            "gem_plants": _make_plants_gdf(source="GEM"),
            "transmission": _make_transmission_gdf(),
            "substations": _make_substations_gdf(),
            "osm_power": _make_osm_gdf(),
        }

        def _side_effect(url):
            for key, val in self.infra._urls.items():
                if url == val:
                    return gdfs[key]
            raise FileNotFoundError(url)

        with patch(
            "climakitae.new_core.data_access.infrastructure_layers.gpd.read_parquet",
            side_effect=_side_effect,
        ):
            self.infra.preload_all()

        assert "plants_by_name" in self.infra._lookup_cache
        assert "plants_by_fuel_type" in self.infra._lookup_cache
        assert "plants_by_operator" in self.infra._lookup_cache
