"""
Explicitly test the DataCatalog class and its methods.
This module contains unit tests for the DataCatalog class, which is part of the climakitae.new_core.data_access.data_access module.
These tests cover the initialization, default values, update methods, getting data, listing and printing clip boundaries, and resetting.
"""

from typing import Generator, Tuple
from unittest.mock import MagicMock, PropertyMock, patch

import geopandas as gpd
import pandas as pd
import pytest
import xarray as xr

from climakitae.core.paths import (
    BOUNDARY_CATALOG_URL,
    DATA_CATALOG_URL,
    RENEWABLES_CATALOG_URL,
)
from climakitae.new_core.data_access.data_access import (
    CATALOG_BOUNDARY,
    CATALOG_CADCAT,
    CATALOG_REN_ENERGY_GEN,
    UNSET,
    DataCatalog,
    _get_closest_options,
)


def make_mock_objects() -> Tuple[MagicMock, MagicMock, MagicMock, pd.DataFrame]:
    """Create mock objects for testing DataCatalog."""
    # ESM catalog
    mock_esm_catalog = MagicMock()
    mock_esm_catalog.df = pd.DataFrame(
        {"id": [1, 2, 3], "name": ["Alice", "Bob", "Charlie"], "age": [25, 30, 35]}
    )

    mock_subset = MagicMock()
    mock_esm_catalog.search.return_value = mock_subset

    ds1 = xr.Dataset(
        data_vars={"tasmax": (["time"], [1.0, 2.0])},
        coords={"time": [0, 1]},
        attrs={"source_id": "MODEL1", "experiment_id": "historical"},
    )
    ds2 = xr.Dataset(
        data_vars={"tasmax": (["time"], [3.0, 4.0])},
        coords={"time": [0, 1]},
        attrs={"source_id": "MODEL2", "experiment_id": "historical"},
    )
    mock_subset.to_dataset_dict.return_value = {"dataset1": ds1, "dataset2": ds2}

    # Boundary catalog
    mock_boundary_catalog = MagicMock()
    mock_boundaries = MagicMock()
    boundary_dict = {
        "none": {"entire domain": 0},
        "states": {"NV": 28, "CA": 16},
        "CA watersheds": {
            "Antelope-Fremont Valleys": 11,
            "Aliso-San Onofre": 128,
        },
        "lat/lon": {"coordinate selection": 0},
        "CA counties": {"Alameda County": 54, "Alpine County": 25},
        "CA Electric Load Serving Entities (IOU & POU)": {
            "San Diego Gas & Electric": 3,
            "Pacific Gas & Electric Company": 2,
        },
        "CA Electricity Demand Forecast Zones": {
            "LADWP Coastal": 0,
            "Greater Bay Area": 1,
        },
        "CA Electric Balancing Authority Areas": {"BANC": 0, "CALISO": 1},
    }
    mock_boundaries.boundary_dict.return_value = boundary_dict

    # Stations dataframe
    mock_stations_df = pd.DataFrame(
        {
            "LON_X": [-120.0, -121.0],
            "LAT_Y": [35.0, 36.0],
            "station_name": ["Station1", "Station2"],
        }
    )

    return mock_esm_catalog, mock_boundary_catalog, mock_boundaries, mock_stations_df


@pytest.fixture
def mock_data_catalog_and_objs() -> Generator:
    """Fixture to provide a mock DataCatalog instance along with its dependencies."""
    mock_esm_catalog, mock_boundary_catalog, mock_boundaries, mock_stations_df = (
        make_mock_objects()
    )

    with patch(
        "climakitae.new_core.data_access.data_access.intake.open_esm_datastore",
        return_value=mock_esm_catalog,
    ) as mock_open_esm, patch(
        "climakitae.new_core.data_access.data_access.intake.open_catalog",
        return_value=mock_boundary_catalog,
    ) as mock_open_catalog, patch(
        "climakitae.new_core.data_access.data_access.read_csv_file",
        return_value=mock_stations_df,
    ):
        # Reset singleton for clean test state
        DataCatalog._instance = UNSET
        catalog_instance = DataCatalog()

        with patch.object(
            type(catalog_instance),
            "boundaries",
            new_callable=PropertyMock,
            return_value=mock_boundaries,
        ):
            yield catalog_instance, mock_esm_catalog, mock_boundary_catalog, mock_stations_df, mock_open_esm, mock_open_catalog


@pytest.fixture
def mock_data_catalog(mock_data_catalog_and_objs: Generator) -> DataCatalog:
    """Fixture to provide a mock DataCatalog instance."""
    catalog_instance, *_ = mock_data_catalog_and_objs
    return catalog_instance


class TestDataCatalogInitialization:
    """Test the initialization and basic properties of the DataCatalog class."""

    def test_intake_functions_called(self, mock_data_catalog_and_objs: Tuple):
        """Intake functions should be called with correct URLs during initialization."""
        *_, mock_open_esm, mock_open_catalog = mock_data_catalog_and_objs

        # Intake functions called with expected args
        assert mock_open_esm.call_count == 2
        mock_open_esm.assert_any_call(DATA_CATALOG_URL)
        mock_open_esm.assert_any_call(RENEWABLES_CATALOG_URL)
        mock_open_catalog.assert_called_once_with(BOUNDARY_CATALOG_URL)

    def test_contains_expected_catalog_keys(self, mock_data_catalog_and_objs: Tuple):
        """The DataCatalog should store catalogs under expected keys after initialization."""
        catalog_instance, mock_esm_catalog, mock_boundary_catalog, *_ = (
            mock_data_catalog_and_objs
        )

        # Catalogs stored correctly
        assert catalog_instance[CATALOG_CADCAT] == mock_esm_catalog
        assert catalog_instance[CATALOG_BOUNDARY] == mock_boundary_catalog
        assert catalog_instance[CATALOG_REN_ENERGY_GEN] == mock_esm_catalog

    def test_contains_expected_entries(self, mock_data_catalog_and_objs: Tuple):
        """The DataCatalog should contain expected entries after initialization."""
        catalog_instance, *_ = mock_data_catalog_and_objs

        # Contains expected entries
        for key in (
            CATALOG_CADCAT,
            CATALOG_BOUNDARY,
            CATALOG_REN_ENERGY_GEN,
            "stations",
        ):
            assert key in catalog_instance

    def test_stations_is_gdf(self, mock_data_catalog_and_objs: Tuple):
        """The 'stations' entry should be a GeoDataFrame."""
        catalog_instance, *_ = mock_data_catalog_and_objs
        stations = catalog_instance["stations"]
        assert isinstance(stations, gpd.GeoDataFrame)

    def test_catalog_df_structure(self, mock_data_catalog_and_objs: Tuple):
        """The catalog_df property should return a DataFrame with expected structure."""
        catalog_instance, *_ = mock_data_catalog_and_objs
        df = catalog_instance.catalog_df

        assert isinstance(df, pd.DataFrame)
        assert set(df["catalog"].unique()).issubset(
            [CATALOG_REN_ENERGY_GEN, CATALOG_CADCAT]
        )

    def test_initialized_state(self, mock_data_catalog_and_objs: Tuple):
        """Test that the DataCatalog instance is marked as initialized after creation."""
        catalog_instance, *_ = mock_data_catalog_and_objs

        assert catalog_instance._initialized is True
        assert catalog_instance.catalog_key == UNSET

    def test_singleton_pattern(self):
        """Test that DataCatalog truly implements singleton pattern."""
        catalog1 = DataCatalog()
        catalog2 = DataCatalog()
        assert catalog1 is catalog2  # Same object reference
        # delete the objects after test to avoid side effects
        del catalog1
        del catalog2

    def test_data_property_returns_cadcat_catalog(
        self, mock_data_catalog_and_objs: Tuple
    ):
        """The .data property should return the CADCAT catalog."""
        catalog_instance, *_ = mock_data_catalog_and_objs
        assert catalog_instance.data is catalog_instance[CATALOG_CADCAT]

    def test_boundaries_property_lazy_loads(self, mock_data_catalog_and_objs: Tuple):
        """The .boundaries property should lazy-load Boundaries object."""
        catalog_instance, *_ = mock_data_catalog_and_objs
        assert catalog_instance._boundaries is UNSET
        boundaries = catalog_instance.boundaries
        assert catalog_instance.boundaries is boundaries  # Same object on second access

    def test_get_data_queries_correct_catalog(self, mock_data_catalog_and_objs: Tuple):
        """Test that get_data queries the correct catalog based on catalog_key."""

        catalog_instance, mock_esm_catalog, *_ = mock_data_catalog_and_objs

        # Setup mock to track calls
        mock_search = MagicMock()
        mock_to_dataset = MagicMock(return_value={"key": xr.Dataset()})
        mock_search.to_dataset_dict = mock_to_dataset
        mock_esm_catalog.search = MagicMock(return_value=mock_search)

        catalog_instance.set_catalog_key("cadcat")
        _ = catalog_instance.get_data({"variable_id": "t2max"})

        # Verify the chain of calls
        mock_esm_catalog.search.assert_called_once_with(variable_id="t2max")
        mock_to_dataset.assert_called_once()


class TestDataCatalogCatalogKeyManagement:
    """Test setting and getting `catalog_key` in DataCatalog."""

    def test_get_data_with_unset_catalog_key(self, mock_data_catalog: DataCatalog):
        """Getting data with UNSET catalog_key should raise error."""
        with pytest.raises(KeyError):
            mock_data_catalog.get_data({"variable_id": "test"})

    @pytest.mark.parametrize("valid_key", ["cadcat", "boundary"])
    def test_setting_valid_catalog_key(
        self, mock_data_catalog: DataCatalog, valid_key: str
    ):
        """Setting a valid catalog key updates the catalog_key attribute."""
        mock_data_catalog.set_catalog_key(valid_key)
        assert mock_data_catalog.catalog_key == valid_key

    def test_setting_catalog_key_with_typo_warns_and_fixes(
        self, mock_data_catalog: DataCatalog
    ):
        """Setting a misspelled key should warn and fallback to closest match."""
        typo = "staaations"

        with pytest.warns(UserWarning) as record:
            mock_data_catalog.set_catalog_key(typo)

        assert len(record) == 2
        assert str(record[0].message) == (
            f"\n\nCatalog key '{typo}' not found.\n"
            f"Attempting to find intended catalog key.\n\n"
        )
        assert str(record[1].message) == (
            f"\n\nUsing closest match 'stations' for validator '{typo}'."
        )
        assert mock_data_catalog.catalog_key == "stations"

    def test_setting_catalog_key_with_ambiguous_match_warns(
        self, mock_data_catalog: DataCatalog
    ):
        """Ambiguous misspelling should warn about multiple possible matches."""
        too_similar_key = "cadctation"

        with pytest.warns(UserWarning) as record:
            mock_data_catalog.set_catalog_key(too_similar_key)

        assert len(record) == 2
        assert str(record[0].message) == (
            f"\n\nCatalog key '{too_similar_key}' not found.\n"
            f"Attempting to find intended catalog key.\n\n"
        )
        assert str(record[1].message) == (
            f"Multiple closest matches found for '{too_similar_key}': "
            f"{['cadcat', 'stations']}. Please specify a more precise key."
        )

    def test_helpful_error_on_invalid_catalog_key(
        self, mock_data_catalog: DataCatalog, capfd: pytest.CaptureFixture
    ):
        """Invalid catalog keys should produce helpful error messages."""
        with pytest.warns(UserWarning) as warning_info:
            mock_data_catalog.set_catalog_key("nonexistent")

        # Check that the print statement and warnings helps the user
        out, _ = capfd.readouterr()

        assert "Available catalog keys" in out
        assert "Available options:" in str(warning_info[1].message)
        assert "cadcat" in str(warning_info[1].message)

    def test_set_new_catalog(self, mock_data_catalog: DataCatalog):
        """Test `set_catalog` adds a new catalog to the DataCatalog instance."""
        assert "new fake catalog" not in mock_data_catalog.keys()
        mock_data_catalog.set_catalog("new fake catalog", DATA_CATALOG_URL)
        assert "new fake catalog" in mock_data_catalog.keys()
        assert mock_data_catalog["new fake catalog"]

    def test_list_clip_boundaries_excludes_special_keys(
        self, mock_data_catalog: DataCatalog
    ):
        """Test `list_clip_boundaries` excludes special keys like 'none' and 'lat/lon'."""
        clip_boundaries = mock_data_catalog.list_clip_boundaries()
        mock_data_catalog.boundaries.boundary_dict.assert_called_once()
        assert "none" not in clip_boundaries
        assert "lat/lon" not in clip_boundaries

    def test_list_clip_boundaries_contains_expected_keys(
        self, mock_data_catalog: DataCatalog
    ):
        """Test `list_clip_boundaries` contains expected boundary keys."""
        clip_boundaries = mock_data_catalog.list_clip_boundaries()
        expected_names = [
            "states",
            "CA counties",
            "CA watersheds",
            "CA Electric Load Serving Entities (IOU & POU)",
            "CA Electricity Demand Forecast Zones",
            "CA Electric Balancing Authority Areas",
        ]
        assert all([k in expected_names for k in list(clip_boundaries.keys())])

    def test_list_clip_boundaries_returns_sorted_values(
        self, mock_data_catalog: DataCatalog
    ):
        """Test `list_clip_boundaries` returns sorted values for each key."""
        clip_boundaries = mock_data_catalog.list_clip_boundaries()
        assert clip_boundaries["states"] == ["CA", "NV"]
        assert clip_boundaries["CA Electricity Demand Forecast Zones"] == [
            "Greater Bay Area",
            "LADWP Coastal",
        ]

    def test_list_clip_boundaries_sets_available_boundaries_property(
        self, mock_data_catalog: DataCatalog
    ):
        """Test `list_clip_boundaries` sets the `available_boundaries` property."""
        mock_data_catalog.list_clip_boundaries()
        assert hasattr(mock_data_catalog, "available_boundaries")

    def test_print_clip_boundaries(
        self, mock_data_catalog: DataCatalog, capfd: pytest.CaptureFixture
    ) -> None:
        """Test `print_clip_boundaries` prints the available boundary options."""
        mock_data_catalog.print_clip_boundaries()
        out, err = capfd.readouterr()
        assert "Available Boundary Options for Clipping" in out

    def test_resetting(self, mock_data_catalog: DataCatalog) -> None:
        """
        Test the resetting functionality of the mock_data_catalog.

        This test verifies that:
        1. The initial state of the catalog_key is UNSET.
        2. The catalog_key can be set to a specific value ("cadcat").
        3. The reset method restores the catalog_key to its initial state (UNSET).
        """
        assert mock_data_catalog.catalog_key == UNSET
        mock_data_catalog.set_catalog_key("cadcat")
        assert mock_data_catalog.catalog_key == "cadcat"
        mock_data_catalog.reset()
        assert mock_data_catalog.catalog_key == UNSET

    @pytest.mark.parametrize(
        "key, options, cutoff, expected",
        [
            ("firrrsst", ["first", "second", "third"], 0.6, ["first"]),
            ("First", ["first", "second", "third"], 0.6, ["first"]),
            ("fir", ["first", "second", "third"], 0.6, ["first"]),
            ("firrrsst", ["first", "second", "third"], 0.9, None),
            (
                "firrrsst",
                ["first", "second", "third"],
                0.1,
                ["first", "second", "third"],
            ),
        ],
    )
    def test_get_closest_options(
        self, key: str, options: list[str], cutoff: float, expected: list[str] | None
    ) -> None:
        """Test `_get_closest_options` for fuzzy matches."""
        retval = _get_closest_options(key, options, cutoff=cutoff)

        if expected is None:
            assert retval is None
        else:
            assert (
                set(retval) == set(expected)
                if len(expected) > 1
                else retval[0] == expected[0]
            )
