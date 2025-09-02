"""
Explicitly test the DataCatalog class and its methods.
This module contains unit tests for the DataCatalog class, which is part of the climakitae.new_core.data_access.data_access module.
These tests cover the initialization, default values, update methods, getting data, listing and printing clip boundaries, and resetting.
"""
from unittest.mock import MagicMock, Mock, patch, PropertyMock

import pytest
import pandas as pd
import geopandas as gpd
import xarray as xr
import warnings

from climakitae.new_core.data_access.data_access import (
    DataCatalog,
    CATALOG_BOUNDARY,
    CATALOG_CADCAT,
    CATALOG_REN_ENERGY_GEN,
    UNSET,
    _get_closest_options,
)

from climakitae.core.paths import (
    BOUNDARY_CATALOG_URL,
    DATA_CATALOG_URL,
    RENEWABLES_CATALOG_URL,
    STATIONS_CSV_PATH,
)
from climakitae.new_core.data_access.boundaries import Boundaries
import intake

def create_mock_objs():
    
    # Create mock objects
    mock_esm_catalog = MagicMock()
    mock_esm_catalog.df = pd.DataFrame({
        "id": [1, 2, 3],
        "name": ["Alice", "Bob", "Charlie"],
        "age": [25, 30, 35]
    })

    mock_subset = MagicMock()
    mock_esm_catalog.search.return_value = mock_subset

    # Create mock dataset dictionary with two xr.Datasets dataset
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
    mock_data_dict = {"dataset1": ds1, "dataset2": ds2}
    mock_subset.to_dataset_dict.return_value = mock_data_dict
    
    # Create mock boundary catalog
    mock_boundary_catalog = MagicMock(spec=intake.catalog.Catalog)
    mock_boundaries = MagicMock()
    boundary_dict = {
        'none': {'entire domain': 0},
        'lat/lon': {'coordinate selection': 0},
        'states': {'CA': 16, 'NV': 28},
        'CA counties': {'Alameda County': 54, 'Alpine County': 25},
        'CA watersheds': {'Aliso-San Onofre': 128, 'Antelope-Fremont Valleys': 11},
        'CA Electric Load Serving Entities (IOU & POU)': {'Pacific Gas & Electric Company': 2,
        'San Diego Gas & Electric': 3},
        'CA Electricity Demand Forecast Zones': {'LADWP Coastal': 0,
        'Greater Bay Area': 1},
        'CA Electric Balancing Authority Areas': {'BANC': 0, 'CALISO': 1},

    }
    mock_boundaries.boundary_dict.return_value = boundary_dict

    # Create mock stations df
    mock_stations_df = pd.DataFrame({
        "LON_X": [-120.0, -121.0],
        "LAT_Y": [35.0, 36.0],
        "station_name": ["Station1", "Station2"]
    })

    with patch(
        "climakitae.new_core.data_access.data_access.intake.open_esm_datastore",
        return_value=mock_esm_catalog
    ) as mock_open_esm, patch(
        "climakitae.new_core.data_access.data_access.intake.open_catalog",
        return_value=mock_boundary_catalog
    ) as mock_open_catalog, patch(
        "climakitae.new_core.data_access.data_access.read_csv_file",
        return_value=mock_stations_df
    ):
        # Reset singleton
        DataCatalog._instance = UNSET
        catalog_instance = DataCatalog()

        with patch.object(
            type(catalog_instance),
            "boundaries",
            new_callable=PropertyMock,
            return_value=mock_boundaries,
        ):
            return catalog_instance, mock_esm_catalog, mock_boundary_catalog, mock_stations_df, mock_open_esm, mock_open_catalog


@pytest.fixture
def mock_data_catalog_and_objs():
    return create_mock_objs()


@pytest.fixture
def mock_data_catalog():
    mock_data_catalog, *_ = create_mock_objs()
    return mock_data_catalog


class TestDataCatalogInitialization:

    def test_intake_functions_called(self, mock_data_catalog_and_objs):
        mock_data_catalog, _, _, _, mock_open_esm, mock_open_catalog = mock_data_catalog_and_objs

        # Intake functions called with expected args
        assert mock_open_esm.call_count == 2
        mock_open_esm.assert_any_call(DATA_CATALOG_URL)
        mock_open_esm.assert_any_call(RENEWABLES_CATALOG_URL)
        mock_open_catalog.assert_called_once_with(BOUNDARY_CATALOG_URL)

    def test_contains_expected_catalog_keys(self, mock_data_catalog_and_objs):
        mock_data_catalog, mock_esm_catalog, mock_boundary_catalog, *_ = mock_data_catalog_and_objs

        # Catalogs stored correctly
        assert mock_data_catalog[CATALOG_CADCAT] == mock_esm_catalog
        assert mock_data_catalog[CATALOG_BOUNDARY] == mock_boundary_catalog
        assert mock_data_catalog[CATALOG_REN_ENERGY_GEN] == mock_esm_catalog

    def test_contains_expected_entries(self, mock_data_catalog_and_objs):
        mock_data_catalog, *_ = mock_data_catalog_and_objs

        # Contains expected entries
        for key in (CATALOG_CADCAT, CATALOG_BOUNDARY, CATALOG_REN_ENERGY_GEN, "stations"):
            assert key in mock_data_catalog

    def test_stations_is_gdf(self, mock_data_catalog_and_objs):
        mock_data_catalog, *_ = mock_data_catalog_and_objs
        stations = mock_data_catalog["stations"]
        assert isinstance(stations, gpd.GeoDataFrame)

    def test_catalog_df_structure(self, mock_data_catalog_and_objs):
        mock_data_catalog, *_ = mock_data_catalog_and_objs
        df = mock_data_catalog.catalog_df

        assert isinstance(df, pd.DataFrame)
        assert set(df["catalog"].unique()).issubset([CATALOG_REN_ENERGY_GEN, CATALOG_CADCAT])

    def test_initialized_state(self, mock_data_catalog_and_objs):
        mock_data_catalog, *_ = mock_data_catalog_and_objs

        assert mock_data_catalog._initialized is True
        assert mock_data_catalog.catalog_key == UNSET

    def test_singleton_pattern(self):
        """Test that DataCatalog truly implements singleton pattern."""
        catalog1 = DataCatalog()
        catalog2 = DataCatalog()
        assert catalog1 is catalog2  # Same object reference

    def test_data_property_returns_cadcat_catalog(self, mock_data_catalog_and_objs):
        """The .data property should return the CADCAT catalog."""
        mock_data_catalog, *_ = mock_data_catalog_and_objs
        assert mock_data_catalog.data is mock_data_catalog[CATALOG_CADCAT]

    def test_boundaries_property_lazy_loads(self, mock_data_catalog_and_objs):
        """The .boundaries property should lazy-load Boundaries object."""
        mock_data_catalog, *_ = mock_data_catalog_and_objs
        # mock_data_catalog = DataCatalog()
        assert mock_data_catalog._boundaries is UNSET
        boundaries = mock_data_catalog.boundaries  # First access
        assert mock_data_catalog._boundaries is not UNSET
        assert mock_data_catalog.boundaries is boundaries  # Same object on second access

    def test_get_data_queries_correct_catalog(self, mock_data_catalog_and_objs):

        mock_data_catalog, mock_esm_catalog, *_ = mock_data_catalog_and_objs

        # Setup mock to track calls
        mock_search = MagicMock()
        mock_to_dataset = MagicMock(return_value={"key": xr.Dataset()})
        mock_search.to_dataset_dict = mock_to_dataset
        mock_esm_catalog.search = MagicMock(return_value=mock_search)
        
        mock_data_catalog.set_catalog_key("cadcat")
        result = mock_data_catalog.get_data({"variable_id": "t2max"})
        
        # Verify the chain of calls
        mock_esm_catalog.search.assert_called_once_with(variable_id="t2max")
        mock_to_dataset.assert_called_once()



class TestDataCatalogCatalogKeyManagement:
    
    def test_get_data_with_unset_catalog_key(self, mock_data_catalog):
        with pytest.raises(KeyError):
            mock_data_catalog.get_data({'variable_id': 'test'})

    @pytest.mark.parametrize("valid_key", ["cadcat", "boundary"])
    def test_setting_valid_catalog_key(self, mock_data_catalog, valid_key):
        """Setting a valid catalog key updates the catalog_key attribute."""
        mock_data_catalog.set_catalog_key(valid_key)
        assert mock_data_catalog.catalog_key == valid_key

    def test_setting_catalog_key_with_typo_warns_and_fixes(self, mock_data_catalog):
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

    def test_setting_catalog_key_with_ambiguous_match_warns(self, mock_data_catalog):
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

    def test_helpful_error_on_invalid_catalog_key(self, mock_data_catalog, capfd):
        """Invalid catalog keys should produce helpful error messages."""
        with pytest.warns(UserWarning) as warning_info:
            mock_data_catalog.set_catalog_key("nonexistent")

        # Check that the print statement and warnings helps the user
        out, _ = capfd.readouterr()

        assert "Available catalog keys" in out
        assert "Available options:" in str(warning_info[1].message) 
        assert "cadcat" in str(warning_info[1].message) 


    def test_set_new_catalog(self, mock_data_catalog):

        mock_data_catalog.set_catalog('new fake catalog', DATA_CATALOG_URL)
        assert 'new fake catalog' in mock_data_catalog.keys()
        assert mock_data_catalog['new fake catalog']

    def test_list_clip_boundaries(self, mock_data_catalog):

        # Check that this is being called, check that return types are correct, check that special values are being filtered (none, lat/lon)
        # Make sure that they sort after they come out
        # Check that available_boundaries property is set on the catalog

        clip_boundaries = mock_data_catalog.list_clip_boundaries()

        expected_names = [
            'states',
            'CA counties',
            'CA watersheds',
            'CA Electric Load Serving Entities (IOU & POU)',
            'CA Electricity Demand Forecast Zones',
            'CA Electric Balancing Authority Areas'
        ]
        assert all (k in expected_names for k in list(clip_boundaries.keys()))

    def test_print_clip_boundaries(self, mock_data_catalog, capfd):

        mock_data_catalog.print_clip_boundaries()
        out, err = capfd.readouterr()
        assert 'Available Boundary Options for Clipping' in out

    def test_resetting(self, mock_data_catalog):

        assert mock_data_catalog.catalog_key == UNSET
        mock_data_catalog.set_catalog_key('cadcat')
        assert mock_data_catalog.catalog_key == 'cadcat'
        mock_data_catalog.reset()
        assert mock_data_catalog.catalog_key == UNSET

    @pytest.mark.parametrize(
        "key, options, cutoff, expected",
        [
            ("firrrsst", ["first", "second", "third"], 0.6, ["first"]),  # fuzzy match
            ("First", ["first", "second", "third"], 0.6, ["first"]),     # case-insensitive
            ("fir", ["first", "second", "third"], 0.6, ["first"]),       # short prefix
            ("firrrsst", ["first", "second", "third"], 0.9, None),       # too strict cutoff
            ("firrrsst", ["first", "second", "third"], 0.1, ["first", "second", "third"]),  # very loose cutoff
        ],
    )
    def test_get_closest_options(self, key, options, cutoff, expected):
        """_get_closest_options should return best fuzzy matches based on cutoff."""
        retval = _get_closest_options(key, options, cutoff=cutoff)

        if expected is None:
            assert retval is None
        else:
            # Compare sets if multiple, otherwise just check first element
            if len(expected) == 1:
                assert retval[0] == expected[0]
            else:
                assert set(retval) == set(expected)
