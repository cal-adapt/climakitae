"""
Explicitly test the DataCatalog class and its methods.
This module contains unit tests for the DataCatalog class, which is part of the climakitae.new_core.data_access.data_access module.
These tests cover the initialization, default values, update methods, getting data, listing and printing clip boundaries, and resetting.
"""
from unittest.mock import MagicMock, Mock, patch

import pytest
import pandas as pd
import geopandas as gpd
import xarray as xr

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


class TestDataCatalog:

    @pytest.fixture
    def mock_data_catalog(self):
        catalog = DataCatalog()

        # # Create mock DataCatalog
        # mock_instance = Mock()
        # mock_instance[CATALOG_CADCAT] = 
        yield catalog

    # 1. Initialization tests
    def test_catalog_initialization(self, mock_data_catalog):
        assert CATALOG_CADCAT in mock_data_catalog
        assert CATALOG_BOUNDARY in mock_data_catalog
        assert CATALOG_REN_ENERGY_GEN in mock_data_catalog
        assert "stations" in mock_data_catalog
        assert isinstance(mock_data_catalog["stations"], gpd.GeoDataFrame)
        assert isinstance(mock_data_catalog.catalog_df, pd.DataFrame)
        assert set(mock_data_catalog.catalog_df["catalog"].unique()).issubset(
            [CATALOG_REN_ENERGY_GEN, CATALOG_CADCAT]
        )
        assert mock_data_catalog._initialized == True
        assert mock_data_catalog.catalog_key == UNSET

    def test_setting_catalog_key(self, mock_data_catalog):

        # 2. Test setting the catalog key
        mock_data_catalog.set_catalog_key('cadcat')
        assert mock_data_catalog.catalog_key == 'cadcat'

        mock_data_catalog.set_catalog_key('boundary')
        assert mock_data_catalog.catalog_key == 'boundary'

        # 2.1 Testing spelling errors for setting specific catalog key
        typo = 'staaations'
        with warnings.catch_warnings(record=True) as warnings_caught:
            mock_data_catalog.set_catalog_key(typo)
            assert len(warnings_caught) == 2
            assert str(warnings_caught[0].message) == f"\n\nCatalog key '{typo}' not found.\nAttempting to find intended catalog key.\n\n"
            assert str(warnings_caught[1].message) == f"\n\nUsing closest match 'stations' for validator '{typo}'."

        # 2.2 Make sure setting catalog key that is not close to any existing keys fails
        nonexistent_key = 'no keys look like this'
        with warnings.catch_warnings(record=True) as warnings_caught:
            warnings.simplefilter("always")
            with pytest.raises(TypeError, match="object of type 'NoneType' has no len()"):
                mock_data_catalog.set_catalog_key(nonexistent_key)

            assert len(warnings_caught) == 2
            assert str(warnings_caught[0].message) == f"\n\nCatalog key '{nonexistent_key}' not found.\nAttempting to find intended catalog key.\n\n"
            assert str(warnings_caught[1].message) == f"No validator registered for '{nonexistent_key}'. Available options: {list(mock_data_catalog.keys())}"


    def test_set_new_catalog(self, mock_data_catalog):
        mock_data_catalog.set_catalog('new fake catalog', DATA_CATALOG_URL)
        assert 'new fake catalog' in mock_data_catalog.keys()
        assert mock_data_catalog['new fake catalog']

    def test_getting_data(self, mock_data_catalog):
        mock_data_catalog.set_catalog_key('cadcat')
        var_id = 't2max'
        retval = mock_data_catalog.get_data({'variable_id': var_id})

        # Assert that the objects returned are xr.Datasets with the `var_id` as the data variable
        one_ds = list(retval.items())[0]
        assert list(one_ds[1].data_vars)[0] == var_id

    def test_list_clip_boundaries(self, mock_data_catalog):
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


    # 6. Check that something is printed when listing boundaries list, and that the title of what is printed is correct
    def test_print_clip_boundaries(self, mock_data_catalog, capfd):
        mock_data_catalog.print_clip_boundaries()
        out, err = capfd.readouterr()
        assert 'Available Boundary Options for Clipping' in out


    def test_resetting(self, mock_data_catalog):
        # 7. Test resetting
        assert mock_data_catalog.catalog_key == 'cadcat'
        mock_data_catalog.reset()
        assert mock_data_catalog.catalog_key == UNSET


    def test_get_closest_options(self):
        retval = _get_closest_options('firrrsst', ['first', 'second', 'third'])
        assert retval[0] == 'first'

        # Increase the threshold and make sure that the closest option can't be found anymore
        retval = _get_closest_options('firrrsst', ['first', 'second', 'third'], cutoff=0.9)
        assert retval == None

        # Decrease the threshold and see that more options fit given key
        retval = _get_closest_options('firrrsst', ['first', 'second', 'third'], cutoff=0.1)
        assert len(retval) == 3