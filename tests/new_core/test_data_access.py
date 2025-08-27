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


class TestDataCatalog:

    @pytest.fixture
    def mock_data_catalog_and_objs(self):
        # Create mock objects
        mock_esm_catalog = MagicMock()
        mock_esm_catalog.df = pd.DataFrame(
            {"variable_id": ["t2max", "tasmin"]})

        mock_boundary_catalog = {
            'states': ['CA'], 
            'CA counties': ['Alameda County'], 
            'CA watersheds': ['Hello1'], 
            'CA Electric Load Serving Entities (IOU & POU)': ['Hello2'], 
            'CA Electricity Demand Forecast Zones': ['Hello3'], 
            'CA Electric Balancing Authority Areas': ['Hello4']
        }

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
            yield catalog_instance, mock_esm_catalog, mock_boundary_catalog, mock_stations_df, mock_open_esm, mock_open_catalog


    def test_catalog_initialization(self, mock_data_catalog_and_objs):

        mock_data_catalog, mock_esm_catalog, mock_boundary_catalog, mock_stations_df, mock_open_esm, mock_open_catalog = mock_data_catalog_and_objs

        # Verify the intake functions were called correctly
        assert mock_open_esm.call_count == 2
        mock_open_esm.assert_any_call(DATA_CATALOG_URL)
        mock_open_esm.assert_any_call(RENEWABLES_CATALOG_URL)
        mock_open_catalog.assert_called_once_with(BOUNDARY_CATALOG_URL)

        # Verify DataCatalog works
        assert "stations" in mock_data_catalog

        # Verify catalogs are stored correctly
        assert mock_data_catalog[CATALOG_CADCAT] == mock_esm_catalog
        assert mock_data_catalog[CATALOG_BOUNDARY] == mock_boundary_catalog
        assert mock_data_catalog[CATALOG_REN_ENERGY_GEN] == mock_esm_catalog

        # Assert objects are instantiated correctly
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


    def test_setting_catalog_key(self, mock_data_catalog_and_objs):

        mock_data_catalog, mock_esm_catalog, mock_boundary_catalog, mock_stations_df, mock_open_esm, mock_open_catalog = mock_data_catalog_and_objs

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

        # 2.3 Try setting catalog key that is too close to multiple existing keys
        too_similar_key = 'cadctation'
        mock_data_catalog.set_catalog_key(too_similar_key)
        with warnings.catch_warnings(record=True) as warnings_caught:
            warnings.simplefilter("always")
            mock_data_catalog.set_catalog_key(too_similar_key)

            assert len(warnings_caught) == 2
            assert str(warnings_caught[0].message) == f"\n\nCatalog key '{too_similar_key}' not found.\nAttempting to find intended catalog key.\n\n"
            assert str(warnings_caught[1].message) == f"Multiple closest matches found for '{too_similar_key}': {['cadcat', 'stations']}. Please specify a more precise key."

    def test_set_new_catalog(self, mock_data_catalog_and_objs):

        mock_data_catalog, mock_esm_catalog, mock_boundary_catalog, mock_stations_df, mock_open_esm, mock_open_catalog = mock_data_catalog_and_objs

        mock_data_catalog.set_catalog('new fake catalog', DATA_CATALOG_URL)
        assert 'new fake catalog' in mock_data_catalog.keys()
        assert mock_data_catalog['new fake catalog']


    def test_getting_data(self, mock_data_catalog_and_objs):

        mock_data_catalog, mock_esm_catalog, mock_boundary_catalog, mock_stations_df, mock_open_esm, mock_open_catalog = mock_data_catalog_and_objs

        mock_data_catalog.set_catalog_key('cadcat')

        var_id = 't2max'
        retval = mock_data_catalog.get_data({'variable_id': var_id})

        # Assert that the objects returned are xr.Datasets with the `var_id` as the data variable
        one_ds = list(retval.items())[0]
        assert list(one_ds[1].data_vars)[0] == var_id


    def test_list_clip_boundaries(self, mock_data_catalog_and_objs):

        mock_data_catalog, mock_esm_catalog, mock_boundary_catalog, mock_stations_df, mock_open_esm, mock_open_catalog = mock_data_catalog_and_objs

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
    def test_print_clip_boundaries(self, mock_data_catalog_and_objs, capfd):

        mock_data_catalog, mock_esm_catalog, mock_boundary_catalog, mock_stations_df, mock_open_esm, mock_open_catalog = mock_data_catalog_and_objs

        mock_data_catalog.print_clip_boundaries()
        out, err = capfd.readouterr()
        assert 'Available Boundary Options for Clipping' in out


    def test_resetting(self, mock_data_catalog_and_objs):

        mock_data_catalog, mock_esm_catalog, mock_boundary_catalog, mock_stations_df, mock_open_esm, mock_open_catalog = mock_data_catalog_and_objs

        assert mock_data_catalog.catalog_key == UNSET
        mock_data_catalog.set_catalog_key('cadcat')
        assert mock_data_catalog.catalog_key == 'cadcat'
        mock_data_catalog.reset()
        assert mock_data_catalog.catalog_key == UNSET


    def test_get_closest_options(self):

        retval = _get_closest_options('firrrsst', ['first', 'second', 'third'])
        assert retval[0] == 'first'

        retval = _get_closest_options('First', ['first', 'second', 'third'])
        assert retval[0] == 'first'

        retval = _get_closest_options('fir', ['first', 'second', 'third'])
        assert retval[0] == 'first'

        # Increase the threshold and make sure that the closest option can't be found anymore
        retval = _get_closest_options('firrrsst', ['first', 'second', 'third'], cutoff=0.9)
        assert retval == None

        # Decrease the threshold and see that more options fit given key
        retval = _get_closest_options('firrrsst', ['first', 'second', 'third'], cutoff=0.1)
        assert len(retval) == 3