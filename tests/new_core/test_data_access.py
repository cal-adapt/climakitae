"""
Explicitly test the DataCatalog class and its methods.
This module contains unit tests for the DataCatalog class, which is part of the climakitae.new_core.data_access.data_access module.
These tests cover the initialization, default values, update methods, getting data, listing and printing clip boundaries, and resetting.
"""
import pytest
import pandas as pd
import geopandas as gpd
import xarray as xr
from unittest.mock import patch, MagicMock

import difflib
import warnings
from typing import Any, Dict

import intake
import intake_esm

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

    catalog = DataCatalog()

    # 1. Initialization tests
    def test_catalog_initialization(self):
        catalog = self.catalog
        assert CATALOG_CADCAT in catalog
        assert CATALOG_BOUNDARY in catalog
        assert CATALOG_REN_ENERGY_GEN in catalog
        assert "stations" in catalog
        assert isinstance(catalog["stations"], gpd.GeoDataFrame)
        assert isinstance(catalog.catalog_df, pd.DataFrame)
        assert set(catalog.catalog_df["catalog"].unique()).issubset(
            [CATALOG_REN_ENERGY_GEN, CATALOG_CADCAT]
        )
        assert catalog._initialized == True
        assert catalog.catalog_key == UNSET

    def test_setting_catalog_key(self):
        catalog = self.catalog

        # 2. Test setting the catalog key
        self.catalog.set_catalog_key('cadcat')
        assert catalog.catalog_key == 'cadcat'

        catalog.set_catalog_key('boundary')
        assert catalog.catalog_key == 'boundary'

        # 2.1 Testing spelling errors for setting specific catalog key
        typo = 'staaations'
        with warnings.catch_warnings(record=True) as warnings_caught:
            catalog.set_catalog_key(typo)
            assert len(warnings_caught) == 2
            assert str(warnings_caught[0].message) == f"\n\nCatalog key '{typo}' not found.\nAttempting to find intended catalog key.\n\n"
            assert str(warnings_caught[1].message) == f"\n\nUsing closest match 'stations' for validator '{typo}'."

        # 2.2 Make sure setting catalog key that is not close to any existing keys fails
        nonexistent_key = 'no keys look like this'
        with warnings.catch_warnings(record=True) as warnings_caught:
            warnings.simplefilter("always")
            with pytest.raises(TypeError, match="object of type 'NoneType' has no len()"):
                catalog.set_catalog_key(nonexistent_key)

            assert len(warnings_caught) == 2
            assert str(warnings_caught[0].message) == f"\n\nCatalog key '{nonexistent_key}' not found.\nAttempting to find intended catalog key.\n\n"
            assert str(warnings_caught[1].message) == f"No validator registered for '{nonexistent_key}'. Available options: {list(catalog.keys())}"


    def test_set_new_catalog(self):
        catalog = self.catalog
        catalog.set_catalog('new fake catalog', DATA_CATALOG_URL)
        assert 'new fake catalog' in catalog.keys()
        assert catalog['new fake catalog']

    def test_getting_data(self):
        catalog = self.catalog
        catalog.set_catalog_key('cadcat')
        var_id = 't2max'
        retval = catalog.get_data({'variable_id': var_id})

        # Assert that the objects returned are xr.Datasets with the `var_id` as the data variable
        one_ds = list(retval.items())[0]
        assert list(one_ds[1].data_vars)[0] == var_id

    def test_list_clip_boundaries(self):
        
        catalog = self.catalog
        clip_boundaries = catalog.list_clip_boundaries()

        expected_names = [
            'states',
            'CA counties',
            'CA watersheds',
            'CA Electric Load Serving Entities (IOU & POU)',
            'CA Electricity Demand Forecast Zones',
            'CA Electric Balancing Authority Areas'
        ]
        assert all (k in expected_names for k in list(clip_boundaries.keys()))


    # # 6. Check that something is printed when listing boundaries list, and that the title of what is printed is correct
    # def test_print_clip_boundaries(self, capsys):
    #     catalog = self.catalog
    #     catalog.print_clip_boundaries()
    #     captured = capsys.readouterr()
    #     print(captured)
    #     assert 'Available Boundary Options for Clipping' in captured


    def test_resetting(self):
        catalog = self.catalog
        # 7. Test resetting
        assert catalog.catalog_key == 'cadcat'
        catalog.reset()
        assert catalog.catalog_key == UNSET


    def test_get_closest_options(self):
        retval = _get_closest_options('firrrsst', ['first', 'second', 'third'])
        assert retval[0] == 'first'

        # Increase the threshold and make sure that the closest option can't be found anymore
        retval = _get_closest_options('firrrsst', ['first', 'second', 'third'], cutoff=0.9)
        assert retval == None

        # Decrease the threshold and see that more options fit given key
        retval = _get_closest_options('firrrsst', ['first', 'second', 'third'], cutoff=0.1)
        assert len(retval) == 3