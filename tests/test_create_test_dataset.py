"""This script tests that the create_test_dataset script runs without error. """

import pytest
from create_test_dataset import _read_data_for_var
from climakitae.data_loaders import _read_from_catalog
from climakitae.selectors import DataSelector, LocSelectorArea
from climakitae.core import _get_catalog_contents


def test_read_data_for_var(_cat, _selections, _location):
    """Test that the function runs without error"""
    _cat = _intake.open_catalog("https://cadcat.s3.amazonaws.com/cae.yaml")
    _selections = DataSelector(choices=_get_catalog_contents(_cat))
    _location = LocSelectorArea()
    _read_data_for_var(cat=_cat, selections=_selections, location=_location)