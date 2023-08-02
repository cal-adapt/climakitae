import pytest
import intake
import pandas as pd
from climakitae.core.data_interface import DataParameters
from climakitae.core.catalog_convert import _scenario_to_experiment_id
from climakitae.core.data_loader import _get_cat_subset, _scenarios_in_data_dict


@pytest.fixture
def test_SEL(test_CAT):
    # Create a DataParameters object
    test_selections = DataParameters()

    return test_selections


# testing that the contents of the catalog subset are consistent with the selections
def test_scenario_dim(test_SEL):
    # Set various non-default selections:
    test_SEL.scenario_ssp = [
        "SSP 3-7.0 -- Business as Usual",
        "SSP 2-4.5 -- Middle of the Road",
    ]

    # Get the corresponding dataset dictionary:
    cat_subset = _get_cat_subset(selections=test_SEL)
    ds_names = cat_subset.keys()

    result = set(_scenarios_in_data_dict(ds_names))
    assert result == set(
        [_scenario_to_experiment_id(item) for item in test_SEL.scenario_ssp]
    )


# and now add more of these with different selections
