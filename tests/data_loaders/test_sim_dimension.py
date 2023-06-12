import pytest
import intake
import pkg_resources
import pandas as pd
from climakitae.selectors import _DataSelector
from .catalog_convert import _scenario_to_experiment_id
from climakitae.data_loaders import _get_cat_subset, _scenarios_in_data_dict


@pytest.fixture
def test_CAT(rootdir):
    # Access the catalog
    catalog = intake.open_esm_datastore(
        "https://cadcat.s3.amazonaws.com/cae-collection.json"
    )
    return catalog


@pytest.fixture
def test_SEL(test_CAT):
    # Create an object as in app.selections
    var_catalog_resource = pkg_resources.resource_filename(
        "climakitae", "data/variable_descriptions.csv"
    )
    var_config = pd.read_csv(var_catalog_resource, index_col=None)
    test_selections = _DataSelector(cat=test_CAT, var_config=var_config)

    return test_selections


# testing that the contents of the catalog subset are consistent with the selections
def test_scenario_dim(test_SEL):
    # Set various non-default selections:
    test_SEL.scenario_ssp = [
        "SSP 3-7.0 -- Business as Usual",
        "SSP 2-4.5 -- Middle of the Road",
    ]

    # Get the corresponding dataset dictionary:
    cat_subset = _get_cat_subset(selections=test_SEL, cat=test_CAT)
    ds_names = cat_subset.keys()

    result = _scenarios_in_data_dict(ds_names)
    assert result == set(
        [_scenario_to_experiment_id(item) for item in selections.scenario_ssp]
    )


# and now add more of these with different selections
