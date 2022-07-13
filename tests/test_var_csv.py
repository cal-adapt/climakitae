import pandas as pd
import pytest
from climakitae.selectors import CatalogContents
import pkg_resources
DATA_PATH = pkg_resources.resource_filename('climakitae', 'data/')
CSV_FILE = pkg_resources.resource_filename('climakitae', 'data/variable_descriptions.csv')

@pytest.fixture
def cat_contents():  
    """Load variable choices listed in Analytics Engine catalog. """
    return CatalogContents()._variable_choices

@pytest.fixture
def descrip_dict_formatted():
    """Read in csv file formatted with the same columns as CatalogContents()._variable_choices. """
    csv = pd.read_csv(CSV_FILE, index_col="description", usecols=["name","description"])
    return csv.to_dict(orient="dict")["name"]

@pytest.fixture
def descrip_pd(): 
    """Read in csv file as pandas dataframe object. """
    pd_df = pd.read_csv(CSV_FILE)
    return pd_df 

def test_csv_matches_cat_contents(cat_contents, descrip_dict_formatted):
    """Ensure that items in catalog are the same as (or a smaller subset of) the items in the variable csv. """
    assert cat_contents["hourly"]["Dynamical"].items() <= descrip_dict_formatted.items(), "Catalog contents are not the same as (or a smaller subset of) the items in the variable descriptions csv."
    
def test_csv_contains_all_columns(descrip_pd): 
    columns = ["name","description","extended_description"]
    assert all(descrip_pd.columns.values == columns), "Variable description csv does not contain the correct columns."