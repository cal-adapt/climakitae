""" This script runs tests using the variable descriptions csv file and the catalog contents variable choices. 
This script tests that the variables and descriptions listed in the catalog are the same as (or a smaller subset of) the variables and descriptions listed in the csv file. 
It also tests that the variable descriptions csv file contains the expected column names. 
"""

import pandas as pd
import pytest
from climakitae.core import _get_catalog_contents
from climakitae.utils import _read_var_csv
import intake
import pkg_resources
CSV_FILE = pkg_resources.resource_filename('climakitae', 'data/variable_descriptions.csv')

@pytest.fixture
def cat_contents():  
    """Load variable choices listed in Analytics Engine catalog. """
    catalog_info = intake.open_catalog("https://cadcat.s3.amazonaws.com/cae.yaml")
    cat_contents = _get_catalog_contents(catalog_info)
    return cat_contents["variable_choices"]

@pytest.fixture
def descrip_dict_formatted():
    """Read in csv file formatted with the same columns as AWS catalog """
    csv = pd.read_csv(CSV_FILE, index_col="description", usecols=["name","description"])
    return csv.to_dict(orient="dict")["name"]

    
@pytest.fixture
def descrip_pd(): 
    """Read in csv file as pandas dataframe object. """
    pd_df = pd.read_csv(CSV_FILE)
    return pd_df 

@pytest.fixture
def descrip_pd(): 
    """Read in csv file as pandas dataframe object. """
    pd_df = pd.read_csv(CSV_FILE)
    return pd_df

@pytest.mark.parametrize("csv_file,index_col,usecols", [(CSV_FILE, "name", ["name","description","extended_description"]), (CSV_FILE, "description", ["description","extended_description","name"]),(CSV_FILE, "extended_description", ["extended_description","name","description"])])
def test_read_var_function_output(csv_file,index_col,usecols):
    """Confirm that any modifications to the function _read_var_csv do not break code and that return object is what is expected. """
    descrip_dict = _read_var_csv(csv_file=csv_file, index_col=index_col, usecols=usecols)
    dict_values = dict(descrip_dict.values())
    assert (dict_values == {usecols[1]: usecols[2]}) or (dict_values == {usecols[2]: usecols[1]}), "Function _read_var_csv does not return expected formatted dictionary object."
    
def test_csv_matches_cat_contents(cat_contents, descrip_dict_formatted):
    """Ensure that items in catalog are the same as (or a smaller subset of) the items in the variable csv. """
    assert cat_contents["hourly"]["Dynamical"].items() <= descrip_dict_formatted.items(), "Catalog contents are not the same as (or a smaller subset of) the items in the variable descriptions csv."
    
def test_csv_contains_all_columns(descrip_pd): 
    columns = ["name","description","extended_description"]
    assert all(descrip_pd.columns.values == columns), "Variable description csv does not contain the correct columns."
