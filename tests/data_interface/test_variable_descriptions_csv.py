"""Test to see that variable descriptions csv is formatted appropriately"""

import pytest

from climakitae.core.data_interface import DataInterface


@pytest.fixture
def var_descrip_df():
    """Return variable descriptions DataFrame from DataInterface object"""
    data_interface = DataInterface()
    return data_interface.variable_descriptions


def test_expected_column_names(var_descrip_df):
    """Ensure that the column names are as expected by the code base"""
    col_names = list(var_descrip_df.columns)
    assert col_names == [
        "variable_id",
        "downscaling_method",
        "timescale",
        "unit",
        "display_name",
        "extended_description",
        "colormap",
        "show",
        "dependencies",
    ]


def test_index_derived_substring_variable_id_for_derived_indices(var_descrip_df):
    """Indices should have "_index_derived" (not just "_index") in their variable_id
    Both of these substrings are used in the code base to identify the type of variable and modify the selection options based on that.
    For example, a derived variable (that is not also an index) has variable_type = "Variable" in the DataInterface object
    For example, a derived variable (that is ALSO an index) has variable_type = "Derived Index" in the DataInterface object
    The logic has slight differences in the code base and must be preserved if we add more derived variables/indices
    """

    index_vars = var_descrip_df[var_descrip_df["variable_id"].str.contains("_index")]

    def _check_substring(row):
        return row.str.contains("_index_derived")

    # All the index variables should contain the full substring "_index_derived", not just "_index"
    contains_index_derived = all(
        index_vars[["variable_id"]].apply(_check_substring, axis=1)
    )

    assert contains_index_derived == True


def test_expected_dependency_column(var_descrip_df):
    """Catalog variables should all have dependencies set to 'None'
    Derived variables should have one or more dependency
    Dependency refers to a variable dependency; i.e. a derived variable is dependent on one or more variables
    """
    derived_vars = var_descrip_df[
        var_descrip_df["variable_id"].str.contains("_derived")
    ]
    catalog_vars = var_descrip_df[
        ~var_descrip_df["variable_id"].str.contains("_derived")
    ]

    def _is_none(row):
        return row == "None"

    # None of the derived variables should have dependencies = "None"
    derived_vars_none = any(
        derived_vars[["dependencies"]].apply(_is_none, axis=1).values
    )

    # All of the catalog variables could have dependencies = "None"
    catalog_vars_none = all(
        catalog_vars[["dependencies"]].apply(_is_none, axis=1).values
    )

    assert derived_vars_none == False
    assert catalog_vars_none == True
