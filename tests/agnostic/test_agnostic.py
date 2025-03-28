import numpy as np
import pandas as pd
import pytest
import xarray as xr

# Start DataInterface up here to avoid reinitializing it in every test
from climakitae.data import DataInterface
from climakitae.explore.agnostic import (
    _create_time_lut,
    _create_warming_level_lut,
    create_lookup_tables,
)

DATA_INTERFACE = DataInterface()


def test__create_time_lut() -> None:
    """
    Test _create_time_lut function from agnostic.py

    Notes
    -----
    The function _create_time_lut is called by create_lookup_tables.
    """

    # Note: this is really obscure, and increases the complexity of the test
    # because it requires that developers intimately know the data structure
    # of the DataInterface class.
    gcms = DATA_INTERFACE.data_catalog.df.source_id.unique()
    time_lut = _create_time_lut(gcms)

    # assert time_lut is a pandas DataFrame
    assert isinstance(time_lut, pd.DataFrame)
    # assert time_lut has the correct columns
    assert all(
        col in time_lut.columns
        for col in [
            "GCM",
            "run",
            "scenario",
            "0.8",
            "1.0",
            "1.2",
            "1.5",
            "2.0",
            "2.5",
            "3.0",
            "4.0",
        ]  # these are all the columns in data/gwl_1850-1900ref.csv
    )
    # assert time_lut has the correct number of rows
    assert len(time_lut) > 0


def test_create_lookup_tables() -> None:
    """
    Test create_lookup_tables function from agnostic.py

    Notes
    -----
    The function create_lookup_tables calls _create_time_lut and
    _create_warming_level_lut.
    """
    lookup_table = create_lookup_tables()

    # assert lookup table is a dictionary
    assert isinstance(lookup_table, dict)
    # assert lookup table has the correct keys
    assert "time lookup table" in lookup_table
    assert "warming level lookup table" in lookup_table
