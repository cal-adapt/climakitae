"""Tests capability to read CMIP6 simulation information from aws"""

import pandas as pd
import pytest

from climakitae.util.generate_gwl_tables import get_sims_on_aws


def test_get_sims_on_aws():
    """Check that all expected scenarios and models are returned."""
    df = pd.read_csv("https://cmip6-pds.s3.amazonaws.com/pangeo-cmip6.csv")
    df_subset = df[
        (df.table_id == "Amon")
        & (df.variable_id == "tas")
        & (df.experiment_id == "historical")
    ]
    sims_on_aws = get_sims_on_aws(df)
    rows = sims_on_aws.index.tolist()
    cols = sims_on_aws.columns.tolist()

    for scenario in ["historical", "ssp585", "ssp370", "ssp245", "ssp126"]:
        assert scenario in cols

    assert sims_on_aws["historical"]["TaiESM1"] == ["r1i1p1f1"]
