"""Tests capability to read CMIP6 simulation information from aws"""

import pandas as pd
import pytest

from climakitae.util.generate_gwl_tables import GWLGenerator


@pytest.mark.advanced
def test_get_sims_on_aws():
    """Check that expected scenarios and models are returned."""
    df = pd.read_csv("https://cmip6-pds.s3.amazonaws.com/pangeo-cmip6.csv")
    gwl_generator = GWLGenerator(df)
    sims_on_aws = gwl_generator.get_sims_on_aws()
    cols = sims_on_aws.columns.tolist()

    for scenario in ["historical", "ssp585", "ssp370", "ssp245", "ssp126"]:
        assert scenario in cols

    # Spot checking each scenario
    assert sims_on_aws["historical"]["TaiESM1"] == ["r1i1p1f1"]
    test_data = sims_on_aws["ssp585"]["MPI-ESM1-2-HR"]
    test_data.sort()
    assert test_data == ["r1i1p1f1", "r2i1p1f1"]
    assert sims_on_aws["ssp370"]["GFDL-CM4"] == []
    assert len(sims_on_aws["ssp245"]["ACCESS-ESM1-5"]) == 10
    test_data = sims_on_aws["ssp126"]["CESM2"]
    test_data.sort()
    assert test_data == ["r10i1p1f1", "r11i1p1f1", "r4i1p1f1"]
