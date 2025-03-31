"""Test CMIOpt class."""

import pytest

from climakitae.explore.uncertainty import CmipOpt


def test_cmipopt_init():
    """Test CMIOpt class initialization."""
    # Test with default parameters
    cmiopt = CmipOpt()
    assert cmiopt.variable == "tas"
    assert cmiopt.area_subset == "states"
    assert cmiopt.location == "California"
    assert cmiopt.timescale == "monthly"
    assert cmiopt.area_average == True


def test_cmipopt_init_with_params():
    """Test CMIOpt class initialization with parameters."""
    # Test with custom parameters
    cmiopt = CmipOpt(
        variable="pr",
        area_subset="countries",
        location="USA",
        timescale="annual",
        area_average=False,
    )
    assert cmiopt.variable == "pr"
    assert cmiopt.area_subset == "countries"
    assert cmiopt.location == "USA"
    assert cmiopt.timescale == "annual"
    assert cmiopt.area_average == False
