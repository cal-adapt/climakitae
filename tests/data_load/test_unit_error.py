"""Manually setting a bad/invalid unit in a DataParameters object should not allow you to retrieve data.
Test that an appropriate error is raised.
"""

import pytest

from climakitae.core.data_interface import DataParameters


class TestErrorRaisedIfBadUnitsManuallySet_Precipitation:
    """Check for precipitation"""

    @pytest.fixture
    def selections(self):
        # Create a DataParameters object
        test_selections = DataParameters()
        test_selections.variable = "Precipitation (total)"
        test_selections.units = "cats&dogs"

        return test_selections

    def test_error_raised(self, selections):
        with pytest.raises(Exception) as e_info:
            selections.retrieve()


class TestErrorRaisedIfBadUnitsManuallySet_AirTemp2m:
    """Check for air temp at 2m"""

    @pytest.fixture
    def selections(self):
        # Create a DataParameters object
        test_selections = DataParameters()
        test_selections.variable = "Air Temperature at 2m"
        test_selections.units = "degreeees Kelllviinnnn"

        return test_selections

    def test_error_raised(self, selections):
        with pytest.raises(Exception) as e_info:
            selections.retrieve()


class TestErrorRaisedIfBadUnitsManuallySet_SurfacePressure:
    """Check for Surface Pressure"""

    @pytest.fixture
    def selections(self):
        # Create a DataParameters object
        test_selections = DataParameters()
        test_selections.variable = "Surface Pressure"
        test_selections.units = "snails per second"

        return test_selections

    def test_error_raised(self, selections):
        with pytest.raises(Exception) as e_info:
            selections.retrieve()
