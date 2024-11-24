"""Test the get_data() function 
"""

import pytest
import io
import sys
from climakitae.core.data_interface import get_data


class TestDerivedVariablesGetData:
    """Test that derived variables/indices can be retrieved
    Some of the use cases below used to raise an error and have since been fixed.
    """

    def test_ffwi_retrieval(self):
        """Make sure that the fosberg fire weather index can be retrieved!
        Previously raised an error in Nov 2024 (has since been fixed)
        """
        try:
            ds = get_data(
                variable="Fosberg fire weather index",
                timescale="hourly",
                resolution="9 km",
                downscaling_method="Dynamical",
                area_subset="states",
                cached_area="CA",
                approach="Warming Level",
                warming_level=[3.0],
            )
        except:
            pytest.fail(
                "Data for Fosberg Fire Weather Index with Warming Levels approach could not be retrieved"
            )

        def test_hourly_relative_humidity_warming_levels_approach_retrieval(self):
            """Make sure that the hourly relative humidity (a derived variable) can be retrieved!
            Previously raised an error in Nov 2024 (has since been fixed)
            """
            try:
                ds = get_data(
                    variable="Relative humidity",
                    timescale="hourly",
                    resolution="9 km",
                    downscaling_method="Dynamical",
                    area_subset="states",
                    cached_area="CA",
                    approach="Warming Level",
                    warming_level=[3.0, 4.0],
                )
            except:
                pytest.fail(
                    "Data for hourly Relative Humidity with a Warming Levels approach could not be retrieved"
                )

        def test_hourly_relative_humidity_time_based_approach_retrieval(self):
            """Make sure that the hourly relative humidity (a derived variable) can be retrieved!
            Previously raised an error in Nov 2024 (has since been fixed)
            """
            try:
                ds = get_data(
                    variable="Relative humidity",
                    timescale="hourly",
                    resolution="9 km",
                    downscaling_method="Dynamical",
                    area_subset="states",
                    cached_area="CA",
                    approach="Time",
                )
            except:
                pytest.fail(
                    "Data for hourly Relative Humidity with a Time-based approach could not be retrieved"
                )


def TestAppropriateStringErrorReturnedIfBadInputGetData():
    """Test that an appropriate error message is printed to the user describing the issue and how to resolve it."""

    def test_error_raised_string_input_warming_level(self):
        """Warming level should be a float input! Make sure the function prints the appropriate error message"""

        # Error message we expect to be printed by the function
        expected_print_message = "ERROR: Function argument warming_level requires a float/int or list of floats/ints input. Your input: <class 'str'> \nReturning None\n"

        # NOTE: function PRINTS this message-- it does not return it as en error
        # Because of this, we have to use sys to capture the print message
        capture = io.StringIO()
        save, sys.stdout = sys.stdout, capture
        get_data(
            variable="Precipitation (total)",
            downscaling_method="Dynamical",
            resolution="45 km",
            timescale="monthly",
            cached_area="San Bernardino County",
            approach="Warming Level",
            warming_level="20",
        )
        sys.stdout = save

        assert capture.getvalue() == expected_print_message
