"""Test the get_data() function"""

import io
import sys

import pytest

from climakitae.core.data_interface import get_data


@pytest.mark.advanced
class TestStationDataRetrievalGetData:
    """Test that the get_data function retrieves station data without error."""

    def test_sandiego_station(self):
        """Test that station data for a single station can be retrieved"""
        try:
            ds = get_data(
                variable="Air Temperature at 2m",  # required argument
                resolution="9 km",  # required argument. Options: "9 km" or "3 km"
                timescale="hourly",  # required argument
                data_type="Stations",  # required argument
                stations="San Diego Lindbergh Field (KSAN)",  # optional argument. If no input, all weather stations are retrieved
            )
        except:
            pytest.fail(
                "Station data for San Diego Lindbergh Field using a 9km resolution could not be retrieved"
            )

    def test_multiple_weather_stations(self):
        """Test retrieveing more than one station and with more complex function arguments"""
        try:
            get_data(
                variable="Air Temperature at 2m",
                resolution="3 km",
                timescale="hourly",
                data_type="Stations",
                stations=[
                    "San Francisco International Airport (KSFO)",
                    "Oakland Metro International Airport (KOAK)",
                ],
                units="degF",
                time_slice=(2000, 2005),
            )
        except:
            pytest.fail(
                "Station data from 2000-2005 for three Bay Area weather stations using a 3km resolution and a unit conversion to degF could not be retrieved"
            )


@pytest.mark.advanced
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
                warming_level=[3.0],
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
                time_slice=(1990, 1991),
                scenario="Historical Climate",
            )
        except:
            pytest.fail(
                "Data for hourly Relative Humidity with a Time-based approach could not be retrieved"
            )


@pytest.mark.advanced
class TestAppropriateStringErrorReturnedIfBadInputGetData:
    """Test that an appropriate error message is printed to the user describing the issue and how to resolve it."""

    def test_station_data_bad_variable_input(self):
        """Test that the function raises the appropriate error message if you input a variable that is not Air Temperature at 2m"""

        # Error message we expect to be printed by the function
        expected_print_message = "Weather station data can only be retrieved for variable=Air Temperature at 2m \nYour input: Peanut Butter and Jellyfishes in the Sky \nRetrieving data for variable=Air Temperature at 2m\n"

        # NOTE: function PRINTS this message-- it does not return it as an error
        # Because of this, we have to use sys to capture the print message
        capture = io.StringIO()
        save, sys.stdout = sys.stdout, capture
        ds = get_data(
            variable="Peanut Butter and Jellyfishes in the Sky",
            resolution="9 km",
            timescale="hourly",
            data_type="Stations",
            stations="San Francisco International Airport (KSFO)",
        )
        sys.stdout = save

        assert capture.getvalue() == expected_print_message

    def test_station_data_bad_station_correct_guess(self):
        """Test that the function can correctly 'guess' an appropriate weather station if the user inputs something that is close in name to an existing weather station in our catalog"""

        expected_print_message = "Input station='San Francisco Airport' is not a valid option.\nClosest option: 'San Francisco International Airport (KSFO)'\nOutputting data for station='San Francisco International Airport (KSFO)'\n"

        # NOTE: function PRINTS this message-- it does not return it as an error
        # Because of this, we have to use sys to capture the print message
        capture = io.StringIO()
        save, sys.stdout = sys.stdout, capture
        ds = get_data(
            variable="Air Temperature at 2m",
            resolution="9 km",
            timescale="hourly",
            data_type="Stations",
            stations="San Francisco Airport",  # not the name of the station, but its close so the function should be able to guess
        )
        sys.stdout = save

        assert capture.getvalue() == expected_print_message

    def test_error_raised_for_reallllyyyy_bad_input_station_data(self):
        """If the function can't even make a reasonable guess as to the user's guess, it should throw a ValueError"""
        # Error message we expect to be printed by the function
        # This is just the first line of the print message
        # Cut after new line because the actual message is really long. It lists all the available stations.
        expected_print_message = (
            "Input station='the US international space station' is not a valid option."
        )

        # NOTE: function PRINTS this message-- it does not return it as an error
        # Because of this, we have to use sys to capture the print message
        capture = io.StringIO()
        save, sys.stdout = sys.stdout, capture
        try:
            ds = get_data(
                variable="Air Temperature at 2m",
                resolution="9 km",
                timescale="hourly",
                data_type="Stations",
                stations="the US international space station",  # Not a good weather station input... silly user!
            )
        except Exception:
            # This function raises a Value Error AND prints a message (before raising the error)
            # Just ignore the error
            pass

        sys.stdout = save

        assert capture.getvalue().split("\n")[0] == expected_print_message

    def test_error_raised_string_input_warming_level(self):
        """Warming level should be a float input! Make sure the function prints the appropriate error message"""

        # Error message we expect to be printed by the function
        expected_print_message = "ERROR: Function argument warming_level requires a float/int or list \n                    of floats/ints input. Your input: <class 'str'> \nReturning None\n"

        # NOTE: function PRINTS this message-- it does not return it as an error
        # Because of this, we have to use sys to capture the print message
        capture = io.StringIO()
        save, sys.stdout = sys.stdout, capture
        ds = get_data(
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

    def test_error_raised_bad_timescale_input(self):
        """Test that an appropriate message is printed if bad input for timescale
        Previously, this input would raise a confusing error!
        Fixed in November 2024 to return a useful print message and None instead
        """

        # Error message we expect to be printed by the function
        expected_print_message = "ERROR: No data found for your input values. Please modify your data request. \nReturning None\n"

        # NOTE: function PRINTS this message-- it does not return it as an error
        # Because of this, we have to use sys to capture the print message
        capture = io.StringIO()
        save, sys.stdout = sys.stdout, capture
        ds = get_data(
            variable="Effective Temperature",
            timescale="hourly",
            resolution="9 km",
            downscaling_method="Dynamical",
            scenario="Historical Climate",
            area_subset="states",
            cached_area="CA",
            time_slice=(1990, 1991),
        )
        sys.stdout = save

        assert capture.getvalue() == expected_print_message
