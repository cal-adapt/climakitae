""" This script runs tests that unit conversions perform as expected """

import pytest
from climakitae.unit_conversions import _convert_units



class TestTempConversions:
    """Ensure that all conversions for temperature units run as expected. """

    @pytest.fixture
    def air_temp_K(self, test_data_2022_monthly_45km):
        """Grab the 2m air temp variable. Native units are in Kelvin. """
        return test_data_2022_monthly_45km["T2"]

    def test_temp_conversion_Celcius(self, air_temp_K):
        """Test that the _convert_units function correctly converts from Kelvin to Celcius. """
        da_converted = _convert_units(da=air_temp_K, selected_units="degC")
        correct_conversion = (air_temp_K - 273.15)
        assert correct_conversion.equals(da_converted)

    def test_temp_conversion_Fahrenheit(self, air_temp_K):
        """Test that the _convert_units function correctly converts from Kelvin to Fahrenheit. """
        da_converted = _convert_units(da=air_temp_K, selected_units="degF")
        correct_conversion = ((1.8 * (air_temp_K - 273.15)) + 32)
        assert correct_conversion.equals(da_converted)



class TestPressureConversions:
    """Ensure that all conversions for pressure units run as expected. """

    @pytest.fixture
    def surf_pres_pasc(self, test_data_2022_monthly_45km):
        """Grab the surface pressure variable. Native units are in Pascals. """
        return test_data_2022_monthly_45km["PSFC"]

    def test_pressure_conversion_hPa(self, surf_pres_pasc):
        """Test that the _convert_units function correctly converts from Pa to hPa. """
        da_converted = _convert_units(da=surf_pres_pasc, selected_units="hPa")
        correct_conversion = (surf_pres_pasc / 100.)
        assert correct_conversion.equals(da_converted)

    def test_pressure_conversion_inHg(self, surf_pres_pasc):
        """Test that the _convert_units function correctly converts from Pa to inHg. """
        da_converted = _convert_units(da=surf_pres_pasc, selected_units="inHg")
        correct_conversion = (surf_pres_pasc * 0.000295300)
        assert correct_conversion.equals(da_converted)

    def test_pressure_conversion_mb(self, surf_pres_pasc):
        """Test that the _convert_units function correctly converts from Pa to mb. """
        da_converted = _convert_units(da=surf_pres_pasc, selected_units="mb")
        correct_conversion = (surf_pres_pasc / 100.)
        assert correct_conversion.equals(da_converted)



class TestWindSpeedConversions:
    """Ensure that all conversions for windspeed units run as expected. """

    @pytest.fixture
    def wind_u_ms(self, test_data_2022_monthly_45km):
        """Grab the wind u variable. Native units are in m/s. """
        return test_data_2022_monthly_45km["U10"]

    def test_wind_u10_conversion_knots(self, wind_u_ms):
        """Test that the _convert_units function correctly converts from m/s to knots. """
        da_converted = _convert_units(da=wind_u_ms, selected_units="knots")
        correct_conversion = wind_u_ms * 1.9438445
        assert correct_conversion.equals(da_converted)



class TestWaterVaporConversions:
    """Ensure that all conversions for water vapor units run as expected. """

    @pytest.fixture
    def water_vapor_kgkg(self, test_data_2022_monthly_45km):
        """Grab the 2m Water Vapor Mixing Ratio variable. Native units are in kg/kg. """
        return test_data_2022_monthly_45km["Q2"]

    def test_water_vapor_ratio_conversion(self, water_vapor_kgkg):
        """Test that the _convert_units function correctly converts from kg/kg to g/kg. """
        da_converted = _convert_units(da=water_vapor_kgkg, selected_units="g kg-1")
        correct_conversion = water_vapor_kgkg * 1000
        assert correct_conversion.equals(da_converted)



class TestPrecipitationConversions:
    """Ensure that all conversions for precipitation units run as expected. """

    @pytest.fixture
    def cumulus_precip_mm(self, test_data_2022_monthly_45km):
        """Grab the cumulus precipitation variable. Native units are in mm. """
        return test_data_2022_monthly_45km["RAINC"]

    def test_precip_conversion(self, cumulus_precip_mm):
        """Test that the _convert_units function correctly converts from mm to inches. """
        da_converted = _convert_units(da=cumulus_precip_mm, selected_units="inches")
        correct_conversion = cumulus_precip_mm / 25.4
        assert correct_conversion.equals(da_converted)
