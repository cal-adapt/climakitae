""" This script runs tests that unit conversions perform as expected """

import pytest
from climakitae.unit_conversions import _convert_units

@pytest.fixture
def air_temp_K(test_data_2022_monthly_45km):
    """Grab the 2m air temp variable. Native units are in Kelvin. """
    return test_data_2022_monthly_45km["T2"]

@pytest.fixture
def surf_pres_pasc(test_data_2022_monthly_45km):
    """Grab the surface pressure variable. Native units are in Pascals. """
    return test_data_2022_monthly_45km["PSFC"]

@pytest.fixture
def wind_u_ms(test_data_2022_monthly_45km):
    """Grab the wind u variable. Native units are in m/s. """
    return test_data_2022_monthly_45km["U10"]

@pytest.fixture
def water_vapor_kgkg(test_data_2022_monthly_45km):
    """Grab the 2m Water Vapor Mixing Ratio variable. Native units are in kg/kg. """
    return test_data_2022_monthly_45km["Q2"]

@pytest.fixture
def cumulus_precip_mm(test_data_2022_monthly_45km):
    """Grab the cumulus precipitation variable. Native units are in mm. """
    return test_data_2022_monthly_45km["RAINC"]

def test_precip_conversion(cumulus_precip_mm):
    """Test that the _convert_units function correctly converts from mm to inches. """
    da_converted = _convert_units(
        da=cumulus_precip_mm, native_units="mm", selected_units="inches"
    )
    correct_conversion = cumulus_precip_mm / 25.4
    assert correct_conversion.equals(da_converted)

def test_water_vapor_ratio_conversion(water_vapor_kgkg):
    """Test that the _convert_units function correctly converts from kg/kg to g/kg. """
    da_converted = _convert_units(
        da=water_vapor_kgkg, native_units="kg/kg", selected_units="g/kg"
    )
    correct_conversion = water_vapor_kgkg * 1000
    assert correct_conversion.equals(da_converted)

def test_wind_u10_conversion_knots(wind_u_ms):
    """Test that the _convert_units function correctly converts from m/s to knots. """
    da_converted = _convert_units(
        da=wind_u_ms, native_units="m/s", selected_units="knots"
    )
    correct_conversion = wind_u_ms * 1.94
    assert correct_conversion.equals(da_converted)

def test_pressure_conversion_hPa(surf_pres_pasc):
    """Test that the _convert_units function correctly converts from Pa to hPa. """

    da_converted = _convert_units(
        da=surf_pres_pasc, native_units="Pa", selected_units="hPa"
    )
    correct_conversion = (surf_pres_pasc / 100.)
    assert correct_conversion.equals(da_converted)

def test_pressure_conversion_inHg(surf_pres_pasc):
    """Test that the _convert_units function correctly converts from Pa to inHg. """

    da_converted = _convert_units(
        da=surf_pres_pasc, native_units="Pa", selected_units="inHg"
    )
    correct_conversion = (surf_pres_pasc / 3386.39)
    assert correct_conversion.equals(da_converted)

def test_pressure_conversion_mb(surf_pres_pasc):
    """Test that the _convert_units function correctly converts from Pa to mb. """

    da_converted = _convert_units(
        da=surf_pres_pasc, native_units="Pa", selected_units="mb"
    )
    correct_conversion = (surf_pres_pasc / 100.)
    assert correct_conversion.equals(da_converted)

def test_temp_conversion_Celcius(air_temp_K):
    """Test that the _convert_units function correctly converts from Kelvin to Celcius. """
    da_converted = _convert_units(
        da=air_temp_K, native_units="K", selected_units="degC"
    )
    correct_conversion = (air_temp_K - 273.15)
    assert correct_conversion.equals(da_converted)

def test_temp_conversion_Fahrenheit(air_temp_K):
    """Test that the _convert_units function correctly converts from Kelvin to Fahrenheit. """
    da_converted = _convert_units(
        da=air_temp_K, native_units="K", selected_units="degF"
    )
    correct_conversion = ((1.8 * (air_temp_K - 273.15)) + 32)
    assert correct_conversion.equals(da_converted)

def test_temp_conversion_Rankine(air_temp_K):
    """Test that the _convert_units function correctly converts from Kelvin to Rankine. """
    da_converted = _convert_units(
        da=air_temp_K, native_units="K", selected_units="degR"
    )
    correct_conversion = (1.8 * air_temp_K)
    assert correct_conversion.equals(da_converted)
