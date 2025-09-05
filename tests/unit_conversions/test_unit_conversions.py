"""This script runs tests that unit conversions perform as expected"""

import cftime
import pytest
import xarray as xr

from climakitae.util.unit_conversions import convert_units, get_unit_conversion_options


class TestConversionOptions:
    def test_options_type(self):
        options = get_unit_conversion_options()
        assert isinstance(options, dict)

    def test_options_temperature(self):
        options = get_unit_conversion_options()
        assert options["K"] == ["K", "degC", "degF"]


class TestErrors:
    def test_native_units_error(self):
        da = xr.DataArray()
        da.attrs = dict()
        with pytest.raises(ValueError):
            convert_units(da, selected_units="degC")


class TestTempConversions:
    """Ensure that all conversions for temperature units run as expected."""

    @pytest.fixture
    def air_temp_K(self, test_data_2022_monthly_45km):
        """Grab the 2m air temp variable. Native units are in Kelvin."""
        return test_data_2022_monthly_45km["T2"]

    def test_temp_conversion_Celcius(self, air_temp_K):
        """Test that the convert_units function correctly converts from Kelvin to Celcius."""
        da_converted = convert_units(da=air_temp_K, selected_units="degC")
        correct_conversion = air_temp_K - 273.15
        assert correct_conversion.equals(da_converted)

    def test_temp_conversion_Fahrenheit(self, air_temp_K):
        """Test that the convert_units function correctly converts from Kelvin to Fahrenheit."""
        da_converted = convert_units(da=air_temp_K, selected_units="degF")
        correct_conversion = (1.8 * (air_temp_K - 273.15)) + 32
        assert correct_conversion.equals(da_converted)

    def test_temp_conversion_degC_to_K(self):
        """Test that the convert_units function correctly converts from Celcius to Kelvin."""
        da = xr.DataArray(data=34.0, attrs={"units": "degC"})
        da_converted = convert_units(da, selected_units="K")
        correct_conversion = da + 273.15
        assert correct_conversion.equals(da_converted)

    def test_temp_conversion_degC_to_degF(self):
        """Test that the convert_units function correctly converts from Celcius to Fahrenheit."""
        da = xr.DataArray(data=34.0, attrs={"units": "degC"})
        da_converted = convert_units(da, selected_units="degF")
        correct_conversion = (1.8 * da) + 32
        assert correct_conversion.equals(da_converted)

    def test_temp_conversion_degF_to_degC(self):
        """Test that the convert_units function correctly converts from Fahrenheit to Celcius."""
        da = xr.DataArray(data=80.0, attrs={"units": "degF"})
        da_converted = convert_units(da, selected_units="degC")
        correct_conversion = (da - 32) / 1.8
        assert correct_conversion.equals(da_converted)

    def test_temp_conversion_degF_to_K(self):
        """Test that the convert_units function correctly converts from Fahrenheit to Kelvin."""
        da = xr.DataArray(data=80.0, attrs={"units": "degF"})
        da_converted = convert_units(da, selected_units="K")
        correct_conversion = ((da - 32) / 1.8) + 273.15
        assert correct_conversion.equals(da_converted)


class TestPressureConversions:
    """Ensure that all conversions for pressure units run as expected."""

    @pytest.fixture
    def surf_pres_pasc(self, test_data_2022_monthly_45km):
        """Grab the surface pressure variable. Native units are in Pascals."""
        return test_data_2022_monthly_45km["PSFC"]

    def test_pressure_conversion_hPa(self, surf_pres_pasc):
        """Test that the convert_units function correctly converts from Pa to hPa."""
        da_converted = convert_units(da=surf_pres_pasc, selected_units="hPa")
        correct_conversion = surf_pres_pasc / 100.0
        assert correct_conversion.equals(da_converted)

    def test_pressure_conversion_inHg(self, surf_pres_pasc):
        """Test that the convert_units function correctly converts from Pa to inHg."""
        da_converted = convert_units(da=surf_pres_pasc, selected_units="inHg")
        correct_conversion = surf_pres_pasc * 0.000295300
        assert correct_conversion.equals(da_converted)

    def test_pressure_conversion_mb(self, surf_pres_pasc):
        """Test that the convert_units function correctly converts from Pa to mb."""
        da_converted = convert_units(da=surf_pres_pasc, selected_units="mb")
        correct_conversion = surf_pres_pasc / 100.0
        assert correct_conversion.equals(da_converted)

    def test_pressure_same_units(self, surf_pres_pasc):
        """Test that the convert_units function returns identical data if selected unit match native units."""
        da_converted = convert_units(da=surf_pres_pasc, selected_units="Pa")
        correct_conversion = surf_pres_pasc
        assert correct_conversion.equals(da_converted)


class TestWindSpeedConversions:
    """Ensure that all conversions for windspeed units run as expected."""

    @pytest.fixture
    def wind_u_ms(self, test_data_2022_monthly_45km):
        """Grab the wind u variable. Native units are in m/s."""
        return test_data_2022_monthly_45km["U10"]

    def test_wind_u10_conversion_knots(self, wind_u_ms):
        """Test that the convert_units function correctly converts from m/s to knots."""
        da_converted = convert_units(da=wind_u_ms, selected_units="knots")
        correct_conversion = wind_u_ms * 1.9438445
        assert correct_conversion.equals(da_converted)

    def test_wind_u10_conversion_mph(self, wind_u_ms):
        """Test that the convert_units function correctly converts from m/s to knots."""
        da_converted = convert_units(da=wind_u_ms, selected_units="mph")
        correct_conversion = wind_u_ms * 2.236936
        assert correct_conversion.equals(da_converted)


class TestWaterVaporConversions:
    """Ensure that all conversions for water vapor units run as expected."""

    @pytest.fixture
    def water_vapor_kgkg(self, test_data_2022_monthly_45km):
        """Grab the 2m Water Vapor Mixing Ratio variable. Native units are in kg/kg."""
        return test_data_2022_monthly_45km["Q2"]

    def test_water_vapor_ratio_conversion(self, water_vapor_kgkg):
        """Test that the convert_units function correctly converts from kg/kg to g/kg."""
        da_converted = convert_units(da=water_vapor_kgkg, selected_units="g kg-1")
        correct_conversion = water_vapor_kgkg * 1000
        assert correct_conversion.equals(da_converted)

    def test_specific_humidity_conversion(self):
        """Test that the convert_units function correctly converts from g/kg to kg/kg."""
        da = xr.DataArray(data=500.0, attrs={"units": "g/kg"})
        da_converted = convert_units(da, selected_units="kg/kg")
        correct_conversion = da / 1000
        assert correct_conversion.equals(da_converted)

    def test_relative_humidity_conversion(self):
        """Test that the convert_units function correctly converts from percent to fraction."""
        da = xr.DataArray(data=50.0, attrs={"units": "[0 to 100]"})
        da_converted = convert_units(da, selected_units="fraction")
        correct_conversion = da / 100.0
        assert correct_conversion.equals(da_converted)


class TestPrecipitationConversions:
    """Ensure that all conversions for precipitation units run as expected."""

    @pytest.fixture
    def cumulus_precip_mm(self, test_data_2022_monthly_45km):
        """Grab the cumulus precipitation variable. Native units are in mm."""
        return test_data_2022_monthly_45km["RAINC"]

    def test_precip_conversion(self, cumulus_precip_mm):
        """Test that the convert_units function correctly converts from mm to inches."""
        da_converted = convert_units(da=cumulus_precip_mm, selected_units="inches")
        correct_conversion = cumulus_precip_mm / 25.4
        assert correct_conversion.equals(da_converted)

    def test_precip_conversion_kg_m2_s1(self, cumulus_precip_mm):
        """Test that the convert_units function correctly converts from mm to inches."""
        da_converted = convert_units(da=cumulus_precip_mm, selected_units="kg m-2 s-1")
        correct_conversion = cumulus_precip_mm / 86400
        assert correct_conversion.equals(da_converted)

    def test_precip_conversion_kg_m2_s1_to_mm(self):
        """Test that the convert_units function correctly converts monthly total from kg m-2 s-1."""
        time = cftime.datetime(2000, 1, 1, calendar="standard")  # single month Jan
        da = xr.DataArray(
            data=0.00011574,
            coords={"time": time},
            attrs={"units": "kg m-2 s-1", "frequency": "monthly"},
        )
        da_converted = convert_units(da, selected_units="mm")
        correct_conversion = da * 86400 * 31
        assert correct_conversion.equals(da_converted)
