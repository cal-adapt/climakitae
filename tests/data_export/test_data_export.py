"""Test the get_data() function"""

import datetime
import os
from unittest import mock
from unittest.mock import mock_open, patch

import numpy as np
import pandas as pd
import pytest
import xarray as xr

import climakitae.core.data_export as export
from climakitae.util.utils import read_csv_file
from climakitae.core.paths import stations_csv_path


class TestExportErrors:

    def test_export_type(self):
        """Non xarray dataset/data array input."""
        with pytest.raises(Exception):
            test_data = np.zeros(1, 1)
            export(test_data)

    def test_export_filename(self):
        """Filename is not a string type."""
        with pytest.raises(Exception):
            test_data = xr.DataArray()
            test_filename = []
            export.export(test_data, test_filename)

    def test_export_format(self):
        """Output format not valid."""
        with pytest.raises(Exception):
            test_data = xr.DataArray()
            test_filename = "test.nc"
            test_format = "nc"
            export.export(test_data, test_filename, test_format)

    @patch("climakitae.core.data_export._export_to_netcdf")
    def test_export_zarr_s3(self, mocked_to_netcdf):
        """Output choice of s3 without zarr format. Patched to avoid
        writing file if error not raised."""
        with pytest.raises(Exception):
            test_data = xr.DataArray()
            test_filename = "test.nc"
            test_format = "NetCDF"
            test_mode = "s3"
            export.export(test_data, test_filename, test_format, test_mode)


class TestExport:

    @patch("xarray.core.dataset.Dataset.to_netcdf")
    def test_export_netcdf(self, mock_to_netcdf):
        test_data = xr.DataArray()
        test_filename = "test.nc"
        test_format = "NetCDF"
        path = os.path.join(os.getcwd(), test_filename)
        export.export(test_data, test_filename, test_format)

        # mock_to_netcdf.assert_called_once_with(test_data, "test.nc")
        mock_to_netcdf.assert_called_once_with(
            path,
            format="NETCDF4",
            engine="netcdf4",
            encoding={"data": {"zlib": True, "complevel": 6}},
        )

    @patch("xarray.core.dataset.Dataset.to_zarr")
    def test_export_zarr(self, mock_to_zarr):
        test_data = xr.DataArray()
        test_filename = "test.zarr"
        test_format = "Zarr"
        path = os.path.join(os.getcwd(), test_filename)
        export.export(test_data, test_filename, test_format)

        mock_to_zarr.assert_called_once_with(
            path,
            encoding={},
        )

    @patch("shutil.rmtree")
    def test_remove_zarr(self, mock_remove):
        fake_file = xr.Dataset()
        with pytest.raises(Exception):
            export.remove_zarr(fake_file)

        fake_file = "test.zarr"
        export.remove_zarr(fake_file)
        mock_remove.assert_called()


class TestHidden:

    @pytest.fixture()
    def test_array(self):
        test_array = xr.DataArray(np.zeros((1)))
        return test_array

    @pytest.fixture()
    def test_ds(self, test_array):
        test_array.name = "data"
        ds = test_array.to_dataset()
        ds = ds.assign_coords({"time": np.array([0])})
        return ds

    def test__convert_da_to_ds(self, test_array):
        ds = export._convert_da_to_ds(test_array)
        assert isinstance(ds, xr.core.dataset.Dataset)

    def test__convert_da_to_ds_ds(self, test_ds):
        ds = export._convert_da_to_ds(test_ds)
        assert ds.equals(test_ds)

    def test__add_metadata(self):
        test_data = xr.Dataset({"data": np.zeros((1))})
        export._add_metadata(test_data)
        for item in [
            "Data_exported_from",
            "Data_export_timestamp",
            "Analysis_package_name",
            "Version",
            "Author",
            "Author_email",
            "Home_page",
            "License",
        ]:
            assert item in test_data.attrs

    def test__dataarray_to_dataframe(self, test_array):
        df = export._dataarray_to_dataframe(test_array)
        assert isinstance(df, pd.core.frame.DataFrame)

    def test__get_unit(self, test_array):
        test_array.attrs["units"] = "mm"
        units = export._get_unit(test_array)
        assert units == "mm"

    def test__get_unit_none(self, test_array):
        units = export._get_unit(test_array)
        assert units == ""

    def test__ease_acces_in_R(self):
        column_name = "_(_)_ _-_"
        result = "_______"
        test_result = export._ease_access_in_R(column_name)
        assert test_result == result

    def test__update_header(self):
        p = pd.DataFrame(np.array([1, 1]))
        unit_map = [("Precipitation", "mm")]
        export._update_header(p, unit_map)
        assert isinstance(p.columns, pd.core.indexes.multi.MultiIndex)
        assert unit_map[0] in p.columns

    def test__dataset_to_dataframe_stations(self):
        """Check that a station dataset is correctly handled."""
        station = "Fresno Yosemite International Airport (KFAT)"
        station_eased = "Fresno_Yosemite_International_Airport_KFAT"
        varname = "Air Temperature at 2m"
        unit = "K"
        test_data = xr.Dataset(
            data_vars={station: (["time"], np.array([1, 1, 1]))},
            coords={"time": np.array([0, 1, 2])},
            attrs={"name": "data"},
        )
        test_data[station].attrs = {"variable_id": "t2", "units": unit}

        df = export._dataset_to_dataframe(test_data)

        assert isinstance(df, pd.core.frame.DataFrame)
        assert (station_eased, varname, unit) in df.columns

    def test__compression_encoding(self):
        test_data = xr.Dataset(
            data_vars={"data": (["time"], np.zeros((1)))},
            coords={"time": np.array([0])},
        )
        compdict = export._compression_encoding(test_data)
        expected = {"data": {"zlib": True, "complevel": 6}}
        assert compdict == expected

    def test__estimate_file_size(self, test_array):
        nc_size = export._estimate_file_size(test_array, "NetCDF")
        zarr_size = export._estimate_file_size(test_array, "Zarr")
        csv_size = export._estimate_file_size(test_array, "CSV")

        assert nc_size > 0
        assert zarr_size > 0
        assert csv_size > 0
        assert nc_size == zarr_size

    def test__estimate_file_size_dataset(self, test_ds):
        csv_size = export._estimate_file_size(test_ds, "CSV")
        assert csv_size > 0

    def test__warn_large_export(self, capfd):
        file_size = 7
        export._warn_large_export(file_size)
        out, err = capfd.readouterr()
        expected = (
            "WARNING: Estimated file size is "
            + str(round(file_size, 2))
            + " GB. This might take a while!\n"
        )
        assert out == expected

    def test__update_encoding(self, test_ds):
        export._update_encoding(test_ds)
        assert "missing_value" not in test_ds.encoding

    def test__fillvalue_encoding(self, test_ds):
        result = export._fillvalue_encoding(test_ds)
        assert result["time"] == {"_FillValue": None}

    @patch("xarray.core.dataset.Dataset.to_netcdf")
    def test__export_netcdf(self, mock_to_netcdf):
        test_array = xr.DataArray(np.zeros((1)))
        save_name = "test.nc"
        export._export_to_netcdf(test_array, save_name)
        path = os.path.join(os.getcwd(), save_name)

        mock_to_netcdf.assert_called_once_with(
            path,
            format="NETCDF4",
            engine="netcdf4",
            encoding={"data": {"zlib": True, "complevel": 6}},
        )

    @patch("shutil.disk_usage", return_value=(1.3e-8, 1.3e-8, 1.3e-8))
    def test__export_netcdf_large(self, mock_shutil):
        """Patch shutil to return a very small available disk space value
        and force Exception."""
        test_array = xr.DataArray(np.zeros((1)))
        save_name = "test.nc"
        with pytest.raises(Exception):
            export._export_to_netcdf(test_array, save_name)

    @patch("shutil.disk_usage", return_value=(1.3e-8, 1.3e-8, 1.3e-8))
    def test__export_zarr_large(self, mock_shutil):
        """Patch shutil to return a very small available disk space value
        and force Exception."""
        test_array = xr.DataArray(np.zeros((1)))
        save_name = "test.zarr"
        with pytest.raises(Exception):
            export._export_to_zarr(test_array, save_name, "local")


# init method or constructor
def input():
    # When mocking file open we still want to be able to read the stations
    # from the stations_csv_path, so getting that input here.
    with open(os.path.join("climakitae", stations_csv_path), "r") as f:
        input = f.read()
    return input


class TestTYM:

    @pytest.fixture
    def tym_df(self):
        datelist = pd.date_range(
            datetime.datetime(2024, 1, 1, 0),
            datetime.datetime(2024, 12, 31, 23),
            freq="h",
        )
        varlist = [
            "Air Temperature at 2m",
            "Dew point temperature",
            "Relative Humidity",
            "Instantaneous downwelling shortwave flux at bottom",
            "Shortwave surface downward direct normal irradiance",
            "Shortwave surface downward diffue irradiance",
            "Instantaneous downwelling longwave flux at bottom",
            "Windspeed at 10m",
            "Wind direction at 10m",
            "Surface Pressure",
        ]

        fake_data = [1 for x in range(0, len(datelist))]
        data = {}
        data["simulation"] = "WRF_MPI-ESM1-2-HR_r3i1p1f1"
        data["time"] = datelist
        data["scenario"] = "Historical + SSP 3-7.0"
        data["lat"] = [33.93816 for x in range(0, len(datelist))]
        data["lon"] = [118.3866 for x in range(0, len(datelist))]
        for item in varlist:
            data[item] = fake_data
        df = pd.DataFrame(data)

        stn_name = "Los Angeles International Airport (KLAX)"
        stn_code_int = 72295023174
        stn_code_str = "KLAX"
        return (df, stn_name, stn_code_int, stn_code_str)

    # @patch("builtins.open", new_callable=mock_open, read_data=input())
    # def test_write_tmy(self, mock_file):
    def test_write_tmy(self, tym_df):
        df, stn_name, stn_code_int, stn_code_str = tym_df
        # Invalid extension
        export.write_tmy_file(
            "test",
            df,
            stn_name,
            stn_code_int,
            df["lat"][0],
            df["lon"][0],
            "CA",
            file_ext="",
        )
        # Default format
        export.write_tmy_file(
            "test", df, stn_name, stn_code_int, df["lat"][0], df["lon"][0], "CA"
        )
        # Station code str
        export.write_tmy_file(
            "test", df, stn_name, stn_code_str, df["lat"][0], df["lon"][0], "CA"
        )
        # EPW format
        export.write_tmy_file(
            "test",
            df,
            stn_name,
            stn_code_int,
            df["lat"][0],
            df["lon"][0],
            "CA",
            file_ext="epw",
        )
        # mock_file.assert_called()


class TestTMYHidden:

    def test__find_missing_val_month(self):
        datelist = pd.date_range(
            datetime.datetime(2010, 1, 1, 0),
            datetime.datetime(2010, 12, 31, 23),
            freq="h",
        )
        df = pd.DataFrame(datelist, columns=["time"])
        no_missing = export._find_missing_val_month(df)

        jan_missing = export._find_missing_val_month(df.drop(index=3))

        assert no_missing == None
        assert jan_missing == 1

    def test__leap_day_fix_TaiESM1(self):
        # Pick year with leap day
        datelist = pd.date_range(
            datetime.datetime(2024, 1, 1, 0),
            datetime.datetime(2024, 12, 31, 23),
            freq="h",
        )
        df = pd.DataFrame(datelist, columns=["time"])
        df["simulation"] = "WRF_TaiESM1_r1i1p1f1"
        df_fixed = export._leap_day_fix(df)

        # Feb 29 renamed to Feb 28
        test1 = df_fixed[
            df_fixed.time == pd.Timestamp(datetime.datetime(2024, 2, 28, 0))
        ]
        assert len(test1) == 2

        # No Feb 29
        test2 = df_fixed[
            df_fixed.time == pd.Timestamp(datetime.datetime(2024, 2, 29, 0))
        ]
        assert len(test2) == 0

    def test__leap_day_fix_other(self):
        # Pick year with leap day
        datelist = pd.date_range(
            datetime.datetime(2024, 1, 1, 0),
            datetime.datetime(2024, 12, 31, 23),
            freq="h",
        )
        df = pd.DataFrame(datelist, columns=["time"])
        df["simulation"] = "WRF_ACCESS-CM2_r1i1p1f1"
        df_fixed = export._leap_day_fix(df)

        # Feb 29 renamed to Feb 28
        test1 = df_fixed[
            df_fixed.time == pd.Timestamp(datetime.datetime(2024, 2, 28, 0))
        ]
        assert len(test1) == 1

        # No Feb 29
        test2 = df_fixed[
            df_fixed.time == pd.Timestamp(datetime.datetime(2024, 2, 29, 0))
        ]
        assert len(test2) == 0

    def test__missing_hour_fix(self):
        datelist = pd.date_range(
            datetime.datetime(2024, 1, 1, 0),
            datetime.datetime(2024, 12, 31, 23),
            freq="h",
        )
        df = pd.DataFrame(datelist, columns=["time"])
        result = export._missing_hour_fix(df.drop(index=3))

        # Assert dropped index exists
        assert isinstance(result["time"][3], pd.Timestamp)

    def test__tmy_8760_size_check_8760(self):
        datelist = pd.date_range(
            datetime.datetime(2023, 1, 1, 0),
            datetime.datetime(2023, 12, 31, 23),
            freq="h",
        )
        df = pd.DataFrame(datelist, columns=["time"])
        df["simulation"] = "WRF_ACCESS-CM2_r1i1p1f1"
        result = export._tmy_8760_size_check(df)

        assert result.equals(df)

    '''def test__tmy_8760_size_check_8759(self):
        datelist = pd.date_range(
            datetime.datetime(2023, 1, 1, 0),
            datetime.datetime(2023, 12, 31, 23),
            freq="h",
        )
        df = pd.DataFrame(datelist, columns=["time"])
        df["simulation"] = "WRF_ACCESS-CM2_r1i1p1f1"
        result = export._tmy_8760_size_check(df.drop(index=100))

        # Assert dropped index exists
        assert len(result) == 8760
        assert isinstance(result["time"][100], pd.Timestamp)'''

    def test__tmy_8760_size_check_8758(self):
        datelist = pd.date_range(
            datetime.datetime(2023, 1, 1, 0),
            datetime.datetime(2023, 12, 31, 23),
            freq="h",
        )
        df = pd.DataFrame(datelist, columns=["time"])
        df["simulation"] = "WRF_ACCESS-CM2_r1i1p1f1"
        result = export._tmy_8760_size_check(df.drop([100, 200]))

        # Assert dropped index exists
        assert len(result) == 8760
        assert isinstance(result["time"][100], pd.Timestamp)

    def test__tmy_8760_size_check_8784(self):
        datelist = pd.date_range(
            datetime.datetime(2024, 1, 1, 0),
            datetime.datetime(2024, 12, 31, 23),
            freq="h",
        )
        df = pd.DataFrame(datelist, columns=["time"])
        df["simulation"] = "WRF_ACCESS-CM2_r1i1p1f1"

        result = export._tmy_8760_size_check(df)
        assert len(result) == 8760

        result = export._tmy_8760_size_check(df.drop(index=3))
        assert len(result) == 8760

        # TODO: Why isn't this 8760?
        # df["simulation"] = "WRF_TaiESM1_r1i1p1f1"
        # result = export._tmy_8760_size_check(df)
        # assert len(result) == 8760
