"""Test the get_data() function"""

import io
import sys
from unittest import mock
from unittest.mock import mock_open, patch

import numpy as np
import pandas as pd
import pytest
import xarray as xr

import climakitae.core.data_export as export


class TestExportErrors:

    @pytest.fixture()
    def mocked_to_zarr(self):
        pass

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

    @mock.patch("climakitae.core.data_export._export_to_zarr")
    def test_export_zarr_s3(self, mocked_to_zarr):
        """Output choice of s3 without zarr format. Mocked to avoid actually
        writing to zarr if error not raised."""
        with pytest.raises(Exception):
            test_data = xr.DataArray()
            test_filename = "test.nc"
            test_format = "NetCDF"
            test_mode = "s3"
            export.export(test_data, test_filename, test_format, test_mode)


class TestExport:

    @pytest.fixture()
    def mocked_to_netcdf(self):
        pass

    @mock.patch("climakitae.core.data_export._export_to_netcdf")
    def test_export_netcdf(self, mocked_to_netcdf):
        test_data = xr.DataArray()
        test_filename = "test.nc"
        test_format = "NetCDF"
        export.export(test_data, test_filename, test_format)
        # TODO: finish test


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

    def test_convert_da_to_ds(self, test_array):
        ds = export._convert_da_to_ds(test_array)
        assert isinstance(ds, xr.core.dataset.Dataset)

    def test_add_metadata(self):
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

    def test_dataarray_to_dataframe(self, test_array):
        df = export._dataarray_to_dataframe(test_array)
        assert isinstance(df, pd.core.frame.DataFrame)

    def test_get_unit(self, test_array):
        test_array.attrs["units"] = "mm"
        units = export._get_unit(test_array)
        assert units == "mm"

    def test_get_unit_none(self, test_array):
        units = export._get_unit(test_array)
        assert units == ""

    def test_ease_acces_in_R(self):
        column_name = "_(_)_ _-_"
        result = "_______"
        test_result = export._ease_access_in_R(column_name)
        assert test_result == result

    def test_update_header(self):
        p = pd.DataFrame(np.array([1, 1]))
        unit_map = [("Precipitation", "mm")]
        export._update_header(p, unit_map)
        assert isinstance(p.columns, pd.core.indexes.multi.MultiIndex)
        assert unit_map[0] in p.columns

    def test_dataset_to_dataframe_stations(self):
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

    def test_compression_encoding(self):
        test_data = xr.Dataset(
            data_vars={"data": (["time"], np.zeros((1)))},
            coords={"time": np.array([0])},
        )
        compdict = export._compression_encoding(test_data)
        expected = {"data": {"zlib": True, "complevel": 6}}
        assert compdict == expected

    def test_estimate_file_size(self, test_array):
        nc_size = export._estimate_file_size(test_array, "NetCDF")
        zarr_size = export._estimate_file_size(test_array, "Zarr")
        csv_size = export._estimate_file_size(test_array, "CSV")

        assert nc_size > 0
        assert zarr_size > 0
        assert csv_size > 0
        assert nc_size == zarr_size

    def test_estimate_file_size_dataset(self, test_ds):
        csv_size = export._estimate_file_size(test_ds, "CSV")
        assert csv_size > 0

    def test_warn_large_export(self, capfd):
        file_size = 7
        export._warn_large_export(file_size)
        out, err = capfd.readouterr()
        expected = (
            "WARNING: Estimated file size is "
            + str(round(file_size, 2))
            + " GB. This might take a while!\n"
        )
        assert out == expected

    def test_update_encoding(self, test_ds):
        export._update_encoding(test_ds)
        assert "missing_value" not in test_ds.encoding

    def test_fillvalue_encoding(self, test_ds):
        result = export._fillvalue_encoding(test_ds)
        assert result["time"] == {"_FillValue": None}
