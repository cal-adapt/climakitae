import os
import tempfile
from unittest.mock import PropertyMock, patch

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from climakitae.util.utils import (
    _package_file_path,
    area_average,
    downscaling_method_as_list,
    get_closest_gridcell,
    read_csv_file,
    write_csv_file,
)


class TestUtils:

    def test_downscaling_method_as_list(self):
        options = {
            "Dynamical": ["Dynamical"],
            "Statistical": ["Statistical"],
            "Dynamical+Statistical": ["Dynamical", "Statistical"],
        }
        for key, value in options.items():
            assert (
                downscaling_method_as_list(key) == value
            ), f"Expected {value}, but got {downscaling_method_as_list(key)}"

    def test_area_average(self):
        # Mock data with x, y dimensions (needs lat for weighting)
        data_xy = xr.Dataset(
            {"var1": (("time", "x", "y"), [[[1, 2], [3, 4]], [[5, 6], [7, 8]]])},
            coords={
                "time": pd.to_datetime(["2023-01-01", "2023-01-02"]),
                "x": [10, 20],
                "y": [30, 40],
                "lat": (("x", "y"), [[30, 31], [32, 33]]),  # Example lat values
                "lon": (("x", "y"), [[-120, -119], [-118, -117]]),  # Example lon values
            },
        )

        # Mock data with lat, lon dimensions
        data_latlon = xr.Dataset(
            {
                "var2": (
                    ("time", "lat", "lon"),
                    [[[10, 20], [30, 40]], [[50, 60], [70, 80]]],
                )
            },
            coords={
                "time": pd.to_datetime(["2023-01-01", "2023-01-02"]),
                "lat": [34, 35],
                "lon": [-118, -117],
            },
        )

        # Test with x, y dimensions
        result_xy = area_average(data_xy)
        assert isinstance(result_xy, xr.Dataset)
        assert "x" not in result_xy.dims
        assert "y" not in result_xy.dims
        assert "time" in result_xy.dims
        # Check if the calculation ran (values should be floats after weighted mean)
        assert result_xy["var1"].dtype == float

        # Test with lat, lon dimensions
        result_latlon = area_average(data_latlon)
        assert isinstance(result_latlon, xr.Dataset)
        assert "lat" not in result_latlon.dims
        assert "lon" not in result_latlon.dims
        assert "time" in result_latlon.dims
        # Check if the calculation ran
        assert result_latlon["var2"].dtype == float

    def test_read_csv_file(self):
        # Create a temporary CSV file for testing

        with tempfile.NamedTemporaryFile(suffix=".csv", delete=False) as tmp:
            # Create test data
            data = {
                "column1": [1, 2, 3],
                "column2": [4, 5, 6],
                "column3": [7, 8, 9],
            }
            df = pd.DataFrame(data)
            df.to_csv(tmp.name, index=False)

            # Test the read_csv_file function
            with patch(
                "climakitae.util.utils._package_file_path", return_value=tmp.name
            ):
                result = read_csv_file("dummy_path")  # The actual path will be mocked

                # Verify results
                assert isinstance(result, pd.DataFrame)
                assert len(result) == 3
                assert list(result.columns) == ["column1", "column2", "column3"]

        # Clean up
        os.unlink(tmp.name)

    @patch("pandas.DataFrame.to_csv")
    @patch("climakitae.util.utils._package_file_path")
    def test_write_csv_file(self, mock_package_file_path, mock_to_csv):
        # Create a temporary CSV file for testing
        with tempfile.NamedTemporaryFile(suffix=".csv", delete=False) as tmp:
            mock_package_file_path.return_value = tmp.name

            # Create test data
            data = {
                "column1": [1, 2, 3],
                "column2": [4, 5, 6],
                "column3": [7, 8, 9],
            }
            df = pd.DataFrame(data)

            # Test the write_csv_file function
            write_csv_file(df, "dummy_path")  # The actual path will be mocked

            # Verify that to_csv was called with the correct arguments
            mock_to_csv.assert_called_once_with(tmp.name)

        # Clean up
        os.unlink(tmp.name)

    def test_package_file_path(self):
        # Test with a valid package name and file name
        package_name = "climakitae"
        file_name = "test_file.txt"
        expected_path = os.path.join(
            os.path.dirname(__file__), "..", "..", package_name, file_name
        )
        # Normalize the expected path to remove the relative path components
        expected_path = os.path.normpath(expected_path)
        assert _package_file_path(file_name) == expected_path

    def test_get_closest_gridcell(self):
        # Create a mock dataset
        ds = xr.Dataset(
            {
            "var": (("lat", "lon"), np.random.rand(5, 5)),
            },
            coords={
            "lat": [10, 20, 30, 40, 50],
            "lon": [100, 110, 120, 130, 140],
            },
        )
        # Add resolution attributes in km (approx. 111 km per degree)
        ds.attrs["resolution"] = "1110 km"
        ds.lat.attrs["resolution"] = 1110.0  # resolution in km (10 degrees ≈ 1110 km)
        ds.lon.attrs["resolution"] = 1110.0  # resolution in km (10 degrees ≈ 1110 km)

        # Test with a point inside the grid
        lat = 25
        lon = 115
        closest_ds = get_closest_gridcell(ds, lat, lon)
        assert isinstance(closest_ds, xr.Dataset)
        assert closest_ds.coords["lat"].item() in ds.coords["lat"].values
        assert closest_ds.coords["lon"].item() in ds.coords["lon"].values

        # Test with a point outside the grid
        lat = 60
        lon = 150
        closest_ds = get_closest_gridcell(ds, lat, lon)
        assert isinstance(closest_ds, xr.Dataset)
        assert closest_ds.coords["lat"].item() in ds.coords["lat"].values
        assert closest_ds.coords["lon"].item() in ds.coords["lon"].values

