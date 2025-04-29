import datetime
import os
import tempfile
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from climakitae.util.utils import (
    _package_file_path,
    area_average,
    downscaling_method_as_list,
    get_closest_gridcell,
    get_closest_gridcells,
    julianDay_to_date,
    read_csv_file,
    readable_bytes,
    write_csv_file,
)


class TestUtils:
    """
    Class for testing functions in the utils module.
    """

    def test_downscaling_method_as_list(self):
        """tests the downscaling_method_as_list function"""
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
        """tests the area_average function"""
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
        """tests the read_csv_file function"""
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
    def test_write_csv_file(
        self, mock_package_file_path: MagicMock, mock_to_csv: MagicMock
    ):
        """tests the write_csv_file function"""
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
        """tests the _package_file_path function"""
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
        """tests the get_closest_gridcell function"""
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

    def test_get_closest_gridcells(self):
        """tests the get_closest_gridcells function"""
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
        ds.lat.attrs["resolution"] = 1110.0
        ds.lon.attrs["resolution"] = 1110.0

        # Test with a list of points inside the grid
        lats = [25, 35]
        lons = [115, 125]
        closest_dss = get_closest_gridcells(ds, lats, lons)

        # Function returns a dataset with a "points" dimension
        assert isinstance(closest_dss, xr.Dataset)
        assert "points" in closest_dss.dims
        assert closest_dss.sizes["points"] == len(lats)

        # Check that each point has coordinates from the original dataset
        for i in range(len(lats)):
            point_ds = closest_dss.isel(points=i)
            assert point_ds.coords["lat"].item() in ds.coords["lat"].values
            assert point_ds.coords["lon"].item() in ds.coords["lon"].values

        # Test with a list of points outside the grid
        lats = [60, 70]
        lons = [150, 160]
        closest_dss = get_closest_gridcells(ds, lats, lons)

        # Function returns a dataset with a "points" dimension
        assert closest_dss is None

    def test_julianDay_to_date(self):
        """tests the julianDay_to_date function"""
        # Test default return_type (str) with default format
        assert julianDay_to_date(1, year=2023) == "Jan-01"
        assert julianDay_to_date(32, year=2023) == "Feb-01"
        assert julianDay_to_date(365, year=2023) == "Dec-31"

        # Test with custom str_format
        assert julianDay_to_date(1, year=2023, str_format="%Y-%m-%d") == "2023-01-01"
        assert (
            julianDay_to_date(60, year=2024, str_format="%Y-%m-%d") == "2024-02-29"
        )  # Leap year

        # Test datetime return type
        dt_result = julianDay_to_date(1, year=2023, return_type="datetime")
        assert dt_result.year == 2023
        assert dt_result.month == 1
        assert dt_result.day == 1

        # Test date return type
        date_result = julianDay_to_date(32, year=2023, return_type="date")
        assert date_result.year == 2023
        assert date_result.month == 2
        assert date_result.day == 1

        # Test leap year handling
        feb29 = julianDay_to_date(
            60, year=2024, return_type="date"
        )  # 2024 is leap year
        assert feb29.month == 2
        assert feb29.day == 29

        # Test invalid return_type raises ValueError
        with pytest.raises(
            ValueError, match="return_type must be 'str', 'datetime', or 'date'"
        ):
            julianDay_to_date(1, return_type="invalid")

        # Test automatic year determination (using mock to avoid year-dependent tests)
        with patch("climakitae.util.utils.datetime.datetime", autospec=True) as mock_dt:
            # Set up the return value for now()
            mock_dt.now.return_value = datetime.datetime(2023, 5, 15)

            # Set up the mock chain for strptime().strftime()
            mock_date_obj = MagicMock()
            mock_dt.strptime.return_value = mock_date_obj
            mock_date_obj.strftime.return_value = "2023-01-01"

            assert julianDay_to_date(1, str_format="%Y-%m-%d") == "2023-01-01"

    @staticmethod
    def test_readable_bytes():
        """Tests the readable_bytes function"""

        # Test bytes
        assert readable_bytes(500) == "500.0 bytes"
        assert readable_bytes(0) == "0.0 bytes"

        # Test KB
        assert readable_bytes(1024) == "1.00 KB"
        assert readable_bytes(1500) == "1.46 KB"
        assert readable_bytes(10240) == "10.00 KB"

        # Test MB
        assert readable_bytes(1048576) == "1.00 MB"  # 1024^2
        assert readable_bytes(2097152) == "2.00 MB"  # 2 * 1024^2
        assert readable_bytes(5242880) == "5.00 MB"  # 5 * 1024^2

        # Test GB
        assert readable_bytes(1073741824) == "1.00 GB"  # 1024^3
        assert readable_bytes(3221225472) == "3.00 GB"  # 3 * 1024^3

        # Test TB
        assert readable_bytes(1099511627776) == "1.00 TB"  # 1024^4
        assert readable_bytes(2199023255552) == "2.00 TB"  # 2 * 1024^4

        # Test edge cases
        assert readable_bytes(1023) == "1023.0 bytes"
        assert readable_bytes(1024**2 - 1) == "1024.00 KB"
        assert readable_bytes(1024**3 - 1) == "1024.00 MB"
        assert readable_bytes(1024**4 - 1) == "1024.00 GB"

