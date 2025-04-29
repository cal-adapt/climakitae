import datetime
import os
import tempfile
from unittest.mock import MagicMock, PropertyMock, patch

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from climakitae.util.utils import (
    _package_file_path,
    area_average,
    combine_hdd_cdd,
    compute_annual_aggreggate,
    compute_multimodel_stats,
    downscaling_method_as_list,
    get_closest_gridcell,
    get_closest_gridcells,
    julianDay_to_date,
    read_csv_file,
    readable_bytes,
    reproject_data,
    trendline,
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

    def test_compute_annual_aggreggate(self):
        """Tests the compute_annual_aggreggate function with various inputs"""
        # Create a mock dataset with time dimension spanning multiple years
        time = pd.date_range("2020-01-01", periods=730, freq="D")  # 2 years of data
        lat = [10, 20]
        lon = [100, 110]
        data = np.ones((len(time), len(lat), len(lon))) * 2  # All values are 2
        da = xr.DataArray(
            data,
            dims=("time", "lat", "lon"),
            coords={"time": time, "lat": lat, "lon": lon},
        )

        # Test with name parameter and grid cell count
        name = "HDD"
        num_grid_cells = 4  # 2x2 grid
        result = compute_annual_aggreggate(da, name, num_grid_cells)

        # Check the result has expected properties
        assert isinstance(result, xr.DataArray)
        assert "year" in result.dims
        assert result.sizes["year"] == 2  # Should have 2 years
        assert result.name == name

        # Expected values: 365 days * 2 (value per day) / 4 (grid cells) = 182.5 for non-leap year
        # and 366 days * 2 / 4 = 183 for leap year (2020)
        np.testing.assert_allclose(result.sel(year=2020).values, 183, rtol=1e-2)
        np.testing.assert_allclose(result.sel(year=2021).values, 182.5, rtol=1e-2)

        # Test with different name and grid cell count
        name2 = "CDD"
        num_grid_cells2 = 2
        result2 = compute_annual_aggreggate(da, name2, num_grid_cells2)
        assert result2.name == name2

        # Expected values doubled since we're dividing by half as many grid cells
        np.testing.assert_allclose(result2.sel(year=2020).values, 366, rtol=1e-2)
        np.testing.assert_allclose(result2.sel(year=2021).values, 365, rtol=1e-2)

        # Test with extra dimensions that should be squeezed
        da_extra_dim = da.expand_dims(dim={"model": ["A"]})
        result3 = compute_annual_aggreggate(da_extra_dim, name, num_grid_cells)
        assert "model" not in result3.dims

    def test_compute_multimodel_stats(self):
        """Tests the compute_multimodel_stats function"""
        # Create a mock dataset with simulation dimension
        time = pd.date_range("2020-01-01", periods=30)
        simulations = ["sim1", "sim2", "sim3"]

        # Create random data with different values for each simulation
        data = np.random.rand(len(time), len(simulations))
        # Make sure values are distinct to test min/max properly
        data[:, 0] *= 2  # sim1 values are doubled
        data[:, 2] += 3  # sim3 values are shifted upward

        da = xr.DataArray(
            data,
            dims=("time", "simulation"),
            coords={"time": time, "simulation": simulations},
            name="test_variable",
        )

        # Run the function
        result = compute_multimodel_stats(da)

        # Verify the result has the expected structure
        assert isinstance(result, xr.DataArray)
        assert "simulation" in result.dims
        assert (
            len(result.simulation) == len(simulations) + 4
        )  # original + mean, min, max, median

        # Check the result contains original data
        for sim in simulations:
            assert sim in result.simulation.values
            np.testing.assert_array_equal(
                result.sel(simulation=sim).values, da.sel(simulation=sim).values
            )

        # Check the computed stats
        # Mean
        assert "simulation mean" in result.simulation.values
        np.testing.assert_allclose(
            result.sel(simulation="simulation mean").values,
            da.mean(dim="simulation").values,
        )

        # Min
        assert "simulation min" in result.simulation.values
        np.testing.assert_allclose(
            result.sel(simulation="simulation min").values,
            da.min(dim="simulation").values,
        )

        # Max
        assert "simulation max" in result.simulation.values
        np.testing.assert_allclose(
            result.sel(simulation="simulation max").values,
            da.max(dim="simulation").values,
        )

        # Median
        assert "simulation median" in result.simulation.values
        np.testing.assert_allclose(
            result.sel(simulation="simulation median").values,
            da.median(dim="simulation").values,
        )

    def test_trendline(self):
        """Tests the trendline function with various inputs"""
        # Create test data with year dimension and simulation coordinates
        years = np.array([2020, 2021, 2022, 2023])
        # Create synthetic data with known trend
        # Mean values follow y = 2x + 3 (m=2, b=3) where x is years starting from 2020
        mean_vals = 2 * (years - 2020) + 3  # [3, 5, 7, 9]
        # Median values follow y = x + 5 (m=1, b=5)
        median_vals = 1 * (years - 2020) + 5  # [5, 6, 7, 8]

        # Create simulations that will give us the desired mean and median
        sim1 = np.array([2, 3, 4, 5])  # Below median
        sim2 = np.array([5, 6, 7, 8])  # median
        sim3 = np.array([8, 9, 10, 11])  # Above median

        # Stack into a single array
        sim_data = np.stack([sim1, sim2, sim3])

        # Verify our data has the expected mean and median
        # The mean should be approximately [5, 6, 7, 8]
        assert np.allclose(np.mean(sim_data, axis=0), [5, 6, 7, 8])
        # The median should be exactly [5, 6, 7, 8]
        assert np.all(np.median(sim_data, axis=0) == median_vals)

        # Create MultiModel stats data that would be output by compute_multimodel_stats
        sim_mean = np.mean(sim_data, axis=0)
        sim_min = np.min(sim_data, axis=0)
        sim_max = np.max(sim_data, axis=0)
        sim_median = np.median(sim_data, axis=0)

        # Stack into array with all simulations and stats
        all_data = np.vstack([sim_data, sim_mean, sim_min, sim_max, sim_median])

        # Create DataArray with expected structure
        simulations = [
            "sim1",
            "sim2",
            "sim3",
            "simulation mean",
            "simulation min",
            "simulation max",
            "simulation median",
        ]
        da = xr.DataArray(
            all_data,
            dims=("simulation", "year"),
            coords={"simulation": simulations, "year": years},
            name="test_variable",
        )

        # Test mean trendline
        mean_trend = trendline(da, kind="mean")

        # Verify basic properties
        assert isinstance(mean_trend, xr.DataArray)
        assert "year" in mean_trend.dims
        assert mean_trend.name == "trendline"
        assert len(mean_trend) == len(years)

        # The mean follows y = x + 5 (close enough with our simulated data)
        expected_mean_values = 1 * (years - 2020) + 5
        np.testing.assert_allclose(mean_trend.values, expected_mean_values, rtol=1e-1)

        # Test median trendline
        median_trend = trendline(da, kind="median")

        # Verify basic properties
        assert isinstance(median_trend, xr.DataArray)
        assert "year" in median_trend.dims
        assert median_trend.name == "trendline"

        # Verify values follow expected trend (m=1, b=5)
        expected_median_values = 1 * (years - 2020) + 5
        np.testing.assert_allclose(
            median_trend.values, expected_median_values, rtol=1e-5
        )

        # Test error handling when simulation stats not available
        da_no_stats = da.sel(simulation=["sim1", "sim2", "sim3"])

        # Check that the function raises an exception for missing mean stats
        with pytest.raises(
            ValueError,
            match="Invalid data provided, please pass the multimodel stats from compute_multimodel_stats",
        ):
            trendline(da_no_stats, kind="mean")

        # Check that the function raises an exception for missing median stats
        with pytest.raises(
            ValueError,
            match="Invalid data provided, please pass the multimodel stats from compute_multimodel_stats",
        ):
            trendline(da_no_stats, kind="median")

        # Test with invalid kind parameter
        with pytest.raises(ValueError):
            trendline(da, kind="invalid_kind")

    def test_combine_hdd_cdd(self):
        """Tests the combine_hdd_cdd function with various input conditions"""

        # Test with valid HDD/CDD data
        valid_names = [
            "Annual Heating Degree Days (HDD)",
            "Annual Cooling Degree Days (CDD)",
            "Heating Degree Hours",
            "Cooling Degree Hours",
        ]

        for name in valid_names:
            # Create mock data array with the coordinates to be dropped
            mock_data = xr.DataArray(
                np.ones((2, 3)),
                dims=["time", "location"],
                coords={
                    "time": [1, 2],
                    "location": ["A", "B", "C"],
                    "scenario": "test_scenario",
                    "Lambert_Conformal": 123,
                    "variable": "test_variable",
                },
                name=name,
            )

            result = combine_hdd_cdd(mock_data)

            # Check correct coordinates were dropped
            assert "scenario" not in result.coords
            assert "Lambert_Conformal" not in result.coords
            assert "variable" not in result.coords

            # Check the data wasn't altered
            np.testing.assert_array_equal(result.values, mock_data.values)

            # Check the name was preserved
            assert result.name == name

        # Test with data missing some of the coordinates to drop
        partial_coords_data = xr.DataArray(
            np.ones((2, 3)),
            dims=["time", "location"],
            coords={
                "time": [1, 2],
                "location": ["A", "B", "C"],
                "scenario": "test_scenario",  # only has one of the coordinates to drop
            },
            name="Annual Heating Degree Days (HDD)",
        )

        result = combine_hdd_cdd(partial_coords_data)
        assert "scenario" not in result.coords
        assert len(result.coords) == 2  # Only time and location remain

        # Test with data having none of the coordinates to drop
        no_coords_data = xr.DataArray(
            np.ones((2, 3)),
            dims=["time", "location"],
            coords={
                "time": [1, 2],
                "location": ["A", "B", "C"],
            },
            name="Annual Cooling Degree Days (CDD)",
        )

        result = combine_hdd_cdd(no_coords_data)
        # Function should not modify the data
        assert result.equals(no_coords_data)

        # Test with invalid data name
        invalid_data = xr.DataArray(np.ones(3), dims=["x"], name="Temperature")

        with pytest.raises(
            ValueError,
            match="Invalid data provided, please pass cooling/heating degree data",
        ):
            combine_hdd_cdd(invalid_data)


class TestReprojectData:
    """
    Class for testing the reproject_data function.
    """

    def test_reproject_data_2d(self):
        """test reproject_data with 2D data"""
        # Create a mock return value for reproject
        mock_reprojected = xr.DataArray(
            [[1, 2], [3, 4]], coords={"y": [1, 2], "x": [3, 4]}
        )

        # Create a mock accessor with the necessary methods
        mock_rio_accessor = MagicMock()
        mock_rio_accessor.crs = "EPSG:3857"
        mock_rio_accessor.reproject.return_value = mock_reprojected

        # Patch the actual accessor property to return our mock
        with patch("rioxarray.open_rasterio", autospec=True), patch.object(
            xr.DataArray,
            "rio",
            new_callable=PropertyMock,
            return_value=mock_rio_accessor,
        ), patch.object(xr.DataArray.rio, "write_crs", return_value=None):

            data_2d = xr.DataArray([[1, 2], [3, 4]], coords={"x": [1, 2], "y": [3, 4]})
            result_2d = reproject_data(data_2d)
            assert result_2d.attrs["grid_mapping"] == "EPSG:4326"

    def test_reproject_data_3d(self):
        """Test reproject_data with 3D data"""
        # Create a mock return value for reproject
        mock_reprojected = xr.DataArray(
            [[1, 2], [3, 4]], coords={"y": [1, 2], "x": [3, 4]}
        )

        # Create a mock accessor with the necessary methods
        mock_rio_accessor = MagicMock()
        mock_rio_accessor.crs = "EPSG:3857"
        mock_rio_accessor.reproject.return_value = mock_reprojected

        # Patch the actual accessor property to return our mock
        with patch("rioxarray.open_rasterio", autospec=True), patch.object(
            xr.DataArray,
            "rio",
            new_callable=PropertyMock,
            return_value=mock_rio_accessor,
        ), patch.object(xr.DataArray.rio, "write_crs", return_value=None):
            data_3d = xr.DataArray(
                np.zeros((3, 2, 2)),
                coords={
                    "time": [1, 2, 3],
                    "x": [1, 2],
                    "y": [3, 4],
                },
            )
            result_3d = reproject_data(data_3d)
            assert result_3d.attrs["grid_mapping"] == "EPSG:4326"

    def test_reproject_data_4d_5d(self):
        """Test reproject_data with 4D and 5D data"""
        # Create a mock return value for reproject
        mock_reprojected = xr.DataArray(
            [[1, 2], [3, 4]], coords={"y": [1, 2], "x": [3, 4]}
        )

        # Create mock accessor with the necessary methods
        mock_rio_accessor = MagicMock()
        mock_rio_accessor.crs = "EPSG:3857"
        mock_rio_accessor.reproject.return_value = mock_reprojected

        # Create mock concat return values for 4D and 5D
        mock_4d = xr.DataArray(
            np.zeros((2, 2, 2, 2)),
            coords={"time": [1, 2], "extra_dim": [1, 2], "x": [1, 2], "y": [3, 4]},
        )
        mock_5d = xr.DataArray(
            np.zeros((2, 2, 2, 2, 2)),
            coords={
                "scenario": [1, 2],
                "time": [1, 2],
                "extra_dim": [1, 2],
                "x": [1, 2],
                "y": [3, 4],
            },
        )

        # Test 4D case
        with patch("rioxarray.open_rasterio", autospec=True), patch.object(
            xr.DataArray,
            "rio",
            new_callable=PropertyMock,
            return_value=mock_rio_accessor,
        ), patch.object(xr.DataArray.rio, "write_crs", return_value=None), patch(
            "xarray.concat", return_value=mock_4d
        ):

            data_4d = xr.DataArray(
                np.zeros((2, 3, 2, 2)),
                coords={
                    "extra_dim": [1, 2],
                    "time": [1, 2, 3],
                    "x": [1, 2],
                    "y": [3, 4],
                },
            )
            result_4d = reproject_data(data_4d)
            assert result_4d.attrs["grid_mapping"] == "EPSG:4326"

        # Test 5D case
        with patch("rioxarray.open_rasterio", autospec=True), patch.object(
            xr.DataArray,
            "rio",
            new_callable=PropertyMock,
            return_value=mock_rio_accessor,
        ), patch.object(xr.DataArray.rio, "write_crs", return_value=None), patch(
            "xarray.concat", return_value=mock_5d
        ):

            data_5d = xr.DataArray(
                np.zeros((2, 2, 3, 2, 2)),
                coords={
                    "scenario": [1, 2],
                    "extra_dim": [1, 2],
                    "time": [1, 2, 3],
                    "x": [1, 2],
                    "y": [3, 4],
                },
            )
            result_5d = reproject_data(data_5d)
            assert result_5d.attrs["grid_mapping"] == "EPSG:4326"

    def test_reproject_data_errors(self):
        """Test error cases in reproject_data"""
        # Create a mock accessor with the necessary methods
        mock_rio_accessor = MagicMock()
        mock_rio_accessor.crs = "EPSG:3857"

        # Test missing spatial dimensions
        with patch.object(
            xr.DataArray,
            "rio",
            new_callable=PropertyMock,
            return_value=mock_rio_accessor,
        ):
            data_invalid = xr.DataArray(
                [[1, 2], [3, 4]], coords={"lat": [1, 2], "lon": [3, 4]}
            )
            with pytest.raises(
                ValueError, match="does not contain spatial dimensions x,y"
            ):
                reproject_data(data_invalid)

        # Test too many dimensions
        with patch.object(
            xr.DataArray,
            "rio",
            new_callable=PropertyMock,
            return_value=mock_rio_accessor,
        ):
            data_6d = xr.DataArray(
                np.zeros((2, 2, 2, 2, 2, 2)),
                coords={
                    "model": [1, 2],
                    "scenario": [1, 2],
                    "extra_dim": [1, 2],
                    "time": [1, 2],
                    "x": [1, 2],
                    "y": [3, 4],
                },
            )
            with pytest.raises(
                ValueError,
                match="dimensions greater than 5 are not currently supported",
            ):
                reproject_data(data_6d)
