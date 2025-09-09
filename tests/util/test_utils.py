import datetime
import os
import tempfile
from unittest.mock import MagicMock, PropertyMock, patch

import geopandas as gpd
import numpy as np
import pandas as pd
import pyproj
import pytest
import xarray as xr
from shapely.geometry import Point, box

from climakitae.util.utils import (  # stack_sims_across_locs, # TODO: Uncomment when implemented
    _get_cat_subset,
    _get_scenario_from_selections,
    _package_file_path,
    add_dummy_time_to_wl,
    area_average,
    clip_gpd_to_shapefile,
    clip_to_shapefile,
    combine_hdd_cdd,
    compute_annual_aggreggate,
    compute_multimodel_stats,
    convert_to_local_time,
    downscaling_method_as_list,
    downscaling_method_to_activity_id,
    get_closest_gridcell,
    get_closest_gridcells,
    julianDay_to_date,
    read_csv_file,
    readable_bytes,
    reproject_data,
    resolution_to_gridlabel,
    scenario_to_experiment_id,
    summary_table,
    timescale_to_table_id,
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

    def test_get_closest_gridcell_with_projection(self):
        """Tests the coordinate transformation in get_closest_gridcell function"""
        # Create a mock dataset with x, y dimensions and lat/lon coordinates to satisfy the function
        ds = xr.Dataset(
            {
                "var": (("x", "y"), np.random.rand(5, 5)),
            },
            coords={
                "x": [10, 20, 30, 40, 50],
                "y": [100, 110, 120, 130, 140],
                # Add these coordinates to satisfy the function's attempt to access lat/lon
                "lat": 37.5,
                "lon": -122.5,
            },
        )
        # Add resolution attributes in km
        ds.attrs["resolution"] = "1110 km"
        ds.x.attrs["resolution"] = 1110.0
        ds.y.attrs["resolution"] = 1110.0

        # Mock the rio accessor and its attributes/methods
        mock_rio_accessor = MagicMock()
        mock_rio_accessor.crs = "EPSG:3857"  # Web Mercator projection

        # Mock the Transformer object
        mock_transformer = MagicMock()
        mock_transformer.transform.return_value = (
            50,
            140,
        )  # Values in our coordinate range

        # Test coordinates (these would be transformed)
        lat, lon = 37.7749, -122.4194  # San Francisco coordinates

        # Apply patches for the test
        with patch.object(
            xr.Dataset, "rio", new_callable=PropertyMock, return_value=mock_rio_accessor
        ), patch("pyproj.Transformer.from_crs", return_value=mock_transformer):

            result = get_closest_gridcell(ds, lat, lon)

            # Check that from_crs was called with the correct parameters
            pyproj.Transformer.from_crs.assert_called_once_with(
                crs_from="epsg:4326", crs_to=mock_rio_accessor.crs, always_xy=True
            )

            # Check that transform was called with the correct parameters
            mock_transformer.transform.assert_called_once_with(lon, lat)

            # Check that the correct grid cell was selected
            assert result.x.item() == 50
            assert result.y.item() == 140

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

    @pytest.mark.advanced
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

    def test_summary_table(self):
        """Tests the summary_table function with both time and year dimensions"""

        # Test with time dimension
        time_data = xr.Dataset(
            {
                "var1": (
                    ("time", "scenario", "simulation"),
                    np.random.rand(3, 2, 2),
                ),
            },
            coords={
                "time": pd.date_range("2020-01-01", periods=3),
                "scenario": ["ssp245", "ssp585"],
                "simulation": ["sim1", "sim2"],
                "lakemask": 0,
                "landmask": 1,
                "lat": 35.0,
                "lon": -120.0,
                "Lambert_Conformal": "projection",
                "x": 100,
                "y": 200,
            },
        )

        # Test with year dimension
        year_data = xr.Dataset(
            {
                "var1": (
                    ("year", "scenario", "simulation"),
                    np.random.rand(3, 2, 2),
                ),
            },
            coords={
                "year": [2020, 2021, 2022],
                "scenario": ["ssp245", "ssp585"],
                "simulation": ["sim1", "sim2"],
                "lakemask": 0,
                "landmask": 1,
                "lat": 35.0,
                "lon": -120.0,
                "Lambert_Conformal": "projection",
                "x": 100,
                "y": 200,
            },
        )

        # Run the function on time data
        time_df = summary_table(time_data)

        # Run the function on year data
        year_df = summary_table(year_data)

        # Check that output is a dataframe
        assert isinstance(time_df, pd.DataFrame)
        assert isinstance(year_df, pd.DataFrame)

        # Check that coordinates were dropped
        for df in [time_df, year_df]:
            for coord in [
                "lakemask",
                "landmask",
                "lat",
                "lon",
                "Lambert_Conformal",
                "x",
                "y",
            ]:
                assert coord not in df.index.names

        # Check that the dataframe has the expected structure
        # For time dimension data
        assert "time" in time_df.index.names
        assert set(time_df.columns.levels[0]) == {"var1"}
        assert set(time_df.columns.levels[1]) == {"sim1", "sim2"}
        assert (
            time_df.index.get_level_values("time")[0]
            <= time_df.index.get_level_values("time")[-1]
        )

        # For year dimension data
        assert "year" in year_df.index.names
        assert set(year_df.columns.levels[0]) == {"var1"}
        assert set(year_df.columns.levels[1]) == {"sim1", "sim2"}
        assert (
            year_df.index.get_level_values("year")[0]
            <= year_df.index.get_level_values("year")[-1]
        )

    def test_add_dummy_time_to_wl(self):
        """Tests the add_dummy_time_to_wl function with various inputs"""

        # Test 1: DataArray with days_from_center dimension
        days = np.arange(-10, 11)
        data = np.random.rand(
            len(days), 2, 3
        )  # Sample data with days and two other dimensions
        coords = {
            "days_from_center": days,
            "simulation": ["sim1", "sim2"],
            "variable": ["var1", "var2", "var3"],
        }
        da_days = xr.DataArray(
            data, dims=["days_from_center", "simulation", "variable"], coords=coords
        )

        result_days = add_dummy_time_to_wl(da_days)

        # Check that the time dimension replaced days_from_center
        assert "days_from_center" not in result_days.dims
        assert "time" in result_days.dims
        # Check that the length of time dimension matches the original
        assert len(result_days.time) == len(days)
        # Check that the timestamps are daily frequency starting from 2000-01-01
        expected_timestamps = pd.date_range("2000-01-01", periods=len(days), freq="D")
        np.testing.assert_array_equal(
            result_days.time.values, expected_timestamps.values
        )

        # Test 2: DataArray with time_delta dimension
        time_delta = np.arange(-5, 6)
        data = np.random.rand(len(time_delta), 2)
        da_time_delta = xr.DataArray(
            data,
            dims=["time_delta", "simulation"],
            coords={
                "time_delta": time_delta,
                "simulation": ["sim1", "sim2"],
            },
        )
        # Add frequency attribute required for time_delta dimension
        da_time_delta.attrs["frequency"] = "hourly"

        result_time_delta = add_dummy_time_to_wl(da_time_delta)

        # Check that time dimension replaced time_delta
        assert "time_delta" not in result_time_delta.dims
        assert "time" in result_time_delta.dims
        # Check that timestamps are hourly frequency
        expected_hourly = pd.date_range("2000-01-01", periods=len(time_delta), freq="h")
        np.testing.assert_array_equal(
            result_time_delta.time.values, expected_hourly.values
        )

        # Test 3: DataArray with months_from_center dimension
        months = np.arange(-3, 4)
        data = np.random.rand(len(months), 3)
        da_months = xr.DataArray(
            data,
            dims=["months_from_center", "simulation"],
            coords={
                "months_from_center": months,
                "simulation": ["sim1", "sim2", "sim3"],
            },
        )

        result_months = add_dummy_time_to_wl(da_months)

        # Check that time dimension replaced months_from_center
        assert "months_from_center" not in result_months.dims
        assert "time" in result_months.dims
        # Check that timestamps are month-end frequency
        expected_months = pd.date_range("2000-01-01", periods=len(months), freq="MS")
        np.testing.assert_array_equal(result_months.time.values, expected_months.values)

        # Test 4: DataArray with hours_from_center dimension
        hours = np.arange(-12, 13)
        data = np.random.rand(len(hours))
        da_hours = xr.DataArray(
            data,
            dims=["hours_from_center"],
            coords={"hours_from_center": hours},
        )

        result_hours = add_dummy_time_to_wl(da_hours)

        # Check that time dimension replaced hours_from_center
        assert "hours_from_center" not in result_hours.dims
        assert "time" in result_hours.dims
        # Check that timestamps are hourly frequency
        expected_hours = pd.date_range("2000-01-01", periods=len(hours), freq="h")
        np.testing.assert_array_equal(result_hours.time.values, expected_hours.values)

        # Test 5: Error handling for DataArray without proper time dimension
        invalid_da = xr.DataArray(
            np.random.rand(5, 5),
            dims=["x", "y"],
            coords={"x": range(5), "y": range(5)},
        )

        with pytest.raises(
            ValueError,
            match="DataArray does not contain necessary warming level information",
        ):
            add_dummy_time_to_wl(invalid_da)

    def test_downscaling_method_to_activity_id(self):
        """Test downscaling_method_to_activity_id with different inputs"""
        # Test forward mapping (default behavior)
        assert downscaling_method_to_activity_id("Dynamical") == "WRF"
        assert downscaling_method_to_activity_id("Statistical") == "LOCA2"

        # Test reverse mapping
        assert downscaling_method_to_activity_id("WRF", reverse=True) == "Dynamical"
        assert downscaling_method_to_activity_id("LOCA2", reverse=True) == "Statistical"

        # Test error case with invalid input
        with pytest.raises(KeyError):
            downscaling_method_to_activity_id("Invalid Method")

        # Test error case with invalid input in reverse mode
        with pytest.raises(KeyError):
            downscaling_method_to_activity_id("Invalid ID", reverse=True)

    def test_resolution_to_gridlabel(self):
        """Test resolution_to_gridlabel with different inputs"""
        # Test forward mapping (default behavior)
        assert resolution_to_gridlabel("45 km") == "d01"
        assert resolution_to_gridlabel("9 km") == "d02"
        assert resolution_to_gridlabel("3 km") == "d03"

        # Test reverse mapping
        assert resolution_to_gridlabel("d01", reverse=True) == "45 km"
        assert resolution_to_gridlabel("d02", reverse=True) == "9 km"
        assert resolution_to_gridlabel("d03", reverse=True) == "3 km"

        # Test error case with invalid input
        with pytest.raises(KeyError):
            resolution_to_gridlabel("invalid resolution")

        # Test error case with invalid input in reverse mode
        with pytest.raises(KeyError):
            resolution_to_gridlabel("invalid grid label", reverse=True)

    def test_timescale_to_table_id(self):
        """Test timescale_to_table_id with different inputs and reverse parameter"""
        # Test forward mapping (default behavior)
        assert timescale_to_table_id("monthly") == "mon"
        assert timescale_to_table_id("daily") == "day"
        assert timescale_to_table_id("hourly") == "1hr"
        assert timescale_to_table_id("yearly_max") == "yrmax"

        # Test reverse mapping
        assert timescale_to_table_id("mon", reverse=True) == "monthly"
        assert timescale_to_table_id("day", reverse=True) == "daily"
        assert timescale_to_table_id("1hr", reverse=True) == "hourly"
        assert timescale_to_table_id("yrmax", reverse=True) == "yearly_max"

        # Test error case with invalid input
        with pytest.raises(KeyError):
            timescale_to_table_id("invalid_timescale")

        # Test error case with invalid input in reverse mode
        with pytest.raises(KeyError):
            timescale_to_table_id("invalid_table_id", reverse=True)

    def test_scenario_to_experiment_id(self):
        """Test scenario_to_experiment_id with different inputs and in reverse mode"""
        # Test forward mapping (default behavior)
        assert scenario_to_experiment_id("Historical Reconstruction") == "reanalysis"
        assert scenario_to_experiment_id("Historical Climate") == "historical"
        assert scenario_to_experiment_id("SSP 2-4.5") == "ssp245"
        assert scenario_to_experiment_id("SSP 5-8.5") == "ssp585"
        assert scenario_to_experiment_id("SSP 3-7.0") == "ssp370"

        # Test reverse mapping
        assert (
            scenario_to_experiment_id("reanalysis", reverse=True)
            == "Historical Reconstruction"
        )
        assert (
            scenario_to_experiment_id("historical", reverse=True)
            == "Historical Climate"
        )
        assert scenario_to_experiment_id("ssp245", reverse=True) == "SSP 2-4.5"
        assert scenario_to_experiment_id("ssp585", reverse=True) == "SSP 5-8.5"
        assert scenario_to_experiment_id("ssp370", reverse=True) == "SSP 3-7.0"

        # Test error case with invalid input
        with pytest.raises(KeyError):
            scenario_to_experiment_id("Invalid Scenario")

        # Test error case with invalid input in reverse mode
        with pytest.raises(KeyError):
            scenario_to_experiment_id("invalid_experiment_id", reverse=True)

    def test_get_cat_subset(self):
        """Tests the _get_cat_subset function with different selection cases"""
        # Create mock DataParameters
        mock_selections = MagicMock()
        mock_selections.variable_id = ["tas"]  # Non-derived variable
        mock_selections.downscaling_method = "Dynamical"
        mock_selections.resolution = "45 km"
        mock_selections.timescale = "monthly"
        mock_selections.simulation = ["sim1"]

        # Mock catalog and search results
        mock_catalog = MagicMock()
        mock_search_result = MagicMock()
        mock_catalog.search.return_value = mock_search_result
        mock_search_result.search.return_value = mock_search_result
        mock_selections._data_catalog = mock_catalog

        # Mock catalog df with institution IDs
        mock_catalog.df = pd.DataFrame({"institution_id": ["UCSD", "Other1", "Other2"]})

        # Mock scenario methods
        with patch(
            "climakitae.util.utils._get_scenario_from_selections",
            return_value=(["SSP 2-4.5"], ["Historical Climate"]),
        ):
            # Test case 1: Standard non-derived variable with Dynamical downscaling
            result = _get_cat_subset(mock_selections)

            # Check that catalog.search was called with correct parameters
            mock_catalog.search.assert_called_once_with(
                activity_id=["WRF"],
                table_id="mon",
                grid_label="d01",
                variable_id=["tas"],
                experiment_id=["ssp245", "historical"],
                source_id=["sim1"],
            )

            # Check that search was called to filter out UCSD institution
            mock_search_result.search.assert_called_once_with(
                institution_id=["Other1", "Other2"]
            )

            assert result == mock_search_result

        # Reset mocks for next test
        mock_catalog.search.reset_mock()
        mock_search_result.search.reset_mock()

        # Test case 2: Derived variable with Statistical downscaling
        mock_selections.variable_id = ["tas_derived"]
        mock_selections.downscaling_method = "Statistical"

        # Create mock variable descriptions dataframe
        var_desc_df = pd.DataFrame(
            {"variable_id": ["tas_derived"], "dependencies": ["tas,pr"]}
        )
        mock_selections._variable_descriptions = var_desc_df

        with patch(
            "climakitae.util.utils._get_scenario_from_selections",
            return_value=(["SSP 2-4.5"], ["Historical Climate"]),
        ):
            result = _get_cat_subset(mock_selections)

            # Check that catalog.search was called with correct parameters (first dependency)
            mock_catalog.search.assert_called_once_with(
                activity_id=["LOCA2"],
                table_id="mon",
                grid_label="d01",
                variable_id=["tas"],  # First dependency from "tas,pr"
                experiment_id=["ssp245", "historical"],
                source_id=["sim1"],
            )

            # Check that search was called to filter for UCSD institution
            mock_search_result.search.assert_called_once_with(institution_id="UCSD")

            assert result == mock_search_result

        # Reset mocks for next test
        mock_catalog.search.reset_mock()
        mock_search_result.search.reset_mock()

        # Test case 3: Combined downscaling methods
        mock_selections.variable_id = ["tas"]  # Non-derived
        mock_selections.downscaling_method = "Dynamical+Statistical"

        with patch(
            "climakitae.util.utils._get_scenario_from_selections",
            return_value=(["SSP 2-4.5"], ["Historical Climate"]),
        ), patch(
            "climakitae.util.utils.downscaling_method_as_list",
            return_value=["Dynamical", "Statistical"],
        ):
            result = _get_cat_subset(mock_selections)

            # Check that catalog.search was called with correct parameters (both methods)
            mock_catalog.search.assert_called_once_with(
                activity_id=["WRF", "LOCA2"],
                table_id="mon",
                grid_label="d01",
                variable_id=["tas"],
                experiment_id=["ssp245", "historical"],
                source_id=["sim1"],
            )

            # For combined methods, the UCSD filter would be applied
            mock_search_result.search.assert_called_once_with(institution_id="UCSD")

            assert result == mock_search_result

    def test_get_scenario_from_selections(self):
        """Tests the _get_scenario_from_selections function with Time and Warming Level approaches"""

        # Test case 1: Time approach
        mock_selections_time = MagicMock()
        mock_selections_time.approach = "Time"
        mock_selections_time.scenario_ssp = ["SSP 2-4.5"]
        mock_selections_time.scenario_historical = ["Historical Climate"]

        scenario_ssp, scenario_historical = _get_scenario_from_selections(
            mock_selections_time
        )

        # Verify that it returns the selections directly for Time approach
        assert scenario_ssp == ["SSP 2-4.5"]
        assert scenario_historical == ["Historical Climate"]

        # Test case 2: Warming Level approach
        mock_selections_wl = MagicMock()
        mock_selections_wl.approach = "Warming Level"

        # Mock the SSPS constant
        with patch(
            "climakitae.util.utils.SSPS", ["SSP 2-4.5", "SSP 3-7.0", "SSP 5-8.5"]
        ):
            scenario_ssp, scenario_historical = _get_scenario_from_selections(
                mock_selections_wl
            )

            # Verify that it returns all SSPs and Historical Climate for Warming Level approach
            assert scenario_ssp == ["SSP 2-4.5", "SSP 3-7.0", "SSP 5-8.5"]
            assert scenario_historical == ["Historical Climate"]


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


class TestConvertToLocalTime:
    """
    Class for testing the convert_to_local_time function.
    """

    def test_convert_to_local_time(self):
        """Test the convert_to_local_time function with various conditions."""

        # Create mock data and selections
        time_values = pd.date_range("2020-01-01T00:00:00", periods=24, freq="h")
        data = xr.DataArray(
            np.random.rand(len(time_values)),
            dims=["time"],
            coords={"time": time_values},
        )
        data.attrs = {"data_type": "Stations", "frequency": "monthly"}
        data.name = "SAN FRANCISCO DWTN"

        mock_stations_df = pd.DataFrame(
            {
                "station": ["SAN FRANCISCO DWTN"],
                "LAT_Y": [37.77],
                "LON_X": [-122.42],
                "Unnamed: 0": [1],
            }
        )

        # Test 1: Monthly data (should return original data)
        # Mock the stations dataframe
        with patch(
            "climakitae.util.utils.read_csv_file",
            return_value=mock_stations_df,
        ), patch(
            "climakitae.util.utils.TimezoneFinder.timezone_at",
            return_value="America/Los_Angeles",
        ), patch(
            "builtins.print"
        ) as mock_print:
            result = convert_to_local_time(data)
            assert result.equals(data)
            mock_print.assert_called_once_with(
                "This dataset's timescale is not granular enough to covert to local time. Local timezone conversion requires hourly data."
            )

        # Test 2: Hourly data with no name
        # Mock the stations dataframe
        data.attrs["frequency"] = "hourly"
        data.name = None
        with patch(
            "climakitae.util.utils.read_csv_file",
            return_value=mock_stations_df,
        ), patch(
            "climakitae.util.utils.TimezoneFinder.timezone_at",
            return_value="America/Los_Angeles",
        ), patch(
            "builtins.print"
        ) as mock_print:
            result = convert_to_local_time(data)
            assert result.equals(data)
            mock_print.assert_called_once_with(
                "Station None not found in Stations CSV. Please set Data Array name to valid station name."
            )

        # Test 3: Hourly data with mismatched name
        # Mock the stations dataframe
        data.name = "SAN FRANCISCO"
        with patch(
            "climakitae.util.utils.read_csv_file",
            return_value=mock_stations_df,
        ), patch(
            "climakitae.util.utils.TimezoneFinder.timezone_at",
            return_value="America/Los_Angeles",
        ), patch(
            "builtins.print"
        ) as mock_print:
            result = convert_to_local_time(data)
            assert result.equals(data)
            mock_print.assert_called_once_with(
                f"Station {data.name} not found in Stations CSV. Please set Data Array name to valid station name."
            )

        # Test 4: Station data type with timezone conversion
        # Mock the stations dataframe
        data.name = "SAN FRANCISCO DWTN"
        with patch(
            "climakitae.util.utils.read_csv_file",
            return_value=mock_stations_df,
        ), patch(
            "climakitae.util.utils.TimezoneFinder.timezone_at",
            return_value="America/Los_Angeles",
        ), patch(
            "builtins.print"
        ) as mock_print:

            result = convert_to_local_time(data)

            # Verify the timezone was set as an attribute
            assert result.attrs["timezone"] == "America/Los_Angeles"

            # Check if the print message about timezone conversion was shown
            mock_print.assert_called_with(
                "Data converted to America/Los_Angeles timezone."
            )

        # Test 4: Station data type with timezone conversion
        # where dataset is passed
        data = xr.DataArray(
            np.random.rand(len(time_values)),
            dims=["time"],
            coords={"time": time_values},
        )
        data.attrs = {"data_type": "Stations", "frequency": "hourly"}
        data.name = "SAN FRANCISCO DWTN"
        data = data.to_dataset()
        with patch(
            "climakitae.util.utils.read_csv_file",
            return_value=mock_stations_df,
        ), patch(
            "climakitae.util.utils.TimezoneFinder.timezone_at",
            return_value="America/Los_Angeles",
        ), patch(
            "builtins.print"
        ) as mock_print:

            result = convert_to_local_time(data)

            # Verify the timezone was set as an attribute
            assert (
                result["SAN FRANCISCO DWTN"].attrs["timezone"] == "America/Los_Angeles"
            )

            # Check if the print message about timezone conversion was shown
            mock_print.assert_called_with(
                "Data converted to America/Los_Angeles timezone."
            )

    def test_convert_to_local_time_no_data_type(self):
        """Test convert_to_local_time with data that has no type set."""
        # Create mock data with lat/lon coordinates
        time_values = pd.date_range("2020-01-01T00:00:00", periods=24, freq="h")
        lat_values = [34.0, 35.0]
        lon_values = [-118.0, -117.0]

        data = xr.DataArray(
            np.random.rand(len(time_values), len(lat_values), len(lon_values)),
            dims=["time", "lat", "lon"],
            coords={
                "time": time_values,
                "lat": lat_values,
                "lon": lon_values,
            },
        )
        data.attrs = {"frequency": "hourly"}

        with patch(
            "climakitae.util.utils.TimezoneFinder.timezone_at",
            return_value="America/Los_Angeles",
        ), patch("builtins.print") as mock_print:
            _ = convert_to_local_time(data)
            mock_print.assert_called_once_with(
                "Data Array attribute 'data_type' not found. Please set 'data_type' to 'Stations' or 'Gridded'."
            )

    def test_convert_to_local_time_gridded_data(self):
        """Test convert_to_local_time with gridded data types."""

        # Create mock data with lat/lon coordinates
        time_values = pd.date_range("2020-01-01T00:00:00", periods=24, freq="h")
        lat_values = [34.0, 35.0]
        lon_values = [-118.0, -117.0]

        data = xr.DataArray(
            np.random.rand(len(time_values), len(lat_values), len(lon_values)),
            dims=["time", "lat", "lon"],
            coords={
                "time": time_values,
                "lat": lat_values,
                "lon": lon_values,
            },
        )
        data.attrs = {"data_type": "Gridded", "frequency": "hourly"}

        # Test with gridded data and lat/lon area_subset
        with patch(
            "climakitae.util.utils.TimezoneFinder.timezone_at",
            return_value="America/Los_Angeles",
        ), patch("builtins.print") as mock_print:

            result = convert_to_local_time(data)

            # Verify the timezone was set as an attribute
            assert result.attrs["timezone"] == "America/Los_Angeles"

            # Check if the print message about timezone conversion was shown
            mock_print.assert_called_with(
                "Data converted to America/Los_Angeles timezone."
            )

            # The time dimension length may be modified when converting to local timezone
            # Instead of checking the exact length, we should verify that the timezone conversion
            # was performed correctly and that we have a valid time dimension
            assert "time" in result.dims
            assert len(result.time) > 0  # Ensure we have at least some time values

            # Optional: Verify that we're dealing with local time now (the key purpose)
            assert len(result.time) == len(time_values)
            # 8 hour offset between LA and UTC times
            new_times = time_values - pd.Timedelta(hours=8)
            assert (data.time == new_times).all()

        # Test with missing frequency and daily times
        time_values = pd.date_range(
            start="2020-01-01T00:00:00", end="2020-02-01T00:00:00", freq="D"
        )
        lat_values = [34.0, 35.0]
        lon_values = [-118.0, -117.0]
        data = xr.DataArray(
            np.random.rand(len(time_values), len(lat_values), len(lon_values)),
            dims=["time", "lat", "lon"],
            coords={
                "time": time_values,
                "lat": lat_values,
                "lon": lon_values,
            },
        )
        data.attrs = {"data_type": "Gridded"}

        with patch(
            "climakitae.util.utils.TimezoneFinder.timezone_at",
            return_value="America/Los_Angeles",
        ), patch("builtins.print") as mock_print:

            result = convert_to_local_time(data)

            # Check if the print message about timezone conversion was shown
            mock_print.assert_called_with(
                "This dataset's timescale is not granular enough to covert to local time. Local timezone conversion requires hourly data."
            )

        # Test with lat/lon provided for area averaged data
        time_values = pd.date_range("2020-01-01T00:00:00", periods=24, freq="h")
        data = xr.DataArray(
            np.random.rand(len(time_values)),
            dims=["time"],
            coords={
                "time": time_values,
            },
        )
        data.attrs = {"data_type": "Gridded", "frequency": "hourly"}

        with patch(
            "climakitae.util.utils.TimezoneFinder.timezone_at",
            return_value="America/Los_Angeles",
        ), patch("builtins.print") as mock_print:

            result = convert_to_local_time(data, -118.0, 34.0)

            # Verify the timezone was set as an attribute
            assert result.attrs["timezone"] == "America/Los_Angeles"

            # Check if the print message about timezone conversion was shown
            mock_print.assert_called_with(
                "Data converted to America/Los_Angeles timezone."
            )

    def test_clip_to_shapefile(self):
        # Dataset to trim
        data = xr.DataArray(
            data=np.zeros((10, 10)),
            coords={
                "lat": np.linspace(38.0, 38.9, num=10),
                "lon": np.linspace(-121.9, -121.0, num=10),
            },
        )
        data = data.rio.set_spatial_dims(y_dim="lat", x_dim="lon")
        data = data.rio.write_crs("EPSG:4326")
        # Mock shapefile inputs with this:
        df = pd.DataFrame(
            {
                "Area": ["Box1"],
            }
        )
        geometry = [box(-121.6, 38.3, -121.2, 38.8)]
        gdf = gpd.GeoDataFrame(df, geometry=geometry, crs="EPSG:4326")
        # This should clip successfully
        with patch("geopandas.read_file", return_value=gdf):
            result = clip_to_shapefile(data, "not_a_file.shp")

        assert result["lat"].min().item() == pytest.approx(38.3, 1e-6)
        assert result["lat"].max().item() == pytest.approx(38.7, 1e-6)
        assert result["lon"].min().item() == pytest.approx(-121.6, 1e-6)
        assert result["lon"].max().item() == pytest.approx(-121.3, 1e-6)
        assert result.attrs["location_subset"] == ["user-defined"]
        assert result.shape == (5, 4)

        with patch("geopandas.read_file", return_value=gdf):
            result = clip_to_shapefile(data, "not_a_file.shp", ("Area", "Box1"))
        assert result.attrs["location_subset"] == ["Box1"]

        # Input data lacks CRS
        with pytest.raises(RuntimeError):
            result = clip_to_shapefile(xr.Dataset(), "not_a_file.shp")

        # "Shapefile" data lacks CRS
        gdf = gpd.GeoDataFrame(df, geometry=geometry, crs=None)
        with patch("geopandas.read_file", return_value=gdf):
            with pytest.raises(RuntimeError):
                result = clip_to_shapefile(data, "not_a_file.shp")

        # "Shapefile" feature too small relative to grid
        geometry = [box(-121.39, 38.52, -121.31, 38.56)]
        gdf = gpd.GeoDataFrame(df, geometry=geometry, crs="EPSG:4326")
        with patch("geopandas.read_file", return_value=gdf):
            with pytest.raises(RuntimeError):
                result = clip_to_shapefile(data, "not_a_file.shp")

    def test_clip_gdf_to_shapefile(self):
        """Test for clip_gdf_to_shapefile with geodataframe to shapefile."""

        # Mock geopandas df dataset to trim -- station list
        df = pd.DataFrame(
            {
                "latitude": [38.5, 39.0, 40.0, 41.5],
                "longitude": [-121.5, -121.2, -120.5, -120.1],
            }
        )
        df["geometry"] = [Point(xy) for xy in zip(df["longitude"], df["latitude"])]
        gdf = gpd.GeoDataFrame(df, geometry="geometry", crs="EPSG:3857")

        # Define clipping polygon
        clip_poly = gpd.GeoDataFrame(
            geometry=[box(-121.5, 38.5, -120.5, 40.0)],
            crs="EPSG:3857",
        )

        # This should clip succesfully
        result = clip_gpd_to_shapefile(gdf, clip_poly)

        assert not result.empty
        assert result["latitude"].min() >= 38.5
        assert result["latitude"].max() <= 40.0
        assert result["longitude"].min() >= -121.5
        assert result["longitude"].max() <= -120.5

        # "Shapefile" lacks CRS
        clip_poly_none = gpd.GeoDataFrame(
            geometry=[box(-121.5, 38.5, -120.5, 40.0)], crs=None
        )
        with pytest.raises(RuntimeError, match="CRS"):
            result = clip_gpd_to_shapefile(gdf, clip_poly_none)
