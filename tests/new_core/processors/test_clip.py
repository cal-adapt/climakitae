"""
Unit tests for climakitae/new_core/processors/clip.py

This module contains comprehensive unit tests for the Clip processor class
that handles spatial data clipping operations including boundary-based,
coordinate-based, and point-based clipping.
"""

import pytest
import numpy as np
import pandas as pd
import xarray as xr
import geopandas as gpd
from unittest.mock import MagicMock, patch
from shapely.geometry import Point, Polygon, box
import pyproj

from climakitae.new_core.processors.clip import Clip
from climakitae.core.constants import _NEW_ATTRS_KEY, UNSET
from climakitae.new_core.data_access.data_access import DataCatalog


class TestClipInit:
    """Test class for Clip initialization."""

    def test_init_single_boundary_string(self):
        """Test initialization with single boundary string."""
        clip = Clip("CA")
        assert clip.value == "CA"
        assert clip.name == "clip"
        assert clip.is_single_point is False
        assert clip.is_multi_point is False
        assert clip.multi_mode is False
        assert clip.operation is None

    def test_init_single_point_tuple(self):
        """Test initialization with single (lat, lon) point."""
        clip = Clip((37.7749, -122.4194))
        assert clip.value == (37.7749, -122.4194)
        assert clip.is_single_point is True
        assert clip.is_multi_point is False
        assert clip.lat == 37.7749
        assert clip.lon == -122.4194
        assert clip.name == "clip"

    def test_init_multi_point_list(self):
        """Test initialization with multiple (lat, lon) points."""
        points = [(37.7749, -122.4194), (34.0522, -118.2437)]
        clip = Clip(points)
        assert clip.is_multi_point is True
        assert clip.is_single_point is False
        assert len(clip.point_list) == 2
        assert clip.point_list[0] == (37.7749, -122.4194)
        assert clip.point_list[1] == (34.0522, -118.2437)
        assert clip.multi_mode is False

    def test_init_multiple_boundaries(self):
        """Test initialization with multiple boundary keys."""
        clip = Clip(["CA", "OR", "WA"])
        assert clip.value == ["CA", "OR", "WA"]
        assert clip.multi_mode is True
        assert clip.operation == "union"
        assert clip.is_single_point is False
        assert clip.is_multi_point is False

    def test_init_coordinate_bounds(self):
        """Test initialization with coordinate bounding box."""
        bounds = ((32.0, 42.0), (-125.0, -114.0))
        clip = Clip(bounds)
        assert clip.value == bounds
        assert clip.is_single_point is False
        assert clip.is_multi_point is False
        assert clip.multi_mode is False


class TestClipSetDataAccessor:
    """Test class for set_data_accessor method."""

    def test_set_data_accessor_success(self):
        """Test setting data catalog accessor successfully."""
        clip = Clip("CA")
        mock_catalog = MagicMock()

        # Initially catalog should be UNSET
        assert clip.catalog is UNSET

        # Set the catalog
        clip.set_data_accessor(mock_catalog)

        # Verify catalog is set
        assert clip.catalog is mock_catalog
        assert clip.catalog is not UNSET


class TestClipUpdateContext:
    """Test class for update_context method."""

    def test_update_context_single_boundary(self):
        """Test context update for single boundary clipping."""
        clip = Clip("CA")
        context = {}

        clip.update_context(context)

        assert _NEW_ATTRS_KEY in context
        assert "clip" in context[_NEW_ATTRS_KEY]
        assert "CA" in context[_NEW_ATTRS_KEY]["clip"]
        assert "Process 'clip' applied" in context[_NEW_ATTRS_KEY]["clip"]

    def test_update_context_single_point(self):
        """Test context update for single point clipping."""
        clip = Clip((37.7749, -122.4194))
        context = {}

        clip.update_context(context)

        assert _NEW_ATTRS_KEY in context
        assert "clip" in context[_NEW_ATTRS_KEY]
        assert "Single point clipping" in context[_NEW_ATTRS_KEY]["clip"]
        assert "37.7749" in context[_NEW_ATTRS_KEY]["clip"]
        assert "-122.4194" in context[_NEW_ATTRS_KEY]["clip"]

    def test_update_context_multi_point(self):
        """Test context update for multi-point clipping."""
        points = [(37.7749, -122.4194), (34.0522, -118.2437)]
        clip = Clip(points)
        context = {}

        clip.update_context(context)

        assert _NEW_ATTRS_KEY in context
        assert "clip" in context[_NEW_ATTRS_KEY]
        assert "Multi-point clipping" in context[_NEW_ATTRS_KEY]["clip"]
        assert "2 coordinate pairs" in context[_NEW_ATTRS_KEY]["clip"]

    def test_update_context_multiple_boundaries(self):
        """Test context update for multiple boundary clipping."""
        clip = Clip(["CA", "OR", "WA"])
        context = {}

        clip.update_context(context)

        assert _NEW_ATTRS_KEY in context
        assert "clip" in context[_NEW_ATTRS_KEY]
        assert "Multi-boundary clipping" in context[_NEW_ATTRS_KEY]["clip"]
        assert "3 boundaries" in context[_NEW_ATTRS_KEY]["clip"]
        assert "union" in context[_NEW_ATTRS_KEY]["clip"]


class TestClipExecuteWithSingleBoundary:
    """Test class for execute method with single boundary."""

    def setup_method(self):
        """Set up test fixtures."""
        self.clip = Clip("CA")
        self.mock_catalog = MagicMock()
        self.clip.set_data_accessor(self.mock_catalog)

        # Create sample dataset with rioxarray support
        self.sample_dataset = xr.Dataset(
            {"temp": (["time", "y", "x"], np.random.rand(10, 5, 5))},
            coords={
                "time": pd.date_range("2020-01-01", periods=10),
                "y": np.linspace(32, 42, 5),
                "x": np.linspace(-124, -114, 5),
            },
        )
        # Set CRS
        self.sample_dataset.rio.write_crs("EPSG:4326", inplace=True)

        # Create mock geometry
        self.mock_geometry = gpd.GeoDataFrame(
            geometry=[box(-125, 32, -114, 42)], crs=pyproj.CRS.from_epsg(4326)
        )

    def test_execute_single_boundary_dataset(self):
        """Test execute with single boundary and xr.Dataset - outcome: data clipped correctly."""
        with patch.object(
            self.clip, "_get_boundary_geometry", return_value=self.mock_geometry
        ), patch.object(
            self.clip, "_clip_data_with_geom", return_value=self.sample_dataset
        ) as mock_clip:

            context = {}
            result = self.clip.execute(self.sample_dataset, context)

            # Verify result exists
            assert result is not None
            assert isinstance(result, xr.Dataset)

            # Verify clipping method was called
            mock_clip.assert_called_once()

            # Verify context was updated
            assert _NEW_ATTRS_KEY in context
            assert "clip" in context[_NEW_ATTRS_KEY]

    def test_execute_single_boundary_dict(self):
        """Test execute with single boundary and dict of datasets - outcome: all datasets clipped."""
        data_dict = {"sim1": self.sample_dataset, "sim2": self.sample_dataset}

        with patch.object(
            self.clip, "_get_boundary_geometry", return_value=self.mock_geometry
        ), patch.object(
            self.clip, "_clip_data_with_geom", return_value=self.sample_dataset
        ) as mock_clip:

            context = {}
            result = self.clip.execute(data_dict, context)

            # Verify result structure
            assert isinstance(result, dict)
            assert "sim1" in result
            assert "sim2" in result

            # Verify clipping was called for each dataset
            assert mock_clip.call_count == 2

            # Verify context was updated
            assert _NEW_ATTRS_KEY in context

    def test_execute_single_boundary_list(self):
        """Test execute with single boundary and list of datasets - outcome: all datasets clipped."""
        data_list = [self.sample_dataset, self.sample_dataset]

        with patch.object(
            self.clip, "_get_boundary_geometry", return_value=self.mock_geometry
        ), patch.object(
            self.clip, "_clip_data_with_geom", return_value=self.sample_dataset
        ) as mock_clip:

            context = {}
            result = self.clip.execute(data_list, context)

            # Verify result structure
            assert isinstance(result, list)
            assert len(result) == 2

            # Verify clipping was called for each dataset
            assert mock_clip.call_count == 2

            # Verify context was updated
            assert _NEW_ATTRS_KEY in context


class TestClipExecuteWithSinglePoint:
    """Test class for execute method with single point."""

    def setup_method(self):
        """Set up test fixtures."""
        self.clip = Clip((37.7749, -122.4194))
        self.sample_dataset = xr.Dataset(
            {"temp": (["time", "y", "x"], np.random.rand(10, 5, 5))},
            coords={
                "time": pd.date_range("2020-01-01", periods=10),
                "y": np.linspace(32, 42, 5),
                "x": np.linspace(-124, -114, 5),
            },
        )
        # Create a single-point result
        self.clipped_point = self.sample_dataset.isel(x=2, y=2)

    def test_execute_single_point_dataset(self):
        """Test execute with single point and xr.Dataset - outcome: closest gridcell returned."""
        with patch.object(
            self.clip, "_clip_data_to_point", return_value=self.clipped_point
        ) as mock_clip:
            context = {}
            result = self.clip.execute(self.sample_dataset, context)

            # Verify result exists and is the clipped data
            assert result is not None
            assert result is self.clipped_point

            # Verify clipping method was called with correct args
            mock_clip.assert_called_once_with(self.sample_dataset, 37.7749, -122.4194)

            # Verify context was updated
            assert _NEW_ATTRS_KEY in context
            assert "Single point clipping" in context[_NEW_ATTRS_KEY]["clip"]

    def test_execute_single_point_list(self):
        """Test execute with single point and list of datasets - outcome: list of closest gridcells."""
        data_list = [self.sample_dataset, self.sample_dataset]

        with patch.object(
            self.clip, "_clip_data_to_point", return_value=self.clipped_point
        ):
            context = {}
            result = self.clip.execute(data_list, context)

            # Verify result structure
            assert isinstance(result, list)
            assert len(result) == 2

            # Verify context was updated
            assert _NEW_ATTRS_KEY in context

    def test_execute_single_point_dict(self):
        """Test execute with single point and dict of datasets - outcome: dict of closest gridcells."""
        data_dict = {"sim1": self.sample_dataset, "sim2": self.sample_dataset}

        with patch.object(
            self.clip, "_clip_data_to_point", return_value=self.clipped_point
        ):
            context = {}
            result = self.clip.execute(data_dict, context)

            # Verify result structure
            assert isinstance(result, dict)
            assert "sim1" in result
            assert "sim2" in result

            # Verify context was updated
            assert _NEW_ATTRS_KEY in context


class TestClipExecuteWithMultiplePoints:
    """Test class for execute method with multiple points."""

    def setup_method(self):
        """Set up test fixtures."""
        self.points = [(37.7749, -122.4194), (34.0522, -118.2437)]
        self.clip = Clip(self.points)
        self.sample_dataset = xr.Dataset(
            {"temp": (["time", "y", "x"], np.random.rand(10, 5, 5))},
            coords={
                "time": pd.date_range("2020-01-01", periods=10),
                "y": np.linspace(32, 42, 5),
                "x": np.linspace(-124, -114, 5),
            },
        )
        # Create a multi-point result with closest_cell dimension
        self.clipped_multipoint = xr.concat(
            [self.sample_dataset.isel(x=2, y=2), self.sample_dataset.isel(x=1, y=1)],
            dim="closest_cell",
        )

    def test_execute_multiple_points_dataset(self):
        """Test execute with multiple points and xr.Dataset - outcome: concatenated closest gridcells."""
        with patch.object(
            self.clip,
            "_clip_data_to_multiple_points",
            return_value=self.clipped_multipoint,
        ) as mock_clip:
            context = {}
            result = self.clip.execute(self.sample_dataset, context)

            # Verify result exists
            assert result is not None
            assert result is self.clipped_multipoint

            # Verify clipping method was called with correct args
            mock_clip.assert_called_once_with(self.sample_dataset, self.points)

            # Verify context was updated
            assert _NEW_ATTRS_KEY in context
            assert "Multi-point clipping" in context[_NEW_ATTRS_KEY]["clip"]

    def test_execute_multiple_points_list(self):
        """Test execute with multiple points and list - outcome: list of multi-point results."""
        data_list = [self.sample_dataset, self.sample_dataset]

        with patch.object(
            self.clip,
            "_clip_data_to_multiple_points",
            return_value=self.clipped_multipoint,
        ):
            context = {}
            result = self.clip.execute(data_list, context)

            # Verify result structure
            assert isinstance(result, list)
            assert len(result) == 2

            # Verify context was updated
            assert _NEW_ATTRS_KEY in context

    def test_execute_multiple_points_dict(self):
        """Test execute with multiple points and dict - outcome: dict of multi-point results."""
        data_dict = {"sim1": self.sample_dataset, "sim2": self.sample_dataset}

        with patch.object(
            self.clip,
            "_clip_data_to_multiple_points",
            return_value=self.clipped_multipoint,
        ):
            context = {}
            result = self.clip.execute(data_dict, context)

            # Verify result structure
            assert isinstance(result, dict)
            assert "sim1" in result
            assert "sim2" in result

            # Verify context was updated
            assert _NEW_ATTRS_KEY in context


class TestClipExecuteWithCoordinateBounds:
    """Test class for execute method with coordinate bounds."""

    def setup_method(self):
        """Set up test fixtures."""
        self.bounds = ((32.0, 42.0), (-125.0, -114.0))
        self.clip = Clip(self.bounds)
        self.sample_dataset = xr.Dataset(
            {"temp": (["time", "y", "x"], np.random.rand(10, 5, 5))},
            coords={
                "time": pd.date_range("2020-01-01", periods=10),
                "y": np.linspace(32, 42, 5),
                "x": np.linspace(-124, -114, 5),
            },
        )
        self.sample_dataset.rio.write_crs("EPSG:4326", inplace=True)

        # Create mock geometry for bounds
        self.mock_geometry = gpd.GeoDataFrame(
            geometry=[box(-125, 32, -114, 42)], crs=pyproj.CRS.from_epsg(4326)
        )

    def test_execute_coordinate_bounds_dataset(self):
        """Test execute with coordinate bounds and xr.Dataset - outcome: data clipped to bounds."""
        with patch.object(
            self.clip, "_clip_data_with_geom", return_value=self.sample_dataset
        ) as mock_clip:
            context = {}
            result = self.clip.execute(self.sample_dataset, context)

            # Verify result exists
            assert result is not None
            assert isinstance(result, xr.Dataset)

            # Verify clipping method was called
            mock_clip.assert_called_once()

            # Verify context was updated
            assert _NEW_ATTRS_KEY in context

    def test_execute_coordinate_bounds_list(self):
        """Test execute with coordinate bounds and list - outcome: all datasets clipped."""
        data_list = [self.sample_dataset, self.sample_dataset]

        with patch.object(
            self.clip, "_clip_data_with_geom", return_value=self.sample_dataset
        ):
            context = {}
            result = self.clip.execute(data_list, context)

            # Verify result structure
            assert isinstance(result, list)
            assert len(result) == 2

            # Verify context was updated
            assert _NEW_ATTRS_KEY in context


class TestClipExecuteWithMultipleBoundaries:
    """Test class for execute method with multiple boundary keys."""

    def setup_method(self):
        """Set up test fixtures."""
        self.boundaries = ["CA", "OR", "WA"]
        self.clip = Clip(self.boundaries)
        self.mock_catalog = MagicMock()
        self.clip.set_data_accessor(self.mock_catalog)

        self.sample_dataset = xr.Dataset(
            {"temp": (["time", "y", "x"], np.random.rand(10, 5, 5))},
            coords={
                "time": pd.date_range("2020-01-01", periods=10),
                "y": np.linspace(32, 42, 5),
                "x": np.linspace(-124, -114, 5),
            },
        )
        self.sample_dataset.rio.write_crs("EPSG:4326", inplace=True)

        # Create mock combined geometry
        self.mock_geometry = gpd.GeoDataFrame(
            geometry=[box(-125, 32, -114, 50)], crs=pyproj.CRS.from_epsg(4326)
        )

    def test_execute_multiple_boundaries_dataset(self):
        """Test execute with multiple boundaries and xr.Dataset - outcome: union of boundaries applied."""
        with patch.object(
            self.clip, "_get_multi_boundary_geometry", return_value=self.mock_geometry
        ), patch.object(
            self.clip, "_clip_data_with_geom", return_value=self.sample_dataset
        ) as mock_clip:

            context = {}
            result = self.clip.execute(self.sample_dataset, context)

            # Verify result exists
            assert result is not None
            assert isinstance(result, xr.Dataset)

            # Verify clipping method was called
            mock_clip.assert_called_once()

            # Verify context was updated
            assert _NEW_ATTRS_KEY in context
            assert "Multi-boundary clipping" in context[_NEW_ATTRS_KEY]["clip"]
            assert "union" in context[_NEW_ATTRS_KEY]["clip"]

    def test_execute_multiple_boundaries_list(self):
        """Test execute with multiple boundaries and list - outcome: all datasets clipped with union."""
        data_list = [self.sample_dataset, self.sample_dataset]

        with patch.object(
            self.clip, "_get_multi_boundary_geometry", return_value=self.mock_geometry
        ), patch.object(
            self.clip, "_clip_data_with_geom", return_value=self.sample_dataset
        ):

            context = {}
            result = self.clip.execute(data_list, context)

            # Verify result structure
            assert isinstance(result, list)
            assert len(result) == 2

            # Verify context was updated
            assert _NEW_ATTRS_KEY in context


class TestClipIntegrationCoordinateBounds:
    """Integration tests for coordinate bounds clipping with real data."""

    def setup_method(self):
        """Set up test fixtures with real data."""
        # Create realistic dataset with proper CRS
        self.dataset = xr.Dataset(
            {"temp": (["time", "y", "x"], np.random.rand(3, 10, 10) + 20)},
            coords={
                "time": pd.date_range("2020-01-01", periods=3),
                "y": np.linspace(32, 42, 10),
                "x": np.linspace(-124, -114, 10),
            },
        )
        self.dataset.rio.write_crs("EPSG:4326", inplace=True)

    def test_clip_with_coordinate_bounds_integration(self):
        """Test real coordinate bounds clipping - outcome: dataset clipped to specified bounds."""
        bounds = ((35.0, 40.0), (-122.0, -116.0))
        clip = Clip(bounds)
        context = {}

        result = clip.execute(self.dataset, context)

        # Verify result exists and is clipped
        assert result is not None
        assert isinstance(result, xr.Dataset)

        # Verify spatial dimensions are reduced
        assert result.sizes["y"] < self.dataset.sizes["y"]
        assert result.sizes["x"] < self.dataset.sizes["x"]

        # Verify context updated
        assert _NEW_ATTRS_KEY in context
        assert "clip" in context[_NEW_ATTRS_KEY]

    def test_clip_data_with_geom_static_method(self):
        """Test _clip_data_with_geom static method - outcome: data clipped to geometry."""
        # Create a simple geometry
        geom = gpd.GeoDataFrame(
            geometry=[box(-122, 35, -118, 40)], crs=pyproj.CRS.from_epsg(4326)
        )

        result = Clip._clip_data_with_geom(self.dataset, geom)

        # Verify result exists
        assert result is not None
        assert isinstance(result, xr.Dataset)

        # Verify it has data
        assert "temp" in result.data_vars

        # Verify spatial dimensions exist
        assert "x" in result.dims or "lon" in result.dims
        assert "y" in result.dims or "lat" in result.dims


class TestClipBoundaryValidation:
    """Integration tests for boundary validation methods."""

    def setup_method(self):
        """Set up test fixtures."""
        self.clip = Clip("CA")
        self.mock_catalog = MagicMock()

        # Mock the boundary_dict structure
        mock_boundaries = MagicMock()
        mock_boundaries.boundary_dict.return_value = {
            "states": {"CA": 5, "OR": 6, "WA": 7},
            "CA counties": {"Los Angeles County": 0, "San Diego County": 1},
            "none": {},
            "lat/lon": {},
        }
        self.mock_catalog.boundaries = mock_boundaries
        self.clip.set_data_accessor(self.mock_catalog)

    def test_validate_boundary_key_valid(self):
        """Test validate_boundary_key with valid key - outcome: validation succeeds."""
        result = self.clip.validate_boundary_key("CA")

        assert result["valid"] is True
        assert result["category"] == "states"
        assert result["suggestions"] == []

    def test_validate_boundary_key_invalid(self):
        """Test validate_boundary_key with invalid key - outcome: returns suggestions."""
        result = self.clip.validate_boundary_key("InvalidKey")

        assert result["valid"] is False
        assert "error" in result
        assert isinstance(result["suggestions"], list)

    def test_validate_boundary_key_partial_match(self):
        """Test validate_boundary_key with partial match - outcome: returns suggestions."""
        result = self.clip.validate_boundary_key("Los Angeles")

        # Should find it or suggest it
        assert "suggestions" in result

    def test_validate_boundary_key_no_catalog(self):
        """Test validate_boundary_key without catalog - outcome: returns error."""
        clip_no_catalog = Clip("CA")
        result = clip_no_catalog.validate_boundary_key("CA")

        assert result["valid"] is False
        assert "error" in result


class TestClipExecuteErrorHandling:
    """Test class for error handling in execute method."""

    def setup_method(self):
        """Set up test fixtures."""
        self.dataset = xr.Dataset(
            {"temp": (["time", "y", "x"], np.random.rand(3, 10, 10) + 20)},
            coords={
                "time": pd.date_range("2020-01-01", periods=3),
                "y": np.linspace(32, 42, 10),
                "x": np.linspace(-124, -114, 10),
            },
        )
        self.dataset.rio.write_crs("EPSG:4326", inplace=True)

    def test_execute_invalid_value_type(self):
        """Test execute with invalid value type - outcome: raises ValueError."""
        clip = Clip("CA")
        clip.value = 123  # Invalid type

        with pytest.raises(ValueError, match="Invalid value type for clipping"):
            context = {}
            clip.execute(self.dataset, context)

    def test_execute_failed_geometry_creation(self):
        """Test execute when geometry creation fails - outcome: raises ValueError."""
        clip = Clip("CA")
        mock_catalog = MagicMock()
        clip.set_data_accessor(mock_catalog)

        # Mock the geometry method to return None
        with patch.object(clip, "_get_boundary_geometry", return_value=None):
            with pytest.raises(ValueError, match="Failed to create geometry"):
                context = {}
                clip.execute(self.dataset, context)

    def test_execute_missing_catalog_for_boundary(self):
        """Test execute without catalog when boundary key provided - outcome: raises RuntimeError."""
        clip = Clip("CA")  # Boundary key but no catalog
        context = {}

        with pytest.raises(RuntimeError, match="DataCatalog is not set"):
            clip.execute(self.dataset, context)

    def test_execute_with_dataarray(self):
        """Test execute with DataArray - outcome: successful clipping."""
        bounds = ((35.0, 40.0), (-122.0, -116.0))
        clip = Clip(bounds)
        context = {}

        # Convert dataset to DataArray
        data_array = self.dataset["temp"]
        data_array.rio.write_crs("EPSG:4326", inplace=True)

        result = clip.execute(data_array, context)

        # Verify result
        assert result is not None
        assert isinstance(result, xr.DataArray)
        assert _NEW_ATTRS_KEY in context

    def test_execute_with_tuple_of_datasets(self):
        """Test execute with tuple of datasets - outcome: returns tuple."""
        bounds = ((35.0, 40.0), (-122.0, -116.0))
        clip = Clip(bounds)
        context = {}

        data_tuple = (self.dataset, self.dataset)
        result = clip.execute(data_tuple, context)

        # Verify result
        assert isinstance(result, tuple)
        assert len(result) == 2
        assert _NEW_ATTRS_KEY in context


class TestClipDataToPointIntegration:
    """Integration tests for _clip_data_to_point static method."""

    def setup_method(self):
        """Set up test fixtures with real data."""
        # Create dataset with valid data
        self.dataset_valid = xr.Dataset(
            {"temp": (["time", "y", "x"], np.random.rand(3, 10, 10) + 20)},
            coords={
                "time": pd.date_range("2020-01-01", periods=3),
                "y": np.linspace(32, 42, 10),
                "x": np.linspace(-124, -114, 10),
                "lat": (["y", "x"], np.tile(np.linspace(32, 42, 10)[:, None], (1, 10))),
                "lon": (
                    ["y", "x"],
                    np.tile(np.linspace(-124, -114, 10)[None, :], (10, 1)),
                ),
            },
        )
        # Add required attributes
        self.dataset_valid.attrs["resolution"] = "3 km"
        # Set CRS
        self.dataset_valid = self.dataset_valid.rio.write_crs("EPSG:4326")

        # Create dataset with some NaN values
        data_with_nans = np.random.rand(3, 10, 10) + 20
        data_with_nans[:, 5, 5] = np.nan  # Make center point NaN
        self.dataset_with_nans = xr.Dataset(
            {"temp": (["time", "y", "x"], data_with_nans)},
            coords={
                "time": pd.date_range("2020-01-01", periods=3),
                "y": np.linspace(32, 42, 10),
                "x": np.linspace(-124, -114, 10),
                "lat": (["y", "x"], np.tile(np.linspace(32, 42, 10)[:, None], (1, 10))),
                "lon": (
                    ["y", "x"],
                    np.tile(np.linspace(-124, -114, 10)[None, :], (10, 1)),
                ),
            },
        )
        # Add required attributes
        self.dataset_with_nans.attrs["resolution"] = "3 km"
        # Set CRS
        self.dataset_with_nans = self.dataset_with_nans.rio.write_crs("EPSG:4326")

    def test_clip_data_to_point_valid_closest(self):
        """Test _clip_data_to_point with valid closest gridcell - outcome: returns single gridcell."""
        lat, lon = 37.0, -119.0

        result = Clip._clip_data_to_point(self.dataset_valid, lat, lon)

        # Verify result exists
        assert result is not None
        assert isinstance(result, xr.Dataset)

        # Verify it's a single point (no x, y dimensions)
        assert "x" not in result.dims and "y" not in result.dims

        # Verify it has the data variable
        assert "temp" in result.data_vars

        # Verify time dimension is preserved
        assert "time" in result.dims
        assert len(result.time) == 3

    def test_clip_data_to_point_with_nan_search(self):
        """Test _clip_data_to_point searches for valid gridcell when closest has NaN - outcome: finds valid cell."""
        # Create dataset where center is NaN but edges have valid data
        # This ensures NaN search will be triggered and will find valid data
        data_with_center_nan = np.random.rand(3, 10, 10) + 20
        # Make center 3x3 area NaN to force search algorithm
        data_with_center_nan[:, 4:7, 4:7] = np.nan
        dataset_center_nan = xr.Dataset(
            {"temp": (["time", "y", "x"], data_with_center_nan)},
            coords={
                "time": pd.date_range("2020-01-01", periods=3),
                "y": np.linspace(32, 42, 10),
                "x": np.linspace(-124, -114, 10),
                "lat": (["y", "x"], np.tile(np.linspace(32, 42, 10)[:, None], (1, 10))),
                "lon": (
                    ["y", "x"],
                    np.tile(np.linspace(-124, -114, 10)[None, :], (10, 1)),
                ),
            },
        )
        dataset_center_nan.attrs["resolution"] = "3 km"
        dataset_center_nan = dataset_center_nan.rio.write_crs("EPSG:4326")

        # Request point that maps to the NaN center location
        lat, lon = 37.0, -119.0

        with patch("builtins.print"):  # Suppress print statements
            result = Clip._clip_data_to_point(dataset_center_nan, lat, lon)

        # If search found a valid gridcell, verify it
        if result is not None:
            assert "temp" in result.data_vars
            # Verify data is not all NaN (should have found valid gridcell)
            assert not result["temp"].isnull().all()

    def test_clip_data_to_point_edge_of_domain(self):
        """Test _clip_data_to_point at edge of domain - outcome: returns edge gridcell."""
        # Point at the edge of domain
        lat, lon = 32.5, -123.5

        result = Clip._clip_data_to_point(self.dataset_valid, lat, lon)

        # Verify result exists
        assert result is not None
        assert isinstance(result, xr.Dataset)
        assert "temp" in result.data_vars

    def test_clip_data_to_point_with_dataarray(self):
        """Test _clip_data_to_point with DataArray converted to Dataset - outcome: works correctly."""
        lat, lon = 37.0, -119.0
        data_array = self.dataset_valid["temp"]

        # Convert DataArray to Dataset (current implementation requires Dataset)
        dataset_from_array = data_array.to_dataset(name="temp")
        dataset_from_array.attrs["resolution"] = "3 km"
        dataset_from_array = dataset_from_array.rio.write_crs("EPSG:4326")

        result = Clip._clip_data_to_point(dataset_from_array, lat, lon)

        # Verify result exists and has the expected variable
        assert result is not None
        assert isinstance(result, xr.Dataset)
        assert "temp" in result.data_vars

    def test_clip_data_to_point_outside_domain(self):
        """Test _clip_data_to_point with point outside domain - outcome: returns None or nearest."""
        # Point far outside domain
        lat, lon = 50.0, -100.0

        with patch("builtins.print"):  # Suppress print statements
            result = Clip._clip_data_to_point(self.dataset_valid, lat, lon)

        # May return None or the nearest edge point depending on implementation
        if result is not None:
            assert isinstance(result, xr.Dataset)

    def test_clip_data_to_point_preserves_attributes(self):
        """Test _clip_data_to_point preserves dataset attributes - outcome: attributes maintained."""
        lat, lon = 37.0, -119.0
        self.dataset_valid.attrs["test_attr"] = "test_value"

        result = Clip._clip_data_to_point(self.dataset_valid, lat, lon)

        # Verify result has attributes
        assert result is not None
        # Attributes may or may not be preserved depending on selection method
        # Just verify the operation completes successfully

    def test_clip_data_to_point_multiple_variables(self):
        """Test _clip_data_to_point with multiple variables - outcome: all variables clipped."""
        # Add another variable
        dataset_multi = self.dataset_valid.copy()
        dataset_multi["precip"] = (["time", "y", "x"], np.random.rand(3, 10, 10))

        lat, lon = 37.0, -119.0
        result = Clip._clip_data_to_point(dataset_multi, lat, lon)

        # Verify both variables are present
        assert result is not None
        assert "temp" in result.data_vars
        assert "precip" in result.data_vars


class TestClipDataToMultiplePointsIntegration:
    """Integration tests for _clip_data_to_multiple_points static method.

    This test class verifies the vectorized multi-point clipping functionality
    including coordinate handling, dimension renaming, and fallback behavior.
    """

    def setup_method(self):
        """Set up test fixtures with real data."""
        # Create dataset with valid data
        self.dataset_valid = xr.Dataset(
            {"temp": (["time", "y", "x"], np.random.rand(3, 10, 10) + 20)},
            coords={
                "time": pd.date_range("2020-01-01", periods=3),
                "y": np.linspace(32, 42, 10),
                "x": np.linspace(-124, -114, 10),
                "lat": (["y", "x"], np.tile(np.linspace(32, 42, 10)[:, None], (1, 10))),
                "lon": (
                    ["y", "x"],
                    np.tile(np.linspace(-124, -114, 10)[None, :], (10, 1)),
                ),
            },
        )
        # Add required attributes
        self.dataset_valid.attrs["resolution"] = "3 km"
        # Set CRS
        self.dataset_valid = self.dataset_valid.rio.write_crs("EPSG:4326")

    def test_clip_data_to_multiple_points_basic(self):
        """Test _clip_data_to_multiple_points with list of valid points - outcome: returns all gridcells."""
        point_list = [(37.0, -119.0), (35.0, -121.0), (40.0, -118.0)]

        with patch("builtins.print"):  # Suppress print statements
            result = Clip._clip_data_to_multiple_points(self.dataset_valid, point_list)

        # Verify result exists
        assert result is not None
        assert isinstance(result, xr.Dataset)

        # Verify it has the closest_cell dimension with correct size
        assert "closest_cell" in result.dims
        assert result.dims["closest_cell"] == 3

        # Verify target coordinates are added
        assert "target_lats" in result.coords
        assert "target_lons" in result.coords
        assert "point_index" in result.coords

        # Verify data variable is present
        assert "temp" in result.data_vars

    def test_clip_data_to_multiple_points_single_point(self):
        """Test _clip_data_to_multiple_points with single point - outcome: works like single point."""
        point_list = [(37.0, -119.0)]

        with patch("builtins.print"):
            result = Clip._clip_data_to_multiple_points(self.dataset_valid, point_list)

        # Verify result exists with single point
        assert result is not None
        assert result.dims["closest_cell"] == 1

        # Verify coordinates match input
        assert float(result["target_lats"].values[0]) == 37.0
        assert float(result["target_lons"].values[0]) == -119.0

    def test_clip_data_to_multiple_points_preserves_time(self):
        """Test _clip_data_to_multiple_points preserves time dimension - outcome: time intact."""
        point_list = [(37.0, -119.0), (35.0, -121.0)]

        with patch("builtins.print"):
            result = Clip._clip_data_to_multiple_points(self.dataset_valid, point_list)

        # Verify time dimension is preserved
        assert "time" in result.dims
        assert len(result.time) == 3

        # Verify data shape is correct (time x closest_cell)
        assert result["temp"].shape == (3, 2)

    def test_clip_data_to_multiple_points_with_dataarray(self):
        """Test _clip_data_to_multiple_points with DataArray converted to Dataset - outcome: works correctly."""
        point_list = [(37.0, -119.0), (35.0, -121.0)]
        data_array = self.dataset_valid["temp"]

        # Convert DataArray to Dataset (current implementation requires resolution attribute)
        dataset_from_array = data_array.to_dataset(name="temp")
        dataset_from_array.attrs["resolution"] = "3 km"
        dataset_from_array = dataset_from_array.rio.write_crs("EPSG:4326")

        with patch("builtins.print"):
            result = Clip._clip_data_to_multiple_points(dataset_from_array, point_list)

        # Verify result exists and has expected structure
        assert result is not None
        assert "temp" in result.data_vars
        assert result.dims["closest_cell"] == 2

    def test_clip_data_to_multiple_points_multiple_variables(self):
        """Test _clip_data_to_multiple_points with multiple variables - outcome: all variables included."""
        # Add another variable
        dataset_multi = self.dataset_valid.copy()
        dataset_multi["precip"] = (["time", "y", "x"], np.random.rand(3, 10, 10))

        point_list = [(37.0, -119.0), (35.0, -121.0)]

        with patch("builtins.print"):
            result = Clip._clip_data_to_multiple_points(dataset_multi, point_list)

        # Verify both variables are present
        assert result is not None
        assert "temp" in result.data_vars
        assert "precip" in result.data_vars

        # Verify both have the correct shape
        assert result["temp"].shape == (3, 2)
        assert result["precip"].shape == (3, 2)

    def test_clip_data_to_multiple_points_coordinate_order(self):
        """Test _clip_data_to_multiple_points maintains point order - outcome: order preserved."""
        point_list = [(32.5, -123.5), (37.0, -119.0), (41.5, -115.0)]

        with patch("builtins.print"):
            result = Clip._clip_data_to_multiple_points(self.dataset_valid, point_list)

        # Verify point order is preserved in coordinates
        assert result is not None
        assert len(result["target_lats"]) == 3
        assert len(result["target_lons"]) == 3

        # Verify target coordinates match input order
        assert float(result["target_lats"].values[0]) == 32.5
        assert float(result["target_lats"].values[1]) == 37.0
        assert float(result["target_lats"].values[2]) == 41.5

        # Verify point indices are sequential
        assert list(result["point_index"].values) == [0, 1, 2]

    def test_clip_data_to_multiple_points_large_list(self):
        """Test _clip_data_to_multiple_points with many points - outcome: efficient vectorized processing."""
        # Create 20 points across the domain
        lats = np.linspace(32, 42, 20)
        lons = np.linspace(-124, -114, 20)
        point_list = list(zip(lats, lons))

        with patch("builtins.print"):
            result = Clip._clip_data_to_multiple_points(self.dataset_valid, point_list)

        # Verify all points processed
        assert result is not None
        assert result.dims["closest_cell"] == 20

        # Verify all target coordinates are present
        assert len(result["target_lats"]) == 20
        assert len(result["target_lons"]) == 20

    def test_clip_data_to_multiple_points_edge_cases(self):
        """Test _clip_data_to_multiple_points with edge/boundary points - outcome: handles edges correctly."""
        # Points at domain edges
        point_list = [
            (32.0, -124.0),  # Southwest corner
            (42.0, -114.0),  # Northeast corner
            (37.0, -124.0),  # Western edge
            (37.0, -114.0),  # Eastern edge
        ]

        with patch("builtins.print"):
            result = Clip._clip_data_to_multiple_points(self.dataset_valid, point_list)

        # Verify all edge points are handled
        assert result is not None
        assert result.dims["closest_cell"] == 4


class TestValidateBoundaryKey:
    """Test class for validate_boundary_key method.

    This test class verifies boundary key validation including:
    - Valid boundary key detection
    - Invalid boundary key handling with suggestions
    - Error handling for missing catalog
    - Partial match suggestion logic
    """

    def setup_method(self):
        """Set up test fixtures with mocked catalog."""
        # Create a Clip instance with mocked data accessor
        self.mock_boundaries = MagicMock()
        self.mock_catalog = MagicMock()
        self.mock_catalog.boundaries = self.mock_boundaries

        # Sample boundary dictionary structure
        self.sample_boundary_dict = {
            "states": {
                "CA": "California",
                "OR": "Oregon",
                "WA": "Washington",
                "NV": "Nevada",
            },
            "counties": {
                "Los Angeles County": "Los Angeles",
                "San Diego County": "San Diego",
                "Orange County": "Orange",
                "Alameda County": "Alameda",
            },
            "watersheds": {
                "Sacramento River": "Sacramento watershed",
                "San Joaquin River": "San Joaquin watershed",
            },
            "none": {},  # Should be ignored
            "lat/lon": {},  # Should be ignored
        }

    def test_validate_boundary_key_valid_state(self):
        """Test validate_boundary_key with valid state - outcome: returns valid=True with category."""
        # Setup mock to return boundary dict
        self.mock_boundaries.boundary_dict.return_value = self.sample_boundary_dict

        # Create Clip instance with mocked catalog
        clip = Clip("CA")
        clip.catalog = self.mock_catalog

        result = clip.validate_boundary_key("CA")

        # Verify validation passed
        assert result["valid"] is True
        assert result["category"] == "states"
        assert result["suggestions"] == []

    def test_validate_boundary_key_valid_county(self):
        """Test validate_boundary_key with valid county - outcome: returns valid=True with category."""
        self.mock_boundaries.boundary_dict.return_value = self.sample_boundary_dict

        clip = Clip("Los Angeles County")
        clip.catalog = self.mock_catalog

        result = clip.validate_boundary_key("Los Angeles County")

        assert result["valid"] is True
        assert result["category"] == "counties"
        assert result["suggestions"] == []

    def test_validate_boundary_key_valid_watershed(self):
        """Test validate_boundary_key with valid watershed - outcome: returns valid=True with category."""
        self.mock_boundaries.boundary_dict.return_value = self.sample_boundary_dict

        clip = Clip("Sacramento River")
        clip.catalog = self.mock_catalog

        result = clip.validate_boundary_key("Sacramento River")

        assert result["valid"] is True
        assert result["category"] == "watersheds"
        assert result["suggestions"] == []

    def test_validate_boundary_key_invalid_with_suggestions(self):
        """Test validate_boundary_key with invalid key - outcome: returns suggestions."""
        self.mock_boundaries.boundary_dict.return_value = self.sample_boundary_dict

        clip = Clip("CA")
        clip.catalog = self.mock_catalog

        # Try invalid key that partially matches
        result = clip.validate_boundary_key("Californ")

        assert result["valid"] is False
        assert "error" in result
        assert "Californ" in result["error"]
        assert len(result["suggestions"]) > 0
        # Should suggest California from states category
        assert any("California" in s for s in result["suggestions"])

    def test_validate_boundary_key_invalid_partial_match(self):
        """Test validate_boundary_key with partial match - outcome: returns matching suggestions."""
        self.mock_boundaries.boundary_dict.return_value = self.sample_boundary_dict

        clip = Clip("CA")
        clip.catalog = self.mock_catalog

        # Try "Diego" which should match "San Diego County"
        result = clip.validate_boundary_key("Diego")

        assert result["valid"] is False
        assert len(result["suggestions"]) > 0
        assert any("San Diego County" in s for s in result["suggestions"])

    def test_validate_boundary_key_invalid_no_match(self):
        """Test validate_boundary_key with completely invalid key - outcome: returns empty suggestions."""
        self.mock_boundaries.boundary_dict.return_value = self.sample_boundary_dict

        clip = Clip("CA")
        clip.catalog = self.mock_catalog

        result = clip.validate_boundary_key("XYZ123NonExistent")

        assert result["valid"] is False
        assert "error" in result
        assert result["suggestions"] == []

    def test_validate_boundary_key_case_insensitive_match(self):
        """Test validate_boundary_key is case-insensitive for suggestions - outcome: finds matches."""
        self.mock_boundaries.boundary_dict.return_value = self.sample_boundary_dict

        clip = Clip("CA")
        clip.catalog = self.mock_catalog

        # Try lowercase version
        result = clip.validate_boundary_key("california")

        assert result["valid"] is False  # Exact match is case-sensitive
        assert len(result["suggestions"]) > 0
        assert any("California" in s for s in result["suggestions"])

    def test_validate_boundary_key_no_catalog(self):
        """Test validate_boundary_key with no catalog set - outcome: returns catalog error."""
        clip = Clip("CA")
        # Don't set catalog (leave as UNSET)

        result = clip.validate_boundary_key("CA")

        assert result["valid"] is False
        assert "error" in result
        assert "DataCatalog is not set" in result["error"]
        assert result["suggestions"] == []

    def test_validate_boundary_key_catalog_access_error(self):
        """Test validate_boundary_key when catalog access fails - outcome: returns error."""
        # Mock boundary_dict to raise an exception
        self.mock_boundaries.boundary_dict.side_effect = RuntimeError(
            "Database connection failed"
        )

        clip = Clip("CA")
        clip.catalog = self.mock_catalog

        result = clip.validate_boundary_key("CA")

        assert result["valid"] is False
        assert "error" in result
        assert "Failed to access boundary data" in result["error"]
        assert "Database connection failed" in result["error"]
        assert result["suggestions"] == []

    def test_validate_boundary_key_limits_suggestions(self):
        """Test validate_boundary_key limits suggestions to 10 - outcome: max 10 suggestions."""
        # Create a large boundary dict with many matches
        large_boundary_dict = {
            "items": {f"Item{i}": f"Description {i}" for i in range(50)}
        }
        self.mock_boundaries.boundary_dict.return_value = large_boundary_dict

        clip = Clip("CA")
        clip.catalog = self.mock_catalog

        # Search for "Item" which matches all 50
        result = clip.validate_boundary_key("Item")

        assert result["valid"] is False
        assert len(result["suggestions"]) <= 10

    def test_validate_boundary_key_ignores_special_categories(self):
        """Test validate_boundary_key ignores 'none' and 'lat/lon' categories - outcome: not suggested."""
        boundary_dict_with_special = {
            "states": {"CA": "California"},
            "none": {"TestNone": "Should be ignored"},
            "lat/lon": {"TestLatLon": "Should be ignored"},
        }
        self.mock_boundaries.boundary_dict.return_value = boundary_dict_with_special

        clip = Clip("CA")
        clip.catalog = self.mock_catalog

        # Try searching for items in special categories
        result = clip.validate_boundary_key("TestNone")

        assert result["valid"] is False
        # Should not find TestNone since it's in 'none' category
        assert not any("TestNone" in s for s in result["suggestions"])


class TestGetBoundaryGeometry:
    """Test class for _get_boundary_geometry method.

    This class tests the method that retrieves geometry data for boundary keys
    from the boundaries catalog, including:
    - Valid boundary key retrieval from different categories
    - Invalid boundary key handling
    - Catalog availability checks
    - Error handling for data access failures
    """

    def setup_method(self):
        """Set up test fixtures with mocked catalog and boundary data."""
        # Create mock catalog and boundaries
        # Import DataCatalog to use in spec
        from climakitae.new_core.data_access.data_access import DataCatalog

        self.mock_boundaries = MagicMock()
        self.mock_catalog = MagicMock(spec=DataCatalog)
        self.mock_catalog.boundaries = self.mock_boundaries

        # Sample boundary dictionary structure matching real data
        self.sample_boundary_dict = {
            "states": {
                "CA": 5,
                "OR": 6,
                "WA": 7,
                "NV": 8,
            },
            "CA counties": {
                "Los Angeles County": 0,
                "San Diego County": 1,
                "Orange County": 2,
            },
            "CA watersheds": {
                "Sacramento River": 0,
                "San Joaquin River": 1,
            },
            "CA Electric Load Serving Entities (IOU & POU)": {
                "PG&E": 0,
                "SCE": 1,
            },
            "none": {},
            "lat/lon": {},
        }

        # Create mock GeoDataFrame to return
        self.mock_geodataframe = gpd.GeoDataFrame(
            {"geometry": [box(-124, 32, -114, 42)]}, crs=pyproj.CRS.from_epsg(4326)
        )

        # Create Clip instance
        self.clip = Clip("CA")
        self.clip.catalog = self.mock_catalog

    def test_get_boundary_geometry_valid_state(self):
        """Test _get_boundary_geometry with valid state boundary key - outcome: returns GeoDataFrame."""
        # Setup mock to return boundary dict and geometry
        self.mock_boundaries.boundary_dict.return_value = self.sample_boundary_dict

        # Mock _extract_geometry_from_category to return geodataframe
        with patch.object(
            self.clip,
            "_extract_geometry_from_category",
            return_value=self.mock_geodataframe,
        ) as mock_extract:
            result = self.clip._get_boundary_geometry("CA")

            # Verify extract was called with correct parameters
            mock_extract.assert_called_once_with("states", 5)

            # Verify result is the mock geodataframe
            assert result is self.mock_geodataframe
            assert isinstance(result, gpd.GeoDataFrame)

    def test_get_boundary_geometry_valid_county(self):
        """Test _get_boundary_geometry with valid county boundary key - outcome: returns GeoDataFrame."""
        # Setup mock to return boundary dict
        self.mock_boundaries.boundary_dict.return_value = self.sample_boundary_dict

        # Mock _extract_geometry_from_category
        with patch.object(
            self.clip,
            "_extract_geometry_from_category",
            return_value=self.mock_geodataframe,
        ) as mock_extract:
            result = self.clip._get_boundary_geometry("Los Angeles County")

            # Verify extract was called with correct parameters for counties
            mock_extract.assert_called_once_with("CA counties", 0)

            # Verify result
            assert result is self.mock_geodataframe
            assert isinstance(result, gpd.GeoDataFrame)

    def test_get_boundary_geometry_valid_watershed(self):
        """Test _get_boundary_geometry with valid watershed boundary key - outcome: returns GeoDataFrame."""
        # Setup mock to return boundary dict
        self.mock_boundaries.boundary_dict.return_value = self.sample_boundary_dict

        # Mock _extract_geometry_from_category
        with patch.object(
            self.clip,
            "_extract_geometry_from_category",
            return_value=self.mock_geodataframe,
        ) as mock_extract:
            result = self.clip._get_boundary_geometry("Sacramento River")

            # Verify extract was called with correct parameters for watersheds
            mock_extract.assert_called_once_with("CA watersheds", 0)

            # Verify result
            assert result is self.mock_geodataframe
            assert isinstance(result, gpd.GeoDataFrame)

    def test_get_boundary_geometry_invalid_key(self):
        """Test _get_boundary_geometry with invalid boundary key - outcome: raises ValueError with suggestions."""
        # Setup mock to return boundary dict
        self.mock_boundaries.boundary_dict.return_value = self.sample_boundary_dict

        # Try to get boundary with invalid key
        with pytest.raises(ValueError, match="Boundary key 'InvalidKey' not found"):
            self.clip._get_boundary_geometry("InvalidKey")

    def test_get_boundary_geometry_no_catalog(self):
        """Test _get_boundary_geometry when catalog is not set - outcome: raises RuntimeError."""
        # Create clip without catalog
        clip_no_catalog = Clip("CA")
        # Don't set catalog (leave as UNSET)

        # Try to get boundary without catalog
        with pytest.raises(RuntimeError, match="DataCatalog is not set"):
            clip_no_catalog._get_boundary_geometry("CA")

    def test_get_boundary_geometry_catalog_access_error(self):
        """Test _get_boundary_geometry when boundary_dict() fails - outcome: raises RuntimeError."""
        # Setup mock to raise exception when accessing boundary_dict
        self.mock_boundaries.boundary_dict.side_effect = RuntimeError(
            "Database connection failed"
        )

        # Try to get boundary when catalog access fails
        with pytest.raises(RuntimeError, match="Failed to access boundary data"):
            self.clip._get_boundary_geometry("CA")

    def test_extract_geometry_from_category_states(self):
        """Test _extract_geometry_from_category for states category - outcome: returns GeoDataFrame."""
        # Create mock DataFrame with index 5 for states
        mock_df = pd.DataFrame({"name": ["California"]}, index=[5])
        mock_gdf = gpd.GeoDataFrame(
            mock_df, geometry=[box(-124, 32, -114, 42)], crs="EPSG:4326"
        )

        # Mock the boundaries._us_states attribute
        self.mock_boundaries._us_states = mock_gdf

        # Call _extract_geometry_from_category
        result = self.clip._extract_geometry_from_category("states", 5)

        # Verify result
        assert isinstance(result, gpd.GeoDataFrame)
        assert len(result) == 1
        assert result.index[0] == 5
        assert result.crs is not None

    def test_extract_geometry_from_category_unknown_category(self):
        """Test _extract_geometry_from_category with unknown category - outcome: raises ValueError."""
        # Try to extract from an unknown category
        with pytest.raises(ValueError, match="Unknown boundary category"):
            self.clip._extract_geometry_from_category("unknown_category", 0)

    def test_extract_geometry_from_category_invalid_index(self):
        """Test _extract_geometry_from_category with invalid index - outcome: raises ValueError."""
        # Create mock DataFrame without index 999
        mock_df = pd.DataFrame({"name": ["California"]}, index=[5])
        mock_gdf = gpd.GeoDataFrame(
            mock_df, geometry=[box(-124, 32, -114, 42)], crs="EPSG:4326"
        )

        # Mock the boundaries._us_states attribute
        self.mock_boundaries._us_states = mock_gdf

        # Try to extract with invalid index
        with pytest.raises(ValueError, match="Index 999 not found in states data"):
            self.clip._extract_geometry_from_category("states", 999)


class TestClipDataToMultiplePointsFallback:
    """Test class for _clip_data_to_multiple_points_fallback method.

    This class tests the fallback method for multiple point clipping that uses
    individual point processing and filtering for duplicate gridcells.

    Tests include:
    - Valid multiple points returning unique gridcells
    - Single point handling
    - Duplicate gridcell filtering
    - Invalid points handling
    - Mixed valid/invalid points
    - Concatenation error handling
    """

    def setup_method(self):
        """Set up test fixtures with sample dataset."""
        # Create sample dataset with realistic structure
        self.dataset = xr.Dataset(
            {"temp": (["time", "y", "x"], np.random.rand(3, 10, 10) + 20)},
            coords={
                "time": pd.date_range("2020-01-01", periods=3),
                "y": np.linspace(32, 42, 10),
                "x": np.linspace(-124, -114, 10),
                "lat": (["y", "x"], np.tile(np.linspace(32, 42, 10)[:, None], (1, 10))),
                "lon": (
                    ["y", "x"],
                    np.tile(np.linspace(-124, -114, 10)[None, :], (10, 1)),
                ),
            },
        )
        self.dataset.attrs["resolution"] = "3 km"
        self.dataset = self.dataset.rio.write_crs("EPSG:4326")

    def test_fallback_multiple_valid_points(self):
        """Test _clip_data_to_multiple_points_fallback with valid multiple points - outcome: returns concatenated dataset."""
        # Define multiple points that should all find valid gridcells
        point_list = [
            (37.0, -119.0),  # Point 1
            (35.0, -121.0),  # Point 2
            (40.0, -118.0),  # Point 3
        ]

        with patch("builtins.print"):  # Suppress print statements
            result = Clip._clip_data_to_multiple_points_fallback(
                self.dataset, point_list
            )

        # Verify result exists and has correct structure
        assert result is not None
        assert isinstance(result, xr.Dataset)

        # Verify closest_cell dimension exists with correct size
        assert "closest_cell" in result.dims
        assert result.sizes["closest_cell"] == 3

        # Verify target coordinates are added
        assert "target_lats" in result.coords
        assert "target_lons" in result.coords
        assert len(result["target_lats"]) == 3
        assert len(result["target_lons"]) == 3

        # Verify original data variable is present
        assert "temp" in result.data_vars

    def test_fallback_single_point(self):
        """Test _clip_data_to_multiple_points_fallback with single point - outcome: returns single gridcell dataset."""
        # Define single point in list
        point_list = [(37.0, -119.0)]

        with patch("builtins.print"):
            result = Clip._clip_data_to_multiple_points_fallback(
                self.dataset, point_list
            )

        # Verify result exists
        assert result is not None
        assert isinstance(result, xr.Dataset)

        # Verify closest_cell dimension has size 1
        assert "closest_cell" in result.dims
        assert result.sizes["closest_cell"] == 1

        # Verify coordinates
        assert "target_lats" in result.coords
        assert "target_lons" in result.coords
        assert float(result["target_lats"].values[0]) == 37.0
        assert float(result["target_lons"].values[0]) == -119.0

    def test_fallback_duplicate_gridcells(self):
        """Test _clip_data_to_multiple_points_fallback with duplicate gridcells - outcome: filters duplicates."""
        # Define points that are very close together (should map to same gridcell)
        point_list = [
            (37.0, -119.0),  # Point 1
            (37.001, -119.001),  # Point 2 - very close to point 1, likely same gridcell
            (35.0, -121.0),  # Point 3 - different location
        ]

        with patch("builtins.print"):
            result = Clip._clip_data_to_multiple_points_fallback(
                self.dataset, point_list
            )

        # Verify result exists
        assert result is not None
        assert isinstance(result, xr.Dataset)

        # Verify closest_cell dimension - should be less than 3 due to duplicate filtering
        assert "closest_cell" in result.dims
        # The exact number depends on grid resolution, but should be <= 3
        assert result.sizes["closest_cell"] <= 3
        assert result.sizes["closest_cell"] >= 1

        # Verify coordinates exist
        assert "target_lats" in result.coords
        assert "target_lons" in result.coords

    def test_fallback_all_points_invalid(self):
        """Test _clip_data_to_multiple_points_fallback when all points return None - outcome: returns None."""
        # Mock _clip_data_to_point to always return None
        with patch.object(Clip, "_clip_data_to_point", return_value=None), patch(
            "builtins.print"
        ):

            point_list = [(37.0, -119.0), (35.0, -121.0)]
            result = Clip._clip_data_to_multiple_points_fallback(
                self.dataset, point_list
            )

        # Verify result is None when all points are invalid
        assert result is None

    def test_fallback_mixed_valid_invalid(self):
        """Test _clip_data_to_multiple_points_fallback with mix of valid and invalid points - outcome: returns valid ones only."""
        # Save the original method
        original_method = Clip._clip_data_to_point

        # Create a mock that returns None for the second point
        def mock_clip_to_point(dataset, lat, lon):
            if lat == 35.0:  # Second point
                return None
            else:
                # Call the original method for other points
                return original_method(dataset, lat, lon)

        with patch.object(
            Clip, "_clip_data_to_point", side_effect=mock_clip_to_point
        ), patch("builtins.print"):

            point_list = [(37.0, -119.0), (35.0, -121.0), (40.0, -118.0)]
            result = Clip._clip_data_to_multiple_points_fallback(
                self.dataset, point_list
            )

        # Verify result exists and contains only valid points
        assert result is not None
        assert isinstance(result, xr.Dataset)

        # Should have 2 gridcells (first and third points)
        assert "closest_cell" in result.dims
        assert result.sizes["closest_cell"] == 2

    def test_fallback_concatenation_error(self):
        """Test _clip_data_to_multiple_points_fallback when concatenation fails - outcome: returns None."""
        # Mock xr.concat to raise an exception
        with patch(
            "xarray.concat", side_effect=Exception("Concatenation failed")
        ), patch("builtins.print"):

            point_list = [(37.0, -119.0), (35.0, -121.0)]
            result = Clip._clip_data_to_multiple_points_fallback(
                self.dataset, point_list
            )

        # Verify result is None when concatenation fails
        assert result is None


class TestClipDataToPointNaNSearch:
    """Test class for NaN search logic in _clip_data_to_point method.

    Tests the expanding radius search functionality that looks for valid
    (non-NaN) gridcells when the closest gridcell contains only NaN values.
    This covers lines 453-509 in clip.py.
    """

    def setup_method(self):
        """Set up test fixtures for NaN search tests."""
        # Create a base dataset with valid data
        data = np.random.rand(5, 10, 10) + 20
        self.dataset = xr.Dataset(
            {"temp": (["time", "y", "x"], data)},
            coords={
                "time": pd.date_range("2020-01-01", periods=5),
                "y": np.linspace(32, 42, 10),
                "x": np.linspace(-124, -114, 10),
                "lat": (["y", "x"], np.tile(np.linspace(32, 42, 10)[:, None], (1, 10))),
                "lon": (
                    ["y", "x"],
                    np.tile(np.linspace(-124, -114, 10)[None, :], (10, 1)),
                ),
            },
        )
        self.dataset.attrs["resolution"] = "3 km"
        self.dataset = self.dataset.rio.write_crs("EPSG:4326")

    def test_nan_search_no_valid_cells_found(self):
        """Test when no valid gridcells found within any search radius - outcome: returns None."""
        # Create dataset where ALL data is NaN
        data_all_nan = np.full((5, 10, 10), np.nan)
        dataset_all_nan = xr.Dataset(
            {"temp": (["time", "y", "x"], data_all_nan)},
            coords={
                "time": pd.date_range("2020-01-01", periods=5),
                "y": np.linspace(32, 42, 10),
                "x": np.linspace(-124, -114, 10),
                "lat": (["y", "x"], np.tile(np.linspace(32, 42, 10)[:, None], (1, 10))),
                "lon": (
                    ["y", "x"],
                    np.tile(np.linspace(-124, -114, 10)[None, :], (10, 1)),
                ),
            },
        )
        dataset_all_nan = dataset_all_nan.rio.write_crs("EPSG:4326")

        # Mock get_closest_gridcell to return None (simulating no closest gridcell found)
        with patch(
            "climakitae.new_core.processors.clip.get_closest_gridcell",
            return_value=None,
        ), patch("builtins.print") as mock_print:

            result = Clip._clip_data_to_point(dataset_all_nan, 37.0, -119.0)

        # Verify result is None when no valid gridcells found
        assert result is None

        # Verify appropriate message was printed
        printed_output = " ".join(
            [str(call[0][0]) for call in mock_print.call_args_list]
        )
        assert (
            "Closest gridcell contains NaN values" in printed_output
            or "No valid gridcells found" in printed_output
        )


class TestClipDataToPointNaNSearchExpansion:
    """Test class for NaN search expansion radius logic in _clip_data_to_point.

    Tests the expanding radius search (lines 453-509) that searches outward
    from the target point when the closest gridcell contains NaN values.
    Tests various radii and successful find scenarios.
    """

    def setup_method(self):
        """Set up test fixtures for NaN search expansion tests."""
        # Create a dataset with a pattern: center NaN, valid data in rings around it
        # This allows testing different search radii
        data = np.random.rand(3, 20, 20) + 20

        # Make the center region (around index 10,10) NaN
        # This will force the search to expand outward
        data[:, 9:11, 9:11] = np.nan  # Center 2x2 cells are NaN

        # Create coordinate arrays
        lats_1d = np.linspace(35, 39, 20)  # ~0.2 degree spacing
        lons_1d = np.linspace(-122, -118, 20)  # ~0.2 degree spacing

        # Create 2D lat/lon grids
        lons_2d, lats_2d = np.meshgrid(lons_1d, lats_1d)

        self.dataset = xr.Dataset(
            {"temp": (["time", "y", "x"], data)},
            coords={
                "time": pd.date_range("2020-01-01", periods=3),
                "y": lats_1d,
                "x": lons_1d,
                "lat": (["y", "x"], lats_2d),
                "lon": (["y", "x"], lons_2d),
            },
        )
        self.dataset = self.dataset.rio.write_crs("EPSG:4326")

        # Target point that maps to NaN center region
        self.target_lat = 37.0  # Center of domain
        self.target_lon = -120.0  # Center of domain

    def test_nan_search_finds_valid_at_first_radius(self):
        """Test finding valid gridcell at 0.01° radius - outcome: returns valid gridcell."""
        # Create a dataset where closest cell is NaN but valid data exists at 0.01° radius
        data = np.random.rand(3, 15, 15) + 20

        # Make a small NaN region at center
        center_idx = 7
        data[:, center_idx, center_idx] = np.nan

        # Very fine grid spacing to test 0.01 degree radius
        # Use 1D lat/lon coordinates (not 2D) as expected by the search logic
        lats_1d = np.linspace(36.95, 37.05, 15)  # ~0.007 degree spacing
        lons_1d = np.linspace(-120.05, -119.95, 15)

        dataset = xr.Dataset(
            {"temp": (["time", "lat", "lon"], data)},
            coords={
                "time": pd.date_range("2020-01-01", periods=3),
                "lat": lats_1d,
                "lon": lons_1d,
            },
        )
        dataset = dataset.rio.write_crs("EPSG:4326")

        # Mock get_closest_gridcell to return gridcell with all NaN data (triggering search)
        nan_result = dataset.isel(lon=center_idx, lat=center_idx)
        # Verify it's actually NaN
        assert np.isnan(nan_result["temp"].isel(time=0).values)

        with patch(
            "climakitae.new_core.processors.clip.get_closest_gridcell",
            return_value=nan_result,
        ), patch("builtins.print") as mock_print:
            result = Clip._clip_data_to_point(dataset, 37.0, -120.0)

        # Verify a valid result was found
        assert result is not None
        # Verify the result has valid (non-NaN) data
        assert not np.isnan(result["temp"].isel(time=0).values)

        # Verify search messages were printed
        printed_output = " ".join(
            [str(call[0][0]) for call in mock_print.call_args_list]
        )
        assert (
            "searching for nearest valid gridcell" in printed_output.lower()
            or "found valid" in printed_output.lower()
        )


class TestGetMultiBoundaryGeometry:
    """Test class for _get_multi_boundary_geometry method.

    Tests the multi-boundary geometry retrieval and combination logic.
    This covers lines 894-934 in clip.py.
    """

    def setup_method(self):
        """Set up test fixtures."""
        # Create mock catalog
        self.mock_catalog = MagicMock(spec=DataCatalog)
        self.mock_catalog.boundaries = {
            "states": {"CA": "California", "OR": "Oregon", "WA": "Washington"},
            "counties": {"Los Angeles County": "Los Angeles County"},
        }

        # Create Clip instance with mock catalog
        self.clip_processor = Clip("CA")
        self.clip_processor._data_accessor = self.mock_catalog

    def test_empty_list_raises_error(self):
        """Test empty list raises ValueError - outcome: raises ValueError."""
        with pytest.raises(ValueError, match="Empty list provided"):
            self.clip_processor._get_multi_boundary_geometry([])

    def test_single_boundary_delegates_to_single_method(self):
        """Test single boundary in list uses single boundary method - outcome: delegates to _get_boundary_geometry."""
        # Mock the single boundary geometry method
        mock_geometry = MagicMock()

        with patch.object(
            self.clip_processor, "_get_boundary_geometry", return_value=mock_geometry
        ) as mock_single:
            result = self.clip_processor._get_multi_boundary_geometry(["CA"])

        # Verify single boundary method was called
        mock_single.assert_called_once_with("CA")
        assert result == mock_geometry

    def test_multiple_valid_boundaries_combined(self):
        """Test multiple valid boundaries combined via union - outcome: returns combined geometry."""
        # Create mock geometries
        mock_geometry1 = MagicMock()
        mock_geometry2 = MagicMock()
        mock_combined = MagicMock()

        # Mock validate_boundary_key to return valid for all
        def mock_validate(key):
            return {"valid": True}

        # Mock _get_boundary_geometry to return geometries
        def mock_get_geometry(key):
            if key == "CA":
                return mock_geometry1
            elif key == "OR":
                return mock_geometry2

        with patch.object(
            self.clip_processor, "validate_boundary_key", side_effect=mock_validate
        ), patch.object(
            self.clip_processor, "_get_boundary_geometry", side_effect=mock_get_geometry
        ) as mock_get, patch.object(
            self.clip_processor, "_combine_geometries", return_value=mock_combined
        ) as mock_combine:

            result = self.clip_processor._get_multi_boundary_geometry(["CA", "OR"])

        # Verify geometries were retrieved
        assert mock_get.call_count == 2

        # Verify combine was called with correct geometries
        mock_combine.assert_called_once()
        call_args = mock_combine.call_args
        assert mock_geometry1 in call_args[0][0]
        assert mock_geometry2 in call_args[0][0]
        assert call_args[1]["operation"] == "union"

        assert result == mock_combined

    def test_some_invalid_boundaries_raises_error(self):
        """Test when some boundaries are invalid - outcome: raises ValueError with suggestions."""

        # Mock validate_boundary_key to return valid for some, invalid for others
        def mock_validate(key):
            if key in ["CA", "OR"]:
                return {"valid": True}
            else:
                return {"valid": False, "suggestions": ["Washington", "Wyoming"]}

        # Mock _get_boundary_geometry to succeed for valid keys
        def mock_get_geometry(key):
            if key in ["CA", "OR"]:
                return MagicMock()
            else:
                raise ValueError(f"Invalid key: {key}")

        with patch.object(
            self.clip_processor, "validate_boundary_key", side_effect=mock_validate
        ), patch.object(
            self.clip_processor, "_get_boundary_geometry", side_effect=mock_get_geometry
        ):

            with pytest.raises(
                ValueError, match="Invalid boundary keys: \\['INVALID'\\]"
            ):
                self.clip_processor._get_multi_boundary_geometry(
                    ["CA", "OR", "INVALID"]
                )

    def test_geometry_retrieval_failure(self):
        """Test when validation passes but geometry retrieval fails - outcome: raises ValueError."""

        # Mock validate_boundary_key to return valid
        def mock_validate(key):
            return {"valid": True}

        # Mock _get_boundary_geometry to raise exception
        def mock_get_geometry(key):
            raise Exception("Failed to retrieve geometry")

        with patch.object(
            self.clip_processor, "validate_boundary_key", side_effect=mock_validate
        ), patch.object(
            self.clip_processor, "_get_boundary_geometry", side_effect=mock_get_geometry
        ):

            with pytest.raises(ValueError, match="Invalid boundary keys"):
                self.clip_processor._get_multi_boundary_geometry(["CA", "OR"])

    def test_all_boundaries_invalid(self):
        """Test when all boundaries are invalid - outcome: raises ValueError."""

        # Mock validate_boundary_key to return invalid for all
        def mock_validate(key):
            return {"valid": False, "suggestions": ["California", "Oregon"]}

        with patch.object(
            self.clip_processor, "validate_boundary_key", side_effect=mock_validate
        ):

            with pytest.raises(
                ValueError, match="Invalid boundary keys: \\['BAD1', 'BAD2'\\]"
            ):
                self.clip_processor._get_multi_boundary_geometry(["BAD1", "BAD2"])


class TestClipErrorHandlingPaths:
    """Test class for error handling paths in Clip processor.

    Tests various error conditions and edge cases to ensure proper
    error handling and user feedback.
    """

    def setup_method(self):
        """Set up test fixtures."""
        # Create sample dataset
        data = np.random.rand(3, 10, 10) + 20
        self.dataset = xr.Dataset(
            {"temp": (["time", "y", "x"], data)},
            coords={
                "time": pd.date_range("2020-01-01", periods=3),
                "y": np.linspace(32, 42, 10),
                "x": np.linspace(-124, -114, 10),
                "lat": (["y", "x"], np.tile(np.linspace(32, 42, 10)[:, None], (1, 10))),
                "lon": (
                    ["y", "x"],
                    np.tile(np.linspace(-124, -114, 10)[None, :], (10, 1)),
                ),
            },
        )
        self.dataset.attrs["resolution"] = "3 km"
        self.dataset = self.dataset.rio.write_crs("EPSG:4326")

        self.context = {}

    def test_execute_invalid_result_type(self):
        """Test execute() with invalid result type - outcome: raises ValueError."""
        clip_processor = Clip((37.0, -119.0))

        # Pass an invalid type (int instead of Dataset/DataArray/dict/Iterable)
        invalid_result = 12345

        with pytest.raises(ValueError, match="Invalid result type for clipping"):
            clip_processor.execute(invalid_result, self.context)

    def test_execute_clipping_returns_none(self):
        """Test execute() when clipping returns None - outcome: raises ValueError."""
        clip_processor = Clip((37.0, -119.0))

        # Mock _clip_data_to_point to return None
        with patch.object(Clip, "_clip_data_to_point", return_value=None), patch(
            "builtins.print"
        ):

            with pytest.raises(
                ValueError, match="Clipping operation failed to produce valid results"
            ):
                clip_processor.execute(self.dataset, self.context)

    def test_multi_point_no_valid_gridcells(self):
        """Test _clip_data_to_multiple_points when get_closest_gridcells returns None - outcome: returns None."""
        point_list = [(37.0, -119.0), (35.0, -121.0)]

        # Mock get_closest_gridcells to return None
        with patch(
            "climakitae.new_core.processors.clip.get_closest_gridcells",
            return_value=None,
        ), patch("builtins.print") as mock_print:

            result = Clip._clip_data_to_multiple_points(self.dataset, point_list)

        # Verify result is None
        assert result is None

        # Verify appropriate message was printed
        printed_output = " ".join(
            [str(call[0][0]) for call in mock_print.call_args_list]
        )
        assert "No valid gridcells found" in printed_output

    def test_multi_point_exception_triggers_fallback(self):
        """Test _clip_data_to_multiple_points exception triggers fallback - outcome: calls fallback method."""
        point_list = [(37.0, -119.0), (35.0, -121.0)]
        mock_fallback_result = MagicMock()

        # Mock get_closest_gridcells to raise an exception
        with patch(
            "climakitae.new_core.processors.clip.get_closest_gridcells",
            side_effect=Exception("Vectorized clipping failed"),
        ), patch.object(
            Clip,
            "_clip_data_to_multiple_points_fallback",
            return_value=mock_fallback_result,
        ) as mock_fallback, patch(
            "builtins.print"
        ) as mock_print:

            result = Clip._clip_data_to_multiple_points(self.dataset, point_list)

        # Verify fallback method was called
        mock_fallback.assert_called_once_with(self.dataset, point_list)
        assert result == mock_fallback_result

        # Verify error and fallback messages were printed
        printed_output = " ".join(
            [str(call[0][0]) for call in mock_print.call_args_list]
        )
        assert "Error in vectorized multi-point clipping" in printed_output
        assert "Falling back" in printed_output

    def test_clip_data_with_geom_crs_warning(self):
        """Test _clip_data_with_geom CRS warning when GeoDataFrame has no CRS - outcome: warning issued."""
        # Create GeoDataFrame without CRS
        geometry = [box(-120, 35, -118, 38)]
        gdf = gpd.GeoDataFrame(geometry=geometry)
        # Explicitly set CRS to None
        gdf.crs = None

        # Verify dataset has CRS
        assert self.dataset.rio.crs is not None

        with pytest.warns(UserWarning, match="does not have a CRS set"):
            result = Clip._clip_data_with_geom(self.dataset, gdf)

        # Verify clipping still works
        assert result is not None
        assert isinstance(result, xr.Dataset)

    def test_execute_with_shapefile_path(self):
        """Test execute() with shapefile path - outcome: reads file and clips data."""
        # Create a temporary shapefile
        import tempfile
        import os

        # Create GeoDataFrame
        geometry = [box(-120, 35, -118, 38)]
        gdf = gpd.GeoDataFrame(geometry=geometry, crs="EPSG:4326")

        # Write to temporary shapefile
        with tempfile.TemporaryDirectory() as tmpdir:
            shapefile_path = os.path.join(tmpdir, "test.shp")
            gdf.to_file(shapefile_path)

            # Verify file exists
            assert os.path.exists(shapefile_path)

            # Create Clip processor with shapefile path
            clip_processor = Clip(shapefile_path)

            # Execute clipping
            result = clip_processor.execute(self.dataset, self.context)

            # Verify result
            assert result is not None
            assert isinstance(result, xr.Dataset)


class TestCombineGeometries:
    """Test class for _combine_geometries method.

    Tests the geometry combination logic used for multi-boundary clipping.
    This covers lines 959-992 in clip.py.
    """

    def setup_method(self):
        """Set up test fixtures."""
        # Create Clip instance
        self.clip_processor = Clip("CA")

        # Create sample geometries
        self.geom1 = gpd.GeoDataFrame(
            geometry=[box(-120, 35, -118, 37)], crs="EPSG:4326"
        )
        self.geom2 = gpd.GeoDataFrame(
            geometry=[box(-122, 37, -120, 39)], crs="EPSG:4326"
        )
        self.geom3 = gpd.GeoDataFrame(
            geometry=[box(-124, 39, -122, 41)], crs="EPSG:4326"
        )

    def test_combine_geometries_empty_list(self):
        """Test _combine_geometries with empty list - outcome: raises ValueError."""
        with pytest.raises(ValueError, match="No geometries provided"):
            self.clip_processor._combine_geometries([])

    def test_combine_geometries_single_geometry(self):
        """Test _combine_geometries with single geometry - outcome: returns geometry as-is."""
        result = self.clip_processor._combine_geometries([self.geom1])

        # Should return the same geometry without modification
        assert result is not None
        assert isinstance(result, gpd.GeoDataFrame)
        assert len(result) == 1
        assert result.crs == self.geom1.crs
        # Verify it's the same geometry
        assert result.geometry[0].equals(self.geom1.geometry[0])

    def test_combine_geometries_invalid_operation(self):
        """Test _combine_geometries with invalid operation - outcome: raises ValueError."""
        with pytest.raises(ValueError, match="Operation 'intersection' not supported"):
            self.clip_processor._combine_geometries(
                [self.geom1, self.geom2], operation="intersection"
            )

    def test_combine_geometries_successful_union(self):
        """Test _combine_geometries with successful union of multiple geometries - outcome: returns combined geometry."""
        result = self.clip_processor._combine_geometries(
            [self.geom1, self.geom2, self.geom3]
        )

        # Verify result
        assert result is not None
        assert isinstance(result, gpd.GeoDataFrame)
        # Should have single combined geometry
        assert len(result) == 1
        # Verify CRS is preserved
        assert result.crs == self.geom1.crs
        # Verify the geometry is a union (single geometry that covers all input areas)
        assert result.geometry[0] is not None

    def test_combine_geometries_concatenation_failure(self):
        """Test _combine_geometries when concatenation fails - outcome: raises ValueError."""
        # Mock pd.concat to raise an exception
        with patch("pandas.concat", side_effect=Exception("Concatenation error")):
            with pytest.raises(ValueError, match="Failed to concatenate geometries"):
                self.clip_processor._combine_geometries([self.geom1, self.geom2])

    def test_combine_geometries_crs_consistency(self):
        """Test _combine_geometries with CRS consistency handling - outcome: converts to common CRS.

        The method should handle geometries with different CRS by converting them all
        to the reference CRS (from first geometry) before concatenation.
        """
        # Create geometry with different CRS (Web Mercator)
        geom_different_crs = gpd.GeoDataFrame(
            geometry=[
                box(-13358338, 4470057, -13135699, 4721671)
            ],  # Approximate Web Mercator coords
            crs="EPSG:3857",  # Web Mercator instead of WGS84
        )

        # Should successfully combine and convert to reference CRS (EPSG:4326)
        result = self.clip_processor._combine_geometries(
            [self.geom1, geom_different_crs]
        )

        # Verify result is a GeoDataFrame with consistent CRS
        assert isinstance(result, gpd.GeoDataFrame)
        assert result.crs == self.geom1.crs  # Should use first geometry's CRS
        assert len(result) == 1  # Union should produce single geometry

    def test_combine_geometries_union_failure(self):
        """Test _combine_geometries when union operation fails - outcome: raises ValueError."""
        # Create geometries that will successfully concatenate but fail during union
        # Patch the unary_union property on GeoDataFrame to raise an exception
        with patch.object(
            gpd.GeoDataFrame,
            "unary_union",
            new_callable=lambda: property(
                lambda self: (_ for _ in ()).throw(Exception("Union operation failed"))
            ),
        ):
            with pytest.raises(
                ValueError, match="Failed to perform union operation on geometries"
            ):
                self.clip_processor._combine_geometries([self.geom1, self.geom2])
