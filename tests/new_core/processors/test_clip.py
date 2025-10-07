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
        self.sample_dataset = xr.Dataset({
            'temp': (['time', 'y', 'x'], np.random.rand(10, 5, 5))
        }, coords={
            'time': pd.date_range('2020-01-01', periods=10),
            'y': np.linspace(32, 42, 5),
            'x': np.linspace(-124, -114, 5)
        })
        # Set CRS
        self.sample_dataset.rio.write_crs("EPSG:4326", inplace=True)
        
        # Create mock geometry
        self.mock_geometry = gpd.GeoDataFrame(
            geometry=[box(-125, 32, -114, 42)],
            crs=pyproj.CRS.from_epsg(4326)
        )
    
    def test_execute_single_boundary_dataset(self):
        """Test execute with single boundary and xr.Dataset - outcome: data clipped correctly."""
        with patch.object(self.clip, '_get_boundary_geometry', return_value=self.mock_geometry), \
             patch.object(self.clip, '_clip_data_with_geom', return_value=self.sample_dataset) as mock_clip:
            
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
        data_dict = {'sim1': self.sample_dataset, 'sim2': self.sample_dataset}
        
        with patch.object(self.clip, '_get_boundary_geometry', return_value=self.mock_geometry), \
             patch.object(self.clip, '_clip_data_with_geom', return_value=self.sample_dataset) as mock_clip:
            
            context = {}
            result = self.clip.execute(data_dict, context)
            
            # Verify result structure
            assert isinstance(result, dict)
            assert 'sim1' in result
            assert 'sim2' in result
            
            # Verify clipping was called for each dataset
            assert mock_clip.call_count == 2
            
            # Verify context was updated
            assert _NEW_ATTRS_KEY in context
    
    def test_execute_single_boundary_list(self):
        """Test execute with single boundary and list of datasets - outcome: all datasets clipped."""
        data_list = [self.sample_dataset, self.sample_dataset]
        
        with patch.object(self.clip, '_get_boundary_geometry', return_value=self.mock_geometry), \
             patch.object(self.clip, '_clip_data_with_geom', return_value=self.sample_dataset) as mock_clip:
            
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
        self.sample_dataset = xr.Dataset({
            'temp': (['time', 'y', 'x'], np.random.rand(10, 5, 5))
        }, coords={
            'time': pd.date_range('2020-01-01', periods=10),
            'y': np.linspace(32, 42, 5),
            'x': np.linspace(-124, -114, 5)
        })
        # Create a single-point result
        self.clipped_point = self.sample_dataset.isel(x=2, y=2)
    
    def test_execute_single_point_dataset(self):
        """Test execute with single point and xr.Dataset - outcome: closest gridcell returned."""
        with patch.object(self.clip, '_clip_data_to_point', return_value=self.clipped_point) as mock_clip:
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
        
        with patch.object(self.clip, '_clip_data_to_point', return_value=self.clipped_point):
            context = {}
            result = self.clip.execute(data_list, context)
            
            # Verify result structure
            assert isinstance(result, list)
            assert len(result) == 2
            
            # Verify context was updated
            assert _NEW_ATTRS_KEY in context
    
    def test_execute_single_point_dict(self):
        """Test execute with single point and dict of datasets - outcome: dict of closest gridcells."""
        data_dict = {'sim1': self.sample_dataset, 'sim2': self.sample_dataset}
        
        with patch.object(self.clip, '_clip_data_to_point', return_value=self.clipped_point):
            context = {}
            result = self.clip.execute(data_dict, context)
            
            # Verify result structure
            assert isinstance(result, dict)
            assert 'sim1' in result
            assert 'sim2' in result
            
            # Verify context was updated
            assert _NEW_ATTRS_KEY in context


class TestClipExecuteWithMultiplePoints:
    """Test class for execute method with multiple points."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.points = [(37.7749, -122.4194), (34.0522, -118.2437)]
        self.clip = Clip(self.points)
        self.sample_dataset = xr.Dataset({
            'temp': (['time', 'y', 'x'], np.random.rand(10, 5, 5))
        }, coords={
            'time': pd.date_range('2020-01-01', periods=10),
            'y': np.linspace(32, 42, 5),
            'x': np.linspace(-124, -114, 5)
        })
        # Create a multi-point result with closest_cell dimension
        self.clipped_multipoint = xr.concat([
            self.sample_dataset.isel(x=2, y=2),
            self.sample_dataset.isel(x=1, y=1)
        ], dim='closest_cell')
    
    def test_execute_multiple_points_dataset(self):
        """Test execute with multiple points and xr.Dataset - outcome: concatenated closest gridcells."""
        with patch.object(self.clip, '_clip_data_to_multiple_points', return_value=self.clipped_multipoint) as mock_clip:
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
        
        with patch.object(self.clip, '_clip_data_to_multiple_points', return_value=self.clipped_multipoint):
            context = {}
            result = self.clip.execute(data_list, context)
            
            # Verify result structure
            assert isinstance(result, list)
            assert len(result) == 2
            
            # Verify context was updated
            assert _NEW_ATTRS_KEY in context
    
    def test_execute_multiple_points_dict(self):
        """Test execute with multiple points and dict - outcome: dict of multi-point results."""
        data_dict = {'sim1': self.sample_dataset, 'sim2': self.sample_dataset}
        
        with patch.object(self.clip, '_clip_data_to_multiple_points', return_value=self.clipped_multipoint):
            context = {}
            result = self.clip.execute(data_dict, context)
            
            # Verify result structure
            assert isinstance(result, dict)
            assert 'sim1' in result
            assert 'sim2' in result
            
            # Verify context was updated
            assert _NEW_ATTRS_KEY in context
