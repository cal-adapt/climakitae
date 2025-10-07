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
