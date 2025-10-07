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
