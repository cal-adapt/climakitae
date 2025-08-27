"""
Unit tests for climakitae/new_core/dataset.py.

This module contains comprehensive unit tests for the Dataset class that 
provides the core Dataset class that implements a flexible, pipeline-based
approach for climate data processing.
"""

from climakitae.new_core.dataset import Dataset


class TestDatasetInit:
    """Test class for Dataset initialization"""

    def test_init_successful(self):
        """Test successful initialization."""

        dataset = Dataset()

        assert hasattr(dataset, "data_access")
        assert hasattr(dataset, "parameter_validator")
        assert hasattr(dataset, "processing_pipeline")