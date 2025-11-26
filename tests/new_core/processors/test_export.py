"""
Unit tests for climakitae/new_core/processors/export.py.
"""

import pytest
from climakitae.new_core.processors.export import Export

class TestExportInitialization:
    """Test class for Export processor initialization."""

    def test_init_defaults(self):
        """Test initialization with default values."""
        processor = Export({})
        assert processor.filename == "dataexport"
        assert processor.file_format == "NetCDF"
        assert processor.mode == "local"
        assert processor.separated is False
        assert processor.export_method == "data"
        assert processor.location_based_naming is False
        assert processor.filename_template is None
        assert processor.fail_on_error is True
