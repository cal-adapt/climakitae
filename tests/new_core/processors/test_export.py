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

    def test_init_valid_parameters(self):
        """Test initialization with valid custom parameters."""
        config = {
            "filename": "custom_name",
            "file_format": "Zarr",
            "mode": "s3",
            "separated": True,
            "export_method": "raw",
            "location_based_naming": True,
            "filename_template": "{name}_{lat}",
            "fail_on_error": False,
            "raw_filename": "raw_file",
            "calc_filename": "calc_file",
        }
        processor = Export(config)
        assert processor.filename == "custom_name"
        assert processor.file_format == "Zarr"
        assert processor.mode == "s3"
        assert processor.separated is True
        assert processor.export_method == "raw"
        assert processor.location_based_naming is True
        assert processor.filename_template == "{name}_{lat}"
        assert processor.fail_on_error is False
        assert processor.raw_filename == "raw_file"
        assert processor.calc_filename == "calc_file"

    def test_init_invalid_file_format(self):
        """Test initialization with invalid file format."""
        with pytest.raises(ValueError, match="file_format must be one of"):
            Export({"file_format": "InvalidFormat"})

    def test_init_invalid_mode(self):
        """Test initialization with invalid mode."""
        with pytest.raises(ValueError, match="mode must be one of"):
            Export({"mode": "invalid_mode"})
