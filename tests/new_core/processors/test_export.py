"""
Unit tests for climakitae/new_core/processors/export.py.
"""

import pytest
import xarray as xr
import numpy as np
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

    def test_init_s3_requires_zarr(self):
        """Test that S3 mode requires Zarr format."""
        with pytest.raises(
            ValueError, match='To export to AWS S3 you must use file_format="Zarr"'
        ):
            Export({"mode": "s3", "file_format": "NetCDF"})

    def test_init_invalid_types(self):
        """Test initialization with invalid parameter types."""
        with pytest.raises(ValueError, match="filename must be a string"):
            Export({"filename": 123})

        with pytest.raises(ValueError, match="separated must be a boolean"):
            Export({"separated": "yes"})

        with pytest.raises(ValueError, match="location_based_naming must be a boolean"):
            Export({"location_based_naming": 1})

        with pytest.raises(ValueError, match="filename_template must be a string"):
            Export({"filename_template": 123})

        with pytest.raises(ValueError, match="fail_on_error must be a boolean"):
            Export({"fail_on_error": "true"})

        with pytest.raises(ValueError, match="raw_filename must be a string"):
            Export({"raw_filename": 123})

        with pytest.raises(ValueError, match="calc_filename must be a string"):
            Export({"calc_filename": 123})

    def test_init_invalid_export_method(self):
        """Test initialization with invalid export method."""
        with pytest.raises(ValueError, match="export_method must be one of"):
            Export({"export_method": "invalid_method"})

class TestExportFilenameGeneration:
    """Test class for filename generation logic."""

    def setup_method(self):
        """Set up test fixtures."""
        self.ds = xr.Dataset(
            {"temp": (["time", "lat", "lon"], np.random.rand(2, 2, 2))},
            coords={"lat": [34.0, 35.0], "lon": [-118.0, -117.0], "time": [0, 1]},
        )
        self.da = xr.DataArray(
            np.random.rand(2, 2, 2),
            dims=["time", "lat", "lon"],
            coords={"lat": [34.0, 35.0], "lon": [-118.0, -117.0], "time": [0, 1]},
            name="test_array",
        )

    def test_generate_filename_default(self):
        """Test default filename generation."""
        processor = Export({"filename": "output"})
        filename = processor._generate_filename(self.ds)
        assert filename == "output"

    def test_generate_filename_separated(self):
        """Test filename generation with separated=True."""
        processor = Export({"filename": "output", "separated": True})
        # DataArray has a name
        filename = processor._generate_filename(self.da)
        assert filename == "test_array_output"

        # Dataset usually doesn't have a name attribute that evaluates to True in boolean context unless set?
        # Actually xr.Dataset doesn't have a 'name' attribute.
        # The code checks: hasattr(data, "name") and data.name
        # So for Dataset it should just be "output"
        filename_ds = processor._generate_filename(self.ds)
        assert filename_ds == "output"

    def test_generate_filename_location_based(self):
        """Test filename generation with location_based_naming=True."""
        processor = Export({"filename": "output", "location_based_naming": True})

        # Create a single point dataset for this test
        ds_point = self.ds.isel(lat=0, lon=0)
        filename = processor._generate_filename(ds_point)

        # Expected format: {base_filename}_{lat}N_{lon}W
        # lat=34.0 -> 340N, lon=-118.0 -> 1180W
        assert filename == "output_340N_1180W"

    def test_generate_filename_template(self):
        """Test filename generation with custom template."""
        processor = Export(
            {
                "filename": "output",
                "filename_template": "{name}_{filename}_{lat}N_{lon}W",
            }
        )

        ds_point = self.da.isel(lat=0, lon=0)
        filename = processor._generate_filename(ds_point)

        # name="test_array", filename="output", lat=34.0, lon=-118.0
        assert filename == "test_array_output_340N_1180W"

class TestExportAttributeCleaning:
    """Test class for attribute cleaning logic."""

    def setup_method(self):
        """Set up test fixtures."""
        self.processor = Export({})
        self.ds = xr.Dataset(
            {"temp": (["time"], [1, 2])},
            attrs={
                "string_attr": "value",
                "int_attr": 1,
                "float_attr": 1.5,
                "list_attr": [1, 2],
                "dict_attr": {"key": "value"},
                "none_attr": None,
                "callable_attr": lambda x: x,
                "numpy_attr": np.array([1, 2]),
            },
        )

    def test_clean_attrs_for_netcdf(self):
        """Test attribute cleaning for NetCDF export."""
        cleaned_ds = self.processor._clean_attrs_for_netcdf(self.ds)

        # Check that basic types are preserved
        assert cleaned_ds.attrs["string_attr"] == "value"
        assert cleaned_ds.attrs["int_attr"] == 1
        assert cleaned_ds.attrs["float_attr"] == 1.5
        assert cleaned_ds.attrs["list_attr"] == [1, 2]

        # Check that dicts are converted to string
        assert isinstance(cleaned_ds.attrs["dict_attr"], str)
        assert "{'key': 'value'}" in cleaned_ds.attrs["dict_attr"]

        # Check that None values are removed
        assert "none_attr" not in cleaned_ds.attrs

        # Check that callables are removed
        assert "callable_attr" not in cleaned_ds.attrs

        # Check that numpy arrays are converted to lists
        assert isinstance(cleaned_ds.attrs["numpy_attr"], list)
        assert cleaned_ds.attrs["numpy_attr"] == [1, 2]
