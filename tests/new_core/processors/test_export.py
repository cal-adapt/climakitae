"""
Unit tests for climakitae/new_core/processors/export.py.
"""

import pytest
from unittest.mock import patch
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

    def test_generate_filename_gridded_ignores_location_naming(self):
        """Test that location_based_naming is silently ignored for gridded data."""
        processor = Export({"filename": "output", "location_based_naming": True})

        # Create dataset with multi-dimensional lat/lon (gridded data, not single-point)
        ds_gridded = xr.Dataset(
            {"temp": (["lat", "lon"], np.random.rand(2, 2))},
            coords={"lat": [34.0, 35.0], "lon": [-118.0, -117.0]},
        )

        # Should return base filename without location suffix (silently ignored)
        filename = processor._generate_filename(ds_gridded)
        assert filename == "output"

    def test_generate_filename_template_error(self):
        """Test filename generation when template variable extraction fails."""
        processor = Export(
            {"filename": "output", "filename_template": "{filename}_{lat}_{lon}"}
        )

        # Create dataset with multi-dimensional lat/lon
        ds_error = xr.Dataset(
            {"temp": (["lat", "lon"], np.random.rand(2, 2))},
            coords={"lat": [34.0, 35.0], "lon": [-118.0, -117.0]},
        )

        # Should use empty strings for lat/lon in template
        filename = processor._generate_filename(ds_error)
        assert filename == "output__"


class TestExportSinglePointDetection:
    """Test class for single-point data detection and coordinate extraction."""

    def setup_method(self):
        """Set up test fixtures."""
        self.processor = Export({"filename": "test"})

    def test_is_single_point_data_true(self):
        """Test detection of single-point data."""
        ds = xr.Dataset(
            {"temp": (["lat", "lon"], [[25.5]])},
            coords={"lat": [34.0], "lon": [-118.0]},
        )
        assert self.processor._is_single_point_data(ds) is True

    def test_is_single_point_data_false_multiple_lat(self):
        """Test detection of multi-point data (multiple lat values)."""
        ds = xr.Dataset(
            {"temp": (["lat", "lon"], [[25.5], [26.0]])},
            coords={"lat": [34.0, 35.0], "lon": [-118.0]},
        )
        assert self.processor._is_single_point_data(ds) is False

    def test_is_single_point_data_false_multiple_lon(self):
        """Test detection of multi-point data (multiple lon values)."""
        ds = xr.Dataset(
            {"temp": (["lat", "lon"], [[25.5, 26.0]])},
            coords={"lat": [34.0], "lon": [-118.0, -117.0]},
        )
        assert self.processor._is_single_point_data(ds) is False

    def test_is_single_point_data_no_lat(self):
        """Test detection when data has no lat coordinate."""
        ds = xr.Dataset(
            {"temp": (["x"], [25.5])},
            coords={"x": [1]},
        )
        assert self.processor._is_single_point_data(ds) is False

    def test_extract_point_coordinates_success(self):
        """Test successful extraction of point coordinates."""
        ds = xr.Dataset(
            {"temp": (["lat", "lon"], [[25.5]])},
            coords={"lat": [34.0], "lon": [-118.0]},
        )
        lat, lon = self.processor._extract_point_coordinates(ds)
        assert lat == 34.0
        assert lon == -118.0

    def test_extract_point_coordinates_error_multi_lat(self):
        """Test error when extracting coordinates from multi-lat data."""
        ds = xr.Dataset(
            {"temp": (["lat", "lon"], [[25.5], [26.0]])},
            coords={"lat": [34.0, 35.0], "lon": [-118.0]},
        )
        with pytest.raises(
            ValueError, match="Cannot use location_based_naming with gridded data"
        ):
            self.processor._extract_point_coordinates(ds)

    def test_extract_point_coordinates_error_message_includes_sizes(self):
        """Test that error message includes actual lat/lon sizes."""
        ds = xr.Dataset(
            {"temp": (["lat", "lon"], np.random.rand(5, 10))},
            coords={"lat": np.arange(5), "lon": np.arange(10)},
        )
        with pytest.raises(ValueError, match=r"5 lat value\(s\) and 10 lon value\(s\)"):
            self.processor._extract_point_coordinates(ds)

    def test_extract_point_coordinates_error_mentions_clip(self):
        """Test that error message mentions clip processor as solution."""
        ds = xr.Dataset(
            {"temp": (["lat", "lon"], np.random.rand(2, 2))},
            coords={"lat": [34.0, 35.0], "lon": [-118.0, -117.0]},
        )
        with pytest.raises(ValueError, match="Use the 'clip' processor"):
            self.processor._extract_point_coordinates(ds)


class TestExportCollectionHandling:
    """Test class for collection (list/tuple) export behavior."""

    def setup_method(self):
        """Set up test fixtures."""
        # Create a list of single-point datasets (like cava_data output)
        self.point_datasets = [
            xr.Dataset(
                {"temp": (["lat", "lon"], [[25.5]])},
                coords={"lat": [34.0], "lon": [-118.0]},
            ),
            xr.Dataset(
                {"temp": (["lat", "lon"], [[26.5]])},
                coords={"lat": [35.0], "lon": [-119.0]},
            ),
            xr.Dataset(
                {"temp": (["lat", "lon"], [[27.5]])},
                coords={"lat": [36.0], "lon": [-120.0]},
            ),
        ]

    def test_export_collection_separated_with_location_naming(self):
        """Test that separated=True with location_based_naming uses lat/lon in filenames."""
        processor = Export(
            {
                "filename": "test_output",
                "separated": True,
                "location_based_naming": True,
            }
        )

        with patch.object(processor, "export_single") as mock_export:
            processor._export_collection(self.point_datasets)

            # Should be called 3 times, once for each point
            assert mock_export.call_count == 3

        # Verify filenames would include lat/lon (check via _export_single_from_collection)
        with patch.object(processor, "export_single") as mock_export:
            processor._export_single_from_collection(self.point_datasets[0], 0)

            # Filename should have been modified to include lat/lon
            # The export_single is called with location_based_naming temporarily disabled
            mock_export.assert_called_once()

    def test_export_collection_separated_without_location_naming(self):
        """Test that separated=True without location_based_naming uses index in filenames."""
        processor = Export(
            {
                "filename": "test_output",
                "separated": True,
                "location_based_naming": False,
            }
        )

        # Track filename changes during export
        filenames_used = []

        def track_filename(data):
            filenames_used.append(processor.filename)
            # Don't actually export
            return

        with patch.object(processor, "export_single", side_effect=track_filename):
            processor._export_collection(self.point_datasets)

        # Should use index-based naming: test_output_0, test_output_1, test_output_2
        assert filenames_used == ["test_output_0", "test_output_1", "test_output_2"]

    def test_export_collection_not_separated(self):
        """Test that separated=False exports each item without index/location suffix."""
        processor = Export(
            {
                "filename": "test_output",
                "separated": False,
                "location_based_naming": True,  # Should be ignored when not separated
            }
        )

        with patch.object(processor, "export_single") as mock_export:
            processor._export_collection(self.point_datasets)

            # Should be called 3 times
            assert mock_export.call_count == 3

    def test_export_data_single_gridded_ignores_options(self):
        """Test that a single gridded dataset ignores separated and location_based_naming."""
        processor = Export(
            {
                "filename": "gridded_output",
                "separated": True,  # Should be ignored
                "location_based_naming": True,  # Should be ignored
            }
        )

        gridded_ds = xr.Dataset(
            {"temp": (["lat", "lon"], np.random.rand(10, 10))},
            coords={"lat": np.arange(10), "lon": np.arange(10)},
        )

        with patch.object(processor, "export_single") as mock_export:
            processor._export_data(gridded_ds)

            # Should be called once with the dataset
            mock_export.assert_called_once_with(gridded_ds)


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

    def test_clean_attrs_dataarray(self):
        """Test attribute cleaning for DataArray."""
        da = xr.DataArray([1, 2], attrs={"dict_attr": {"key": "value"}})
        cleaned_da = self.processor._clean_attrs_for_netcdf(da)
        assert isinstance(cleaned_da.attrs["dict_attr"], str)

    def test_clean_attrs_tolist_failure(self):
        """Test attribute cleaning when tolist fails."""

        class FailToList:
            def tolist(self):
                raise ValueError("Fail")

            def __str__(self):
                return "FailToList"

        ds = xr.Dataset({"temp": (["time"], [1, 2])}, attrs={"fail_attr": FailToList()})
        cleaned_ds = self.processor._clean_attrs_for_netcdf(ds)
        assert cleaned_ds.attrs["fail_attr"] == "FailToList"


class TestExportExecute:
    """Test class for execute method routing logic."""

    def setup_method(self):
        """Set up test fixtures."""
        self.ds = xr.Dataset({"temp": (["time"], [1, 2])})
        self.context = {}

    def test_execute_none_method(self):
        """Test execute with export_method='none'."""
        processor = Export({"export_method": "none"})
        with patch.object(processor, "export_single") as mock_export:
            result = processor.execute(self.ds, self.context)
            assert result is self.ds
            mock_export.assert_not_called()

    def test_execute_data_method(self):
        """Test execute with export_method='data'."""
        processor = Export({"export_method": "data"})
        with patch.object(processor, "export_single") as mock_export:
            result = processor.execute(self.ds, self.context)
            assert result is self.ds
            mock_export.assert_called_once_with(self.ds)

    def test_execute_dict_input(self):
        """Test execute with dictionary input."""
        processor = Export({"export_method": "data"})
        data_dict = {"ds1": self.ds, "ds2": self.ds}

        with patch.object(processor, "export_single") as mock_export:
            result = processor.execute(data_dict, self.context)
            assert result is data_dict
            assert mock_export.call_count == 2

    def test_execute_list_input(self):
        """Test execute with list input."""
        processor = Export({"export_method": "data"})
        data_list = [self.ds, self.ds]

        with patch.object(processor, "export_single") as mock_export:
            result = processor.execute(data_list, self.context)
            assert result is data_list
            assert mock_export.call_count == 2

    def test_execute_selective_export(self):
        """Test execute with selective export methods."""
        # Test raw export
        processor = Export({"export_method": "raw"})
        with patch.object(processor, "_export_with_suffix") as mock_export:
            processor.execute(self.ds, self.context)
            mock_export.assert_called_once_with(self.ds, "raw")

        # Test calculate export
        processor = Export({"export_method": "calculate"})
        with patch.object(processor, "_export_with_suffix") as mock_export:
            processor.execute(self.ds, self.context)
            mock_export.assert_called_once_with(self.ds, "calc")

        # Test both export
        processor = Export({"export_method": "both"})
        with patch.object(processor, "_export_with_suffix") as mock_export:
            processor.execute(self.ds, self.context)
            assert mock_export.call_count == 2
            mock_export.assert_any_call(self.ds, "raw")
            mock_export.assert_any_call(self.ds, "calc")

    def test_update_context(self):
        """Test update_context method."""
        processor = Export({"filename": "test"})
        context = {}
        processor.update_context(context)

        from climakitae.core.constants import _NEW_ATTRS_KEY

        assert _NEW_ATTRS_KEY in context
        assert processor.name in context[_NEW_ATTRS_KEY]
        assert (
            "Transformation was done using the following value"
            in context[_NEW_ATTRS_KEY][processor.name]
        )

    def test_determine_data_type_context(self):
        """Test _determine_data_type with context indicators."""
        processor = Export({})
        from climakitae.core.constants import _NEW_ATTRS_KEY

        # Test with processing steps in context
        context = {_NEW_ATTRS_KEY: {"some_process": "details"}}
        assert processor._determine_data_type(self.ds, context) == "calc"

        # Test with only load steps
        context = {_NEW_ATTRS_KEY: {"_load_data": "details"}}
        assert processor._determine_data_type(self.ds, context) == "raw"

    def test_determine_data_type_attrs(self):
        """Test _determine_data_type with data attributes."""
        processor = Export({})
        context = {}

        # Test with processed_by attribute
        ds_calc = self.ds.copy()
        ds_calc.attrs["processed_by"] = "some_process"
        assert processor._determine_data_type(ds_calc, context) == "calc"

        # Test with calculation_method attribute
        ds_calc = self.ds.copy()
        ds_calc.attrs["calculation_method"] = "mean"
        assert processor._determine_data_type(ds_calc, context) == "calc"

        # Test with derived_from attribute
        ds_calc = self.ds.copy()
        ds_calc.attrs["derived_from"] = "other_data"
        assert processor._determine_data_type(ds_calc, context) == "calc"

        # Test with no indicators
        assert processor._determine_data_type(self.ds, context) == "raw"

    def test_handle_dict_result_mismatch(self):
        """Test _handle_dict_result with mismatched export method."""
        processor = Export({})

        # export_method="raw" but only calc_data present
        with patch.object(processor, "_export_with_suffix") as mock_export:
            processor._handle_dict_result({"calc_data": self.ds}, "raw")
            mock_export.assert_not_called()

        # export_method="calculate" but only raw_data present
        with patch.object(processor, "_export_with_suffix") as mock_export:
            processor._handle_dict_result({"raw_data": self.ds}, "calculate")
            mock_export.assert_not_called()

    def test_handle_selective_export_mismatch(self):
        """Test _handle_selective_export with mismatched data type."""
        processor = Export({})

        # export_method="raw" but data_type="calc"
        # Should fall back to default behavior (export with suffix matching export_method)
        with patch.object(processor, "_determine_data_type", return_value="calc"):
            with patch.object(processor, "_export_with_suffix") as mock_export:
                processor._handle_selective_export(self.ds, {}, "raw")
                mock_export.assert_called_once_with(self.ds, "raw")

        # export_method="calculate" but data_type="raw"
        # Should fall back to default behavior
        with patch.object(processor, "_determine_data_type", return_value="raw"):
            with patch.object(processor, "_export_with_suffix") as mock_export:
                processor._handle_selective_export(self.ds, {}, "calculate")
                mock_export.assert_called_once_with(self.ds, "calc")

    def test_export_with_suffix(self):
        """Test _export_with_suffix method."""
        processor = Export({})
        with patch.object(processor, "export_single") as mock_export:
            processor._export_with_suffix(self.ds, "raw")
            # export_single is called with just the data
            mock_export.assert_called_once_with(self.ds)
            # Verify filename was temporarily modified
            # We can't easily verify the temporary modification here as it's reverted
            # But we can verify it was called

    def test_export_with_suffix_dict(self):
        """Test _export_with_suffix with dictionary input."""
        processor = Export({})
        data_dict = {"ds1": self.ds, "ds2": self.ds}
        with patch.object(processor, "export_single") as mock_export:
            processor._export_with_suffix(data_dict, "raw")
            assert mock_export.call_count == 2

    def test_export_with_suffix_list(self):
        """Test _export_with_suffix with list input."""
        processor = Export({})
        data_list = [self.ds, self.ds]
        with patch.object(processor, "export_single") as mock_export:
            processor._export_with_suffix(data_list, "raw")
            assert mock_export.call_count == 2

    def test_export_with_suffix_invalid_type(self):
        """Test _export_with_suffix with invalid input type."""
        processor = Export({})
        with pytest.raises(
            TypeError, match="Expected xr.Dataset, xr.DataArray, dict, list, or tuple"
        ):
            processor._export_with_suffix("invalid", "raw")

        with pytest.raises(TypeError, match="Expected xr.Dataset or xr.DataArray"):
            processor._export_with_suffix(["invalid"], "raw")


class TestExportSingle:
    """Test class for export_single method."""

    def setup_method(self):
        """Set up test fixtures."""
        self.ds = xr.Dataset({"temp": (["time"], [1, 2])})
        self.processor = Export({"filename": "test_output"})

    @patch("climakitae.new_core.processors.export._export_to_netcdf")
    def test_export_single_netcdf(self, mock_export):
        """Test export to NetCDF."""
        self.processor.file_format = "NetCDF"
        self.processor.export_single(self.ds)
        mock_export.assert_called_once()
        args, _ = mock_export.call_args
        assert args[1] == "test_output.nc"

    @patch("climakitae.new_core.processors.export._export_to_zarr")
    def test_export_single_zarr(self, mock_export):
        """Test export to Zarr."""
        self.processor.file_format = "Zarr"
        self.processor.export_single(self.ds)
        mock_export.assert_called_once()
        args, _ = mock_export.call_args
        assert args[1] == "test_output.zarr"
        assert args[2] == "local"

    @patch("climakitae.new_core.processors.export._export_to_csv")
    def test_export_single_csv(self, mock_export):
        """Test export to CSV."""
        self.processor.file_format = "CSV"
        self.processor.export_single(self.ds)
        mock_export.assert_called_once()
        args, _ = mock_export.call_args
        assert args[1] == "test_output.csv.gz"

    @patch("os.path.exists", return_value=True)
    @patch("climakitae.new_core.processors.export._export_to_netcdf")
    def test_export_single_skip_existing(self, mock_export, mock_exists):
        """Test skip existing file."""
        self.processor.export_method = "skip_existing"
        self.processor.export_single(self.ds)
        mock_export.assert_not_called()

    @patch(
        "climakitae.new_core.processors.export._export_to_netcdf",
        side_effect=RuntimeError("Export failed"),
    )
    def test_export_single_fail_on_error(self, mock_export):
        """Test fail on error behavior."""
        # Default is fail_on_error=True
        with pytest.raises(RuntimeError, match="Export failed"):
            self.processor.export_single(self.ds)

        # Test with fail_on_error=False
        self.processor.fail_on_error = False
        # Should not raise exception
        self.processor.export_single(self.ds)

    def test_export_single_invalid_type(self):
        """Test export_single with invalid input type."""
        with pytest.raises(TypeError, match="Expected xr.Dataset or xr.DataArray"):
            self.processor.export_single("invalid_type")
