"""
Unit tests for climakitae/new_core/processors/export.py.
"""

from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest
import xarray as xr

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


class TestExportClosestCellHandling:
    """Test class for multi-point clip results (closest_cell dimension) export behavior."""

    def setup_method(self):
        """Set up test fixtures with closest_cell dimension data (like from clip processor)."""
        # Simulate output from clipping to multiple lat/lon points
        # The clip processor creates a closest_cell dimension with target_lats/lons coords
        self.multi_point_ds = xr.Dataset(
            {
                "temp": (
                    ["time", "closest_cell"],
                    [[25.5, 26.5, 27.5], [25.6, 26.6, 27.6]],
                )
            },
            coords={
                "time": [0, 1],
                "closest_cell": [0, 1, 2],
                "target_lats": ("closest_cell", [34.05, 37.77, 32.72]),
                "target_lons": ("closest_cell", [-118.25, -122.42, -117.16]),
                "point_index": ("closest_cell", [0, 1, 2]),
            },
        )

        # Dataset with closest_cell but no target coords (edge case)
        self.multi_point_ds_no_target = xr.Dataset(
            {"temp": (["time", "closest_cell"], [[25.5, 26.5], [25.6, 26.6]])},
            coords={
                "time": [0, 1],
                "closest_cell": [0, 1],
            },
        )

    def test_has_closest_cell_dimension_true(self):
        """Test detection of closest_cell dimension."""
        processor = Export({})
        assert processor._has_closest_cell_dimension(self.multi_point_ds) is True

    def test_has_closest_cell_dimension_false_no_dim(self):
        """Test detection returns False when no closest_cell dimension."""
        processor = Export({})
        gridded_ds = xr.Dataset(
            {"temp": (["lat", "lon"], np.random.rand(10, 10))},
            coords={"lat": np.arange(10), "lon": np.arange(10)},
        )
        assert processor._has_closest_cell_dimension(gridded_ds) is False

    def test_has_closest_cell_dimension_false_size_one(self):
        """Test detection returns False when closest_cell has size 1."""
        processor = Export({})
        single_point_ds = xr.Dataset(
            {"temp": (["closest_cell"], [25.5])},
            coords={"closest_cell": [0]},
        )
        assert processor._has_closest_cell_dimension(single_point_ds) is False

    def test_export_data_separated_splits_closest_cell(self):
        """Test that separated=True splits data along closest_cell dimension."""
        processor = Export(
            {
                "filename": "test_output",
                "separated": True,
                "location_based_naming": False,
            }
        )

        with patch.object(processor, "export_single") as mock_export:
            processor._export_data(self.multi_point_ds)

            # Should be called 3 times (once per closest_cell)
            assert mock_export.call_count == 3

    def test_export_data_not_separated_keeps_together(self):
        """Test that separated=False exports the full dataset as one file."""
        processor = Export(
            {
                "filename": "test_output",
                "separated": False,
            }
        )

        with patch.object(processor, "export_single") as mock_export:
            processor._export_data(self.multi_point_ds)

            # Should be called once with the full dataset
            mock_export.assert_called_once()

    def test_split_closest_cells_with_location_naming(self):
        """Test location-based naming when splitting closest_cell dimension."""
        processor = Export(
            {
                "filename": "station",
                "separated": True,
                "location_based_naming": True,
            }
        )

        filenames_used = []

        def track_filename(data):
            filenames_used.append(processor.filename)

        with patch.object(processor, "export_single", side_effect=track_filename):
            processor._split_and_export_closest_cells(self.multi_point_ds)

        # Should use target_lats/lons for naming
        # Format: {base}_{lat}{N/S}_{lon}{E/W}
        assert len(filenames_used) == 3
        assert filenames_used[0] == "station_34-05N_118-25W"
        assert filenames_used[1] == "station_37-77N_122-42W"
        assert filenames_used[2] == "station_32-72N_117-16W"

    def test_split_closest_cells_with_index_naming(self):
        """Test index-based naming when splitting closest_cell dimension."""
        processor = Export(
            {
                "filename": "station",
                "separated": True,
                "location_based_naming": False,
            }
        )

        filenames_used = []

        def track_filename(data):
            filenames_used.append(processor.filename)

        with patch.object(processor, "export_single", side_effect=track_filename):
            processor._split_and_export_closest_cells(self.multi_point_ds)

        # Should use index-based naming
        assert filenames_used == ["station_0", "station_1", "station_2"]

    def test_split_closest_cells_location_naming_no_target_coords(self):
        """Test location naming falls back to index when target coords missing."""
        processor = Export(
            {
                "filename": "station",
                "separated": True,
                "location_based_naming": True,  # Should fallback to index
            }
        )

        filenames_used = []

        def track_filename(data):
            filenames_used.append(processor.filename)

        with patch.object(processor, "export_single", side_effect=track_filename):
            processor._split_and_export_closest_cells(self.multi_point_ds_no_target)

        # Should fall back to index naming since no target_lats/lons
        assert filenames_used == ["station_0", "station_1"]

    def test_split_closest_cells_restores_filename(self):
        """Test that original filename is restored after split export."""
        processor = Export(
            {
                "filename": "original_name",
                "separated": True,
            }
        )

        with patch.object(processor, "export_single"):
            processor._split_and_export_closest_cells(self.multi_point_ds)

        # Original filename should be restored
        assert processor.filename == "original_name"


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


class TestExportPointsDimension:
    """Test class for export processor handling of 'points' dimension."""

    def setup_method(self):
        """Set up test fixtures with points dimension."""
        # Create dataset with 'points' dimension (from clip processor)
        self.points_ds = xr.Dataset(
            {"temp": (["points", "time"], np.random.rand(2, 3))},
            coords={
                "time": pd.date_range("2020-01-01", periods=3),
                "point_lat": ("points", [34.05, 37.77]),
                "point_lon": ("points", [-118.25, -122.42]),
                "point_index": ("points", [0, 1]),
            },
        )

    def test_has_closest_cell_dimension_with_points(self):
        """Test _has_closest_cell_dimension returns True for 'points' dimension."""
        processor = Export({})
        assert processor._has_closest_cell_dimension(self.points_ds) is True

    def test_has_closest_cell_dimension_with_closest_cell(self):
        """Test _has_closest_cell_dimension returns True for 'closest_cell' dimension."""
        # Dataset with closest_cell dimension (from old clip code path)
        closest_cell_ds = xr.Dataset(
            {"temp": (["closest_cell", "time"], np.random.rand(2, 3))},
            coords={
                "time": pd.date_range("2020-01-01", periods=3),
                "target_lats": ("closest_cell", [34.05, 37.77]),
                "target_lons": ("closest_cell", [-118.25, -122.42]),
            },
        )
        processor = Export({})
        assert processor._has_closest_cell_dimension(closest_cell_ds) is True

    def test_has_closest_cell_dimension_with_neither(self):
        """Test _has_closest_cell_dimension returns False when neither dimension exists."""
        regular_ds = xr.Dataset(
            {"temp": (["time", "lat", "lon"], np.random.rand(3, 2, 2))},
            coords={
                "time": pd.date_range("2020-01-01", periods=3),
                "lat": [34.0, 35.0],
                "lon": [-118.0, -117.0],
            },
        )
        processor = Export({})
        assert processor._has_closest_cell_dimension(regular_ds) is False

    @patch("climakitae.new_core.processors.export.Export.export_single")
    def test_split_and_export_closest_cells_with_points_dimension(self, mock_export):
        """Test _split_and_export_closest_cells handles 'points' dimension correctly."""
        processor = Export(
            {"filename": "output", "separated": True, "location_based_naming": True}
        )

        filenames_used = []

        def track_filename(data):
            filenames_used.append(processor.filename)

        mock_export.side_effect = track_filename

        processor._split_and_export_closest_cells(self.points_ds)

        # Should call export_single twice (once per point)
        assert mock_export.call_count == 2

        # Verify filenames contain geographic coordinates
        assert len(filenames_used) == 2
        assert any(
            "34" in filename and "118" in filename for filename in filenames_used
        )
        assert any(
            "37" in filename and "122" in filename for filename in filenames_used
        )

    @patch("climakitae.new_core.processors.export.Export.export_single")
    def test_split_and_export_uses_point_lat_lon_coords(self, mock_export):
        """Test that export uses point_lat/point_lon coordinates for filenames."""
        processor = Export(
            {
                "filename": "point_export",
                "separated": True,
                "location_based_naming": True,
            }
        )

        filenames_used = []

        def track_filename(data):
            filenames_used.append(processor.filename)

        mock_export.side_effect = track_filename

        processor._split_and_export_closest_cells(self.points_ds)

        # Both filenames should be present
        assert len(filenames_used) == 2

        # Check that filenames contain lat/lon coordinates (not grid coordinates)
        # Filenames should look like: point_export_34041N_118239W
        for filename in filenames_used:
            # Should contain N/S and W/E indicators (geographic format)
            assert (
                "N" in filename or "S" in filename
            ), f"Filename {filename} missing N/S indicator"
            assert (
                "W" in filename or "E" in filename
            ), f"Filename {filename} missing W/E indicator"

            # Should NOT contain million-scale numbers (grid coordinates)
            # Extract numbers from filename
            import re

            numbers = re.findall(r"\d+", filename)
            for num in numbers:
                assert (
                    int(num) < 100000
                ), f"Filename {filename} contains grid coordinate {num}"


class TestClipExportIntegration:
    """Integration tests for full clip→export workflow."""

    def test_wrf_points_clip_to_export_geographic_coordinates(self):
        """Test full WRF clip→export flow preserves geographic coordinates in filenames."""
        from climakitae.new_core.processors.clip import Clip

        # Create WRF-style dataset with Lambert Conformal projection
        y_vals = np.array([4176113.66, 4179113.66, 4182113.66])
        x_vals = np.array([1393911.73, 1396911.73, 1399911.73])
        lat_vals = np.array([34.05, 34.08, 34.11])
        lon_vals = np.array([-118.25, -118.22, -118.19])

        wrf_dataset = xr.Dataset(
            {
                "t2max": (["time", "y", "x"], np.random.rand(2, 3, 3)),
                "lat": (["y", "x"], np.broadcast_to(lat_vals[:, None], (3, 3))),
                "lon": (["y", "x"], np.broadcast_to(lon_vals, (3, 3))),
            },
            coords={
                "time": pd.date_range("2020-01-01", periods=2),
                "y": y_vals,
                "x": x_vals,
            },
        )

        # User-provided points (geographic coordinates)
        user_points = [(34.05, -118.25)]

        # Step 1: Clip to points with separated=True
        clip_processor = Clip({"points": user_points, "separated": True})
        context = {}
        clipped_data = clip_processor.execute(wrf_dataset, context)

        # Verify clip produced points dimension with geographic coords
        assert "points" in clipped_data.dims
        assert "point_lat" in clipped_data.coords
        assert "point_lon" in clipped_data.coords

        point_lat = float(clipped_data["point_lat"].values[0])
        point_lon = float(clipped_data["point_lon"].values[0])

        # Coordinates should be geographic
        assert 30 < point_lat < 40
        assert -125 < point_lon < -110

        # Step 2: Export with location-based naming
        export_processor = Export(
            {
                "filename": "wrf_point_export",
                "separated": True,
                "location_based_naming": True,
            }
        )

        filenames_used = []

        def track_filename(data):
            filenames_used.append(export_processor.filename)

        with patch.object(
            export_processor, "export_single", side_effect=track_filename
        ):
            export_processor._split_and_export_closest_cells(clipped_data)

        # Verify filename contains geographic coordinates
        assert len(filenames_used) == 1
        filename = filenames_used[0]

        # Should contain N/S and W/E indicators
        assert "N" in filename, f"Filename {filename} missing N indicator"
        assert "W" in filename, f"Filename {filename} missing W indicator"

        # Should contain coordinate values close to user input
        assert "34" in filename, f"Filename {filename} missing latitude"
        assert "118" in filename, f"Filename {filename} missing longitude"

        # Should NOT contain grid coordinates (million-scale numbers)
        import re

        numbers = re.findall(r"\d+", filename)
        for num in numbers:
            assert (
                int(num) < 100000
            ), f"Filename {filename} contains grid coordinate value {num}"

        # Expected format: wrf_point_export_34041N_118239W or similar
        assert filename.startswith(
            "wrf_point_export_"
        ), f"Filename {filename} doesn't have expected prefix"
