"""
DataProcessor Export
"""

import os
from typing import Any, Dict, Iterable, Union

import xarray as xr

from climakitae.core.constants import _NEW_ATTRS_KEY
from climakitae.core.data_export import (
    _export_to_csv,
    _export_to_netcdf,
    _export_to_zarr,
)
from climakitae.new_core.data_access.data_access import DataCatalog
from climakitae.new_core.processors.abc_data_processor import (
    DataProcessor,
    register_processor,
)


# last possible step
@register_processor("export", priority=9999)
class Export(DataProcessor):
    """
    Export data to various file formats.

    This processor exports xarray datasets and data arrays to NetCDF, Zarr, or CSV formats.
    It supports both local file export and AWS S3 export for Zarr files.

    Parameters
    ----------        value : dict[str, Any]
            Configuration dictionary with the following supported keys:

        - filename (str, optional): Output filename without extension. Default: "dataexport"
        - file_format (str, optional): File format to export to. Supported values:
          "NetCDF", "Zarr", "CSV". Default: "NetCDF"
        - mode (str, optional): Save location for Zarr files. Supported values:
          "local" (save to local filesystem), "s3" (save to AWS S3). Default: "local"
        - separated (bool, optional): Whether to create separate files when exporting
          multiple datasets. If True, each dataset will use its name as part of the filename.
          Default: False
        - export_method (str, optional): What type of data to export. Supported values:
          "data" (export all data), "raw" (export raw data only), "calculate" (export calculated data only),
          "both" (export both raw and calculated data), "skip_existing" (skip if file exists), "None" (no export).
          Default: "data"
        - raw_filename (str, optional): Filename for raw data when using "raw" or "both" export methods.
          If not provided, defaults to "{filename}_raw". Default: None
        - calc_filename (str, optional): Filename for calculated data when using "calculate" or "both" export methods.
          If not provided, defaults to "{filename}_calc". Default: None
        - location_based_naming (bool, optional): Whether to include lat/lon coordinates
          in filenames for spatial data. Default: False
        - filename_template (str, optional): Template for generating filenames. Can include
          placeholders like {filename}, {lat}, {lon}, {name}. Default: None
        - fail_on_error (bool, optional): Whether to raise exceptions on export errors.
          If False, errors are logged but execution continues. Default: True

    Examples
    --------
    Export to NetCDF (default):
    >>> export_proc = Export({"filename": "my_data"})

    Export to Zarr on S3:
    >>> export_proc = Export({
    ...     "filename": "climate_data",
    ...     "file_format": "Zarr",
    ...     "mode": "s3"
    ... })

    Export to CSV with location-based naming:
    >>> export_proc = Export({
    ...     "filename": "temperature",
    ...     "file_format": "CSV",
    ...     "separated": True,
    ...     "location_based_naming": True
    ... })

    Export with custom filename template:
    >>> export_proc = Export({
    ...     "filename_template": "{name}_data_{lat}N_{lon}W",
    ...     "file_format": "NetCDF"
    ... })

    Export with data type separation:
    >>> export_proc = Export({
    ...     "filename": "climate_data",
    ...     "export_method": "both",
    ...     "raw_filename": "raw_climate_data",
    ...     "calc_filename": "processed_climate_data"
    ... })

    Export only calculated data:
    >>> export_proc = Export({
    ...     "filename": "temperature",
    ...     "export_method": "calculate"
    ... })

    Export only raw data:
    >>> export_proc = Export({
    ...     "filename": "temperature",
    ...     "export_method": "raw"
    ... })

    Notes
    -----
    - S3 export is only available for Zarr format
    - Large files may trigger warnings about disk space and file size
    - The processor adds metadata to exported datasets including timestamps and package information
    - When location_based_naming is True, coordinates are formatted as {lat}N_{lon}W
    - Custom filename templates support placeholders: {filename}, {lat}, {lon}, {name}
    - export_method="skip_existing" allows graceful handling of existing files
    - For "raw", "calculate", and "both" export methods, the processor looks for context indicators
      to distinguish data types. Raw data is exported with "_raw" suffix, calculated with "_calc" suffix
    - When using "both", two separate files are created for raw and calculated data
    """

    def __init__(self, value: Dict[str, Any]):
        """
        Initialize the processor.

        Parameters
        ----------
        value : dict[str, Any]
            Configuration values for the export operation. Expected keys:
            - filename (str, optional): Output filename without extension. Default: "dataexport"
            - file_format (str, optional): File format ("NetCDF", "Zarr", "CSV"). Default: "NetCDF"
            - mode (str, optional): Save location for Zarr files ("local", "s3"). Default: "local"
            - separated (bool, optional): Whether to create separate files when exporting multiple datasets. Default: False
            - export_method (str, optional): What type of data to export. Default: "data"
            - raw_filename (str, optional): Filename for raw data exports. Default: None
            - calc_filename (str, optional): Filename for calculated data exports. Default: None

        Raises
        ------
        ValueError
            If invalid file_format or mode values are provided
        """
        self.value = value
        self.name = "_export"
        self.filename = value.get("filename", "dataexport")
        self.file_format = value.get("file_format", "NetCDF")
        self.mode = value.get("mode", "local")
        self.separated = value.get("separated", False)
        self.export_method = value.get("export_method", "data")
        self.location_based_naming = value.get("location_based_naming", False)
        self.filename_template = value.get("filename_template", None)
        self.fail_on_error = value.get("fail_on_error", True)
        self.raw_filename = value.get("raw_filename", None)
        self.calc_filename = value.get("calc_filename", None)

        # Validate inputs during initialization
        self._validate_parameters()

    def _validate_parameters(self):
        """
        Validate the export parameters.

        Raises
        ------
        ValueError
            If invalid parameter values are provided
        """
        # Validate file format
        valid_formats = ["zarr", "netcdf", "csv"]
        if self.file_format.lower() not in valid_formats:
            raise ValueError(
                f'file_format must be one of {valid_formats}, got "{self.file_format}"'
            )

        # Validate mode
        valid_modes = ["local", "s3"]
        if self.mode.lower() not in valid_modes:
            raise ValueError(f'mode must be one of {valid_modes}, got "{self.mode}"')

        # Validate S3 + format combination
        if (self.mode.lower() == "s3") and (self.file_format.lower() != "zarr"):
            raise ValueError('To export to AWS S3 you must use file_format="Zarr"')

        # Validate filename type
        if not isinstance(self.filename, str):
            raise ValueError(f"filename must be a string, got {type(self.filename)}")

        # Validate separated type
        if not isinstance(self.separated, bool):
            raise ValueError(f"separated must be a boolean, got {type(self.separated)}")

        # Validate export_method
        valid_export_methods = [
            "data",
            "raw",
            "calculate",
            "both",
            "skip_existing",
            "none",
        ]
        if self.export_method.lower() not in valid_export_methods:
            raise ValueError(
                f'export_method must be one of {valid_export_methods}, got "{self.export_method}"'
            )

        # Validate location_based_naming type
        if not isinstance(self.location_based_naming, bool):
            raise ValueError(
                f"location_based_naming must be a boolean, got {type(self.location_based_naming)}"
            )

        # Validate filename_template type
        if self.filename_template is not None and not isinstance(
            self.filename_template, str
        ):
            raise ValueError(
                f"filename_template must be a string or None, got {type(self.filename_template)}"
            )

        # Validate fail_on_error type
        if not isinstance(self.fail_on_error, bool):
            raise ValueError(
                f"fail_on_error must be a boolean, got {type(self.fail_on_error)}"
            )

        # Validate filename types
        if self.raw_filename is not None and not isinstance(self.raw_filename, str):
            raise ValueError(
                f"raw_filename must be a string or None, got {type(self.raw_filename)}"
            )

        if self.calc_filename is not None and not isinstance(self.calc_filename, str):
            raise ValueError(
                f"calc_filename must be a string or None, got {type(self.calc_filename)}"
            )

    def execute(
        self,
        result: Union[
            xr.Dataset, xr.DataArray, Iterable[Union[xr.Dataset, xr.DataArray]]
        ],
        context: Dict[str, Any],
    ) -> Union[xr.Dataset, xr.DataArray, Iterable[Union[xr.Dataset, xr.DataArray]]]:
        """
        Run the processor

        Parameters
        ----------
        result : xr.Dataset | xr.DataArray | Iterable[xr.Dataset | xr.DataArray]
            The data to be exported.

        context : dict
            The context for the processor. This is not used in this
            implementation but is included for consistency with the
            DataProcessor interface.

        Returns
        -------
        Union[xr.Dataset, xr.DataArray, Iterable[xr.Dataset | xr.DataArray]]
            The data written to file.
        """
        # Skip export if method is "none"
        if self.export_method.lower() == "none":
            print("Export method set to 'none'; no data is exported!")
            return result

        # Handle different export methods
        export_method_lower = self.export_method.lower()

        if export_method_lower in ["raw", "calculate", "both"]:
            # For these methods, we need to handle data based on context or data structure
            # Note: We check if result is actually a dict (for cases like cava_data output)
            # even though type annotation doesn't include Dict to maintain base class compatibility
            if isinstance(result, dict):
                self._handle_dict_result(result, export_method_lower)
            else:
                self._handle_selective_export(result, context, export_method_lower)
        else:
            # Standard export for "data" and "skip_existing" methods
            match result:
                case xr.Dataset() | xr.DataArray():
                    self.export_single(result)
                case dict():
                    for _, value in result.items():
                        self.export_single(value)
                case list() | tuple():
                    for item in result:
                        if isinstance(item, (xr.Dataset, xr.DataArray)):
                            self.export_single(item)
                        else:
                            raise TypeError(
                                f"Expected xr.Dataset or xr.DataArray, got {type(item)}"
                            )
                case _:
                    raise TypeError(
                        f"Expected xr.Dataset, xr.DataArray, dict, list, or tuple, got {type(result)}"
                    )

        return result

    def _handle_dict_result(self, result: dict, export_method: str):
        """
        Handle dictionary results (like those from cava_data function).
        """
        raw_data = result.get("raw_data")
        calc_data = result.get("calc_data")

        if export_method == "raw" and raw_data is not None:
            self._export_with_suffix(raw_data, "raw")
        elif export_method == "calculate" and calc_data is not None:
            self._export_with_suffix(calc_data, "calc")
        elif export_method == "both":
            if raw_data is not None:
                self._export_with_suffix(raw_data, "raw")
            if calc_data is not None:
                self._export_with_suffix(calc_data, "calc")

    def _handle_selective_export(
        self,
        result: Union[
            xr.Dataset, xr.DataArray, Iterable[Union[xr.Dataset, xr.DataArray]]
        ],
        context: Dict[str, Any],
        export_method: str,
    ):
        """
        Handle selective export methods (raw, calculate, both).

        This method looks for indicators in the data or context to determine
        whether data should be treated as raw or calculated.
        """
        # For other data structures, check context for data type indicators
        data_type = self._determine_data_type(result, context)

        if export_method == "raw" and data_type == "raw":
            self._export_with_suffix(result, "raw")
        elif export_method == "calculate" and data_type == "calc":
            self._export_with_suffix(result, "calc")
        elif export_method == "both":
            # If we can't determine type, export as both with different suffixes
            self._export_with_suffix(result, "raw")
            self._export_with_suffix(result, "calc")
        else:
            # Default behavior - export with appropriate suffix
            suffix = "raw" if export_method == "raw" else "calc"
            self._export_with_suffix(result, suffix)

    def _determine_data_type(self, result, context: Dict[str, Any]) -> str:
        """
        Determine if data should be treated as raw or calculated based on context clues.

        Returns "raw" or "calc" based on available information.
        """
        # Check context for processing history
        if _NEW_ATTRS_KEY in context:
            processing_steps = context[_NEW_ATTRS_KEY]
            # If there are processing steps beyond data loading, treat as calculated
            if any(
                step for step in processing_steps.keys() if not step.startswith("_load")
            ):
                return "calc"

        # Check data attributes for processing indicators
        if hasattr(result, "attrs"):
            attrs = result.attrs if hasattr(result, "attrs") else {}
            # Look for indicators of processed data
            if any(
                key in attrs
                for key in ["processed_by", "calculation_method", "derived_from"]
            ):
                return "calc"

        # Default to raw if no processing indicators found
        return "raw"

    def _export_with_suffix(
        self,
        data: Union[
            xr.Dataset, xr.DataArray, Iterable[Union[xr.Dataset, xr.DataArray]]
        ],
        suffix: str,
    ):
        """
        Export data with appropriate filename suffix.
        """
        # Temporarily modify filename for this export
        original_filename = self.filename

        # Use custom filename if provided, otherwise add suffix
        if suffix == "raw" and self.raw_filename:
            self.filename = self.raw_filename
        elif suffix == "calc" and self.calc_filename:
            self.filename = self.calc_filename
        else:
            self.filename = f"{original_filename}_{suffix}"

        try:
            # Export the data using standard export logic
            match data:
                case xr.Dataset() | xr.DataArray():
                    self.export_single(data)
                case dict():
                    for _, value in data.items():
                        self.export_single(value)
                case list() | tuple():
                    for item in data:
                        if isinstance(item, (xr.Dataset, xr.DataArray)):
                            self.export_single(item)
                        else:
                            raise TypeError(
                                f"Expected xr.Dataset or xr.DataArray, got {type(item)}"
                            )
                case _:
                    raise TypeError(
                        f"Expected xr.Dataset, xr.DataArray, dict, list, or tuple, got {type(data)}"
                    )
        finally:
            # Restore original filename
            self.filename = original_filename

    def update_context(self, context: Dict[str, Any]):
        """
        Update the context with information about the transformation.

        Parameters
        ----------
        context : dict[str, Any]
            Parameters for processing the data.

        Note
        ----
        The context is updated in place. This method does not return anything.
        """

        if _NEW_ATTRS_KEY not in context:
            context[_NEW_ATTRS_KEY] = {}

        value_str = str(self.value)
        context[_NEW_ATTRS_KEY][
            self.name
        ] = f"""Process '{self.name}' applied to the data. Transformation was done using the following value: {value_str}."""

    def set_data_accessor(self, catalog: DataCatalog):
        # Placeholder for setting data accessor
        pass

    def _generate_filename(self, data: Union[xr.Dataset, xr.DataArray]) -> str:
        """
        Generate a filename for the data based on configuration options.

        Parameters
        ----------
        data : xr.Dataset | xr.DataArray
            The data to generate a filename for.

        Returns
        -------
        str
            The generated filename without extension.
        """
        # Start with base filename
        base_filename = self.filename

        # Apply custom template if provided
        if self.filename_template:
            template_vars = {
                "filename": self.filename,
                "name": getattr(data, "name", "data") or "data",
                "lat": "",
                "lon": "",
            }

            # Extract lat/lon if available
            if hasattr(data, "lat") and hasattr(data, "lon"):
                try:
                    lat_val = float(
                        data.lat.item()
                        if hasattr(data.lat, "item")
                        else data.lat.values.flat[0]
                    )
                    lon_val = float(
                        data.lon.item()
                        if hasattr(data.lon, "item")
                        else data.lon.values.flat[0]
                    )
                    template_vars["lat"] = str(round(lat_val, 3)).replace(".", "")
                    template_vars["lon"] = str(round(abs(lon_val), 3)).replace(".", "")
                except (AttributeError, ValueError, IndexError):
                    pass

            base_filename = self.filename_template.format(**template_vars)

        # Apply separated naming (dataset name prefix)
        elif self.separated and hasattr(data, "name") and data.name:
            base_filename = f"{data.name}_{base_filename}"

        # Apply location-based naming if requested
        if self.location_based_naming and hasattr(data, "lat") and hasattr(data, "lon"):
            try:
                lat_val = float(
                    data.lat.item()
                    if hasattr(data.lat, "item")
                    else data.lat.values.flat[0]
                )
                lon_val = float(
                    data.lon.item()
                    if hasattr(data.lon, "item")
                    else data.lon.values.flat[0]
                )
                lat_str = str(round(lat_val, 3)).replace(".", "")
                lon_str = str(round(abs(lon_val), 3)).replace(".", "")
                base_filename = f"{base_filename}_{lat_str}N_{lon_str}W"
            except (AttributeError, ValueError, IndexError):
                # If lat/lon extraction fails, continue without location suffix
                pass

        # Remove any existing extension from filename
        base_filename = base_filename.split(".")[0]

        return base_filename

    def export_single(self, data: Union[xr.Dataset, xr.DataArray]):
        """
        Export a single xr.Dataset or xr.DataArray to file.

        Parameters
        ----------
        data : xr.Dataset | xr.DataArray
            The data to be exported.

        Raises
        ------
        RuntimeError
            If the export operation fails and fail_on_error is True
        """
        # Get normalized format for processing
        req_format = self.file_format.lower()

        # Generate filename using smart naming logic
        base_filename = self._generate_filename(data)

        # Create full filename with appropriate extension
        extension_dict = {"zarr": ".zarr", "netcdf": ".nc", "csv": ".csv.gz"}
        save_name = base_filename + extension_dict[req_format]

        # Check if file exists and handle based on export_method
        if self.export_method.lower() == "skip_existing" and os.path.exists(save_name):
            print(f"File {save_name} already exists, skipping export.")
            return

        # Export using the appropriate function from data_export.py
        for k, v in data.attrs.items():
            if isinstance(v, dict):
                # Convert dictionary attributes to string to avoid serialization issues
                data.attrs[k] = str(v)
        try:
            match req_format:
                case "zarr":
                    _export_to_zarr(data, save_name, self.mode)
                case "netcdf":
                    _export_to_netcdf(data, save_name)
                case "csv":
                    _export_to_csv(data, save_name)
                case _:
                    # This should never happen due to validation in __init__
                    raise ValueError(f"Unsupported file format: {self.file_format}")
        except (OSError, ValueError, RuntimeError) as e:
            error_msg = f"Export failed for {save_name}: {str(e)}"
            if self.fail_on_error:
                raise RuntimeError(error_msg) from e
            else:
                print(f"Warning: {error_msg}")
                return

    @classmethod
    def export_no_error(
        cls,
        data: Union[xr.Dataset, xr.DataArray],
        filename: str = "dataexport",
        file_format: str = "NetCDF",
        **kwargs,
    ) -> None:
        """
        Export data without raising exceptions if files already exist.

        This is a convenience method that mimics the behavior of _export_no_e
        from the cava_data function for backward compatibility.

        Parameters
        ----------
        data : xr.Dataset | xr.DataArray
            The data to export.
        filename : str, optional
            Output filename without extension. Default: "dataexport"
        file_format : str, optional
            File format ("NetCDF", "Zarr", "CSV"). Default: "NetCDF"
        **kwargs
            Additional parameters passed to the Export processor.
        """
        export_config = {
            "filename": filename,
            "file_format": file_format,
            "export_method": "skip_existing",
            "fail_on_error": False,
            **kwargs,
        }

        exporter = cls(export_config)
        exporter.export_single(data)

    @classmethod
    def export_raw_calc_data(
        cls,
        raw_data: Union[xr.Dataset, xr.DataArray, None] = None,
        calc_data: Union[xr.Dataset, xr.DataArray, None] = None,
        filename: str = "dataexport",
        file_format: str = "NetCDF",
        export_method: str = "both",
        **kwargs,
    ) -> None:
        """
        Export raw and/or calculated data similar to cava_data behavior.

        Parameters
        ----------
        raw_data : xr.Dataset | xr.DataArray, optional
            The raw data to export. Only exported if export_method includes "raw".
        calc_data : xr.Dataset | xr.DataArray, optional
            The calculated data to export. Only exported if export_method includes "calculate".
        filename : str, optional
            Base filename without extension. Default: "dataexport"
        file_format : str, optional
            File format ("NetCDF", "Zarr", "CSV"). Default: "NetCDF"
        export_method : str, optional
            What to export: "raw", "calculate", or "both". Default: "both"
        **kwargs
            Additional parameters passed to the Export processor.
        """
        export_config = {
            "filename": filename,
            "file_format": file_format,
            "export_method": export_method,
            "fail_on_error": False,
            **kwargs,
        }

        exporter = cls(export_config)

        # Create a dict similar to cava_data output
        data_dict = {}
        if raw_data is not None:
            data_dict["raw_data"] = raw_data
        if calc_data is not None:
            data_dict["calc_data"] = calc_data

        if data_dict:
            exporter._handle_dict_result(data_dict, export_method)
