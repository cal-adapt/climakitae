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
          "data" (export all data), "skip_existing" (skip if file exists), "None" (no export).
          Default: "data"
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

    Skip export if files already exist:
    >>> export_proc = Export({
    ...     "filename": "climate_data",
    ...     "export_method": "skip_existing"
    ... })

    Notes
    -----
    - S3 export is only available for Zarr format
    - Large files may trigger warnings about disk space and file size
    - The processor adds metadata to exported datasets including timestamps and package information
    - When location_based_naming is True, coordinates are formatted as {lat}N_{lon}W
    - Custom filename templates support placeholders: {filename}, {lat}, {lon}, {name}
    - export_method="skip_existing" allows graceful handling of existing files
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

        Raises
        ------
        ValueError
            If invalid file_format or mode values are provided
        """
        self.value = value
        self.name = "export"
        self.filename = value.get("filename", "dataexport")
        self.file_format = value.get("file_format", "NetCDF")
        self.mode = value.get("mode", "local")
        self.separated = value.get("separated", False)
        self.export_method = value.get("export_method", "data")
        self.location_based_naming = value.get("location_based_naming", False)
        self.filename_template = value.get("filename_template", None)
        self.fail_on_error = value.get("fail_on_error", True)

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
        valid_export_methods = ["data", "skip_existing", "none"]
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

        context[_NEW_ATTRS_KEY][
            self.name
        ] = f"""Process '{self.name}' applied to the data. Transformation was done using the following value: {self.value}."""

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
        exporter.export_single(data)
