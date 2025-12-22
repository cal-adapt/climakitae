"""
DataProcessor Export
"""

import logging
import os
from typing import Any, Dict, Iterable, Union

import xarray as xr

from climakitae.core.constants import _NEW_ATTRS_KEY
from climakitae.core.data_export import (_export_to_csv, _export_to_netcdf,
                                         _export_to_zarr)
from climakitae.new_core.data_access.data_access import DataCatalog
from climakitae.new_core.param_validation.export_param_validator import \
    _infer_file_format
from climakitae.new_core.processors.abc_data_processor import (
    DataProcessor, register_processor)

# Module logger
logger = logging.getLogger(__name__)


# last possible step
@register_processor("export", priority=9999)
class Export(DataProcessor):
    """
    Export climate data to various file formats.

    This processor exports xarray Datasets and DataArrays to NetCDF, Zarr, or CSV formats.
    It supports both local filesystem export and AWS S3 export (Zarr only).

    The processor handles three distinct data structures:

    1. Gridded datasets: A single xr.Dataset/xr.DataArray with `lat` and `lon`
       as coordinate dimensions containing multiple values (e.g., shape
       ``(time, lat, lon)``). Options like ``separated`` and ``location_based_naming``
       are silently ignored since they don't apply to gridded data.

    2. Multi-point clip results: A single xr.Dataset/xr.DataArray with a
       ``closest_cell`` dimension (output from clipping to multiple lat/lon points).
       When ``separated=True``, the data is split along the ``closest_cell``
       dimension and each slice is exported to its own file:

       - ``separated=True`` + ``location_based_naming=True``: Filenames include
         the target lat/lon coordinates (e.g., ``myfile_34-05N_118-25W.nc``)
       - ``separated=True`` + ``location_based_naming=False``: Filenames include
         index numbers (e.g., ``myfile_0.nc``, ``myfile_1.nc``)
       - ``separated=False``: Single file with all points along ``closest_cell``

    3. Point-based data collections: A **list** of xr.Dataset/xr.DataArray objects,
       where each item represents a single spatial point (scalar `lat`/`lon`
       coordinates with size 1). This is the output format from ``cava_data``
       when ``separate_files=True``. For collections:

       - ``separated=True``: Each point is exported to its own file
       - ``separated=True`` + ``location_based_naming=True``: Filenames include lat/lon
         coordinates (e.g., ``myfile_34-0N_118-0W.nc``)
       - ``separated=True`` + ``location_based_naming=False``: Filenames include index
         numbers (e.g., ``myfile_0.nc``, ``myfile_1.nc``)
       - ``separated=False``: Each item exported normally (unique filename if conflicts)

    Note: When ``cava_data`` uses ``separate_files=False`` or ``batch_mode=True``,
    it concatenates points into a single dataset along a ``simulation`` dimension.
    In this case, the data is treated as a single dataset (like case 1 above).

    Parameters
    ----------
    value : dict[str, Any]
        Configuration dictionary with the following supported keys:

        filename (str, optional): Base output filename without extension.
            Default: "dataexport"
        file_format (str, optional): Output file format. Supported values:
            "NetCDF", "Zarr", "CSV". Case-insensitive. Default: "NetCDF"
        mode (str, optional): Storage location for Zarr files.
            "local" saves to local filesystem, "s3" saves to AWS S3.
            Default: "local"
        separated (bool, optional): When exporting a collection of point datasets,
            whether to create separate files for each point. If True, each dataset
            gets its own file with either lat/lon or index suffix. If False, all
            items are exported with the base filename (unique suffixes added if needed).
            Ignored for single gridded datasets. Default: False
        location_based_naming (bool, optional): When separated=True and
            exporting point-based data, include lat/lon coordinates in filenames
            (e.g., filename_34-0N_118-0W.nc). If False, uses index numbers
            instead (e.g., filename_0.nc). Silently ignored for gridded datasets.
            Default: False
        export_method (str, optional): Controls what data to export. Options:
            "data": Export all provided data (default)
            "raw": Export only raw/unprocessed data
            "calculate": Export only calculated/processed data
            "both": Export both raw and calculated data to separate files
            "skip_existing": Skip export if file already exists
            "none": Skip export entirely
            Default: "data"
        raw_filename (str, optional): Custom filename for raw data when using
            export_method="raw" or "both". If not provided, uses
            "{filename}_raw". Default: None
        calc_filename (str, optional): Custom filename for calculated data when
            using export_method="calculate" or "both". If not provided, uses
            "{filename}_calc".
            Default: None
        filename_template (str, optional): Custom template for generating filenames.
            Supports placeholders: {filename}, {lat}, {lon}, {name}.
            Lat/lon placeholders only populated for single-point data.
            Example: "{name}_data_{lat}N_{lon}W".
            Default: None
        fail_on_error (bool, optional): If True, raise exceptions on export
            errors. If False, log warnings and continue.
            Default: True

    Examples
    --------
    Basic export to NetCDF:

    >>> export_proc = Export({"filename": "climate_output"})

    Export gridded data (separated/location_based_naming ignored):

    >>> # These options are silently ignored for gridded data
    >>> export_proc = Export({
    ...     "filename": "gridded_data",
    ...     "separated": True,  # ignored
    ...     "location_based_naming": True,  # ignored
    ... })

    Export point collection with lat/lon in filenames:

    >>> # For a list of single-point datasets (e.g., from cava_data)
    >>> export_proc = Export({
    ...     "filename": "station_data",
    ...     "separated": True,
    ...     "location_based_naming": True,
    ... })
    >>> # Results in: station_data_34-0N_118-0W.nc, station_data_35-5N_119-5W.nc, ...

    Export point collection with index-based filenames:

    >>> export_proc = Export({
    ...     "filename": "station_data",
    ...     "separated": True,
    ...     "location_based_naming": False,
    ... })
    >>> # Results in: station_data_0.nc, station_data_1.nc, ...

    Export to Zarr on S3:

    >>> export_proc = Export({
    ...     "filename": "climate_data",
    ...     "file_format": "Zarr",
    ...     "mode": "s3",
    ... })

    Export raw and calculated data separately (e.g., for CAVA workflow):

    >>> export_proc = Export({
    ...     "filename": "cava_output",
    ...     "export_method": "both",
    ...     "raw_filename": "raw_observations",
    ...     "calc_filename": "processed_metrics",
    ... })

    Custom filename template:

    >>> export_proc = Export({
    ...     "filename_template": "{name}_analysis_{lat}N_{lon}W",
    ...     "file_format": "NetCDF",
    ... })

    Notes
    -----
    - S3 export requires file_format="Zarr"; other formats only support local export
    - File extensions are automatically added based on format: .nc (NetCDF),
      .zarr (Zarr), .csv.gz (CSV, gzip compressed)
    - Duplicate filenames: If a file already exists, a unique suffix _1, _2,
      etc. is appended (unless export_method="skip_existing")
    - Lat/lon format in filenames uses dashes instead of dots to avoid extension
      confusion: 34.5 becomes 34-5
    - Gridded data: When exporting a single dataset with `lat`/`lon` dimensions
      containing multiple values, the ``separated`` and ``location_based_naming``
      options are silently ignored
    - Point collections: When exporting a **list** of single-point datasets (e.g.,
      from ``cava_data`` with ``separate_files=True``), use ``separated=True`` to
      create individual files per location
    - Concatenated points: When ``cava_data`` uses ``separate_files=False``, points
      are concatenated along a ``simulation`` dimension into a single dataset;
      this is treated as gridded data for export purposes
    """

    def __init__(self, value: Dict[str, Any]):
        """
        Initialize the Export processor.

        Parameters
        ----------
        value : dict[str, Any]
            Configuration dictionary. See class docstring for full parameter details.

            Common keys:

            - filename (str): Base output filename. Default: "dataexport"
            - file_format (str): "NetCDF", "Zarr", or "CSV". Default: "NetCDF"
            - mode (str): "local" or "s3" (Zarr only). Default: "local"
            - separated (bool): Export collection items to separate files. Default: False
            - location_based_naming (bool): Use lat/lon in filenames. Default: False
            - export_method (str): "data", "raw", "calculate", "both",
              "skip_existing", or "none". Default: "data"

        Raises
        ------
        ValueError
            If invalid parameter values are provided (e.g., unknown file_format,
            S3 mode with non-Zarr format).
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
        # Validate and potentially auto-correct file format
        valid_formats = ["zarr", "netcdf", "csv"]
        if self.file_format.lower() not in valid_formats:
            # Try to infer the intended format
            inferred = _infer_file_format(self.file_format)
            if inferred:
                logger.info(
                    "Interpreted file_format '%s' as '%s'.",
                    self.file_format,
                    inferred.upper(),
                )
                self.file_format = inferred.capitalize()
            else:
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
        Execute the export processor on the provided data.

        This method is the main entry point for exporting climate data. It supports
        multiple data structures and export modes, routing each case to the
        appropriate internal handler.

        Parameters
        ----------
        result : xr.Dataset | xr.DataArray | Iterable[xr.Dataset | xr.DataArray]
            The data to be exported. Can be:

            - A single xr.Dataset or xr.DataArray (gridded data)
            - A list or tuple of xr.Dataset/xr.DataArray (e.g., from cava_data)
            - A dict with "raw_data" and/or "calc_data" keys

        context : dict
            The processing context. Used for determining data type when
            export_method is "raw", "calculate", or "both".

        Returns
        -------
        xr.Dataset | xr.DataArray | Iterable[xr.Dataset | xr.DataArray]
            The same data passed in (unchanged). This allows chaining processors.

        Notes
        -----
        The export_method parameter controls what gets exported:

        - "data" / "skip_existing": Standard export via `_export_data`
        - "raw" / "calculate" / "both": Selective export based on
          data type (uses `_handle_dict_result` or `_handle_selective_export`)
        - "none": No export; prints a message and returns data unchanged

        See Also
        --------
        _export_data : Main export dispatcher for standard export methods.
        _handle_dict_result : Handles dict results (e.g., from cava_data).
        """
        # Skip export if method is "none"
        if self.export_method.lower() == "none":
            logger.info("Export method set to 'none'; no data is exported!")
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
            self._export_data(result)

        return result

    def _export_data(
        self,
        result: Union[
            xr.Dataset, xr.DataArray, Iterable[Union[xr.Dataset, xr.DataArray]]
        ],
    ):
        """
        Route data to the appropriate export method based on its structure.

        This is the main dispatcher that handles different data structures:

        - xr.Dataset / xr.DataArray with ``closest_cell`` dimension: When
          ``separated=True``, splits along the dimension and exports each
          slice to a separate file via ``_split_and_export_closest_cells``.
          This handles multi-point clip results.
        - xr.Dataset / xr.DataArray (gridded): Exported directly via ``export_single``.
        - list / tuple: Treated as a collection (e.g., from ``cava_data``).
          Routed to ``_export_collection`` which respects ``separated`` and
          ``location_based_naming`` settings.
        - dict: Each value is processed recursively. List/tuple values are
          routed to ``_export_collection``; others to ``export_single``.

        Parameters
        ----------
        result : xr.Dataset | xr.DataArray | Iterable[xr.Dataset | xr.DataArray]
            The data to export.

        Raises
        ------
        TypeError
            If result is not a supported type (Dataset, DataArray, dict, list, tuple).

        See Also
        --------
        export_single : Exports a single dataset/dataarray.
        _export_collection : Handles collections with separated/location_based_naming.
        _split_and_export_closest_cells : Handles multi-point clip results.
        """
        match result:
            case xr.Dataset() | xr.DataArray():
                # Check if this is multi-point data with closest_cell dimension
                # that should be split into separate files
                if self.separated and self._has_closest_cell_dimension(result):
                    self._split_and_export_closest_cells(result)
                else:
                    # Single dataset - export directly
                    self.export_single(result)
            case dict():
                # Dict of datasets - export each value
                for _, value in result.items():
                    if isinstance(value, (list, tuple)):
                        self._export_collection(value)
                    else:
                        self.export_single(value)
            case list() | tuple():
                # Collection of datasets - handle separated/location_based_naming
                self._export_collection(result)
            case _:
                raise TypeError(
                    f"Expected xr.Dataset, xr.DataArray, dict, list, or tuple, got {type(result)}"
                )

    def _export_collection(
        self,
        items: Iterable[Union[xr.Dataset, xr.DataArray]],
    ):
        """
        Export a collection of datasets, respecting separated and location_based_naming.

        This method handles the export of lists of datasets, such as those from
        ``cava_data(..., separate_files=True)``, where each item is a single-point
        dataset with scalar ``lat``/``lon`` coordinates. The behavior depends on
        the ``separated`` configuration option:

        - ``separated=True``: Each item is exported to its own file with a unique
          suffix (either lat/lon coordinates or an index number).
        - ``separated=False``: Each item is exported using the base filename. If
          multiple items have the same base filename, later files will get
          incrementing numeric suffixes (_1, _2, etc.) to avoid overwrites.

        Parameters
        ----------
        items : Iterable[xr.Dataset | xr.DataArray]
            Collection of datasets to export. Each item should be a single-point
            dataset (scalar ``lat``/``lon``) for ``location_based_naming`` to work.

        Raises
        ------
        TypeError
            If any item in the collection is not an xr.Dataset or xr.DataArray.

        See Also
        --------
        _export_single_from_collection : Handles individual item export with naming.
        """
        items_list = list(items)

        if not items_list:
            logger.warning("Empty collection provided for export, nothing to export.")
            return

        if self.separated:
            # Export each item as a separate file
            for idx, item in enumerate(items_list):
                if not isinstance(item, (xr.Dataset, xr.DataArray)):
                    raise TypeError(
                        f"Expected xr.Dataset or xr.DataArray, got {type(item)}"
                    )
                self._export_single_from_collection(item, idx)
        else:
            # Export all items (each as its own file, but without index/location suffix)
            for item in items_list:
                if not isinstance(item, (xr.Dataset, xr.DataArray)):
                    raise TypeError(
                        f"Expected xr.Dataset or xr.DataArray, got {type(item)}"
                    )
                self.export_single(item)

    def _export_single_from_collection(
        self,
        data: Union[xr.Dataset, xr.DataArray],
        index: int,
    ):
        """
        Export a single item from a collection with appropriate naming.

        This method is called when `separated=True` for each item in a collection.
        It modifies the filename based on the `location_based_naming` setting:

        - location_based_naming=True: Appends _<lat>N_<lon>W to the filename
          (e.g., output_37-7749N_122-4194W.nc).
        - location_based_naming=False: Appends _<index> to the filename
          (e.g., output_0.nc, output_1.nc).

        Parameters
        ----------
        data : xr.Dataset | xr.DataArray
            The data to export.
        index : int
            The index of this item in the collection (used when location_based_naming=False).

        Notes
        -----
        Latitude and longitude values are rounded to 6 decimal places. Decimal points
        are replaced with hyphens for filesystem compatibility (e.g., 37.7749 becomes
        37-7749).

        See Also
        --------
        _export_collection : The parent method that calls this for each item.
        """
        # Store original filename
        original_filename = self.filename

        try:
            if self.location_based_naming and self._is_single_point_data(data):
                # Use lat/lon in filename
                lat_val, lon_val = self._extract_point_coordinates(data)
                lat_str = str(round(lat_val, 6)).replace(".", "-")
                lon_str = str(round(abs(lon_val), 6)).replace(".", "-")
                self.filename = f"{original_filename}_{lat_str}N_{lon_str}W"
            else:
                # Use index in filename
                self.filename = f"{original_filename}_{index}"

            # Temporarily disable location_based_naming for export_single
            # since we've already handled the naming here
            original_location_naming = self.location_based_naming
            self.location_based_naming = False

            try:
                self.export_single(data)
            finally:
                self.location_based_naming = original_location_naming

        finally:
            self.filename = original_filename

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
            # Export the data using the standard export logic
            self._export_data(data)
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

    def _clean_attrs_for_netcdf(
        self, data: Union[xr.Dataset, xr.DataArray]
    ) -> Union[xr.Dataset, xr.DataArray]:
        """
        Clean attributes to ensure they can be serialized to NetCDF.

        Parameters
        ----------
        data : xr.Dataset | xr.DataArray
            The data with potentially problematic attributes.

        Returns
        -------
        xr.Dataset | xr.DataArray
            The data with cleaned attributes.
        """
        # Make a copy to avoid modifying the original
        data = data.copy()

        def clean_attrs_dict(attrs_dict):
            """Clean a dictionary of attributes."""
            cleaned = {}
            for k, v in attrs_dict.items():
                if v is None:
                    # Skip None values
                    continue
                elif isinstance(v, dict):
                    # Convert dictionary attributes to string
                    cleaned[k] = str(v)
                elif callable(v):
                    # Skip callable objects
                    continue
                elif isinstance(v, (str, int, float, list, tuple, bytes)):
                    # Keep basic types
                    cleaned[k] = v
                elif hasattr(v, "tolist"):
                    # Convert numpy arrays to lists
                    try:
                        cleaned[k] = v.tolist()
                    except Exception:
                        # If conversion fails, convert to string
                        cleaned[k] = str(v)
                else:
                    # Convert other types to string
                    cleaned[k] = str(v)
            return cleaned

        # Clean top-level attributes
        data.attrs = clean_attrs_dict(data.attrs)

        # Clean variable attributes for datasets
        if isinstance(data, xr.Dataset):
            for var_name in data.data_vars:
                data[var_name].attrs = clean_attrs_dict(data[var_name].attrs)

            # Clean coordinate attributes
            for coord_name in data.coords:
                data[coord_name].attrs = clean_attrs_dict(data[coord_name].attrs)

        return data

    def _is_single_point_data(self, data: Union[xr.Dataset, xr.DataArray]) -> bool:
        """
        Check if the data represents a single spatial point.

        Parameters
        ----------
        data : xr.Dataset | xr.DataArray
            The data to check.

        Returns
        -------
        bool
            True if data has exactly one lat and one lon coordinate value.
        """
        if not (hasattr(data, "lat") and hasattr(data, "lon")):
            return False

        try:
            lat_size = data.lat.size if hasattr(data.lat, "size") else len(data.lat)
            lon_size = data.lon.size if hasattr(data.lon, "size") else len(data.lon)
            return lat_size == 1 and lon_size == 1
        except (AttributeError, TypeError):
            return False

    def _has_closest_cell_dimension(
        self, data: Union[xr.Dataset, xr.DataArray]
    ) -> bool:
        """
        Check if data has a closest_cell dimension (from multi-point clipping).

        Parameters
        ----------
        data : xr.Dataset | xr.DataArray
            The data to check.

        Returns
        -------
        bool
            True if data has a 'closest_cell' dimension with size > 1.
        """
        if not hasattr(data, "dims"):
            return False
        return "closest_cell" in data.dims and data.sizes.get("closest_cell", 0) > 1

    def _split_and_export_closest_cells(self, data: Union[xr.Dataset, xr.DataArray]):
        """
        Split data along closest_cell dimension and export each slice separately.

        This handles multi-point clip results where data has a 'closest_cell'
        dimension. Each slice is exported with appropriate naming based on
        the `location_based_naming` setting:

        - location_based_naming=True: Uses target_lats/target_lons coordinates
          if available, otherwise falls back to index-based naming
        - location_based_naming=False: Uses index-based naming (0, 1, 2...)

        Parameters
        ----------
        data : xr.Dataset | xr.DataArray
            Data with a 'closest_cell' dimension to split and export.

        Notes
        -----
        The clip processor adds `target_lats` and `target_lons` coordinates
        along the `closest_cell` dimension, which are used for location-based
        naming when available.
        """
        n_points = data.sizes["closest_cell"]
        original_filename = self.filename

        # Check for target coordinate availability (added by clip processor)
        has_target_coords = (
            hasattr(data, "target_lats")
            and hasattr(data, "target_lons")
            and "closest_cell" in data.target_lats.dims
        )

        try:
            for idx in range(n_points):
                # Select this slice along closest_cell
                slice_data = data.isel(closest_cell=idx)

                # Determine filename suffix
                if self.location_based_naming and has_target_coords:
                    # Use target coordinates from clip processor
                    lat_val = float(data.target_lats.isel(closest_cell=idx).values)
                    lon_val = float(data.target_lons.isel(closest_cell=idx).values)
                    # Format: replace decimal point with hyphen for filesystem safety
                    # Use absolute values and add N/S, E/W suffixes
                    lat_str = str(round(abs(lat_val), 6)).replace(".", "-")
                    lon_str = str(round(abs(lon_val), 6)).replace(".", "-")
                    lat_suffix = "N" if lat_val >= 0 else "S"
                    lon_suffix = "W" if lon_val < 0 else "E"
                    self.filename = f"{original_filename}_{lat_str}{lat_suffix}_{lon_str}{lon_suffix}"
                else:
                    # Use index-based naming
                    self.filename = f"{original_filename}_{idx}"

                # Temporarily disable location_based_naming since we've handled it
                original_location_naming = self.location_based_naming
                self.location_based_naming = False

                try:
                    self.export_single(slice_data)
                finally:
                    self.location_based_naming = original_location_naming

        finally:
            self.filename = original_filename

    def _extract_point_coordinates(
        self, data: Union[xr.Dataset, xr.DataArray]
    ) -> tuple[float, float]:
        """
        Extract lat/lon coordinates from single-point data.

        Parameters
        ----------
        data : xr.Dataset | xr.DataArray
            The data to extract coordinates from. Must be single-point data.

        Returns
        -------
        tuple[float, float]
            A tuple of (lat, lon) values.

        Raises
        ------
        ValueError
            If the data is not single-point or coordinates cannot be extracted.
        """
        if not self._is_single_point_data(data):
            lat_size = (
                data.lat.size
                if hasattr(data, "lat") and hasattr(data.lat, "size")
                else "N/A"
            )
            lon_size = (
                data.lon.size
                if hasattr(data, "lon") and hasattr(data.lon, "size")
                else "N/A"
            )
            raise ValueError(
                f"Cannot use location_based_naming with gridded data. "
                f"Your data has {lat_size} lat value(s) and {lon_size} lon value(s), "
                f"but location_based_naming requires a single spatial point to include in the filename.\n\n"
                f"Options:\n"
                f"  1. Use the 'clip' processor with a single (lat, lon) tuple to extract one location\n"
                f"  2. Set location_based_naming=False to export the full grid without coordinates in the filename"
            )

        try:
            # Use .values to materialize dask arrays, then extract the scalar
            lat_values = data.lat.values
            lon_values = data.lon.values

            # Handle both scalar and 1-element array cases
            lat_val = float(
                lat_values.item() if hasattr(lat_values, "item") else lat_values
            )
            lon_val = float(
                lon_values.item() if hasattr(lon_values, "item") else lon_values
            )

            return lat_val, lon_val

        except Exception as e:
            raise ValueError(
                f"Failed to extract lat/lon coordinates for location_based_naming: {e}. "
                f"Ensure your data has valid single-point lat/lon coordinates."
            ) from e

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

        Notes
        -----
        For gridded data (multiple lat/lon values), the `location_based_naming`
        option is silently ignored since there's no single location to include
        in the filename. Location-based naming is handled at the collection level
        in `_export_single_from_collection` for point-based data collections.
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

            # Extract lat/lon if available and data is single-point
            if self._is_single_point_data(data):
                try:
                    lat_val, lon_val = self._extract_point_coordinates(data)
                    template_vars["lat"] = str(round(lat_val, 3)).replace(".", "")
                    template_vars["lon"] = str(round(abs(lon_val), 3)).replace(".", "")
                except ValueError:
                    # If extraction fails, leave lat/lon empty in template
                    pass

            base_filename = self.filename_template.format(**template_vars)

        # Apply separated naming (dataset name prefix)
        elif self.separated and hasattr(data, "name") and data.name:
            base_filename = f"{data.name}_{base_filename}"

        # Apply location-based naming if requested AND data is single-point
        # For gridded data, this option is silently ignored (handled at collection level)
        if self.location_based_naming and self._is_single_point_data(data):
            try:
                lat_val, lon_val = self._extract_point_coordinates(data)
                lat_str = str(round(lat_val, 3)).replace(".", "")
                lon_str = str(round(abs(lon_val), 3)).replace(".", "")
                base_filename = f"{base_filename}_{lat_str}N_{lon_str}W"
            except ValueError:
                # If extraction fails, skip location-based naming
                pass

        # Remove any existing extension from filename
        base_filename = base_filename.split(".")[0]

        return base_filename

    def _get_unique_filename(self, base_filename: str, extension: str) -> str:
        """
        Generate a unique filename by appending _N suffix if file exists.

        Follows the Linux convention of appending _1, _2, etc. to find
        the minimum N that guarantees a unique filename.

        Parameters
        ----------
        base_filename : str
            The base filename without extension
        extension : str
            The file extension (including the dot, e.g., ".nc")

        Returns
        -------
        str
            A unique filename that doesn't exist on disk
        """
        # Start checking from _1
        n = 1
        while True:
            candidate = f"{base_filename}_{n}{extension}"
            if not os.path.exists(candidate):
                return candidate
            n += 1
            # Safety limit to prevent infinite loops
            if n > 10000:
                raise RuntimeError(
                    f"Could not find unique filename after 10000 attempts for {base_filename}"
                )

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
        if not isinstance(data, (xr.Dataset, xr.DataArray)):
            raise TypeError(f"Expected xr.Dataset or xr.DataArray, got {type(data)}")

        # Get normalized format for processing
        req_format = self.file_format.lower()

        # Generate filename using smart naming logic
        base_filename = self._generate_filename(data)

        # Create full filename with appropriate extension
        extension_dict = {"zarr": ".zarr", "netcdf": ".nc", "csv": ".csv.gz"}
        extension = extension_dict[req_format]
        save_name = base_filename + extension

        # Handle export_method="skip_existing" - skip if file exists
        if self.export_method.lower() == "skip_existing" and os.path.exists(save_name):
            logger.info("File %s already exists, skipping export.", save_name)
            return

        # For other export methods, find a unique filename if file exists
        if os.path.exists(save_name):
            save_name = self._get_unique_filename(base_filename, extension)
            logger.info("File already exists. Saving to unique filename: %s", save_name)

        try:
            match req_format:
                case "zarr":
                    # Clean up attributes to avoid NetCDF serialization issues
                    data = self._clean_attrs_for_netcdf(data)
                    _export_to_zarr(data, save_name, self.mode)
                case "netcdf":
                    # Clean up attributes to avoid NetCDF serialization issues
                    data = self._clean_attrs_for_netcdf(data)
                    _export_to_netcdf(data, save_name)
                case "csv":
                    # Clean up attributes to avoid NetCDF serialization issues
                    data = self._clean_attrs_for_netcdf(data)
                    _export_to_csv(data, save_name)
                case _:
                    # This should never happen due to validation in __init__
                    raise ValueError(f"Unsupported file format: {self.file_format}")

            logger.info("Export complete: %s", save_name)

        except (OSError, ValueError, RuntimeError) as e:
            error_msg = f"Export failed for {save_name}: {str(e)}"
            if self.fail_on_error:
                raise RuntimeError(error_msg) from e
            else:
                logger.warning(error_msg)
                return
