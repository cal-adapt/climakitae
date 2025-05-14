"""
DataProcessor for clipping data based on spatial boundaries.
"""

import os
import warnings
from typing import Any, Dict, Iterable, Union

import geopandas as gpd
import pyproj
import xarray as xr
from shapely.geometry import mapping

from climakitae.core.constants import UNSET
from climakitae.new_core.data_access import DataCatalog
from climakitae.new_core.processors.data_processor import (
    _PROCESSOR_REGISTRY,
    DataProcessor,
    register_processor,
)


@register_processor("clip")
class Clip(DataProcessor):
    """
    Clip data based on spatial boundaries.

    This class is a placeholder for data subsetting logic.

    Parameters
    ----------
    value : str | tuple
        The value to clip the data by. This can be a string representing a
        shapefile or a tuple representing latitude and longitude bounds.

    Methods:
    -------
    _clip_with_shape_file(shape_file: str, data: xr.Dataset) -> xr.Dataset

    _clip_with_lat_lon(lat: tuple, lon: tuple, data: xr.Dataset) -> xr.Dataset
    """

    def __init__(self, value):
        """
        Initialize the Clip processor.

        Parameters
        ----------
        value : str
            The value to clip the data by.
        """
        self.value = value

    def execute(
        self,
        result: Union[
            xr.Dataset, xr.DataArray, Iterable[Union[xr.Dataset, xr.DataArray]]
        ],
        context: Dict[str, Any],
    ) -> Union[xr.Dataset, xr.DataArray, Iterable[Union[xr.Dataset, xr.DataArray]]]:
        # Placeholder for data subsetting logic
        match self.value:
            case str():
                # open a shape file and clip the data
                return self._clip_with_shape_file(self.value, result)
            case tuple():
                # Clip lat/lot on the dataset based on lat lon tuple
                return self._clip_with_lat_lon(self.value[0], self.value[1], result)
            case _:
                raise ValueError(
                    "Invalid value type for clipping."
                )  # TODO warning not error

    def update_context(self, context: Dict[str, Any]):
        # Placeholder for updating context
        pass

    def set_data_accessor(self, catalog: DataCatalog):
        # Placeholder for setting data accessor
        pass

    @staticmethod
    def _clip_with_shape_file(
        shape_file: str, data: xr.Dataset | Iterable[xr.Dataset]
    ) -> xr.Dataset:
        """
        Clip the data using a shapefile.

        Parameters
        ----------
        shape_file : str
            The path to the shapefile.
        data : xr.Dataset | Iterable[xr.Dataset]
            The data to be clipped.

        Returns
        -------
        xr.Dataset
            The clipped data.
        """
        # Check if the shapefile exists
        if not os.path.exists(shape_file):
            warnings.warn(f"Shapefile {shape_file} does not exist.", UserWarning)
            return data

        match data:
            case xr.Dataset() | xr.DataArray():  # a single dataset was passed
                # open the shape file
                shape_gdf = gpd.read_file(shape_file)

                # Clip the data using the shapefile
                return data.rio.clip(
                    shape_gdf.geometry.apply(mapping), shape_gdf.crs, drop=True
                )
            case dict():
                # open the shape file
                shape_gdf = gpd.read_file(shape_file)

                # Clip the data using the shapefile
                clipped_data = {}
                for key, value in data.items():
                    clipped_data[key] = value.rio.clip(
                        shape_gdf.geometry.apply(mapping), shape_gdf.crs
                    )
                return clipped_data

            case list():
                # open the shape file
                shape_gdf = gpd.read_file(shape_file)

                # Clip the data using the shapefile
                clipped_data = []
                for value in data:
                    clipped_data.append(
                        value.rio.clip(shape_gdf.geometry.apply(mapping), shape_gdf.crs)
                    )
                return clipped_data
            case tuple():
                # open the shape file
                shape_gdf = gpd.read_file(shape_file)

                # Clip the data using the shapefile
                clipped_data = []
                for value in data:
                    clipped_data.append(
                        value.rio.clip(shape_gdf.geometry.apply(mapping), shape_gdf.crs)
                    )
                # convert to tuple
                return tuple(clipped_data)
            case _:
                raise ValueError(
                    "Invalid data type for clipping."
                )  # TODO warning not error

    @staticmethod
    def _get_lat_lon_dims(data_obj: xr.Dataset | xr.DataArray) -> tuple[str, str]:
        """
        Get the lat/lon type dimensions of the data object.

        Parameters
        ----------
        data_obj : xr.Dataset | xr.DataArray
            The data object to get dimensions from.

        Returns
        -------
        tuple[str, str]
            The latitude and longitude dimension names.
        """
        dims = data_obj.dims if hasattr(data_obj, "dims") else None
        if dims is None:
            return None, None

        # Map of possible dimension names
        lat_options = ["lat", "latitude", "y"]
        lon_options = ["lon", "longitude", "x"]

        lat_dim = next((dim for dim in lat_options if dim in dims), None)
        lon_dim = next((dim for dim in lon_options if dim in dims), None)

        return lat_dim, lon_dim

    @staticmethod
    def _convert_lat_lon(lat: tuple, lon: tuple, crs: str | int | pyproj.CRS) -> tuple:
        """
        Convert latitude and longitude bounds to the correct format.

        Parameters
        ----------
        lat : tuple
            The latitude bounds.
        lon : tuple
            The longitude bounds.
        crs : str | int | pyproj.CRS
            The coordinate reference system to convert to.

        Returns
        -------
        tuple
            The converted latitude and longitude bounds.
        """
        lat_lon_to_model_proj = pyproj.Transformer.from_crs(
            crs_from="EPSG:4326",
            crs_to=crs,
            always_xy=True,
        )
        corners = [
            (lon[0], lat[0]),
            (lon[1], lat[0]),
            (lon[1], lat[1]),
            (lon[0], lat[1]),
        ]
        transformed_corners = [
            lat_lon_to_model_proj.transform(lon, lat) for lon, lat in corners
        ]
        return (
            min(transformed_corners, key=lambda x: x[1])[1],
            max(transformed_corners, key=lambda x: x[1])[1],
        ), (
            min(transformed_corners, key=lambda x: x[0])[0],
            max(transformed_corners, key=lambda x: x[0])[0],
        )

    @staticmethod
    def _clip_with_lat_lon(
        lat: tuple, lon: tuple, data: xr.Dataset | Iterable[xr.Dataset]
    ) -> xr.Dataset:
        """
        Clip the data using latitude and longitude bounds.

        Parameters
        ----------
        lat : tuple
            The latitude bounds.
        lon : tuple
            The longitude bounds.
        data : xr.Dataset | Iterable[xr.Dataset]
            The data to be clipped.

        Returns
        -------
        xr.Dataset
            The clipped data.
        """
        lat_dim, lon_dim = UNSET, UNSET

        match data:
            case xr.Dataset() | xr.DataArray():
                # Clip the data using latitude and longitude bounds
                lat_dim, lon_dim = Clip._get_lat_lon_dims(data)
                if lat_dim is None or lon_dim is None:
                    raise ValueError(
                        "Latitude and longitude dimensions not found in the data."
                    )
                if lat_dim == "y" or lon_dim == "x":
                    # Convert to latitude and longitude dimensions
                    lat, lon = Clip._convert_lat_lon(lat, lon, data.rio.crs)
                return data.sel(
                    {lat_dim: slice(lat[0], lat[1]), lon_dim: slice(lon[0], lon[1])}
                )

            case dict():
                # Clip the data using latitude and longitude bounds
                clipped_data = {}
                for key, value in data.items():
                    if lat_dim is UNSET or lon_dim is UNSET:
                        lat_dim, lon_dim = Clip._get_lat_lon_dims(value)
                    if lat_dim is None or lon_dim is None:
                        raise ValueError(
                            "Latitude and longitude dimensions not found in the data."
                        )
                    if lat_dim == "y" or lon_dim == "x":
                        # Convert to latitude and longitude dimensions
                        lat, lon = Clip._convert_lat_lon(lat, lon, value.rio.crs)
                    slice_dict = {
                        lat_dim: slice(lat[0], lat[1]),
                        lon_dim: slice(lon[0], lon[1]),
                    }
                    print(f"Clipping {key} with slice: {slice_dict}")
                    print(
                        f"Lat data before clipping: {min(value[lat_dim].values)}, {max(value[lat_dim].values)}"
                    )
                    print(
                        f"Lon data before clipping: {min(value[lon_dim].values)}, {max(value[lon_dim].values)}"
                    )
                    clipped_data[key] = value.sel(
                        {
                            lat_dim: slice(lat[0], lat[1]),
                            lon_dim: slice(lon[0], lon[1]),
                        }
                    )
                return clipped_data
            case list():
                # Clip the data using latitude and longitude bounds
                clipped_data = []
                for value in data:
                    if lat_dim is UNSET or lon_dim is UNSET:
                        lat_dim, lon_dim = Clip._get_lat_lon_dims(value)
                    if lat_dim is None or lon_dim is None:
                        raise ValueError(
                            "Latitude and longitude dimensions not found in the data."
                        )
                    if lat_dim == "y" or lon_dim == "x":
                        # Convert to latitude and longitude dimensions
                        lat, lon = Clip._convert_lat_lon(lat, lon, value.rio.crs)
                    clipped_data.append(
                        value.sel(
                            {
                                lat_dim: slice(lat[0], lat[1]),
                                lon_dim: slice(lon[0], lon[1]),
                            }
                        )
                    )
                return clipped_data
            case tuple():
                clipped_data = []
                for value in data:
                    if lat_dim is UNSET or lon_dim is UNSET:
                        lat_dim, lon_dim = Clip._get_lat_lon_dims(value)
                    if lat_dim is None or lon_dim is None:
                        raise ValueError(
                            "Latitude and longitude dimensions not found in the data."
                        )
                    if lat_dim == "y" or lon_dim == "x":
                        # Convert to latitude and longitude dimensions
                        lat, lon = Clip._convert_lat_lon(lat, lon, value.rio.crs)
                    clipped_data.append(
                        value.sel(
                            {
                                lat_dim: slice(lat[0], lat[1]),
                                lon_dim: slice(lon[0], lon[1]),
                            }
                        )
                    )
                return tuple(clipped_data)
            case _:
                warnings.warn(
                    f"Invalid data type for clipping. Expected xr.Dataset, dict, list, or tuple but got {type(data)}.",
                    UserWarning,
                )
                return data
