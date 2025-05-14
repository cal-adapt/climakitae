"""
DataProcessor for clipping data based on spatial boundaries.
"""

import os
import warnings
from typing import Any, Dict, Iterable, Union

import geopandas as gpd
import pyproj
import xarray as xr
from shapely.geometry import box, mapping

from climakitae.core.constants import UNSET
from climakitae.new_core.data_access import DataCatalog
from climakitae.new_core.processors.data_processor import (
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
        geom = UNSET
        match self.value:
            case str():
                if not os.path.exists(self.value):
                    raise FileNotFoundError(f"File {self.value} does not exist.")
                geom = gpd.read_file(self.value)
            case tuple():
                geom = gpd.GeoDataFrame(
                    geometry=[
                        box(
                            minx=self.value[1][0],
                            miny=self.value[0][0],
                            maxx=self.value[1][1],
                            maxy=self.value[0][1],
                        )
                    ],
                    crs=pyproj.CRS.from_epsg(4326),
                    # 4326 is WGS84 i.e. lat/lon
                )
            case _:
                raise ValueError(
                    f"Invalid value type for clipping. Expected str or tuple but got {type(self.value)}."
                )

        match result:
            case dict():
                # Clip each dataset in the dictionary
                return {
                    key: self._clip_data_with_geom(value, geom)
                    for key, value in result.items()
                }
            case xr.Dataset() | xr.DataArray():
                # Clip the single dataset
                return self._clip_data_with_geom(result, geom)
            case Iterable():
                # return as passed type
                ret = [self._clip_data_with_geom(data, geom) for data in result]
                return type(result)(ret)
            case _:
                raise ValueError(
                    f"Invalid result type for clipping. Expected dict, xr.Dataset, xr.DataArray, or Iterable but got {type(result)}."
                )

    def update_context(self, context: Dict[str, Any]):
        # Placeholder for updating context
        pass

    def set_data_accessor(self, catalog: DataCatalog):
        # Placeholder for setting data accessor
        pass

    @staticmethod
    def _clip_data_with_geom(data: xr.DataArray | xr.Dataset, gdf: gpd.GeoDataFrame):
        """
        Clip the data using a bounding box.

        Parameters
        ----------
        data : xr.Dataset | Iterable[xr.Dataset]
            The data to be clipped.
        gdf : gpd.GeoDataFrame
            The GeoDataFrame containing the geometry

        Returns
        -------
        xr.Dataset | Iterable[xr.Dataset]
            The clipped data.
        """
        # check crs of geom and data
        if gdf.crs is None and data.rio.crs is not None:
            warnings.warn(
                "The GeoDataFrame does not have a CRS set. Setting the CRS to the data's CRS."
            )
            gdf.set_crs(data.rio.crs, inplace=True)

        return data.rio.clip(
            gdf.geometry.apply(mapping),
            gdf.crs,
            drop=True,
            all_touched=True,
        )
