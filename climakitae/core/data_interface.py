import os
import pkg_resources
import pandas as pd
import geopandas as gpd
import intake
from climakitae.core.constants import (
    variable_descriptions_csv_path,
    stations_csv_path,
    data_catalog_url,
)
from climakitae.utils import read_csv_file
from climakitae.core.boundaries import Boundaries


class DataInterface:
    def __init__(self):
        self.variable_descriptions = read_csv_file(variable_descriptions_csv_path)
        self.stations = read_csv_file(stations_csv_path)
        self.data_catalog = intake.open_esm_datastore(data_catalog_url)

        # Get geography boundaries and selection options
        self.geographies = Boundaries()
        self.geography_choose = self.geographies.boundary_dict()

    def get_stations_gdf(self):
        stations_gpf = gpd.GeoDataFrame(
            self.stations,
            crs="EPSG:4326",
            geometry=gpd.points_from_xy(self.stations.LON_X, self.stations.LAT_Y),
        )
        return stations_gpf
