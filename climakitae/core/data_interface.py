import pkg_resources
import pandas as pd
import geopandas as gpd
import intake
from climakitae.core.constants import variable_descriptions_csv_path, stations_csv_path, data_catalog_url


class DataInterface:
    def __init__(self):
        self.variable_descriptions = pd.read_csv(variable_descriptions_csv_path, index_col=None)
        self.stations = pd.read_csv(stations_csv_path)
        self.data_catalog = intake.open_esm_datastore(data_catalog_url)
        
    def get_stations_gdf(self):
        stations_gpf = gpd.GeoDataFrame(
            self.stations,
            crs="EPSG:4326",
            geometry=gpd.points_from_xy(self.stations.LON_X, self.stations.LAT_Y),
        )
        return stations_gpf
