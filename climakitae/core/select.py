from climakitae.core.data_interface import DataInterface, DataParameters


class Select:
    def __init__(self):
        # self.data_interface = DataInterface()
        # self.select_params = DataSelector(
        #     data_catalog=self.data_interface.data_catalog,
        #     variable_descriptions=self.data_interface.variable_descriptions,
        #     stations_gdf=self.data_interface.get_stations_gdf(self.data_interface.stations),
        #     geographies=self.data_interface.geographies,
        # )
        self.data_selector = DataParameters()
