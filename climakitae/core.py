import intake
from .data_export import _export_to_user
from .explore import AppExplore
from .view import _visualize
from .data_loaders import _read_from_catalog, _compute, _read_data_from_csv
from .selectors import (
    DataSelector,
    _display_select,
    LocSelectorArea,
    UserFileChoices,
    _user_export_select,
    FileTypeSelector,
    _get_simulation_options,
)
from .catalog_convert import (
    _resolution_to_gridlabel,
    _timescale_to_table_id,
    _scenario_to_experiment_id,
)


class Application(object):
    """
    The main control center of the library. Users can select and read-in
    datasets (retrieve), visualize, transform, interpret, and export them.
    """

    def __init__(self):
        self._cat = intake.open_esm_datastore(
            "https://cadcat.s3.amazonaws.com/cae-collection.json"
        )
        self.location = LocSelectorArea(name="Location Selections")
        self.selections = DataSelector(cat=self._cat, location=self.location)
        self.user_export_format = FileTypeSelector()
        self.explore = AppExplore(self.selections, self.location, self._cat)

    # === Select =====================================
    def select(self):
        """
        A top-level convenience method -- calls a method to display a panel of
        choices for the data available to load. Modifies the 'selections' and
        'location' values according to what the user specifies in that GUI.
        """
        # Reset simulation options
        # This will remove ensmean if the use has just called app.explore.amy()
        self.selections.simulation = _get_simulation_options(
            cat=self._cat,
            activity_id=self.selections.downscaling_method,
            table_id=_timescale_to_table_id(self.selections.timescale),
            grid_label=_resolution_to_gridlabel(self.selections.resolution),
            experiment_id=[
                _scenario_to_experiment_id(scen)
                for scen in self.selections.scenario_historical
                + self.selections.scenario_ssp
            ],
        )
        # Display panel
        select_panel = _display_select(self.selections, self.location)
        return select_panel

    # === Read data into memory =====================================
    def load(self, data):
        """Read lazily loaded dask data into memory

        Args: 
            data (xarray.DataArray): input data

        Returns 
            xarray.DataArray: same as input data
        """
        return _compute(data)

    # === Retrieve ===================================
    def retrieve(self):
        """
        The primary and first data loading method, called by
        core.Application.retrieve, it returns a DataArray (which can be quite large)
        containing everything requested by the user (which is stored in 'selections'
        and 'location').

        Returns: 
            DataArray: output data

        """
        return _read_from_catalog(self.selections, self.location, self._cat)

    def retrieve_from_csv(self, csv, merge=True):
        """
        Retrieve data from csv input. Allows user to bypass app.select GUI and allows
        developers to pre-set inputs in a csv file for ease of use in a notebook.

        Args: 
            csv (str): path to local csv file
            merge (bool, optional): if multiple datasets desired, merge to form a single object? Defaults to True

        Returns: 
            Dataset: if multiple rows are in the csv, each row is a data_variable
            DataArray: if csv only has one row
            list: if multiple rows are in the csv and merge=True, multiple DataArrays are returned in a single list.
        """
        return _read_data_from_csv(
            self.selections, self.location, self._cat, csv, merge
        )

    # === View =======================================
    def view(self, data, lat_lon=True, width=None, height=None, cmap=None):
        """Create a generic visualization of the data

        Args:
            data (xarray.DataArray)
            lat_lon (bool, optional): reproject to lat/lon coords? (default to True)
            width (int, optional): width of plot (default to hvplot.image default)
            height (int, optional): hight of plot (default to hvplot.image default)
            test (array-like): test
            cmap (matplotlib colormap name): colormap to apply to data (default to "ae_orange"); applies only to mapped data

        Returns:
            hvplot.image() or matplotlib object, depending on input data
        """
        return _visualize(data, lat_lon=lat_lon, width=width, height=height, cmap=cmap)

    # === Export =====================================
    def export_as(self):
        """
        Displays a panel of choices for export file formats. Modifies the
        'user_export_format' value according to user specification.
        """
        export_select_panel = _user_export_select(self.user_export_format)
        return export_select_panel

    def export_dataset(self, data_to_export, file_name, **kwargs):
        """
        Uses the selection from 'export_as' to create a file in the specified
        format and write it to the working directory.
        """
        return _export_to_user(
            self.user_export_format, data_to_export, file_name, **kwargs
        )
