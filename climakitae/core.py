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
    """The main control center of the library.

    Users can select and read-in datasets (retrieve), visualize,
    transform, interpret, and export them.

    Attributes
    ----------
    _cat: intake_esm.core.esm_datastore
        Data catalog
    location: LocSelectorArea
        Location settings
    selections: DataSelector
        Data settings (variable, unit, timescale, etc)
    explore: AppExplore
        Module hosting the explore panel options

    Methods
    -------
    select
        Display data selection GUI
        Modifies the values in the location and selections attributes
    retrieve
        Retrieve data from catalog
        Will grab data depending on location and selections attributes
    retrieve_from_csv
        Retrieve catalog data from input csv file
    load
        Read lazily-loaded dask array into memory
    view
        Display a generic visualization of the data
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
        """Display data selection panel in Jupyter Notebook environment

        A top-level convenience method.
        Calls a method to display a panel of choices for the data available to load.
        Modifies Application.selections and Application.location' values
        according to what the user specifies in that GUI.

        Returns
        -------
        panel.layout.base.Row
            Selections GUI
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

        Will also print an estimation of the size of the data to be loaded

        Parameters
        ----------
        data: xr.DataArray
            Lazily loaded dask array

        Returns
        -------
        xr.DataArray
            Input data, loaded into memory
        """
        return _compute(data)

    # === Retrieve ===================================
    def retrieve(self):
        """Retrieve data from catalog

        Applications.selections and Applications.location determine data retrieves
        Grabs the data from the AWS S3 bucket, returns lazily loaded dask array
        User-facing function that provides a wrapper for _read_from_catalog

        Returns
        -------
        xr.DataArray
            Lazily loaded dask array

        """
        return _read_from_catalog(self.selections, self.location, self._cat)

    def retrieve_from_csv(self, csv, merge=True):
        """Retrieve data from csv input

        Allows user to bypass app.select GUI and allows
        developers to pre-set inputs in a csv file for ease of use in a notebook.
        Will return ONE of three datatypes depending on input csv

        Parameters
        ----------
            csv: str
                Path to local csv file
            merge: bool, optional
                If multiple datasets desired, merge to form a single object?
                Defaults to True

        Returns
        -------
            xr.Dataset
                If multiple rows are in the csv, each row is a data_variable
            xr.DataArray
                If csv only has one row
            list of xr.DataArray
                If multiple rows are in the csv and merge=True,
                multiple DataArrays are returned in a single list.
        """
        return _read_data_from_csv(
            self.selections, self.location, self._cat, csv, merge
        )

    # === View =======================================
    def view(self, data, lat_lon=True, width=None, height=None, cmap=None):
        """Create a generic visualization of the data

        Visualization will depend on the shape of the input data.
        Works much faster if the data has already been loaded into memory.

        Parameters
        ----------
        data: xr.DataArray
            Input data
        lat_lon: bool, optional
            Reproject to lat/lon coords?
            Default to True.
        width: int, optional
            Width of plot
            Default to hvplot default
        height: int, optional
            Height of plot
            Default to hvplot.image default
        cmap: matplotlib colormap name or AE colormap names
            Colormap to apply to mapped data (will not effect lineplots)
            Default to "ae_orange"

        Returns
        -------
        holoviews.core.spaces.DynamicMap
            Interactive map or lineplot
        matplotlib.figure.Figure
            xarray default map
            Only produced if gridded data doesn't have sufficient cells for hvplot

        Raises
        ------
        UserWarning
            Warn user that the function will be slow if data has not been loaded into memory
        """
        return _visualize(data, lat_lon=lat_lon, width=width, height=height, cmap=cmap)

    # === Export =====================================
    def export_as(self):
        """Displays a panel of choices for export file formats.

        Modifies the Application.user_export_format value according to user specification.

        Returns
        -------
        panel.layout.base.Row
            Panel displayed in notebook
        """
        export_select_panel = _user_export_select(self.user_export_format)
        return export_select_panel

    def export_dataset(self, data_to_export, file_name, **kwargs):
        """Export dataset to desired filetype

        Uses the selection from 'export_as' to create a file in the specified
        format and write it to the working directory.
        File will be automatically created in the working directory.

        Parameters
        ----------
        data_to_export: xr.DataArray or xr.Dataset
            Data to be exported
        file_name: str
            Filename to give output
            Should not include file extension (i.e. "my_filename" instead of "my_filename.nc")
        """
        return _export_to_user(
            self.user_export_format, data_to_export, file_name, **kwargs
        )
