import intake
from .data_export import _export_to_user
from .explore import AppExplore
from .view import _visualize
from .data_loaders import (
    _read_from_catalog, 
    _compute, 
    _read_data_from_csv
) 
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
from .meteo_yr import _retrieve_meteo_yr_data


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
        """Read lazily loaded dask data into memory"""
        return _compute(data)

    # === Retrieve ===================================
    def retrieve(self):
        """
        The primary and first data loading method, called by
        core.Application.retrieve, it returns a DataArray (which can be quite large)
        containing everything requested by the user (which is stored in 'selections'
        and 'location').

        Args:
            selections (DataLoaders): object holding user's selections
            location (LocSelectorArea): object holding user's location selections
            cat (intake_esm.core.esm_datastore): catalog

        Returns:
            da (xr.DataArray): output data
        """
        return _read_from_catalog(self.selections, self.location, self._cat)

    def retrieve_from_csv(self, csv, merge=True):
        """
        Retrieve data from csv input. Allows user to bypass app.select GUI and allows
        developers to pre-set inputs in a csv file for ease of use in a notebook.

        Args:
            selections (DataLoaders): object holding user's data selections
            location (LocSelectorArea): object holding user's location selections
            cat (intake_esm.core.esm_datastore): catalog
            csv (str): path to local csv file
            merge (bool, options): if multiple datasets desired, merge to form a single object?

        Returns: one of the following, depending on csv input and merge
            xr_ds (xr.Dataset): if multiple rows are in the csv, each row is a data_variable
            xr_da (xr.DataArray): if csv only has one row
            xr_list (list of xr.DataArrays): if multiple rows are in the csv and merge=True,
                multiple DataArrays are returned in a single list.
        """
        return _read_data_from_csv(
            self.selections, self.location, self._cat, csv, merge
        )
    
    def retrieve_meteo_yr_data(self, ssp=None, year_start=2015, year_end=None): 
        """User-facing function for retrieving data needed for computing a meteorological year.

        Reads in the hourly ensemble means instead of the hourly data. 
        Reads in future SSP data, historical climate data, or a combination 
        of both, depending on year_start and year_end 

        Parameters
        ----------
        ssp: str, one of "SSP 2-4.5 -- Middle of the Road", "SSP 2-4.5 -- Middle of the Road", "SSP 3-7.0 -- Business as Usual", "SSP 5-8.5 -- Burn it All"
            Shared Socioeconomic Pathway. Defaults to SSP 3-7.0 -- Business as Usual
        year_start: int, optional
            Year between 1980-2095. Default to 2015
        year_end: int, optional
            Year between 1985-2100. Default to year_start+30

        Returns
        -------
        xr.DataArray
            Hourly ensemble means from year_start-year_end for the ssp specified.
        """
        return _retrieve_meteo_yr_data(self.selections, self.location, self._cat, ssp, year_start, year_end)

    # === View =======================================
    def view(self, data, lat_lon=True, width=None, height=None, cmap=None):
        """Create a generic visualization of the data

        Args:
            data (xr.DataArray)
            lat_lon (boolean, optional): reproject to lat/lon coords? (default to True)
            width (int, optional): width of plot (default to hvplot.image default)
            height (int, optional): hight of plot (default to hvplot.image default)
            cmap (str, optional): colormap to apply to data (default to "ae_orange"); applies only to mapped data

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
