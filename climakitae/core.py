"""The main user-facing interfaces of the climakitae library. 
"""
# from dataclasses import dataclass
import intake
import pkg_resources
import pandas as pd
from .data_export import _export_to_user
from .view import _visualize
from .data_loaders import _read_catalog_from_select, _read_catalog_from_csv, _compute
from .selectors import (
    _DataSelector,
    _display_select,
    _user_export_select,
)


# @dataclass()
class _DataInterface(object):
    """
    A singleton class to hold the catalog and shapefile information
    needed by the GetData class.

    Only one instance can exist because of how it is constructed.
    The user never interacts with this. Each additional time it is
    instantiated after the first time, it just returns the same initial
    instance.
    """

    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(_DataInterface, cls).__new__(cls)
            return cls._instance

    def __init__(self):
        var_catalog_resource = pkg_resources.resource_filename(
            "climakitae", "data/variable_descriptions.csv"
        )
        self.var_config = pd.read_csv(var_catalog_resource, index_col=None)
        self.catalog = intake.open_esm_datastore(
            "https://cadcat.s3.amazonaws.com/cae-collection.json"
        )
        # TODO: add shapefile stuff and any other reference tables here as well,
        # which get used in selectors.py and data_loaders.py
        # Initialize empty, and populate as needed on function calls.


# === Select =====================================
class DataChooser(object):
    """
    A class for selections which holds the selections param object, and a method to
    retrieve whatever those selections indicate.

    Can be instantiated many times. Explore modules create their own instance.
    """

    def __init__(self, ref_data):
        self.ref_data = ref_data  # _DataInterface()
        self.selections = _DataSelector(
            cat=self.ref_data.catalog, var_config=self.ref_data.var_config
        )
        # TODO: pull UI stuff out of _DataSelector to extent possible?

    def retrieve(self, config=None, merge=True):
        """Retrieve data from catalog

        To retrieve data using the settings in a configuration csv file, set config to the local
        filepath of the csv.
        Grabs the data from the AWS S3 bucket, returns lazily loaded dask array.
        User-facing function that provides a wrapper for _read_catalog_from_csv and _read_catalog_from_select.

        Parameters
        ----------
        config: str, optional
            Local filepath to configuration csv file
            Default to None-- retrieve settings in self.selections
        merge: bool, optional
            If config is TRUE and multiple datasets desired, merge to form a single object?
            Defaults to True.

        Returns
        -------
        xr.DataArray
            Lazily loaded dask array
            Default if no config file provided
        xr.Dataset
            If multiple rows are in the csv, each row is a data_variable
            Only an option if a config file is provided
        list of xr.DataArray
            If multiple rows are in the csv and merge=True,
            multiple DataArrays are returned in a single list.
            Only an option if a config file is provided.

        """
        if config is not None:
            if type(config) == str:
                return _read_catalog_from_csv(
                    self.selections, self.catalog, config, merge
                )
            else:
                raise ValueError(
                    "To retrieve data specified in a configuration file, please input the path to your local configuration csv as a string"
                )
        return _read_catalog_from_select(self.selections, self.ref_data.catalog)


class Select(DataChooser):
    """
    Adds a method to display the panel user-interface on top of the
    DataChooser class. Create a new instance every time.

    Usage
    --------
    To view the select panel in a notebook environment,
    you can use the following code:

    >>> import climakitae as ck
    >>> my_choices = ck.Select()
    >>> my_choices.interface() #displays selection panel

    And then to retrieve the selected data:
    >>> my_data = my_choices.retrieve()

    """

    def __init__(self, ref_data):
        super().__init__(ref_data)
        # automatically return interface when new instance is defined:
        self.interface()

    def interface(self):
        """Display data selection panel in Jupyter Notebook environment

        A top-level convenience method.
        Calls a method to display a panel of choices for the data available to load.
        Modifies self.selections values
        according to what the user specifies in that GUI.

        Returns
        -------
        panel.layout.base.Row
            Selections GUI
        """

        # select_panel =
        return _display_select(self.selections)


class SimulationFinder(DataChooser):
    """
    Pseudo-code for how this would work under the regime established above.
    """

    def __init__(self, *args, **kwargs):
        super(SimulationFinder, self).__init__(*args, **kwargs)
        self.data = xr.DataArray()
        self.metric_choices = _ParamSubClassTBD
        self.analysis_output = _SomeDataStructure
        self.analysis_toggle = _ParamChoices

    def define_params(self):
        pn.Params(self.selections)

    def compute(self):
        self.data = self.retrieve(self.selections)
        metrics = get_metric(self.data, self.metric_choices)
        self.analysis_output = cred_and_spread(metrics)

    def display_output(self):
        _panel_of_plots(self.analysis_output, self.analysis_toggle)

    def export_data(self):
        if self.analysis_toggle.which_export == "full timeseries":
            export(self.data.sel(simulation=self.analysis_toggle.which_sims))
        elif self.analysis_toggle.which_export == "analysis output":
            export(self.analysis_output)


# === Read data into memory =====================================
def load(data):
    """Read lazily loaded dask data into memory

    Will also print an estimation of the size of the data to be loaded
    Called as ck.load(data)

    Parameters
    ----------
    data: xr.DataArray
        Lazily loaded dask array

    Returns
    -------
    xr.DataArray
        Input data, loaded into memory

    See also
    --------
    xarray.DataArray.compute
    """
    return _compute(data)
    # TODO: move this, rename it load, and put directly in __inti__.py
    # (no need for this wrapper function anymore)


# === View =======================================
def view(data, lat_lon=True, width=None, height=None, cmap=None):
    """Create a generic visualization of the data

    Visualization will depend on the shape of the input data.
    Works much faster if the data has already been loaded into memory.
    Called as ck.view(data)

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
        Colormap to apply to data
        Default to "ae_orange" for mapped data or color-blind friendly "categorical_cb" for timeseries data.

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
    # TODO: as with load, no need for a wrapper function -- just rename _visualize as view
    # and reference directly in __init__.py


# === Export =====================================
# being consolidated into one anyway:
def export_as():
    """Displays a panel of choices for export file formats.

    Modifies the Application.user_export_format value according to user specification.

    Returns
    -------
    panel.layout.base.Row
        Panel displayed in notebook
    """
    export_select_panel = _user_export_select(self.user_export_format)
    return export_select_panel


def export_dataset(data_to_export, file_name, **kwargs):
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
    return _export_to_user(self.user_export_format, data_to_export, file_name, **kwargs)
