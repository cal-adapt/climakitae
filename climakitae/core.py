import intake
from .data_export import _export_to_user
from .data_loaders import _read_from_catalog, _compute
from .explore import AppExplore
from .selectors import (
    DataSelector,
    _display_select,
    LocSelectorArea,
    UserFileChoices,
    _user_export_select,
    FileTypeSelector,
)
from .view import _visualize


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
        select_panel = _display_select(self.selections, self.location)
        return select_panel

    # === Read data into memory =====================================
    def load(self, data):
        """Read lazily loaded dask data into memory"""
        return _compute(data)

    # === Retrieve ===================================
    def retrieve(self):
        """
        Uses the information gathered in 'select' and stored in 'selections' and
        'location' to generate an xarray DataArray as specified, and return that
        DataArray object.
        """
        # TODO: insert additional 'hang in there' statement if it's taking a while
        return _read_from_catalog(self.selections, self.location, self._cat)

    # === View =======================================
    def view(self, data, lat_lon=True, width=None, height=None, cmap=None):
        """Create a generic visualization of the data

        Args:
            data (xr.DataArray)
            lat_lon (boolean): reproject to lat/lon coords? (default to True)
            width (int): width of plot (default to hvplot.image default)
            height (int): hight of plot (default to hvplot.image default)
            cmap (str): colormap to apply to data (default to "viridis");
                        applies only to mapped data

        Returns:
            hvplot.image()

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
