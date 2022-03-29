from .selectors import (DataSelector, _display_select, LocSelectorArea,
                        UserFileChoices, _user_export_select, FileTypeSelector)
from .data_loaders import _read_from_catalog
from .data_export import _export_to_user
from .metadata_update import transform_details
from .example_transform import temporal_mean


class Application(object):
    """
    The main control center of the library. Users can select and read-in datasets (generate),
    visualize, transform, interpret, and export them.
    """

    def __init__(self):
        self.selections = DataSelector()
        self.location = LocSelectorArea()
        self.user_export_format = FileTypeSelector()
        
    # === Select =====================================
    def select(self):
        """
        A top-level convenience method -- calls a method to display a panel of choices for
        the data available to load. Modifies the 'selections' and 'location' values
        according to what the user specifies in that GUI.
        """
        select_panel = _display_select(self.selections, self.location)
        return select_panel

    # === Generate ===================================
    def generate(self):
        """
        Uses the information gathered in 'select' and store in 'selections' and 'location'
        to generate an xarray Dataset as specified, and return that Dataset object.
        """
        # to do: insert additional 'hang in there' statement if it's taking a while
        return _read_from_catalog(self.selections, self.location)

    # === Export ======================================
    def export_as(self):
        """
        Displays a panel of choices for export file formats. Modifies the
        'user_export_format' value according to user specification.
        """
        export_select_panel = _user_export_select(self.user_export_format)
        return export_select_panel

    def export_dataset(self,data_to_export,variable_name,file_name):
        """
        Uses the selection from 'export_as' to create a file in the specified
        format and write it to the working directory.
        """
        return _export_to_user(self.user_export_format,data_to_export,
                               variable_name,file_name)