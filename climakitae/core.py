import xarray as xr

from .selectors import DataSelector, _display_select, LocSelectorArea
from .data_loaders import _read_from_catalog


class Application(object):
    """
    The main control center of the library. Users can select and read-in datasets (generate),
    visualize, transform, interpret, and export them.
    """

    def __init__(self):
        self.selections = DataSelector()
        self.location = LocSelectorArea()
        self.data = xr.Dataset()

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
        print("... reading-in data ...")
        # to do: insert additional 'hang in there' statement if it's taking a while
        self.data = _read_from_catalog(self.selections, self.location)
        print("Data Summary:")
        print(self.data)
