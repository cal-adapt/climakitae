from .selectors import (
    DataSelector,
    _display_select,
    LocSelectorArea,
    UserFileChoices,
    _user_export_select,
    FileTypeSelector,
)
from .data_loaders import _read_from_catalog
from .data_export import _export_to_user
import intake


class Application(object):
    """
    The main control center of the library. Users can select and read-in datasets (retrieve),
    visualize, transform, interpret, and export them.
    """

    def __init__(self):
        self._cat = intake.open_catalog("https://cadcat.s3.amazonaws.com/cae.yaml")
        self.selections = DataSelector(choices=_get_catalog_contents(self._cat))

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

    # === Retrieve ===================================
    def retrieve(self):
        """
        Uses the information gathered in 'select' and stored in 'selections' and 'location'
        to generate an xarray DataArray as specified, and return that DataArray object.
        """
        # to do: insert additional 'hang in there' statement if it's taking a while
        return _read_from_catalog(self.selections, self.location, self._cat)

    # === Export ======================================
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


def _get_catalog_contents(_cat):
    # get the list of data variables from one of the zarr files:
    _ds = _cat[list(_cat)[0]].to_dask()
    _variable_choices_hourly_wrf = {
        v.attrs["description"].capitalize(): k for k, v in _ds.data_vars.items()
    }
    # Add derived variables 
    # Dictionary key (i.e. Precipitation (total)) will appear in list of variable options. 
    _variable_choices_hourly_wrf.update(
            {"Precipitation (total)": "TOT_PRECIP",  
             "Relative Humidity": "REL_HUMIDITY", 
             "Wind Magnitude at 10m": "WIND_MAG"}
    ) 
    # remove some variables from the list, which will be superceded by higher quality hydrology
    _to_drop = ["Surface runoff", "Subsurface runoff", "Snow water equivalent"]
    [_variable_choices_hourly_wrf.pop(k) for k in _to_drop]
    # give better names to some descriptions, and reorder:
    _variable_choices_hourly_wrf["Surface Pressure"] = _variable_choices_hourly_wrf[
        "Sfc pressure"
    ]
    _variable_choices_hourly_wrf.pop("Sfc pressure")
    _variable_choices_hourly_wrf["2m Air Temperature"] = _variable_choices_hourly_wrf[
        "Temp at 2 m"
    ]
    _variable_choices_hourly_wrf.pop("Temp at 2 m")
    _variable_choices_hourly_wrf[
        "2m Water Vapor Mixing Ratio"
    ] = _variable_choices_hourly_wrf["Qv at 2 m"]
    _variable_choices_hourly_wrf.pop("Qv at 2 m")
    _variable_choices_hourly_wrf[
        "West-East component of Wind at 10m"
    ] = _variable_choices_hourly_wrf["U at 10 m"]
    _variable_choices_hourly_wrf.pop("U at 10 m")
    _variable_choices_hourly_wrf[
        "North-South component of Wind at 10m"
    ] = _variable_choices_hourly_wrf["V at 10 m"]
    _variable_choices_hourly_wrf.pop("V at 10 m")
    _variable_choices_hourly_wrf[
        "Snowfall (snow and ice)"
    ] = _variable_choices_hourly_wrf["Accumulated total grid scale snow and ice"]
    _variable_choices_hourly_wrf.pop("Accumulated total grid scale snow and ice")
    _move_to_end = [k for k in _variable_choices_hourly_wrf if "Instantaneous" in k]
    for k in _move_to_end:
        _variable_choices_hourly_wrf[k] = _variable_choices_hourly_wrf.pop(k)

    # expand this dictionary to also be dependent on LOCA vs WRF:
    _variable_choices_daily_loca = [
        "Temperature",
        "Maximum Relative Humidity",
        "Minimum Relative Humidity",
        "Solar Radiation",
        "Wind Speed",
        "Wind Direction",
        "Precipitation",
    ]
    _variable_choices_hourly_loca = ["Temperature", "Precipitation"]

    _variable_choices_hourly = {
        "Dynamical": _variable_choices_hourly_wrf,
        "Statistical": _variable_choices_hourly_loca,
    }
    _variable_choices_daily = {
        "Dynamical": _variable_choices_hourly_wrf,
        "Statistical": _variable_choices_daily_loca,
    }

    _variable_choices = {
        "hourly": _variable_choices_hourly,
        "daily": _variable_choices_daily,
    }

    # hard-coded options:
    _scenario_choices = {
        "historical": "Historical Climate",
        "": "Historical Reconstruction",
        "ssp245": "SSP 2-4.5 -- Middle of the Road",
        "ssp370": "SSP 3-7.0 -- Business as Usual",
        "ssp585": "SSP 5-8.5 -- Burn it All",
    }

    _resolutions = list(set(e.metadata["nominal_resolution"] for e in _cat.values()))

    _scenario_list = []
    for resolution in _resolutions:
        _temp = list(
            set(
                e.metadata["experiment_id"]
                for e in _cat.values()
                if e.metadata["nominal_resolution"] == resolution
            )
        )
        _temp.sort()  # consistent order
        _scenario_subset = [(_scenario_choices[e], e) for e in _temp]
        _scenario_subset = dict(_scenario_subset)
        _scenario_list.append((resolution, _scenario_subset))
    _scenarios = dict(_scenario_list)
    return {
        "scenarios": _scenarios,
        "resolutions": _resolutions,
        "scenario_choices": _scenario_choices,
        "variable_choices": _variable_choices,
    }
