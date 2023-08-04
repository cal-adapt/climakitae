import panel as pn
from shapely.geometry import box, Polygon
from matplotlib.figure import Figure
import matplotlib.ticker as ticker
import cartopy.crs as ccrs
import cartopy.feature as cfeature

from climakitae.core.data_interface import DataParametersWithPanes


class Select:
    def __init__(self):
        self.data_parameters = DataParametersWithPanes()

    def show(self):
        # Show panel visualually
        select_panel = _display_select(self.data_parameters)
        return select_panel


def _selections_param_to_panel(selections):
    """For the _DataSelector object, get parameters and parameter
    descriptions formatted as panel widgets
    """
    area_subset = pn.widgets.Select.from_param(
        selections.param.area_subset, name="Subset the data by..."
    )
    area_average_text = pn.widgets.StaticText(
        value="Compute an area average across grid cells within your selected region?",
        name="",
    )
    area_average = pn.widgets.RadioBoxGroup.from_param(
        selections.param.area_average, inline=True
    )
    cached_area = pn.widgets.Select.from_param(
        selections.param.cached_area, name="Location selection"
    )
    data_type_text = pn.widgets.StaticText(
        value="",
        name="Data type",
    )
    data_type = pn.widgets.RadioBoxGroup.from_param(
        selections.param.data_type, inline=True, name=""
    )
    data_warning = pn.widgets.StaticText.from_param(
        selections.param._data_warning, name="", style={"color": "red"}
    )
    downscaling_method_text = pn.widgets.StaticText(value="", name="Downscaling method")
    downscaling_method = pn.widgets.CheckBoxGroup.from_param(
        selections.param.downscaling_method, inline=True
    )
    historical_selection_text = pn.widgets.StaticText(
        value="<br>Estimates of recent historical climatic conditions",
        name="Historical Data",
    )
    historical_selection = pn.widgets.CheckBoxGroup.from_param(
        selections.param.scenario_historical
    )
    station_data_info = pn.widgets.StaticText.from_param(
        selections.param._station_data_info, name="", style={"color": "red"}
    )
    ssp_selection_text = pn.widgets.StaticText(
        value="<br> Shared Socioeconomic Pathways (SSPs) represent different global emissions scenarios",
        name="Future Model Data",
    )
    ssp_selection = pn.widgets.CheckBoxGroup.from_param(selections.param.scenario_ssp)
    resolution_text = pn.widgets.StaticText(
        value="",
        name="Model Grid-Spacing",
    )
    resolution = pn.widgets.RadioBoxGroup.from_param(
        selections.param.resolution, inline=False
    )
    timescale_text = pn.widgets.StaticText(value="", name="Timescale")
    timescale = pn.widgets.RadioBoxGroup.from_param(
        selections.param.timescale, name="", inline=False
    )
    time_slice = pn.widgets.RangeSlider.from_param(selections.param.time_slice, name="")
    units_text = pn.widgets.StaticText(name="Variable Units", value="")
    units = pn.widgets.RadioBoxGroup.from_param(selections.param.units, inline=False)
    variable = pn.widgets.Select.from_param(selections.param.variable, name="")
    variable_text = pn.widgets.StaticText(name="Variable", value="")
    variable_description = pn.widgets.StaticText.from_param(
        selections.param.extended_description, name=""
    )

    widgets_dict = {
        "area_average": area_average,
        "area_subset": area_subset,
        "cached_area": cached_area,
        "data_type": data_type,
        "data_type_text": data_type_text,
        "data_warning": data_warning,
        "downscaling_method": downscaling_method,
        "historical_selection": historical_selection,
        "latitude": selections.param.latitude,
        "longitude": selections.param.longitude,
        "resolution": resolution,
        "station_data_info": station_data_info,
        "ssp_selection": ssp_selection,
        "resolution": resolution,
        "timescale": timescale,
        "time_slice": time_slice,
        "units": units,
        "variable": variable,
        "variable_description": variable_description,
    }
    text_dict = {
        "area_average_text": area_average_text,
        "downscaling_method_text": downscaling_method_text,
        "historical_selection_text": historical_selection_text,
        "resolution_text": resolution_text,
        "ssp_selection_text": ssp_selection_text,
        "units_text": units_text,
        "timescale_text": timescale_text,
        "variable_text": variable_text,
    }

    return widgets_dict | text_dict


def _display_select(selections):
    """
    Called by 'select' at the beginning of the workflow, to capture user
    selections. Displays panel of widgets from which to make selections.
    Modifies 'selections' object, which is used by retrieve() to build an
    appropriate xarray Dataset.
    """
    # Get formatted panel widgets for each parameter
    widgets = _selections_param_to_panel(selections)

    data_choices = pn.Column(
        widgets["variable_text"],
        widgets["variable"],
        widgets["variable_description"],
        pn.Row(
            pn.Column(
                widgets["historical_selection_text"],
                widgets["historical_selection"],
                widgets["ssp_selection_text"],
                widgets["ssp_selection"],
                pn.Column(
                    selections.scenario_view,
                    widgets["time_slice"],
                    width=220,
                ),
                width=250,
            ),
            pn.Column(
                widgets["units_text"],
                widgets["units"],
                widgets["timescale_text"],
                widgets["timescale"],
                widgets["resolution_text"],
                widgets["resolution"],
                widgets["station_data_info"],
                width=150,
            ),
        ),
        width=380,
    )

    col_1_location = pn.Column(
        selections.map_view,
        widgets["area_subset"],
        widgets["cached_area"],
        widgets["latitude"],
        widgets["longitude"],
        widgets["area_average_text"],
        widgets["area_average"],
        width=220,
    )
    col_2_location = pn.Column(
        pn.Spacer(height=10),
        pn.widgets.StaticText(
            value="",
            name="Weather station",
        ),
        pn.widgets.CheckBoxGroup.from_param(selections.param.station, name=""),
        width=270,
    )
    loc_choices = pn.Row(col_1_location, col_2_location)

    everything_else = pn.Row(data_choices, pn.layout.HSpacer(width=10), loc_choices)

    # Panel overall structure:
    all_things = pn.Column(
        pn.Row(
            pn.Column(
                widgets["data_type_text"],
                widgets["data_type"],
                width=150,
            ),
            pn.Column(
                widgets["downscaling_method_text"],
                widgets["downscaling_method"],
                width=270,
            ),
            pn.Column(
                widgets["data_warning"],
                width=400,
            ),
        ),
        pn.Spacer(background="black", height=1),
        everything_else,
    )

    return pn.Card(
        all_things,
        title="Choose Data Available with the Cal-Adapt Analytics Engine",
        collapsible=False,
    )
