"""Backend functions and classes for building the thresholds GUI."""

import pandas as pd
import panel as pn
import param

from climakitae.core.data_interface import DataParametersWithPanes
from climakitae.core.data_load import read_catalog_from_select
from climakitae.util.unit_conversions import convert_units
from climakitae.explore.threshold_tools import (
    get_exceedance_count,
    plot_exceedance_count,
    exceedance_plot_title,
    exceedance_plot_subtitle,
)

# ============ Class and methods for the explore.thresholds() GUI ==============


def _get_threshold_data(self):
    """
    This function pulls data from the catalog and reads it into memory

    Args:
        selections (DataLoaders): object holding user's selections
        cat (intake_esm.core.esm_datastore): catalog

    Returns:
        data (xr.DataArray): data to use for creating postage stamp data

    """
    # Read data from catalog
    data = read_catalog_from_select(self)
    data = data.compute()  # Read into memory
    return data


class ThresholdParameters(DataParametersWithPanes):
    """
    An object that holds the "Data Options" parameters for the
    explore.thresholds panel.
    """

    # Define the params (before __init__ so that we can access them during __init__)
    threshold_direction = param.Selector(
        default="above", objects=["above", "below"], label="Direction"
    )
    threshold_value = param.Number(default=0, label="")
    duration1_length = param.Integer(default=1, bounds=(0, None), label="")
    duration1_type = param.Selector(
        default="hour", objects=["year", "month", "day", "hour"], label=""
    )
    period_length = param.Integer(default=1, bounds=(0, None), label="")
    period_type = param.Selector(
        default="year", objects=["year", "month", "day"], label=""
    )
    group_length = param.Integer(default=1, bounds=(0, None), label="")
    group_type = param.Selector(
        default="hour", objects=["year", "month", "day", "hour"], label=""
    )
    duration2_length = param.Integer(default=1, bounds=(0, None), label="")
    duration2_type = param.Selector(
        default="hour", objects=["year", "month", "day", "hour"], label=""
    )
    smoothing = param.Selector(
        default="None", objects=["None", "Running mean"], label="Smoothing"
    )
    num_timesteps = param.Integer(
        default=10, bounds=(0, None), label="Number of timesteps"
    )

    def __init__(self, *args, **params):
        super().__init__(*args, **params)

        # Selectors defaults
        self.append_historical = False
        self.area_average = "Yes"
        self.resolution = "45 km"
        self.scenario_ssp = ["SSP 3-7.0 -- Business as Usual"]
        self.scenario_historical = []
        self.time_slice = (2020, 2100)
        self.timescale = "hourly"
        self.variable = "Air Temperature at 2m"

        # Location defaults
        self.area_subset = "CA counties"
        self.cached_area = ["Los Angeles County"]

        # Get the underlying dataarray
        self.da = _get_threshold_data(self)

        self.threshold_value = round(self.da.mean().values.item())
        self.param.threshold_value.label = f"Value (units: {self.units})"

    # For reloading plot
    reload_plot = param.Action(
        lambda x: x.param.trigger("reload_plot"), label="Reload Plot"
    )

    # For reloading data
    reload_data = param.Action(
        lambda x: x.param.trigger("reload_data"), label="Reload Data"
    )
    changed_loc_and_var = param.Boolean(default=True)
    changed_units = param.Boolean(default=False)

    @param.depends(
        "area_subset",
        "cached_area",
        "variable",
        watch=True,
    )
    def _updated_bool_loc_and_var(self):
        """Update boolean if any changes were made to the location or variable"""
        self.changed_loc_and_var = True

    @param.depends("units", watch=True)
    def _updated_units(self):
        """Update boolean if a change was made to the units"""
        self.changed_units = True

    @param.depends("reload_data", watch=True)
    def _update_data(self):
        """If the button was clicked and the location, variable, or units were
        changed, reload the data from AWS"""
        if self.changed_loc_and_var:
            self.da = _get_threshold_data(self)
            self.changed_loc_and_var = False
        if self.changed_units:
            self.da = convert_units(da=self.da, selected_units=self.units)
            self.threshold_value = round(self.da.mean().values.item())
            self.param.threshold_value.label = f"Value (units: {self.units})"
            self.changed_units = False

    def transform_data(self):
        return get_exceedance_count(
            self.da,
            threshold_value=self.threshold_value,
            threshold_direction=self.threshold_direction,
            duration1=(self.duration1_length, self.duration1_type),
            period=(self.period_length, self.period_type),
            groupby=(self.group_length, self.group_type),
            duration2=(self.duration2_length, self.duration2_type),
            smoothing=self.num_timesteps if self.smoothing == "Running mean" else None,
        )

    @param.depends("reload_plot", "reload_data", watch=False)
    def view(self):
        try:
            to_plot = self.transform_data()
            obj = plot_exceedance_count(to_plot)
        except Exception as e:
            # Display any raised Errors (instead of plotting) if any of the
            # user specifications are incompatible or not yet implemented.
            return e
        return pn.Column(
            pn.widgets.Button.from_param(
                self.param.reload_plot, button_type="primary", width=150, height=30
            ),
            exceedance_plot_title(to_plot),
            exceedance_plot_subtitle(to_plot),
            obj,
        )

    @param.depends("smoothing")
    def smoothing_card(self):
        """A reactive panel card used by _exceedance_visualize that only
        displays the num_timesteps option if smoothing is selected."""
        if self.smoothing != "None":
            smooth_row = pn.Row(
                self.param.smoothing, self.param.num_timesteps, width=375
            )
        else:
            smooth_row = pn.Row(self.param.smoothing, width=375)
        return pn.Card(smooth_row, title="Smoothing", collapsible=False)

    @param.depends("duration1_length", "duration1_type", watch=False)
    def group_row(self):
        """A reactive row for duration2 options that updates if group is updated"""
        self.group_length = self.duration1_length
        self.group_type = self.duration1_type
        return pn.Row(self.param.group_length, self.param.group_type, width=375)

    @param.depends("group_length", "group_type", watch=False)
    def duration2_row(self):
        """A reactive row for duration2 options that updates if group is updated"""
        self.duration2_length = self.group_length
        self.duration2_type = self.group_type
        return pn.Row(self.param.duration2_length, self.param.duration2_type, width=375)


def _exceedance_visualize(choices, option=1):
    """
    Uses holoviz 'panel' library to display the parameters and view defined for
    exploring exceedance.
    """
    _left_column_width = 375

    if option == 1:
        plot_card = choices.view
    elif option == 2:
        # For show: potential option to display multiple tabs if we want to
        # build this out as a broader GUI app for all threshold tools
        plot_card = pn.Tabs(
            ("Event counts", choices.view),
            ("Return values", pn.Row()),
            ("Return periods", pn.Row()),
        )
    else:
        raise ValueError("Unknown option")

    options_card = pn.Card(
        # Threshold value and direction
        pn.Row(
            choices.param.threshold_direction,
            choices.param.threshold_value,
            width=_left_column_width,
        ),
        # DURATION 1
        "I'm interested in extreme conditions that last for . . .",
        pn.Row(choices.param.duration1_length, choices.param.duration1_type, width=375),
        pn.layout.Divider(margin=(-10, 0, -10, 0)),
        # PERIOD
        "Show me a timeseries of the number of occurences every . . .",
        pn.Row(
            choices.param.period_length,
            choices.param.period_type,
            width=_left_column_width,
        ),
        "Examples: for an annual timeseries, select '1-year'. For a seasonal timeseries, select '3-month'.",
        pn.layout.Divider(margin=(-10, 0, -10, 0)),
        # GROUP
        "Optional aggregation: I'm interested in the number of ___ that contain at least one occurance.",
        choices.group_row,
        # DURATION 2
        "After aggregation, I'm interested in occurances that last for . . .",
        choices.duration2_row,
        title="Threshold event options",
        collapsible=False,
    )

    exceedance_count_panel = pn.Column(
        pn.Spacer(width=15),
        pn.Row(
            pn.Column(options_card, choices.smoothing_card, width=_left_column_width),
            pn.Spacer(width=15),
            pn.Column(plot_card),
        ),
    )
    return exceedance_count_panel


def thresholds_visualize(self, option=1):
    """
    Function for constructing and displaying the explore.thresholds() panel.
    """
    _first_row_height = 300

    data_options_card = pn.Card(
        pn.Row(
            pn.Column(
                pn.widgets.Select.from_param(self.param.variable, name="Data variable"),
                pn.widgets.RadioButtonGroup.from_param(self.param.units),
                pn.widgets.StaticText.from_param(
                    self.param.extended_description, name=""
                ),
                pn.widgets.Button.from_param(
                    self.param.reload_data,
                    button_type="primary",
                    width=150,
                    height=30,
                ),
                width=230,
            ),
            pn.Column(
                self.param.area_subset,
                self.param.latitude,
                self.param.longitude,
                self.param.cached_area,
                width=230,
            ),
            pn.Column(self.map_view, width=180),
        ),
        title="Data Options",
        collapsible=False,
        width=700,
        height=_first_row_height,
    )

    _thresholds_tool_description = (
        "Select a variable of interest, variable units,"
        " and region of interest to the left. Then, use the dropdowns below to"
        " define the extreme event you are interested in. After clicking"
        " 'Reload Plot', the plot on the lower right will show event frequencies"
        " across different simulations. You can also explore how these"
        " frequencies change under different global emissions scenarios. Save a"
        " plot to come back to later by putting your cursor over the lower right"
        " and clicking the save icon."
    )

    description_box = pn.Card(
        _thresholds_tool_description,
        title="About this tool",
        collapsible=False,
        # width = 400,
        height=_first_row_height,
    )

    plot_panel = _exceedance_visualize(self, option)  # display the holoviz panel
    return pn.Column(pn.Row(data_options_card, description_box), plot_panel)
