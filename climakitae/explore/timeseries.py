import datetime as dt
import xarray as xr
import param
import panel as pn
import hvplot.xarray
import pandas as pd


class TimeSeriesParameters(param.Parameterized):
    """Class to hold TimeSeries params

    An object to hold time-series params, which depends only on the 'param'
    library. Currently used in 'timeseries_visualize', which uses 'panel' to
    draw the GUI, but another UI could in principle be used to update these
    parameters instead.
    """

    resample_period = param.Selector(default="years", objects=dict())
    _time_scales = dict(
        [("hours", "H"), ("days", "D"), ("months", "MS"), ("years", "AS-SEP")]
    )

    def __init__(self, dataset, **params):
        super().__init__(**params)
        self.data = dataset

        _time_resolution = dataset.attrs["frequency"]
        _time_scales = dict([("months", "MS"), ("years", "AS-SEP")])
        if (_time_resolution == "daily") or (_time_resolution == "hourly"):
            _time_scales["days"] = "D"
        if _time_resolution == "hourly":
            _time_scales["hours"] = "H"
        self._time_scales = _time_scales
        self.param["resample_period"].objects = self._time_scales

    anomaly = param.Boolean(default=True, label="Difference from a historical mean")
    reference_range = param.CalendarDateRange(
        default=(dt.datetime(1981, 1, 1), dt.datetime(2010, 12, 31)),
        bounds=(dt.datetime(1980, 1, 1), dt.datetime(2021, 12, 31)),
    )
    remove_seasonal_cycle = param.Boolean(default=False)
    smoothing = param.Selector(default="None", objects=["None", "Running Mean"])
    num_timesteps = param.Integer(default=0, bounds=(0, 240))
    separate_seasons = param.Boolean(
        default=False, label="Disaggregate into four seasons"
    )

    extremes = param.ListSelector(default=[], objects=["Min", "Max", "Percentile"])
    resample_window = param.Integer(default=1, bounds=(1, 30))
    percentile = param.Number(
        default=0,
        bounds=(0, 1),
        step=0.01,
        doc="Relevant if 'running extremes' is 'percentile.",
    )

    @param.depends("anomaly", watch=True)
    def update_seasonal_cycle(self):
        if not self.anomaly:
            self.remove_seasonal_cycle = False

    @param.depends("remove_seasonal_cycle", watch=True)
    def update_anom(self):
        if self.remove_seasonal_cycle:
            self.anomaly = True

    def transform_data(self):
        """
        Returns a dataset that has been transformed in the ways that the params
        indicate, ready to plot in the preview window ("view" method of this
        class), or be saved out.
        """
        if self.remove_seasonal_cycle:
            to_plot = self.data.groupby("time.month") - self.data.groupby(
                "time.month"
            ).mean("time")
        else:
            to_plot = self.data

        def _get_anom(y):
            """
            Returns the difference with respect to the average across a historical range.
            """
            if y.attrs["frequency"] == "1month":
                # If frequency is monthly, then the reference period average needs to be a
                # weighted average, with weights equal to the number of days in each month
                reference_slice = y.sel(time=slice(*self.reference_range))
                month_weights = (
                    reference_slice.time.dt.daysinmonth
                )  # Number of days in each month of the reference range
                reference_avg = reference_slice.weighted(month_weights).mean(
                    "time"
                )  # Calculate the weighted average of this period
                return y - reference_avg  # return the difference
            else:
                return y - y.sel(time=slice(*self.reference_range)).mean("time")

        def _running_mean(y):
            # If timescale is monthly, need to weight the rolling average by the number of days in each month
            if y.attrs["frequency"] == "1month":
                # Access the number of days in each month corresponding to each element of y
                month_weights = y.time.dt.daysinmonth

                # Construct DataArrayRolling objects for both the data and the weights
                rolling_y = y.rolling(time=self.num_timesteps, center=True).construct(
                    "window"
                )
                rolling_weights = month_weights.rolling(
                    time=self.num_timesteps, center=True
                ).construct("window")

                # Build a DataArrayWeighted and collapse across the window dimension with mean
                result = rolling_y.weighted(rolling_weights.fillna(0)).mean(
                    "window", skipna=False
                )
                return result
            else:
                return y.rolling(time=self.num_timesteps, center=True).mean("time")

        if self.anomaly and not self.separate_seasons:
            to_plot = _get_anom(to_plot)
        elif self.anomaly and self.separate_seasons:
            to_plot = to_plot.groupby("time.season").map(_get_anom)
        else:
            to_plot = to_plot

        if self.smoothing == "Running Mean" and self.num_timesteps > 0:
            if self.separate_seasons:
                to_plot = to_plot.groupby("time.season").map(_running_mean)
            else:
                to_plot = _running_mean(to_plot)

        def _extremes_da(y):
            plot_multiple = xr.Dataset()
            if "Max" in self.extremes:
                plot_multiple["max"] = to_plot.resample(
                    time=str(self.resample_window)
                    + self._time_scales[self.resample_period]
                ).max("time")
            if "Min" in self.extremes:
                plot_multiple["min"] = to_plot.resample(
                    time=str(self.resample_window)
                    + self._time_scales[self.resample_period]
                ).min("time")
            if "Percentile" in self.extremes:
                plot_multiple[str(self.percentile) + " percentile"] = to_plot.resample(
                    time=str(self.resample_window)
                    + self._time_scales[self.resample_period]
                ).quantile(q=self.percentile)
            return plot_multiple.to_array("extremes")

        if self.extremes != []:
            return _extremes_da(to_plot)
        else:
            return to_plot

    @param.depends(
        "anomaly",
        "reference_range",
        "separate_seasons",
        "smoothing",
        "num_timesteps",
        "remove_seasonal_cycle",
        "extremes",
        "resample_window",
        "resample_period",
        "percentile",
        watch=False,
    )
    def view(self):
        """
        Does the main work of timeseries.explore(). Updating a plot in real-time
        to enable the user to preview the results of any timeseries transforms.
        """
        to_plot = self.transform_data()

        if self.separate_seasons:
            menu_list = ["scenario", "time.season"]
        else:
            menu_list = ["scenario"]

        # Resample period user-friendly (used in title)
        resample_per_str = str(self.resample_period)[
            :-1
        ]  # Remove plural (i.e. "months" --> "month")

        # Percentile string user-friendly (used in title)
        percentile_int = int(self.percentile * 100)
        ordinal = lambda n: "%d%s" % (
            n,
            "tsnrhtdd"[(n // 10 % 10 != 1) * (n % 10 < 4) * n % 10 :: 4],
        )
        percentrile_str = ordinal(percentile_int)

        # Smoothing string user-friendly (used in title)
        if self.smoothing == "Running Mean":
            smoothing_str = "Smoothed "
        else:
            smoothing_str = ""

        # Get start and end years of input data
        # Use that to build a title
        pd_datetime = pd.DatetimeIndex(self.data.time.values)
        year1, year2 = str(pd_datetime[0].year), str(pd_datetime[-1].year)
        new_title = smoothing_str + "Difference for " + year1 + " - " + year2

        if self.extremes == []:
            plot_by = "simulation"
            if self.anomaly:
                if (
                    self.smoothing == "Running Mean"
                ):  # Smoothed, anomaly timeseries, no extremes
                    new_title = (
                        smoothing_str
                        + "Difference for ".lower()
                        + year1
                        + " - "
                        + year2
                        + " with a "
                        + str(self.num_timesteps)
                        + " timesteps running mean"
                    )
                else:  # Unsmoothed, anomaly timeseries, no extremes
                    new_title = (
                        smoothing_str + "Difference for " + year1 + " - " + year2
                    )
            else:
                if (
                    self.smoothing == "Running Mean"
                ):  # Smoothed, timeseries, no extremes
                    new_title = (
                        smoothing_str
                        + "Timeseries for ".lower()
                        + year1
                        + " - "
                        + year2
                        + " with a "
                        + str(self.num_timesteps)
                        + " timesteps running mean"
                    )
                else:  # Unsmoothed, timeseries, no extremes
                    new_title = (
                        smoothing_str + "Timeseries for " + year1 + " - " + year2
                    )

        elif self.extremes != []:
            plot_by = ["simulation", "extremes"]
            if self.smoothing == "None":
                if self.extremes == "Percentile":  # Unsmoothed, percentile extremes
                    new_title = (
                        smoothing_str
                        + percentrile_str
                        + " percentile extremes over a "
                        + str(self.resample_window)
                        + "-"
                        + resample_per_str
                        + " window"
                    )
                else:  # Unsmoothed, min/max/mean extremes
                    new_title = (
                        smoothing_str
                        + "Extremes over a "
                        + str(self.resample_window)
                        + "-"
                        + resample_per_str
                        + " window"
                    )
            elif self.smoothing != "None":
                if self.extremes == "Percentile":  # Smoothed, percentile extremes
                    new_title = (
                        smoothing_str
                        + percentrile_str
                        + " percentile extremes over a "
                        + str(self.resample_window)
                        + "-"
                        + resample_per_str
                        + " window"
                    )
                else:  # Smoothed, min/max/mean extremes
                    new_title = (
                        smoothing_str
                        + "Extremes over a "
                        + str(self.resample_window)
                        + "-"
                        + resample_per_str
                        + " window"
                    )

        obj = to_plot.hvplot.line(
            x="time",
            widget_location="bottom",
            by=plot_by,
            groupby=menu_list,
            title=new_title,
        )
        return obj


def timeseries_visualize(choices):
    """
    Uses holoviz 'panel' library to display the parameters and view defined in
    an instance of _TimeSeriesParams.
    """
    smooth_text = "Smoothing applies a running mean to remove noise from the data."
    resample_text = "The resample window and period define the length of time over which to calculate the extreme."

    return pn.Column(
        pn.Row(
            pn.Column(
                pn.widgets.StaticText(name="", value="Transformation Options"),
                choices.param.anomaly,
                choices.param.reference_range,
                choices.param.remove_seasonal_cycle,
                choices.param.separate_seasons,
                choices.param.smoothing,
                choices.param.num_timesteps,
                pn.Spacer(height=10),
            ),
            pn.Spacer(width=50),
            pn.Column(
                pn.widgets.CheckBoxGroup.from_param(choices.param.extremes),
                choices.param.percentile,
                pn.Row(
                    choices.param.resample_window,
                    choices.param.resample_period,
                    width=320,
                ),
                pn.widgets.StaticText(name="", value=smooth_text),
                pn.widgets.StaticText(name="", value=resample_text),
            ),
        ),
        choices.view,
    )


def _update_attrs(data_to_output, attrs_to_add):
    """
    This function updates the attributes of the DataArray being output
    so that it contains new attributes that describe the transforms
    that were performed in the timeseries toolkit.
    Called only in Timeseries.output_current
    """
    attributes = data_to_output.attrs
    attrs_to_add.pop("name")
    attrs_to_add.pop("separate_seasons")
    if attrs_to_add["extremes"] != "percentile":
        attrs_to_add.pop("percentile")
    if attrs_to_add["extremes"] == "None":
        attrs_to_add.pop("resample_period")
        attrs_to_add.pop("resample_window")
    if attrs_to_add["smoothing"] != "None":
        attrs_to_add["smoothing_timesteps"] = attrs_to_add["num_timesteps"]
    attrs_to_add.pop("num_timesteps")
    if not attrs_to_add["anomaly"]:
        attrs_to_add.pop("reference_range")
    attrs_to_add = {
        "timeseries: " + k: (str(v) if type(v) == bool or None else v)
        for k, v in attrs_to_add.items()
    }

    datefmt = "%b %d %Y (%H:%M)"
    for att, v in attrs_to_add.items():
        if type(v) == tuple:
            if (type(v[0])) == dt.datetime:
                dates = [atti.strftime(datefmt) for atti in v]
                date_str = " - ".join(dates)
                attrs_to_add[att] = date_str

    attributes.update(attrs_to_add)
    data_to_output.attrs = attributes
    return data_to_output


class TimeSeries:
    """
    Holds the instance of TimeSeriesParameters that is used for the following purposes:
    1) to display a panel that previews various time-series transforms (explore), and
    2) to save the transform represented by the current state of that preview into a new variable (output_current).
    """

    def __init__(self, data):
        if (
            type(data) != xr.core.dataarray.DataArray
        ):  # Data is NOT in the form of xr.DataArray
            raise ValueError(
                "Please pass an xarray DataArray (e.g. as output by DataParameters.retrieve())."
            )
        else:
            raise_error = False
            error_message = ""
            if "lat" in data.coords:  # Data is NOT area averaged
                raise_error = True
                error_message += "Please pass a timeseries (area average)."
            if not any(
                ["Historical + " in v for v in data.scenario.values]
            ):  # Append historical = False
                if raise_error == True:
                    error_message += "\n"
                else:
                    raise_error = True
                error_message += (
                    "Please append the historical period in your data retrieval."
                )
            if raise_error:  # If any errors
                raise ValueError(error_message)

        self.choices = TimeSeriesParameters(data)

    def explore(self):
        """Create an interactive visualization of the timeseries data, dependant on the attributes set in previous steps. Allows user to directly modify the data in the GUI. Only works in a jupyter notebook environment.

        Returns
        -------
        panel.layout.base.Column

        """
        return timeseries_visualize(self.choices)

    def output_current(self):
        """Output the current attributes of the class to a DataArray object.
        Allows the data to be easily accessed by the user after modifying the attributes directly in the explore panel, for example.

        Returns
        -------
        xr.DataArray

        """
        to_output = self.choices.transform_data()
        attrs_to_add = dict(self.choices.get_param_values())
        to_output = _update_attrs(to_output, attrs_to_add)
        return to_output