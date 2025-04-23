import datetime as dt
import xarray as xr
import param
import pandas as pd

# Remove param's parameter descriptions from docstring because
# ANSI escape sequences in them complicate their rendering
param.parameterized.docstring_describe_params = False
# Docstring signatures are also hard to read and therefore removed
param.parameterized.docstring_signature = False


class TimeSeriesParameters(param.Parameterized):
    """Class of python Param to hold parameters for Time Series.

    Parameters
    ----------
    dataset : xr.DataArray
        Timeseries data
    **params:
        Additional arguments to initialize Param class.

    Attributes
    ----------
    data : xr.DataArray
        The time series data provided to the class.
    anomaly : bool, optional
        True to transform timeseries into anomalies (default True).
    extremes: list[str], optional
        List of extremes quantities to compute (options "Max", "Min", "Percentile").
    num_timesteps: int, optional
        Number of timesteps for rolling mean calculations (default 0).
    percentile: int | float, optional
        Percentile to calculate when using the "Percentile" option in extremes (range 0-1).
    reference_range: tuple[dt.datetime,dt.datetime]
        Reference date range (default 1981-01-01 to 2010-12-31).
    remove_seasonal_cycle: bool, optional
        True to remove the seasonal cycle from the timeseries (default False).
    resample_window: int, optional
        Size of resample window (between 1-30, inclusive).
    separate_seasons: bool, optional
        True to disaggregate into four seasons (default False).
    smoothing: str, optional
        Set to "Running Mean" for smoothing (default "None").

    Methods
    -------
    transform_data(self)
        Transform timeseries dataset using user parameters.

    """

    resample_period = param.Selector(default="years", objects=dict())
    _time_scales = dict(
        [("hours", "h"), ("days", "D"), ("months", "MS"), ("years", "YS-SEP")]
    )

    def __init__(self, dataset: xr.DataArray, **params):
        super().__init__(**params)
        self.data = dataset

        _time_resolution = dataset.attrs["frequency"]
        _time_scales = dict([("months", "MS"), ("years", "YS-SEP")])
        if (_time_resolution == "daily") or (_time_resolution == "hourly"):
            _time_scales["days"] = "D"
        if _time_resolution == "hourly":
            _time_scales["hours"] = "h"
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

    def transform_data(self) -> xr.DataArray:
        """Transform timeseries based on parameters.
        Returns a dataset that has been transformed in the ways that the params
        indicate, ready to plot in the preview window ("view" method of this
        class), or be saved out.

        Returns
        -------
        xr.DataArray
            Transformed result.
        """
        if self.remove_seasonal_cycle:
            to_plot = self.data.groupby("time.month") - self.data.groupby(
                "time.month"
            ).mean("time")
        else:
            to_plot = self.data

        def _get_anom(y: xr.Dataset) -> xr.DataArray:
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

        def _running_mean(y: xr.Dataset) -> xr.DataArray:
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

        def _extremes_da(y: xr.Dataset) -> xr.DataArray:
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


def _update_attrs(
    data_to_output: xr.DataArray, attrs_to_add: dict[str, str]
) -> xr.DataArray:
    """Update DataArray attributes.
    This function updates the attributes of the DataArray being output
    so that it contains new attributes that describe the transforms
    that were performed in the timeseries toolkit.
    Called only in Timeseries.output_current

    Parameters
    ----------
    data_to_output : xr.DataArray
        The attributes of this data array will be modified.
    attrs_to_add : dict[str,str]
        Dictionary containing attributes to modify.

    Returns
    -------
    xr.DataArray
        Modified data array.
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

    Parameters
    ----------
    data : xr.DataArray
        Time series array with no spatial coordinates.

    Attributes
    ----------
    choices: TimeSeriesParameters
        Param object containing time series data and analysis parameters.

    """

    def __init__(self, data: xr.DataArray):
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

    def output_current(self) -> xr.DataArray:
        """Output the current attributes of the class to a DataArray object.
        Allows the data to be easily accessed by the user after modifying the attributes directly in the explore panel, for example.

        Returns
        -------
        xr.DataArray

        """
        to_output = self.choices.transform_data()
        attrs_to_add = dict(self.choices.param.values())
        to_output = _update_attrs(to_output, attrs_to_add)
        return to_output
