"""Tests for :mod:`climakitae.visualize.plots`."""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")

import numpy as np
import pytest
import xarray as xr
from matplotlib.axes import Axes

from climakitae.visualize import (
    AreaPlot,
    ClimatologyPlot,
    TimeSeriesPlot,
    create_plot,
    get_plot_class,
)


@pytest.fixture(name="area_dataset")
def _area_dataset_fixture() -> xr.Dataset:
    """Build a small dataset for area and timeseries plotting tests."""
    time = np.arange(5)
    y = np.arange(3)
    x = np.arange(4)
    values = np.arange(5 * 3 * 4, dtype=float).reshape(5, 3, 4)
    return xr.Dataset(
        {"u10": (("time", "y", "x"), values)},
        coords={"time": time, "y": y, "x": x},
    )


@pytest.fixture(name="climatology_dataset")
def _climatology_dataset_fixture() -> xr.Dataset:
    """Build a small dataset for warming-level plotting tests."""
    warming_level = np.array([0.8, 1.5, 2.0, 2.5])
    sim = np.array(["a", "b"])
    y = np.arange(2)
    x = np.arange(2)
    values = np.arange(4 * 2 * 2 * 2, dtype=float).reshape(4, 2, 2, 2)
    return xr.Dataset(
        {"t2max": (("warming_level", "sim", "y", "x"), values)},
        coords={"warming_level": warming_level, "sim": sim, "y": y, "x": x},
    )


def test_area_plot_renders(area_dataset: xr.Dataset) -> None:
    """AreaPlot renders a 2D field and returns matplotlib axes."""
    ax = AreaPlot(area_dataset, variable="u10", x="x", y="y").render()
    assert isinstance(ax, Axes)


def test_timeseries_plot_renders(area_dataset: xr.Dataset) -> None:
    """TimeSeriesPlot reduces non-time dimensions and renders a line."""
    ax = TimeSeriesPlot(area_dataset, y="u10").render()
    assert isinstance(ax, Axes)
    assert len(ax.lines) == 1


def test_climatology_plot_renders(climatology_dataset: xr.Dataset) -> None:
    """ClimatologyPlot infers warming_level x-axis and renders a line."""
    ax = ClimatologyPlot(climatology_dataset, y="t2max").render()
    assert isinstance(ax, Axes)
    assert len(ax.lines) == 1


def test_area_plot_validates_axes_names(area_dataset: xr.Dataset) -> None:
    """AreaPlot fails fast when configured x/y axes are absent."""
    with pytest.raises(ValueError, match="x='lon'"):
        AreaPlot(area_dataset, variable="u10", x="lon", y="lat").render()


def test_climatology_plot_requires_warming_axis(area_dataset: xr.Dataset) -> None:
    """ClimatologyPlot raises when warming-level coordinate is missing."""
    with pytest.raises(ValueError, match="could not infer warming-level axis"):
        ClimatologyPlot(area_dataset, y="u10").render()


def test_get_plot_class_retrieves_known_types() -> None:
    """Class lookup works for public registry names."""
    assert get_plot_class("AreaPlot") is AreaPlot
    assert get_plot_class("TimeSeriesPlot") is TimeSeriesPlot
    assert get_plot_class("ClimatologyPlot") is ClimatologyPlot


def test_get_plot_class_rejects_unknown_type() -> None:
    """Class lookup raises for unknown names."""
    with pytest.raises(ValueError, match="unknown plot type"):
        get_plot_class("UnknownPlot")


def test_create_plot_constructs_expected_subclass(area_dataset: xr.Dataset) -> None:
    """Factory returns the expected concrete class."""
    plot = create_plot("TimeSeriesPlot", area_dataset, y="u10")
    assert isinstance(plot, TimeSeriesPlot)
