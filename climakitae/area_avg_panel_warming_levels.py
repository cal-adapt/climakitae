"""
This script produces an area-avaeraged time plot for all available models for the Cal-Adapt: Analytics Engine Warming Levels
notebok in development.
It is the first panel in the full panel figure, with the GMT context plot provided below it (ipcc_spm_fig8_recreate.py).

Effevtive workflow:
When in the Warming Levels notebook, both the full spatial data and the area-averaged data should be read in.
"""


## Import libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import xarray as xr
import numpy as np

## Pre-selected data: 2m SAT, monthly time scale, degC

## Data needs to be pre-selected at monthly timescale
weights = np.cos(np.deg2rad(data.lat))
area_avg_data = data.weighted(weights).mean("x").mean("y")

## Smoothing with a running mean over 30 year intervals
def _running_mean(y):
    return y.rolling(time=30*12, center=True).mean("time")


## Obtaining the area_averaged data for the line plots - from data_loaders.py
## Need to keep full spatial data separate for postage stamp figures

to_plot = _running_mean(area_avg_data)
to_plot.hvplot.line(
    x="time", y="Air Temperature at 2m", by="simulation", legend="below"
)


## This keeps crashing my kernel -- in development
