"""Compound extreme event indices.

This module provides derived variables for compound extreme event analysis,
including indices for compound heat-drought and heat-rainfall events based on
copula methods.

Derived Variables
-----------------
chtdei
    Compound High Temperature and Drought Event Index.
chtrei
    Compound High Temperature and Rain Event Index.

References
----------
Li et al. (2023) npj Climate and Atmospheric Science
https://doi.org/10.1038/s41612-023-00413-3
"""

import logging

import numpy as np
import xarray as xr
from scipy import stats

from climakitae.new_core.derived_variables.registry import (
    preserve_spatial_metadata,
    register_derived,
)

logger = logging.getLogger(__name__)


def _max_consecutive(binary_series):
    """Calculate maximum consecutive True values in boolean series.

    Parameters
    ----------
    binary_series : np.ndarray
        Boolean array representing event occurrence.

    Returns
    -------
    int
        Maximum number of consecutive True values.
    """
    if not np.any(binary_series):
        return 0

    # Add padding to handle edge cases
    padded = np.concatenate(([False], binary_series, [False]))
    # Find where series changes from False to True and vice versa
    diffs = np.diff(padded.astype(int))
    starts = np.where(diffs == 1)[0]
    ends = np.where(diffs == -1)[0]

    if len(starts) == 0:
        return 0

    return np.max(ends - starts)


def _gringorten_cdf(x):
    """Calculate empirical CDF using Gringorten plotting formula.

    P(x_i) = (m_i - 0.44) / (n + 0.12)

    where m_i is the rank of x_i and n is sample size.

    Parameters
    ----------
    x : np.ndarray
        Input data array.

    Returns
    -------
    np.ndarray
        Empirical CDF values.
    """
    n = len(x)
    ranks = stats.rankdata(x, method="ordinal")
    return (ranks - 0.44) / (n + 0.12)


def _clayton_copula(u, v, theta):
    """Clayton copula: C(u,v) = (u^(-θ) + v^(-θ) - 1)^(-1/θ).

    Parameters
    ----------
    u, v : float or np.ndarray
        Marginal CDF values in [0, 1].
    theta : float
        Clayton copula parameter (must be > 0).

    Returns
    -------
    float or np.ndarray
        Copula value.
    """
    if theta <= 0:
        theta = 0.001  # Avoid numerical issues
    return (u ** (-theta) + v ** (-theta) - 1) ** (-1 / theta)


def _gumbel_copula(u, v, theta):
    """Gumbel copula: C(u,v) = exp(-[(-ln u)^θ + (-ln v)^θ]^(1/θ)).

    Parameters
    ----------
    u, v : float or np.ndarray
        Marginal CDF values in [0, 1].
    theta : float
        Gumbel copula parameter (must be >= 1).

    Returns
    -------
    float or np.ndarray
        Copula value.
    """
    if theta < 1:
        theta = 1.001  # Gumbel requires theta >= 1
    return np.exp(-(((-np.log(u)) ** theta + (-np.log(v)) ** theta) ** (1 / theta)))


def _fit_clayton_copula(u, v):
    """Fit Clayton copula parameter using Kendall's tau method.

    Parameters
    ----------
    u, v : np.ndarray
        Marginal CDF values.

    Returns
    -------
    float
        Estimated Clayton theta parameter.
    """
    tau = stats.kendalltau(u, v)[0]
    # For Clayton: theta = 2*tau / (1 - tau)
    if tau >= 1 or tau <= 0:
        return 0.001
    return 2 * tau / (1 - tau)


def _fit_gumbel_copula(u, v):
    """Fit Gumbel copula parameter using Kendall's tau method.

    Parameters
    ----------
    u, v : np.ndarray
        Marginal CDF values.

    Returns
    -------
    float
        Estimated Gumbel theta parameter.
    """
    tau = stats.kendalltau(u, v)[0]
    # For Gumbel: theta = 1 / (1 - tau)
    if tau >= 1:
        return 1.001
    return 1 / (1 - tau)


@register_derived(
    variable="chtdei",
    query={"variable_id": ["t2max", "prec"]},  # WRF variables
    description="Compound High Temperature and Drought Event Index (bivariate copula-based severity)",
    units="probability",
    source="builtin",
)
def calc_chtdei_chtrei(ds):
    """Calculate Compound High Temperature and Drought/Rain Event Indices.

    This implements the bivariate copula-based severity index from Li et al. (2023)
    for compound extreme events during summer (JJA).

    Parameters
    ----------
    ds : xr.Dataset
        Dataset containing:
        - 't2max' or 'tasmax': Daily maximum temperature (K)
        - 'prec' or 'pr': Daily total precipitation (mm)
        Time dimension should contain daily data spanning multiple summers.

    Returns
    -------
    xr.Dataset
        Dataset with 'chtdei' and 'chtrei' variables added.

    Notes
    -----
    The index uses a bivariate joint probability distribution between:
    - X: Total number of compound event days per summer
    - Y: Maximum duration (consecutive days) of compound events per summer

    The severity index is defined as:
        PI = P(X > x, Y > y) = 1 - F_X(x) - F_Y(y) + C(F_X(x), F_Y(y))

    where C is an Archimedean copula (Clayton for CHTDEI, Gumbel for CHTREI).

    Smaller PI values indicate more severe compound events.

    Compound events are defined as:
    - CHTDE: High temp (>90th percentile) AND low precip (<25th percentile)
    - CHTRE: High temp (>90th percentile) AND high precip (>75th percentile)

    Reference
    ---------
    Li et al. (2023) npj Climate and Atmospheric Science
    https://doi.org/10.1038/s41612-023-00413-3
    """
    logger.debug("Computing CHTDEI and CHTREI from temperature and precipitation")

    # Detect variable names (handle both WRF and LOCA naming)
    if "t2max" in ds:
        tmax = ds.t2max
        tmax_name = "t2max"
    elif "tasmax" in ds:
        tmax = ds.tasmax
        tmax_name = "tasmax"
    else:
        raise ValueError("Dataset must contain 't2max' or 'tasmax' variable")

    if "prec" in ds:
        precip = ds.prec
        precip_name = "prec"
    elif "pr" in ds:
        precip = ds.pr
        precip_name = "pr"
    else:
        raise ValueError("Dataset must contain 'prec' or 'pr' variable")

    # Step 1: Calculate percentile thresholds for JJA period
    logger.debug("Calculating percentile thresholds")
    tmax_p90 = tmax.quantile(0.90, dim="time")
    precip_p25 = precip.quantile(0.25, dim="time")
    precip_p75 = precip.quantile(0.75, dim="time")

    # Identify compound event days
    chtde_days = (tmax > tmax_p90) & (precip < precip_p25)
    chtre_days = (tmax > tmax_p90) & (precip > precip_p75)

    # Step 2: Group by summer and calculate X and Y using xarray groupby
    logger.debug("Grouping by summer seasons")

    # X: Total number of compound event days per year (using groupby)
    chtde_X = chtde_days.groupby("time.year").sum(dim="time")
    chtre_X = chtre_days.groupby("time.year").sum(dim="time")

    # Y: Maximum consecutive days (max duration) per year
    # Need to compute this using apply along time dimension for each year
    def max_consecutive_duration(group):
        """Apply max_consecutive to each spatial location in the group."""
        return xr.apply_ufunc(
            _max_consecutive,
            group,
            input_core_dims=[["time"]],
            vectorize=True,
            dask="parallelized",
            output_dtypes=[float],
        )

    logger.debug("Calculating maximum consecutive day durations")
    chtde_Y = chtde_days.groupby("time.year").apply(max_consecutive_duration)
    chtre_Y = chtre_days.groupby("time.year").apply(max_consecutive_duration)

    # Step 3: Calculate severity indices using copulas
    logger.debug("Fitting copulas and calculating severity indices")

    # Define function to compute copula severity at each location
    def compute_chtdei(x, y):
        """Compute CHTDEI using Clayton copula for a single location time series."""
        x = np.asarray(x, dtype=float)
        y = np.asarray(y, dtype=float)

        # Skip if no events
        if np.all(x == 0) and np.all(y == 0):
            return np.full(x.shape, np.nan, dtype=float)

        # Calculate marginal CDFs
        u = _gringorten_cdf(x)
        v = _gringorten_cdf(y)

        # Fit Clayton copula
        theta = _fit_clayton_copula(u, v)

        # Calculate PI for each year
        result = np.full(x.shape, np.nan, dtype=float)
        for t in range(len(x)):
            C = _clayton_copula(u[t], v[t], theta)
            result[t] = 1 - u[t] - v[t] + C

        return result

    def compute_chtrei(x, y):
        """Compute CHTREI using Gumbel copula for a single location time series."""
        x = np.asarray(x, dtype=float)
        y = np.asarray(y, dtype=float)

        # Skip if no events
        if np.all(x == 0) and np.all(y == 0):
            return np.full(x.shape, np.nan, dtype=float)

        # Calculate marginal CDFs
        u = _gringorten_cdf(x)
        v = _gringorten_cdf(y)

        # Fit Gumbel copula
        theta = _fit_gumbel_copula(u, v)

        # Calculate PI for each year
        result = np.full(x.shape, np.nan, dtype=float)
        for t in range(len(x)):
            C = _gumbel_copula(u[t], v[t], theta)
            result[t] = 1 - u[t] - v[t] + C

        return result

    # Apply copula calculation using xarray's apply_ufunc
    chtdei = xr.apply_ufunc(
        compute_chtdei,
        chtde_X,
        chtde_Y,
        input_core_dims=[["year"], ["year"]],
        output_core_dims=[["year"]],
        vectorize=True,
        dask="parallelized",
        output_dtypes=[float],
        dask_gufunc_kwargs={"allow_rechunk": True},
    )

    chtrei = xr.apply_ufunc(
        compute_chtrei,
        chtre_X,
        chtre_Y,
        input_core_dims=[["year"], ["year"]],
        output_core_dims=[["year"]],
        vectorize=True,
        dask="parallelized",
        output_dtypes=[float],
        dask_gufunc_kwargs={"allow_rechunk": True},
    )

    # Add to dataset
    logger.debug("Adding CHTDEI and CHTREI to dataset")
    ds["chtdei"] = chtdei
    ds["chtrei"] = chtrei

    ds["chtdei"].attrs = {
        "units": "probability",
        "long_name": "Compound High Temperature and Drought Event Index",
        "description": "Joint probability severity index; smaller values = more severe events",
        "copula_type": "Clayton",
        "event_definition": "Temperature >90th percentile AND Precipitation <25th percentile",
        "derived_from": "t2max/tasmax, prec/pr",
        "derived_by": "climakitae (based on Li et al. 2023)",
        "reference": "https://doi.org/10.1038/s41612-023-00413-3",
    }

    ds["chtrei"].attrs = {
        "units": "probability",
        "long_name": "Compound High Temperature and Rain Event Index",
        "description": "Joint probability severity index; smaller values = more severe events",
        "copula_type": "Gumbel",
        "event_definition": "Temperature >90th percentile AND Precipitation >75th percentile",
        "derived_from": "t2max/tasmax, prec/pr",
        "derived_by": "climakitae (based on Li et al. 2023)",
        "reference": "https://doi.org/10.1038/s41612-023-00413-3",
    }

    # Preserve CRS/grid_mapping and related spatial metadata via registry helper
    preserve_spatial_metadata(ds, "chtdei", tmax_name)
    preserve_spatial_metadata(ds, "chtrei", precip_name)

    return ds
