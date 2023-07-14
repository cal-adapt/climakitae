"""Functions for deriving indices"""

import xarray as xr
import numpy as np


def effective_temp(T):
    """Compute effective temperature
    Effective Temp = (1/2)*(yesterday's effective temp) + (1/2)*(today's actual temp)

    Parameters
    ----------
    T: xr.DataArray
        Daily air temperature in any units

    Returns
    --------
    xr.DataArray
        Effective temperature

    References
    ----------
    https://www.nationalgas.com/document/132516/download
    """
    # Get "yesterday" temp by shifting the time index back one time step (1 day)
    T_yesterday = T.shift(time=1)

    # Derive effective temp
    eft_today = T_yesterday * 0.5 + T * 0.5
    eft_yesterday = eft_today.shift(time=1)
    eft = eft_yesterday * 0.5 + T * 0.5

    # Assign same attributes as input data
    # Or else, the output data will have no attributes :(
    eft.attrs = T.attrs

    return eft


def noaa_heat_index(T, RH):
    """Compute the NOAA Heat Index

    Parameters
    ----------
    T: xr.DataArray
        Temperature in deg F
    RH: xr.DataArray
        Relative Humidity in percentage (0-100)

    Returns
    --------
    xr.DataArray
        Heat index per timestep

    References
    -----------
    https://www.wpc.ncep.noaa.gov/html/heatindex_equation.shtml

    """
    T = T.reindex_like(RH)  # Need to have the same dimension/coordinate orders
    HI = (
        -42.379
        + 2.04901523 * T
        + 10.14333127 * RH
        - 0.22475541 * T * RH
        - 0.00683783 * T * T
        - 0.05481717 * RH * RH
        + 0.00122874 * T * T * RH
        + 0.00085282 * T * RH * RH
        - 0.00000199 * T * T * RH * RH
    )

    # Adjust for high temperature, low relative humidity
    # 80 < T < 112 (deg F)
    # RH < 13%
    adj_highT_lowRH = ((13 - RH) / 4) * ((17 - abs(T - 95)) / 17) ** (
        1 / 2
    )  # Adjustment
    HI_highT_lowRH = HI - adj_highT_lowRH  # Subtract adjustment from HI

    # Adjust for low temperature, high relative humidity
    # 80 < T < 87 (deg F)
    # RH > 85%
    adj_lowT_highRH = ((RH - 85) / 10) * ((87 - T) / 5)  # Adjustment
    HI_lowT_highRH = HI + adj_lowT_highRH  # Add adjustment from HI

    # Use different equation if heat index if the heat index value < 80
    low_HI = 0.5 * (T + 61.0 + ((T - 68.0) * 1.2) + (RH * 0.094))

    # Adjust heat index depending on different condions for RH, T, and valid range of HI
    HI = xr.where((RH < 13) & (T > 80) & (T < 112), HI_highT_lowRH, HI)
    HI = xr.where(((RH > 85) & (T < 87) & (T > 80)), HI_lowT_highRH, HI)
    HI = xr.where(HI < 80, low_HI, HI)

    # Assign units attribute
    HI.attrs["units"] = "degF"
    return HI


## ========== FOSBERG FIRE INDEX AND RELATED HELPER FUNCTIONS ==========


def fosberg_fire_index(t2_F, rh_percent, windspeed_mph):
    """Compute the Fosberg Fire Weather Index.
    Use hourly weather as inputs.
    Ensure that the input variables are in the correct units (see below).

    Parameters
    ----------
    t2_F: xr.DataArray
        Air temperature in units of Fahrenheit
    rh_percent: xr.DataArray
        Relative humidity in units of 0-100 (percent)
    windspeed_mph: xr.DataArray
        Windspeed in units of miles per hour

    Returns
    -------
    xr.DataArray
        Fosberg Fire Weather Index computed for each grid cell

    References
    ----------
    https://a.atmos.washington.edu/wrfrt/descript/definitions/fosbergindex.html
    https://github.com/sharppy/SHARPpy/blob/main/sharppy/sharptab/fire.py
    https://www.spc.noaa.gov/exper/firecomp/INFO/fosbinfo.html

    """
    # Compute the equilibrium moisture constant
    m_low, m_mid, m_high = _equilibirum_moisture_constant(h=rh_percent, T=t2_F)

    # For RH < 10%, use the low m value.
    # For RH >= 10%, use the mid value
    m = xr.where(rh_percent < 10, m_low, m_mid)
    # For RH > 50%, use the high m value.
    m = xr.where(rh_percent > 50, m_high, m)

    # Compute the moisture dampening coefficient
    n = _moisture_dampening_coeff(m)

    # Compute the index
    U = windspeed_mph
    FFWI = (n * ((1 + U**2) ** 0.5)) / 0.3002

    # Add descriptive attributes
    FFWI.attrs["units"] = "[0-100]"

    return FFWI


# Define some helper functions
def _equilibirum_moisture_constant(h, T):
    """Compute the equilibrium moisture constant.
    Dependent on relative humidity percent.
    Used to compute Fosberg Fire Weather Index.
    Will return three values corresponding to the level of humidity.

    Args:
        h (xr.DataArray): relative humidity in units of 0-100 (percent)
        T (xr.DataArray): air temperature in units of Fahrenheit

    Returns:
        m_low (xr.DataArray): equilibrium moisture constant for low humidity (<10%)
        m_mid (xr.DataArray): equilibrium moisture constant for 10% < humidity <= 50%
        m_high (xr.DataArray): equilibrium moisture constant for high humidity (>50%)

    """
    # h < 10: Low humidity
    m_low = 0.03229 + 0.281073 * h - 0.000578 * h * T

    # (10 < h <= 50): Mid humiditiy
    m_mid = 2.22749 + 0.160107 * h - 0.01478 * T

    # h > 50: High humidity
    m_high = 21.0606 + 0.005565 * (h**2) - 0.00035 * h * T - 0.483199 * h

    return (m_low, m_mid, m_high)


def _moisture_dampening_coeff(m):
    """Compute the moisture dampening coefficient.
    Used to compute Fosberg Fire Weather Index.

    Args:
        m (xr.DataArray): equilibrium moisture constant

    """
    n = 1 - 2 * (m / 30) + 1.5 * (m / 30) ** 2 - 0.5 * (m / 30) ** 3
    return n
