"""Functions for deriving indices"""

import xarray as xr
import numpy as np


def noaa_heat_index(T, RH):
    """Compute the NOAA Heat Index.
    See references for more information on the derivation on this index.

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
    NOAA: https://www.wpc.ncep.noaa.gov/html/heatindex_equation.shtml
    NCAR NCL documentation: https://www.ncl.ucar.edu/Document/Functions/Heat_stress/heat_index_nws.shtml

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
    HI = xr.where((HI < 80) & (HI > 40), low_HI, HI)

    # Following NCAR documentation (see function references), for temperature values less that 40F, the HI is set to the ambient temperature.
    HI = xr.where((T < 40), T, HI)

    # Reassign coordinate attributes
    # For some reason, these get improperly assigned in the xr.where step
    for coord in list(HI.coords):
        HI[coord].attrs = RH[coord].attrs

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
    m_low, m_mid, m_high = _equilibrium_moisture_constant(h=rh_percent, T=t2_F)

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

    # If fosberg index > 100, reset to 100
    FFWI = xr.where(FFWI < 100, FFWI, 100, keep_attrs=True)

    # If fosberg index is negative, set to 0
    FFWI = xr.where(FFWI > 0, FFWI, 0, keep_attrs=True)

    # Reassign coordinate attributes
    # For some reason, these get improperly assigned in the xr.where step
    for coord in list(FFWI.coords):
        FFWI[coord].attrs = t2_F[coord].attrs

    # Add descriptive attributes
    FFWI.name = "Fosberg Fire Weather Index"
    FFWI.attrs["units"] = "[0 to 100]"

    return FFWI


# Define some helper functions
def _equilibrium_moisture_constant(h, T):
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
