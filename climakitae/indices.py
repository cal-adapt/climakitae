"""Functions for deriving fire indices"""

import xarray as xr


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

    # For RH > 10%, use the low m value.
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
    FFWI.name = "Fosberg Fire Weather Index"
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
