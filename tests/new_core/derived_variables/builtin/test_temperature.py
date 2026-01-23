import numpy as np

from climakitae.new_core.derived_variables.builtin.temperature import (
    calc_heat_index,
    calc_wind_chill,
    calc_diurnal_temperature_range,
    calc_diurnal_temperature_range_wrf,
    calc_hdd_wrf,
    calc_cdd_wrf,
    calc_hdd_loca,
    calc_cdd_loca,
)


def test_heat_index_below_threshold(wrf_dataset):
    ds = wrf_dataset.copy()
    ds["t2"] = 295.0  # ~71°F, below heat-index threshold
    ds["rh"] = 50.0

    out = calc_heat_index(ds.copy())
    # below threshold heat_index should equal original temperature
    assert np.allclose(out.heat_index.values, ds.t2.values)


def test_heat_index_high_temp_high_rh(wrf_dataset):
    ds = wrf_dataset.copy()
    ds["t2"] = 310.0  # ~98°F
    ds["rh"] = 80.0

    out = calc_heat_index(ds.copy())

    # replicate formula used in module for an expected value
    t_f = (ds.t2 - 273.15) * 9.0 / 5.0 + 32.0
    rh = ds.rh
    hi_simple = 0.5 * (t_f + 61.0 + (t_f - 68.0) * 1.2 + rh * 0.094)

    c1 = -42.379
    c2 = 2.04901523
    c3 = 10.14333127
    c4 = -0.22475541
    c5 = -6.83783e-3
    c6 = -5.481717e-2
    c7 = 1.22874e-3
    c8 = 8.5282e-4
    c9 = -1.99e-6

    hi_full = (
        c1
        + c2 * t_f
        + c3 * rh
        + c4 * t_f * rh
        + c5 * t_f**2
        + c6 * rh**2
        + c7 * t_f**2 * rh
        + c8 * t_f * rh**2
        + c9 * t_f**2 * rh**2
    )

    low_rh_mask = (rh < 13) & (t_f >= 80) & (t_f <= 112)
    adjustment1 = ((13 - rh) / 4) * np.sqrt((17 - np.abs(t_f - 95)) / 17)
    hi_full = np.where(low_rh_mask, hi_full - adjustment1, hi_full)

    high_rh_mask = (rh > 85) & (t_f >= 80) & (t_f <= 87)
    adjustment2 = ((rh - 85) / 10) * ((87 - t_f) / 5)
    hi_full = np.where(high_rh_mask, hi_full + adjustment2, hi_full)

    heat_index_f = np.where(hi_simple < 80, hi_simple, hi_full)
    heat_index_f = np.where(t_f < 80, t_f, heat_index_f)
    expected_k = (heat_index_f - 32.0) * 5.0 / 9.0 + 273.15

    assert np.allclose(out.heat_index.values, np.asarray(expected_k))


def test_wind_chill_formula(wrf_dataset):
    ds = wrf_dataset.copy()
    ds["t2"] = 260.0  # cold
    ds["u10"] = 10.0
    ds["v10"] = 0.0

    out = calc_wind_chill(ds.copy())

    wind_speed_ms = np.sqrt(ds.u10.values**2 + ds.v10.values**2)
    wind_speed_mph = wind_speed_ms * 2.237
    t_f = (ds.t2 - 273.15) * 9.0 / 5.0 + 32.0

    wind_chill_f = (
        35.74
        + 0.6215 * t_f
        - 35.75 * (wind_speed_mph**0.16)
        + 0.4275 * t_f * (wind_speed_mph**0.16)
    )

    valid_mask = (t_f <= 50) & (wind_speed_mph > 3)
    wind_chill_f = np.where(valid_mask, wind_chill_f, t_f)
    expected_k = (wind_chill_f - 32.0) * 5.0 / 9.0 + 273.15

    assert np.allclose(out.wind_chill.values, expected_k)


def test_diurnal_ranges_and_degree_days(loca_dataset, wrf_dataset):
    loca = loca_dataset.copy()
    out_loca = calc_diurnal_temperature_range(loca.copy())
    assert np.allclose(
        out_loca.diurnal_temperature_range.values, (loca.tasmax - loca.tasmin).values
    )

    wrf = wrf_dataset.copy()
    out_wrf = calc_diurnal_temperature_range_wrf(wrf.copy())
    assert np.allclose(
        out_wrf.diurnal_temperature_range_wrf.values, (wrf.t2max - wrf.t2min).values
    )

    # HDD / CDD for WRF using t2
    wrf2 = wrf_dataset.copy()
    wrf2["t2"] = 280.0
    out_hdd = calc_hdd_wrf(wrf2.copy())
    # threshold default ~291.483, so t2=280 -> HDD=1
    assert (out_hdd.HDD_wrf.values == 1).all()

    wrf3 = wrf_dataset.copy()
    wrf3["t2"] = 300.0
    out_cdd = calc_cdd_wrf(wrf3.copy())
    assert (out_cdd.CDD_wrf.values == 1).all()

    # LOCA HDD/CDD using tasmax/tasmin average
    loca2 = loca_dataset.copy()
    # Make average below threshold
    loca2["tasmax"] = 285.0
    loca2["tasmin"] = 280.0
    out_hdd_loca = calc_hdd_loca(loca2.copy())
    assert (out_hdd_loca.HDD_loca.values == 1).all()

    loca3 = loca_dataset.copy()
    loca3["tasmax"] = 300.0
    loca3["tasmin"] = 295.0
    out_cdd_loca = calc_cdd_loca(loca3.copy())
    assert (out_cdd_loca.CDD_loca.values == 1).all()
