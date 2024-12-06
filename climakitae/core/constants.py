# constants.py
"""This module defines constants across the codebase"""

WARMING_LEVELS = [0.8, 1.2, 1.5, 2.0, 2.5, 3.0, 4.0]
SSPS = [
    "SSP 2-4.5 -- Middle of the Road",
    "SSP 3-7.0 -- Business as Usual",
    "SSP 5-8.5 -- Burn it All",
]
WRF_BA_MODELS = [
    "WRF_EC-Earth3_r1i1p1f1",
    "WRF_MPI-ESM1-2-HR_r3i1p1f1",
    "WRF_TaiESM1_r1i1p1f1",
    "WRF_MIROC6_r1i1p1f1",
    "WRF_EC-Earth3-Veg_r1i1p1f1",
]
NON_WRF_BA_MODELS = [
    "WRF_FGOALS-g3_r1i1p1f1",
    "WRF_CNRM-ESM2-1_r1i1p1f2",
    "WRF_CESM2_r11i1p1f1",
]
