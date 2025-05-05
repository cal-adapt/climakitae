# constants.py
"""This module defines constants across the codebase"""

# Sentinel for unset values
# This is used to differentiate between a value that is set to None
# and a value that is not set at all.
UNSET = object()

# global warming levels available on AE
WARMING_LEVELS = [0.8, 1.0, 1.2, 1.5, 2.0, 2.5, 3.0, 4.0]

# shared socioeconomic pathways (IPCC)
SSPS = [
    "SSP 2-4.5",
    "SSP 3-7.0",
    "SSP 5-8.5",
]

# WRF models that have a-priori bias adjustment
WRF_BA_MODELS = [
    "WRF_EC-Earth3_r1i1p1f1",
    "WRF_MPI-ESM1-2-HR_r3i1p1f1",
    "WRF_TaiESM1_r1i1p1f1",
    "WRF_MIROC6_r1i1p1f1",
    "WRF_EC-Earth3-Veg_r1i1p1f1",
]

# WRF models that do not have a-priori bias adjustment
NON_WRF_BA_MODELS = [
    "WRF_FGOALS-g3_r1i1p1f1",
    "WRF_CNRM-ESM2-1_r1i1p1f2",
    "WRF_CESM2_r11i1p1f1",
]

# WRF models that do not reach 0.8°C GWL
WRF_NO_0PT8_GWL_MODELS = ["WRF_EC-Earth3-Veg_r1i1p1f1_historical+ssp370"]

# LOCA models that do not reach 0.8°C GWL
LOCA_NO_0PT8_GWL_MODELS = [
    "LOCA2_EC-Earth3_r4i1p1f1_historical+ssp245",
    "LOCA2_EC-Earth3_r4i1p1f1_historical+ssp370",
    "LOCA2_EC-Earth3_r4i1p1f1_historical+ssp585",
    "LOCA2_EC-Earth3-Veg_r3i1p1f1_historical+ssp245",
    "LOCA2_EC-Earth3-Veg_r3i1p1f1_historical+ssp370",
    "LOCA2_EC-Earth3-Veg_r3i1p1f1_historical+ssp585",
    "LOCA2_EC-Earth3-Veg_r5i1p1f1_historical+ssp245",
]
