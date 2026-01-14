# constants.py
"""This module defines constants across the codebase"""


class _UnsetType:
    """Singleton sentinel for unset values.

    This class implements __deepcopy__ to return itself, ensuring that
    `copy.deepcopy(UNSET) is UNSET` remains True. This is critical for
    thread-safe query snapshots in ClimateData.get().
    """

    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __deepcopy__(self, memo):
        # Return the singleton instance, not a copy
        return self

    def __copy__(self):
        # Return the singleton instance, not a copy
        return self

    def __repr__(self):
        return "UNSET"

    def __reduce__(self):
        # Support pickling by returning the class for reconstruction
        return (self.__class__, ())


# Sentinel for unset values
# This is used to differentiate between a value that is set to None
# and a value that is not set at all.
UNSET = _UnsetType()

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
    "WRF_ENSMEAN_r11i1p1f1",
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

# Constant Keys for User Interface
_NEW_ATTRS_KEY = "new_attrs"
PROC_KEY = "processes"

# Constant Keys for Data Catalog
CATALOG_CADCAT = "cadcat"
CATALOG_REN_ENERGY_GEN = "renewable energy generation"
CATALOG_BOUNDARY = "boundary"
CATALOG_HDP = "hdp"

# Boundary Data Constants
WESTERN_STATES_LIST = ["CA", "NV", "OR", "WA", "UT", "MT", "ID", "AZ", "CO", "NM", "WY"]

PRIORITY_UTILITIES = [
    "Pacific Gas & Electric Company",
    "San Diego Gas & Electric",
    "Southern California Edison",
    "Los Angeles Department of Water & Power",
    "Sacramento Municipal Utility District",
]

CALISO_AREA_THRESHOLD = 100

# Valid unit conversion options
UNIT_OPTIONS = {
    "K": ["K", "degC", "degF"],
    "degF": ["K", "degC", "degF"],
    "degC": ["K", "degC", "degF"],
    "hPa": ["Pa", "hPa", "mb", "inHg"],
    "Pa": ["Pa", "hPa", "mb", "inHg"],
    "m/s": ["m/s", "mph", "knots"],
    "m s-1": ["m s-1", "mph", "knots"],
    "[0 to 100]": ["[0 to 100]", "fraction"],
    "mm": ["mm", "inches"],
    "mm/d": ["mm/d", "inches/d"],
    "mm/h": ["mm/h", "inches/h"],
    "kg/kg": ["kg/kg", "g/kg"],
    "kg kg-1": ["kg kg-1", "g kg-1"],
    "kg m-2 s-1": ["kg m-2 s-1", "mm", "inches"],
    "g/kg": ["g/kg", "kg/kg"],
}
# Start and end years for LOCA and WRF data
LOCA_START_YEAR = 1950
LOCA_END_YEAR = 2100

WRF_START_YEAR = 1981
WRF_END_YEAR = 2099

# Constants for data size thresholds and processing (metric_calc)
BYTES_TO_MB_FACTOR = 1e6  # Conversion factor from bytes to megabytes
BYTES_TO_GB_FACTOR = 1e9  # Conversion factor from bytes to gigabytes
SMALL_ARRAY_THRESHOLD_BYTES = 1e7  # 10MB threshold for small arrays
MEDIUM_ARRAY_THRESHOLD_BYTES = 1e9  # 1GB threshold for medium arrays
PERCENTILE_TO_QUANTILE_FACTOR = 100.0  # Convert percentiles to quantiles
MIN_VALID_DATA_POINTS = 3  # Minimum data points required for statistical fitting
NUMERIC_PRECISION_DECIMAL_PLACES = 2  # Decimal places for numeric output formatting
RETURN_VALUE_PRECISION = 5  # Decimal places for return value rounding

# WKT representation of WRF coordinate system
WRF_CRS = """PROJCS["undefined",
        GEOGCS["undefined",
            DATUM["undefined",
            SPHEROID["undefined",6370000,0]
            ],
            PRIMEM["Greenwich",0,AUTHORITY["EPSG","8901"]],
            UNIT["degree",0.0174532925199433]
        ],
        PROJECTION["Lambert_Conformal_Conic_2SP"],
        PARAMETER["standard_parallel_1",30],
        PARAMETER["standard_parallel_2",60],
        PARAMETER["latitude_of_origin",38],
        PARAMETER["central_meridian",-70],
        PARAMETER["false_easting",0],
        PARAMETER["false_northing",0],
        UNIT["metre",1,AUTHORITY["EPSG","9001"]],
        AXIS["Easting",EAST],
        AXIS["Northing",NORTH]
        ]"""
