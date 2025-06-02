# paths.py
"""This module defines package level paths"""

# Local Data Paths
VARIABLE_DESCRIPTIONS_CSV_PATH = "data/variable_descriptions.csv"
STATIONS_CSV_PATH = "data/hadisd_stations.csv"

# Intake Data Calalog URLs
DATA_CATALOG_URL = "https://cadcat.s3.amazonaws.com/cae-collection.json"
BOUNDARY_CATALOG_URL = "https://cadcat.s3.amazonaws.com/parquet/catalog.yaml"
RENEWABLES_CATALOG_URL = (
    "https://wfclimres.s3.amazonaws.com/era/era-ren-collection.json"
)

# S3 scratch bucket for exporting
EXPORT_S3_BUCKET = "cadcat-tmp"

# Colormap text files

AE_ORANGE = "data/cmaps/ae_orange.txt"
AE_DIVERGING = "data/cmaps/ae_diverging.txt"
AE_BLUE = "data/cmaps/ae_blue.txt"
AE_DIVERGING_R = "data/cmaps/ae_diverging_r.txt"
CATEGORICAL_CB = "data/cmaps/categorical_cb.txt"

# Global warming levels files
GWL_1850_1900_FILE = "data/gwl_1850-1900ref.csv"
GWL_1981_2010_FILE = "data/gwl_1981-2010ref.csv"
GWL_1850_1900_TIMEIDX_FILE = "data/gwl_1850-1900ref_timeidx.csv"
GWL_1981_2010_TIMEIDX_FILE = "data/gwl_1981-2010ref_timeidx.csv"

# GMT context plot files
SSP119_FILE = "data/tas_global_SSP1_1_9.csv"
SSP126_FILE = "data/tas_global_SSP1_2_6.csv"
SSP245_FILE = "data/tas_global_SSP2_4_5.csv"
SSP370_FILE = "data/tas_global_SSP3_7_0.csv"
SSP585_FILE = "data/tas_global_SSP5_8_5.csv"
HIST_FILE = "data/tas_global_Historical.csv"
