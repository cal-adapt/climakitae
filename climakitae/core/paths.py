# paths.py
"""This module defines package level paths"""

# Local Data Paths
variable_descriptions_csv_path = "data/variable_descriptions.csv"
stations_csv_path = "data/hadisd_stations.csv"

# Intake Data Calalog URLs
data_catalog_url = "https://cadcat.s3.amazonaws.com/cae-collection.json"
boundary_catalog_url = "https://cadcat.s3.amazonaws.com/parquet/catalog.yaml"

# S3 scratch bucket for exporting
export_s3_bucket = "cadcat-tmp"

# Colormap text files

ae_orange = "data/cmaps/ae_orange.txt"
ae_diverging = "data/cmaps/ae_diverging.txt"
ae_blue = "data/cmaps/ae_blue.txt"
ae_diverging_r = "data/cmaps/ae_diverging_r.txt"
categorical_cb = "data/cmaps/categorical_cb.txt"

# Global warming levels files
gwl_1850_1900_file = "data/gwl_1850-1900ref.csv"
gwl_1981_2010_file = "data/gwl_1981-2010ref.csv"
GWL_1850_1900_TIMEIDX_FILE = "data/gwl_1850-1900ref_timeidx.csv"
GWL_1981_2010_TIMEIDX_FILE = "data/gwl_1981-2010ref_timeidx.csv"

# GMT context plot files
ssp119_file = "data/tas_global_SSP1_1_9.csv"
ssp126_file = "data/tas_global_SSP1_2_6.csv"
ssp245_file = "data/tas_global_SSP2_4_5.csv"
ssp370_file = "data/tas_global_SSP3_7_0.csv"
ssp585_file = "data/tas_global_SSP5_8_5.csv"
hist_file = "data/tas_global_Historical.csv"
