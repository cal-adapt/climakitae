# paths.py
"""This module defines package level paths"""

# Local Data Paths
VARIABLE_DESCRIPTIONS_CSV_PATH = "data/variable_descriptions.csv"
HADISD_STATIONS_URL = "https://cadcat.s3.amazonaws.com/hadisd/hadisd_stations.csv"

# Intake Data Calalog URLs
DATA_CATALOG_URL = "https://cadcat.s3.amazonaws.com/cae-collection.json"
BOUNDARY_CATALOG_URL = "https://cadcat.s3.amazonaws.com/parquet/catalog.yaml"
RENEWABLES_CATALOG_URL = (
    "https://wfclimres.s3.amazonaws.com/era/era-ren-collection.json"
)
HDP_CATALOG_URL = "https://cadcat.s3.amazonaws.com/histwxstns/era-hdp-collection.json"

# S3 scratch bucket for exporting
EXPORT_S3_BUCKET = "cadcat-tmp"

# Infrastructure data layer URLs (CA power plant and grid data, EPSG:4326 GeoParquet)
# TODO(scraping): GEM data is embedded in the ETL script; see
# scripts/build_infrastructure_parquets.py for source URLs and update instructions.
EIA860M_CA_PLANTS_URL = (
    "https://cadcat.s3.amazonaws.com/infrastructure/eia860m_ca_plants.parquet"
)
GEM_CA_PLANTS_URL = (
    "https://cadcat.s3.amazonaws.com/infrastructure/gem_ca_plants.parquet"
)
HIFLD_CA_TRANSMISSION_URL = (
    "https://cadcat.s3.amazonaws.com/infrastructure/hifld_ca_transmission.parquet"
)
HIFLD_CA_SUBSTATIONS_URL = (
    "https://cadcat.s3.amazonaws.com/infrastructure/hifld_ca_substations.parquet"
)
OSM_CA_POWER_URL = "https://cadcat.s3.amazonaws.com/infrastructure/osm_ca_power.parquet"

INFRASTRUCTURE_LAYER_URLS = {
    "eia_plants": EIA860M_CA_PLANTS_URL,
    "gem_plants": GEM_CA_PLANTS_URL,
    "transmission": HIFLD_CA_TRANSMISSION_URL,
    "substations": HIFLD_CA_SUBSTATIONS_URL,
    "osm_power": OSM_CA_POWER_URL,
}

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
