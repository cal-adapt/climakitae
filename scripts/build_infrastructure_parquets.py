#!/usr/bin/env python3
"""Build CA power plant and grid infrastructure GeoParquet files.

Fetches data from 5 public sources (EIA-860M, HIFLD, GEM, OSM) and writes
5 GeoParquet files for California infrastructure. All outputs are EPSG:4326,
snappy-compressed.

Data Sources:
    - EIA-860M: Monthly power plant operating data
    - GEM: Global Energy Monitor tracker data (Solar, Wind, Coal, Gas)
    - HIFLD: Transmission lines and substations
    - OSM: OpenStreetMap power infrastructure

Usage:
    python scripts/build_infrastructure_parquets.py --output-dir /tmp/infra
    python scripts/build_infrastructure_parquets.py --output-dir /tmp/infra --gem-dir /path/to/gem/
"""

import argparse
import io
import json
import logging
import time
import warnings
from pathlib import Path
from typing import Optional

import geopandas as gpd
import pandas as pd
import requests
from shapely.geometry import LineString, Point

# Use a browser-like User-Agent so servers (GEM, EIA, Overpass) don't 403/406 us.
_UA = "Mozilla/5.0 (X11; Linux x86_64; rv:125.0) Gecko/20100101 Firefox/125.0"
_SESSION = requests.Session()
_SESSION.headers.update(
    {
        "User-Agent": _UA,
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
        "Accept-Language": "en-US,en;q=0.9",
        "Accept-Encoding": "gzip, deflate, br",
        "Referer": "https://globalenergymonitor.org/",
        "DNT": "1",
        "Connection": "keep-alive",
    }
)

logger = logging.getLogger(__name__)

# California bounding box: (min_lat, min_lon, max_lat, max_lon)
CA_BBOX = (32.5, -124.5, 42.0, -114.1)

# EIA Fuel Type standardization mapping
EIA_FUEL_TYPE_MAP = {
    "SUN": "Solar",
    "NG": "NaturalGas",
    "WAT": "Hydro",
    "NUC": "Nuclear",
    "BIT": "Coal",
    "SUB": "Coal",
    "LIG": "Coal",
    "DFO": "Oil",
    "RFO": "Oil",
    "GEO": "Geothermal",
    "WND": "Wind",
    "OTH": "Other",
    "OG": "NaturalGas",
    "PG": "NaturalGas",
    "PC": "Coal",
    "WH": "Other",
    "MWH": "Storage",
    "PUR": "Other",
    "WC": "Coal",
    "AB": "Biomass",
    "MSW": "Biomass",
    "LFG": "Biomass",
    "OBS": "Biomass",
    "BFG": "Biomass",
    "OBL": "Biomass",
    "SLW": "Biomass",
    "BLQ": "Biomass",
    "WDL": "Biomass",
    "WDS": "Biomass",
}

# TODO(scraping): These URLs change each GEM release (monthly).
# To automate, scrape https://globalenergymonitor.org/projects/ for download links
# matching each tracker type. Current URLs are the March 2026 release.
GEM_URLS = {
    "Solar": "https://globalenergymonitor.org/wp-content/uploads/2026/02/Global-Solar-Power-Tracker-February-2026.xlsx",
    "Wind": "https://globalenergymonitor.org/wp-content/uploads/2026/02/Global-Wind-Power-Tracker-February-2026.xlsx",
    "Coal": "https://globalenergymonitor.org/wp-content/uploads/2026/01/Global-Coal-Plant-Tracker-January-2026.xlsx",
    "Gas": "https://globalenergymonitor.org/wp-content/uploads/2026/01/Global-Oil-and-Gas-Plant-Tracker-January-2026.xlsx",
    "Geothermal": "https://globalenergymonitor.org/wp-content/uploads/2026/03/Geothermal-Power-Tracker-March-2026.xlsx",
    "Bioenergy": "https://globalenergymonitor.org/wp-content/uploads/2025/09/Global-Bioenergy-Power-Tracker-September-2025.xlsx",
    "Hydropower": "https://globalenergymonitor.org/wp-content/uploads/2026/03/Global-Hydropower-Tracker-March-2026.xlsx",
    "Nuclear": "https://globalenergymonitor.org/wp-content/uploads/2025/09/Global-Nuclear-Power-Tracker-September-2025.xlsx",
}


def fetch_eia860m_ca() -> gpd.GeoDataFrame:
    """Fetch EIA-860M operating power plants in California.

    Downloads the latest EIA-860M monthly generator data Excel file, filters
    to California operating plants, and standardizes fuel types.

    Returns
    -------
    gpd.GeoDataFrame
        California power plants with columns: plant_id, name, fuel_type,
        capacity_mw, status, operator, county, source, geometry (Point).
        CRS is EPSG:4326.

    Notes
    -----
    Falls back from July to June 2025 file if download fails.
    Drops rows with missing latitude/longitude.
    """
    urls = [
        # Try most recent months first; EIA publishes ~1 month lag
        "https://www.eia.gov/electricity/data/eia860m/xls/march_generator2026.xlsx",
        "https://www.eia.gov/electricity/data/eia860m/xls/february_generator2026.xlsx",
        "https://www.eia.gov/electricity/data/eia860m/xls/january_generator2026.xlsx",
        "https://www.eia.gov/electricity/data/eia860m/xls/december_generator2025.xlsx",
        "https://www.eia.gov/electricity/data/eia860m/xls/november_generator2025.xlsx",
    ]

    df = None
    for url in urls:
        try:
            logger.info("  Attempting download from %s", url)
            # Use bare requests (not _SESSION) so GEM Referer header
            # doesn't confuse the EIA download server.
            resp = requests.get(url, headers={"User-Agent": _UA}, timeout=60)
            resp.raise_for_status()
            if resp.content[:2] != b"PK":
                raise ValueError("Response is not an xlsx file (likely HTML redirect)")
            # 2026 format has an extra title row → skiprows=2
            df = pd.read_excel(
                io.BytesIO(resp.content),
                sheet_name="Operating",
                skiprows=2,
                engine="openpyxl",
            )
            logger.info("  Successfully downloaded EIA-860M data")
            break
        except Exception as e:
            logger.warning("  Failed to download from %s: %s", url, e)
            continue

    if df is None:
        logger.error("  All EIA-860M URLs failed")
        return gpd.GeoDataFrame(
            columns=[
                "plant_id",
                "name",
                "fuel_type",
                "capacity_mw",
                "status",
                "operator",
                "county",
                "source",
                "geometry",
            ],
            crs="EPSG:4326",
        )

    # Filter to California
    df = df[df["Plant State"] == "CA"].copy()

    if len(df) == 0:
        logger.warning("  No California plants found in EIA-860M data")
        return gpd.GeoDataFrame(
            columns=[
                "plant_id",
                "name",
                "fuel_type",
                "capacity_mw",
                "status",
                "operator",
                "county",
                "source",
                "geometry",
            ],
            crs="EPSG:4326",
        )

    # Standardize fuel types — column renamed from "Energy Source 1" in 2026 format
    energy_col = (
        "Energy Source Code"
        if "Energy Source Code" in df.columns
        else "Energy Source 1"
    )
    df["fuel_type"] = df[energy_col].map(EIA_FUEL_TYPE_MAP).fillna("Other")

    # Extract required fields — operator renamed from "Utility Name" to "Entity Name" in 2026
    id_col = "Plant ID" if "Plant ID" in df.columns else "Plant Id"
    operator_col = "Entity Name" if "Entity Name" in df.columns else "Utility Name"
    df["plant_id"] = df[id_col].astype(str)
    df["name"] = df["Plant Name"]
    df["capacity_mw"] = pd.to_numeric(df["Nameplate Capacity (MW)"], errors="coerce")
    df["operator"] = df[operator_col]
    df["county"] = df["County"]
    df["status"] = "Operating"
    df["source"] = "EIA-860M"

    # Drop rows with missing coordinates
    df = df.dropna(subset=["Latitude", "Longitude"])

    # Create GeoDataFrame
    gdf = gpd.GeoDataFrame(
        df[
            [
                "plant_id",
                "name",
                "fuel_type",
                "capacity_mw",
                "status",
                "operator",
                "county",
                "source",
            ]
        ],
        geometry=gpd.points_from_xy(df["Longitude"], df["Latitude"]),
        crs="EPSG:4326",
    )

    return gdf


_GEM_PREAMBLE_SHEETS = frozenset(
    {
        "about",
        "column key",
        "below threshold",
        "distributed (<1 mw)",
        "sub-threshold units",
    }
)


def _gem_select_data_sheet(
    all_sheets: dict[str, pd.DataFrame],
) -> pd.DataFrame:
    """Return the first GEM sheet that looks like tabular plant data.

    GEM Excel files always open with an 'About' preamble sheet. The actual
    data lives on a sheet whose columns include 'Country/Area'. This helper
    selects that sheet, falling back to the last sheet if nothing matches.
    """
    for name, df in all_sheets.items():
        if name.lower() in _GEM_PREAMBLE_SHEETS:
            continue
        cols_lower = [str(c).lower() for c in df.columns]
        if any("country" in c for c in cols_lower):
            return df
    # Fallback: last non-preamble sheet
    non_preamble = [
        df
        for name, df in all_sheets.items()
        if name.lower() not in _GEM_PREAMBLE_SHEETS
    ]
    return non_preamble[-1] if non_preamble else next(iter(all_sheets.values()))


def fetch_gem_ca(gem_dir: Optional[str] = None) -> gpd.GeoDataFrame:
    """Fetch GEM (Global Energy Monitor) power plants in California.

    Downloads or reads local GEM tracker Excel files for Solar, Wind, Coal,
    and Natural Gas plants. Filters to operating plants in California.

    Parameters
    ----------
    gem_dir : str, optional
        Path to local directory containing GEM Excel files. If provided,
        reads local files instead of downloading.

    Returns
    -------
    gpd.GeoDataFrame
        California power plants with columns: plant_id, name, fuel_type,
        capacity_mw, status, operator, county, source, geometry (Point).
        CRS is EPSG:4326.

    Notes
    -----
    Handles column name variations across different GEM tracker versions.
    Only includes operating plants.
    """
    all_plants = []

    for fuel_type, url in GEM_URLS.items():
        try:
            # Load from local file or download
            if gem_dir:
                gem_path = Path(gem_dir)
                matching_files = list(gem_path.glob(f"*{fuel_type}*.xlsx"))
                if not matching_files:
                    logger.warning(
                        "  No %s tracker file found in %s", fuel_type, gem_dir
                    )
                    continue
                df = pd.read_excel(
                    matching_files[0], sheet_name=None, engine="openpyxl"
                )
                df = _gem_select_data_sheet(df)
                logger.info("  Loaded %s tracker from local file", fuel_type)
            else:
                logger.info("  Downloading %s tracker from GEM", fuel_type)
                resp = _SESSION.get(url, timeout=120)
                resp.raise_for_status()
                all_sheets = pd.read_excel(
                    io.BytesIO(resp.content), sheet_name=None, engine="openpyxl"
                )
                df = _gem_select_data_sheet(all_sheets)

            # Normalize column names (handle variations)
            df.columns = df.columns.str.strip()

            # Find country and state columns
            country_col = None
            state_col = None
            for col in df.columns:
                if "country" in col.lower():
                    country_col = col
                if "state" in col.lower() or "province" in col.lower():
                    state_col = col

            if country_col is None or state_col is None:
                logger.warning(
                    "  Could not find country/state columns in %s tracker", fuel_type
                )
                continue

            # Filter to California, USA
            df = df[
                (
                    df[country_col]
                    .astype(str)
                    .str.contains("United States", case=False, na=False)
                )
                & (
                    df[state_col]
                    .astype(str)
                    .str.contains("California", case=False, na=False)
                )
            ].copy()

            if len(df) == 0:
                logger.info("  No California plants in %s tracker", fuel_type)
                continue

            # Find status column and filter to operating
            status_col = None
            for col in df.columns:
                if "status" in col.lower():
                    status_col = col
                    break

            if status_col:
                df = df[
                    df[status_col]
                    .astype(str)
                    .str.lower()
                    .str.contains("operat|ops", na=False)
                ]

            # Extract fields with column name variations
            name_col = next(
                (
                    c
                    for c in df.columns
                    if "plant" in c.lower()
                    or "project" in c.lower()
                    and "name" in c.lower()
                ),
                None,
            )
            capacity_col = next(
                (
                    c
                    for c in df.columns
                    if "capacity" in c.lower() and "mw" in c.lower()
                ),
                None,
            )
            owner_col = next((c for c in df.columns if "owner" in c.lower()), None)
            lat_col = next(
                (c for c in df.columns if c.strip().lower() == "latitude"), None
            )
            lon_col = next(
                (c for c in df.columns if c.strip().lower() == "longitude"), None
            )

            df["plant_id"] = "GEM_" + fuel_type + "_" + df.index.astype(str)
            df["name"] = df[name_col] if name_col else "Unknown"
            df["fuel_type"] = fuel_type
            df["capacity_mw"] = (
                pd.to_numeric(df[capacity_col], errors="coerce")
                if capacity_col
                else pd.NA
            )
            df["status"] = df[status_col] if status_col else "Operating"
            df["operator"] = df[owner_col] if owner_col else "Unknown"
            df["county"] = pd.NA
            df["source"] = "GEM"

            # Drop rows with missing coordinates
            if lat_col and lon_col:
                df = df.dropna(subset=[lat_col, lon_col])
                df["latitude"] = pd.to_numeric(df[lat_col], errors="coerce")
                df["longitude"] = pd.to_numeric(df[lon_col], errors="coerce")
                df = df.dropna(subset=["latitude", "longitude"])

                gdf = gpd.GeoDataFrame(
                    df[
                        [
                            "plant_id",
                            "name",
                            "fuel_type",
                            "capacity_mw",
                            "status",
                            "operator",
                            "county",
                            "source",
                        ]
                    ],
                    geometry=gpd.points_from_xy(df["longitude"], df["latitude"]),
                    crs="EPSG:4326",
                )
                all_plants.append(gdf)
            else:
                logger.warning(
                    "  Could not find lat/lon columns in %s tracker", fuel_type
                )

        except Exception as e:
            logger.error("  Failed to fetch %s tracker: %s", fuel_type, e)
            if not gem_dir and isinstance(e, requests.exceptions.HTTPError):
                logger.error(
                    "  GEM uses Cloudflare WAF that blocks automated downloads. "
                    "Manually download the trackers from "
                    "https://globalenergymonitor.org/projects/ and re-run with "
                    "--gem-dir /path/to/dir/"
                )
            continue

    if not all_plants:
        return gpd.GeoDataFrame(
            columns=[
                "plant_id",
                "name",
                "fuel_type",
                "capacity_mw",
                "status",
                "operator",
                "county",
                "source",
                "geometry",
            ],
            crs="EPSG:4326",
        )

    return pd.concat(all_plants, ignore_index=True)


def fetch_hifld_transmission_ca() -> gpd.GeoDataFrame:
    """Fetch HIFLD transmission lines in California.

    Queries the HIFLD Electric Power Transmission Lines ArcGIS REST API
    with pagination, filters to California bounding box.

    Returns
    -------
    gpd.GeoDataFrame
        California transmission lines with columns: line_id, voltage_kv,
        status, owner, source, geometry (LineString).
        CRS is EPSG:4326.

    Notes
    -----
    Uses a server-side spatial envelope filter (CA bounding box) so we only
    download lines that intersect California (~2,100 of ~52,000 total US
    lines). Uses pagination with 2000 records per request.
    """
    base_url = "https://services1.arcgis.com/Hp6G80Pky0om7QvQ/arcgis/rest/services/Electric_Power_Transmission_Lines/FeatureServer/0/query"

    # Server-side spatial filter: CA bounding box envelope in WGS84.
    ca_envelope = json.dumps(
        {
            "xmin": CA_BBOX[1],
            "ymin": CA_BBOX[0],
            "xmax": CA_BBOX[3],
            "ymax": CA_BBOX[2],
            "spatialReference": {"wkid": 4326},
        }
    )

    all_features = []
    offset = 0
    batch_size = 2000

    while True:
        params = {
            "where": "1=1",
            "geometry": ca_envelope,
            "geometryType": "esriGeometryEnvelope",
            "spatialRel": "esriSpatialRelIntersects",
            "inSR": "4326",
            "outFields": "*",
            "outSR": "4326",  # force WGS84 lat/lon output
            "f": "json",
            "resultOffset": offset,
            "resultRecordCount": batch_size,
        }

        try:
            # Use bare requests — _SESSION has GEM Referer that breaks ArcGIS
            response = requests.get(
                base_url,
                params=params,
                headers={"User-Agent": _UA},
                timeout=60,
            )
            response.raise_for_status()
            data = response.json()

            if "error" in data:
                logger.error("  HIFLD transmission API error: %s", data["error"])
                break

            features = data.get("features", [])
            if not features:
                break

            all_features.extend(features)
            offset += batch_size
            logger.info(
                "  Fetched %d transmission line features (total: %d)",
                len(features),
                len(all_features),
            )

        except Exception as e:
            logger.error(
                "  Failed to fetch HIFLD transmission at offset %d: %s", offset, e
            )
            break

    if not all_features:
        return gpd.GeoDataFrame(
            columns=["line_id", "voltage_kv", "status", "owner", "source", "geometry"],
            crs="EPSG:4326",
        )

    # Parse features into GeoDataFrame
    records = []
    for feat in all_features:
        attrs = feat.get("attributes", {})
        geom_data = feat.get("geometry", {})

        # Build LineString from paths
        paths = geom_data.get("paths", [])
        if not paths:
            continue

        # Use first path (most lines have one path)
        coords = paths[0]
        if len(coords) < 2:
            continue

        line = LineString(coords)

        # Filter by centroid in CA bbox
        centroid = line.centroid
        if not (
            CA_BBOX[0] <= centroid.y <= CA_BBOX[2]
            and CA_BBOX[1] <= centroid.x <= CA_BBOX[3]
        ):
            continue

        voltage = attrs.get("VOLTAGE")
        if voltage and voltage != -999:
            voltage_kv = float(voltage)
        else:
            voltage_kv = pd.NA

        records.append(
            {
                "line_id": str(attrs.get("OBJECTID", "")),
                "voltage_kv": voltage_kv,
                "status": attrs.get("STATUS", ""),
                "owner": attrs.get("OWNER", ""),
                "source": "HIFLD",
                "geometry": line,
            }
        )

    if not records:
        return gpd.GeoDataFrame(
            columns=["line_id", "voltage_kv", "status", "owner", "source", "geometry"],
            crs="EPSG:4326",
        )

    return gpd.GeoDataFrame(records, crs="EPSG:4326")


def fetch_hifld_substations_ca() -> gpd.GeoDataFrame:
    """Fetch HIFLD substations in California.

    Queries the HIFLD Electric Substations ArcGIS REST API with pagination,
    filters to California bounding box.

    Returns
    -------
    gpd.GeoDataFrame
        California substations with columns: substation_id, name, voltage_kv,
        status, owner, source, geometry (Point).
        CRS is EPSG:4326.

    Notes
    -----
    Uses the `STATE='CA'` where-clause filter (server-side) so we only download
    California records (~4,300 of ~78,000 total US substations).
    Uses pagination with 2000 records per request.
    Service URL is from the HIFLD Substations 1_9_25 item on ArcGIS Online
    (item id 83397b209bfb4007a2f4c00e70df8e5d).
    """
    base_url = "https://services6.arcgis.com/OO2s4OoyCZkYJ6oE/arcgis/rest/services/Substations/FeatureServer/0/query"

    all_features = []
    offset = 0
    batch_size = 2000

    while True:
        params = {
            "where": "STATE='CA'",
            "outFields": "*",
            "outSR": "4326",  # force WGS84 lat/lon output
            "f": "json",
            "resultOffset": offset,
            "resultRecordCount": batch_size,
        }

        try:
            # Use bare requests — _SESSION has GEM Referer that breaks ArcGIS
            response = requests.get(
                base_url,
                params=params,
                headers={"User-Agent": _UA},
                timeout=60,
            )
            response.raise_for_status()
            data = response.json()

            if "error" in data:
                logger.error("  HIFLD substations API error: %s", data["error"])
                break

            features = data.get("features", [])
            if not features:
                break

            all_features.extend(features)
            offset += batch_size
            logger.info(
                "  Fetched %d substation features (total: %d)",
                len(features),
                len(all_features),
            )

        except Exception as e:
            logger.error(
                "  Failed to fetch HIFLD substations at offset %d: %s", offset, e
            )
            break

    if not all_features:
        return gpd.GeoDataFrame(
            columns=[
                "substation_id",
                "name",
                "voltage_kv",
                "status",
                "owner",
                "source",
                "geometry",
            ],
            crs="EPSG:4326",
        )

    # Parse features into GeoDataFrame
    records = []
    for feat in all_features:
        attrs = feat.get("attributes", {})
        geom_data = feat.get("geometry", {})

        # Handle point or polygon geometry
        if "x" in geom_data and "y" in geom_data:
            point = Point(geom_data["x"], geom_data["y"])
        elif "rings" in geom_data:
            # Polygon: use centroid
            from shapely.geometry import Polygon

            rings = geom_data["rings"]
            if rings and len(rings[0]) >= 3:
                poly = Polygon(rings[0])
                point = poly.centroid
            else:
                continue
        else:
            continue

        # Server-side filter STATE='CA' already restricts to California; no
        # client-side bbox check needed (would clip islands like Catalina).

        voltage = attrs.get("MAX_VOLT")
        if voltage and voltage != -999:
            voltage_kv = float(voltage)
        else:
            voltage_kv = pd.NA

        records.append(
            {
                "substation_id": str(attrs.get("OBJECTID", "")),
                "name": attrs.get("NAME", ""),
                "voltage_kv": voltage_kv,
                "status": attrs.get("STATUS", ""),
                "owner": attrs.get("OWNER", ""),
                "source": "HIFLD",
                "geometry": point,
            }
        )

    if not records:
        return gpd.GeoDataFrame(
            columns=[
                "substation_id",
                "name",
                "voltage_kv",
                "status",
                "owner",
                "source",
                "geometry",
            ],
            crs="EPSG:4326",
        )

    return gpd.GeoDataFrame(records, crs="EPSG:4326")


def _fetch_osm_power_type(
    power_type: str,
    bbox: tuple,
    retries: int = 3,
    timeout: int = 60,
) -> gpd.GeoDataFrame:
    """Fetch OSM power infrastructure elements of a specific type.

    Queries the Overpass API for nodes, ways, and relations with the given
    power tag within the bounding box.

    Parameters
    ----------
    power_type : str
        OSM power tag value (e.g., "plant", "generator", "substation", "line", "cable").
    bbox : tuple
        Bounding box as (min_lat, min_lon, max_lat, max_lon).
    retries : int, optional
        Number of retry attempts on failure, by default 3.
    timeout : int, optional
        Query timeout in seconds, by default 90.

    Returns
    -------
    gpd.GeoDataFrame
        Power infrastructure features with columns: osm_id, name, power_type,
        fuel_type, capacity_mw, voltage_kv, operator, source, geometry.
        CRS is EPSG:4326.

    Notes
    -----
    Skips ways and relations (complex geometry handling).
    Only processes nodes with point geometries.
    Retries with exponential backoff on network failures.
    """
    min_lat, min_lon, max_lat, max_lon = bbox

    query = f"""
    [out:json][timeout:{timeout}];
    (
      node["power"="{power_type}"]({min_lat},{min_lon},{max_lat},{max_lon});
      way["power"="{power_type}"]({min_lat},{min_lon},{max_lat},{max_lon});
      relation["power"="{power_type}"]({min_lat},{min_lon},{max_lat},{max_lon});
    );
    out body;
    >;
    out skel qt;
    """

    # Try multiple Overpass endpoints in order. overpass.openstreetmap.fr is the
    # most permissive for non-browser User-Agents; overpass-api.de often returns
    # 406/403 from datacenter IPs.
    _overpass_endpoints = [
        "https://overpass.openstreetmap.fr/api/interpreter",
        "https://overpass-api.de/api/interpreter",
        "https://overpass.private.coffee/api/interpreter",
    ]
    # Note: do NOT send Accept: application/json — some Overpass mirrors 406 on it.
    # The query itself uses [out:json] so the response is JSON regardless.
    overpass_headers = {
        "Content-Type": "application/x-www-form-urlencoded",
        "User-Agent": _UA,
    }
    data: dict = {}

    for attempt in range(retries):
        endpoint = _overpass_endpoints[attempt % len(_overpass_endpoints)]
        try:
            response = requests.post(
                endpoint,
                data={"data": query.strip()},
                headers=overpass_headers,
                timeout=timeout,
            )
            response.raise_for_status()
            data = response.json()
            break
        except (requests.exceptions.RequestException, ValueError) as e:
            wait_time = 2**attempt
            logger.warning(
                "  OSM %s query failed (attempt %d/%d): %s. Retrying in %ds...",
                power_type,
                attempt + 1,
                retries,
                e,
                wait_time,
            )
            if attempt < retries - 1:
                time.sleep(wait_time)
            else:
                logger.error(
                    "  OSM %s query failed after %d attempts", power_type, retries
                )
                return gpd.GeoDataFrame(
                    columns=[
                        "osm_id",
                        "name",
                        "power_type",
                        "fuel_type",
                        "capacity_mw",
                        "voltage_kv",
                        "operator",
                        "source",
                        "geometry",
                    ],
                    crs="EPSG:4326",
                )

    elements = data.get("elements", [])
    records = []

    for elem in elements:
        elem_type = elem.get("type")
        tags = elem.get("tags", {})

        # Only process nodes (points) for simplicity
        if elem_type == "node":
            lon = elem.get("lon")
            lat = elem.get("lat")
            if lon is None or lat is None:
                continue

            geometry = Point(lon, lat)

        elif elem_type == "way":
            # Ways need separate node lookup - skip for simplicity
            logger.debug("  Skipping way element (complex geometry)")
            continue

        elif elem_type == "relation":
            # Relations are complex multi-polygons - skip
            continue

        else:
            continue

        # Extract tags
        name = tags.get("name", "")
        fuel_type = tags.get("fuel", "")

        # Parse capacity
        capacity_str = tags.get("plant:output:electricity", "")
        capacity_mw = pd.NA
        if capacity_str:
            # Parse "100 MW" -> 100.0
            try:
                capacity_mw = float(capacity_str.split()[0])
            except (ValueError, IndexError):
                pass

        # Parse voltage
        voltage_str = tags.get("voltage", "")
        voltage_kv = pd.NA
        if voltage_str:
            try:
                voltage_val = float(voltage_str)
                # Convert V to kV if needed
                if voltage_val > 1000:
                    voltage_kv = voltage_val / 1000
                else:
                    voltage_kv = voltage_val
            except ValueError:
                pass

        operator = tags.get("operator", "")

        records.append(
            {
                "osm_id": str(elem.get("id", "")),
                "name": name,
                "power_type": power_type,
                "fuel_type": fuel_type,
                "capacity_mw": capacity_mw,
                "voltage_kv": voltage_kv,
                "operator": operator,
                "source": "OSM",
                "geometry": geometry,
            }
        )

    if not records:
        return gpd.GeoDataFrame(
            columns=[
                "osm_id",
                "name",
                "power_type",
                "fuel_type",
                "capacity_mw",
                "voltage_kv",
                "operator",
                "source",
                "geometry",
            ],
            crs="EPSG:4326",
        )

    return gpd.GeoDataFrame(records, crs="EPSG:4326")


def fetch_osm_ca() -> gpd.GeoDataFrame:
    """Fetch OSM power infrastructure in California.

    Queries OpenStreetMap via Overpass API for plants, generators, substations,
    transmission lines, and cables within California.

    Returns
    -------
    gpd.GeoDataFrame
        Combined power infrastructure with columns: osm_id, name, power_type,
        fuel_type, capacity_mw, voltage_kv, operator, source, geometry.
        CRS is EPSG:4326.

    Notes
    -----
    Fetches 5 power types: plant, generator, substation, line, cable.
    Adds 1-second delay between requests to avoid rate limiting.
    """
    power_types = ["plant", "generator", "substation", "line", "cable"]
    all_features = []

    for i, ptype in enumerate(power_types):
        logger.info(
            "  Fetching OSM power type: %s (%d/%d)", ptype, i + 1, len(power_types)
        )
        gdf = _fetch_osm_power_type(ptype, CA_BBOX)
        all_features.append(gdf)

        # Rate limiting
        if i < len(power_types) - 1:
            time.sleep(1)

    if not all_features:
        return gpd.GeoDataFrame(
            columns=[
                "osm_id",
                "name",
                "power_type",
                "fuel_type",
                "capacity_mw",
                "voltage_kv",
                "operator",
                "source",
                "geometry",
            ],
            crs="EPSG:4326",
        )

    return pd.concat(all_features, ignore_index=True)


def main():
    """Main entry point for the script."""
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--output-dir", required=True, help="Output directory for parquet files"
    )
    parser.add_argument(
        "--gem-dir",
        default=None,
        help="Local directory with GEM Excel files (skips download)",
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    sources = {
        "eia860m_ca_plants": lambda: fetch_eia860m_ca(),
        "gem_ca_plants": lambda: fetch_gem_ca(args.gem_dir),
        "hifld_ca_transmission": lambda: fetch_hifld_transmission_ca(),
        "hifld_ca_substations": lambda: fetch_hifld_substations_ca(),
        "osm_ca_power": lambda: fetch_osm_ca(),
    }

    for name, fetcher in sources.items():
        logger.info("Fetching %s ...", name)
        try:
            gdf = fetcher()
            out_path = output_dir / f"{name}.parquet"
            gdf.to_parquet(out_path, engine="pyarrow", compression="snappy")
            logger.info("  Wrote %d rows → %s", len(gdf), out_path)
        except Exception as e:
            logger.error("  Failed to fetch %s: %s", name, e)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    main()
