# Available Boundary Options for Clipping in ClimakitAE

This document lists all available boundary options that can be used with the `Clip` processor in ClimakitAE for spatial data subsetting.

## Overview

The `Clip` processor supports several types of spatial boundaries for data clipping:

1. **File-based boundaries**: Any shapefile or other geospatial file
2. **Predefined boundaries**: Built-in boundary datasets from the ClimakitAE catalog
3. **Coordinate boxes**: Latitude/longitude bounding boxes

## Usage Examples

```python
from climakitae.new_core.processors.clip import Clip

# Using a shapefile
clip1 = Clip("/path/to/my_boundary.shp")

# Using predefined boundaries (examples)
clip2 = Clip("CA")                    # California state
clip3 = Clip("Los Angeles County")    # LA County
clip4 = Clip("Russian River")         # Russian River watershed  
clip5 = Clip("PG&E")                  # PG&E utility service area
clip6 = Clip("CALISO")               # California ISO balancing area

# Using lat/lon bounding box
clip7 = Clip(((32.0, 42.0), (-125.0, -114.0)))  # ((lat_min, lat_max), (lon_min, lon_max))
```

## Available Predefined Boundaries

### 1. Western US States (11 options)
**Category**: `states`
**Available options**:
- AZ (Arizona)
- CA (California)  
- CO (Colorado)
- ID (Idaho)
- MT (Montana)
- NV (Nevada)
- NM (New Mexico)
- OR (Oregon)
- UT (Utah)
- WA (Washington)
- WY (Wyoming)

### 2. California Counties (58 options)
**Category**: `CA counties`
**Available options** (partial list):
- Alameda County
- Alpine County
- Amador County
- Butte County
- Calaveras County
- Contra Costa County
- Del Norte County
- El Dorado County
- Fresno County
- Glenn County
- Humboldt County
- Imperial County
- Inyo County
- Kern County
- Kings County
- Lake County
- Lassen County
- Los Angeles County
- Madera County
- Marin County
- Mariposa County
- Mendocino County
- Merced County
- Modoc County
- Mono County
- Monterey County
- Napa County
- Nevada County
- Orange County
- Placer County
- Plumas County
- Riverside County
- Sacramento County
- San Benito County
- San Bernardino County
- San Diego County
- San Francisco County
- San Joaquin County
- San Luis Obispo County
- San Mateo County
- Santa Barbara County
- Santa Clara County
- Santa Cruz County
- Shasta County
- Sierra County
- Siskiyou County
- Solano County
- Sonoma County
- Stanislaus County
- Sutter County
- Tehama County
- Trinity County
- Tulare County
- Tuolumne County
- Ventura County
- Yolo County
- Yuba County

### 3. California Watersheds (140+ options)
**Category**: `CA watersheds`
**Examples** (HUC-8 watersheds):
- Aliso-San Onofre
- Antelope-Fremont Valleys
- Applegate
- Battle Creek
- Big Chico Creek-Sacramento River
- Russian River
- Sacramento River
- San Francisco Bay
- Los Angeles River
- Santa Ana River
- Colorado River
- Salinas River
- And 130+ more...

### 4. California Electric Load Serving Entities (52 options)
**Category**: `CA Electric Load Serving Entities (IOU & POU)`
**Examples**:
- Alameda Power & Telecom
- Azusa Light & Power
- Bear Valley Electric Service
- Biggs Municipal Utilities
- Burbank Water & Power
- Desert Electric Cooperative
- Gridley Municipal Utilities
- Healdsburg Electric Department
- Imperial Irrigation District
- Lodi Electric Utility
- Los Angeles Department of Water and Power
- Modesto Irrigation District
- Pacific Gas & Electric (listed as a full utility name)
- Pasadena Water & Power
- Redding Electric Utility
- Riverside Public Utilities
- Sacramento Municipal Utility District
- San Diego Gas & Electric (listed as full utility name)
- Santa Clara Electric Department
- Southern California Edison (listed as full utility name)
- Turlock Irrigation District
- And 30+ more municipal utilities and cooperatives...

### 5. California Electricity Demand Forecast Zones (28 options)
**Category**: `CA Electricity Demand Forecast Zones`
**Available options**:
- Big Creek East
- Big Creek West
- Burbank/Glendale
- Central Coast
- Central Valley
- East Bay
- Fresno
- Humboldt
- Imperial Valley
- Kern
- Los Angeles Basin
- North Coast
- North Valley
- Orange County
- Other CEC
- Other SCE
- PG&E Bay Area
- Riverside
- Sacramento Valley
- San Diego
- Santa Barbara
- Sierra
- Stockton
- Ventura County
- And 4 more...

### 6. California Electric Balancing Authority Areas (8 options)
**Category**: `CA Electric Balancing Authority Areas`
**Available options**:
- BANC (Balancing Authority of Northern California)
- CALISO (California Independent System Operator)
- IID (Imperial Irrigation District)
- LADWP (Los Angeles Department of Water and Power)
- NV Energy
- PACE (PacifiCorp East)
- PACW (PacifiCorp West)
- TIDC (Turlock Irrigation District)

## Programmatic Access

You can programmatically get all available boundaries:

```python
from climakitae.new_core.processors.clip import Clip
from climakitae.new_core.data_access.data_access import DataCatalog

# Get static information (no catalog needed)
categories = Clip.get_supported_boundary_categories()
examples = Clip.get_boundary_examples()
Clip.print_boundary_usage_examples()

# Get dynamic information (requires catalog)
catalog = DataCatalog()
clip_processor = Clip("dummy")
clip_processor.set_data_accessor(catalog)

# List all available boundaries
all_boundaries = clip_processor.list_available_boundaries()

# Pretty print for users
clip_processor.print_available_boundaries()

# Validate a boundary key
validation = clip_processor.validate_boundary_key("CA")
```

## Key Validation and Error Handling

The `Clip` processor includes robust validation:

- **Valid keys**: Returns geometry successfully
- **Invalid keys**: Provides helpful error messages and suggestions
- **Partial matches**: Suggests similar boundary names when exact matches fail
- **Case sensitivity**: Generally case-sensitive, but validation provides suggestions

## Notes

1. **Coordinate Reference System**: All boundary data uses WGS84 (EPSG:4326) coordinate system
2. **File formats**: Any format supported by GeoPandas can be used for file-based boundaries
3. **Performance**: Boundary data is lazy-loaded for efficiency
4. **Updates**: Boundary catalogs are updated periodically; use the validation methods to check current availability

## Error Examples

```python
# These will provide helpful error messages:
clip_processor.validate_boundary_key("PGE")        # Suggests "PG&E" alternatives
clip_processor.validate_boundary_key("russian")    # Suggests "Russian River"
clip_processor.validate_boundary_key("InvalidKey") # Lists similar options
```
