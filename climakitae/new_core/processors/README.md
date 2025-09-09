# ClimakitAE Data Processors

This directory contains the data processing pipeline components for the ClimakitAE new core architecture. Processors are modular, extensible components that transform, filter, or otherwise process xarray data objects in a sequential pipeline.

## Architecture Overview

### Registry Pattern
All processors use a decorator-based registry system with priority-based execution:

```python
@register_processor("processor_name", priority=100)
class MyProcessor(DataProcessor):
    ...
```

**Lower priority values = higher execution priority** (executed first).

### Abstract Base Class
All processors inherit from `DataProcessor` and must implement:
- `execute(result, context)`: Process the data
- `update_context(context)`: Record processing steps for provenance
- `set_data_accessor(catalog)`: Set data access interface

### Processing Pipeline Flow
```
Data Input → Filter → Transform → Localize → Calculate → Export
    ↓           ↓         ↓         ↓          ↓        ↓
Priority:   0-10     50-200    100-750    7500     9998-9999
```

## Processor Catalog

### Early Filtering (Priority 0-10)

#### FilterUnAdjustedModels (Priority: 0)
**File**: `filter_unbiased_models.py`  
**Purpose**: Filters out climate models that lack a-priori bias adjustment  

**How it works**:
- Checks for models in the `NON_WRF_BA_MODELS` list
- Removes datasets containing unadjusted models from the pipeline
- Applied by default unless explicitly disabled with `value="no"`
- Essential for ensuring data quality in bias-sensitive analyses

**Parameters**:
- `value` (str): `"yes"` (default) to filter, `"no"` to include all models

**Usage Context**: Automatically applied to maintain data quality standards.

---

#### WarmingLevel (Priority: 10)
**File**: `warming_level.py`  
**Purpose**: Transforms time-series data to warming level-centered approach  

**How it works**:
- Converts calendar year-based data to global warming level periods
- Uses lookup tables (`GWL_1850_1900_FILE`, `GWL_1981_2010_TIMEIDX_FILE`) to map warming levels to time periods
- Applies configurable time windows around central warming years
- Handles multiple warming levels simultaneously

**Parameters**:
- `warming_levels` (list[float]): Global warming levels in °C (e.g., [1.5, 2.0, 3.0])
- `warming_level_months` (list[int], optional): Months to include (1-12)
- `warming_level_window` (int, optional): Years before/after central year (default: 15)

**Usage Context**: Essential for climate impact studies focused on temperature thresholds rather than specific calendar years.

---

### Data Concatenation (Priority 50)

#### Concat (Priority: 50)
**File**: `concatenate.py`  
**Purpose**: Concatenates multiple datasets along a new ensemble dimension  

**How it works**:
- Takes iterable of xarray datasets/arrays
- Creates new "sim" dimension using `source_id` attributes
- Enables ensemble analysis across multiple climate models
- Handles time domain extension for consistent temporal coverage

**Parameters**:
- `value` (str): Dimension name for concatenation (default: "time")

**Usage Context**: Combines multiple climate model outputs into ensemble datasets for statistical analysis.

---

### Spatial Operations (Priority 100-200)

#### TimeSlice (Priority: 100)
**File**: `time_slice.py`  
**Purpose**: Subset data based on temporal ranges  

**How it works**:
- Applies `xarray.sel()` with time slice operations
- Handles multiple data types (Dataset, DataArray, dict, list/tuple)
- Coerces input dates to pandas Timestamps for consistency
- Updates context with slicing information for provenance

**Parameters**:
- `value` (Iterable[date-like]): Tuple of start and end dates

**Usage Context**: One of the most commonly used processors for temporal data subsetting.

---

#### Localize (Priority: 155)
**File**: `localize.py`  
**Purpose**: Extract gridded data at weather station locations with optional bias correction  

**How it works**:
- Loads station metadata from CSV files
- Finds nearest gridcells to station coordinates using geodesic distance
- Optionally applies quantile delta mapping (QDM) bias correction
- Uses xclim library for advanced statistical bias correction
- Accesses HadISD observational data from S3 for bias correction reference

**Parameters**:
- `value`: Station specification (str, list, or dict with config)
- `bias_correction` (bool): Enable/disable bias correction
- `method` (str): Bias correction method ('quantile_delta_mapping')
- `window` (int): Window size for seasonal grouping (default: 30)
- `nquantiles` (int): Number of quantiles for QDM (default: 100)

**Usage Context**: Critical for point-based climate analysis and validation against observational data.

---

#### Clip (Priority: 200)
**File**: `clip.py`  
**Purpose**: Spatial clipping operations for geographic subsetting  

**How it works**:
- **Boundary clipping**: Uses predefined administrative boundaries (states, counties, watersheds)
- **Coordinate clipping**: Bounding box operations with lat/lon coordinates  
- **Point extraction**: Finds closest gridcells with valid data using expanding search radii
- **Multi-geometry support**: Unions multiple boundaries, handles multiple points efficiently
- **Smart point handling**: Searches for nearest valid (non-NaN) data within expanding radii
- Uses geopandas, shapely, and rioxarray for spatial operations

**Parameters**:
- Single boundary: `"CA"` or `"Los Angeles County"`
- Multiple boundaries: `["CA", "OR", "WA"]` (combined using union)
- Bounding box: `((lat_min, lat_max), (lon_min, lon_max))`
- Single point: `(lat, lon)` - returns closest gridcell
- Multiple points: `[(lat1, lon1), (lat2, lon2)]`
- Custom geometry: `"/path/to/shapefile.shp"`

**Usage Context**: Essential for regional climate analysis and geographic data subsetting.

---

### Unit Conversion (Priority 750)

#### ConvertUnits (Priority: 750)
**File**: `convert_units.py`  
**Purpose**: Convert climate variable units to user-specified formats  

**How it works**:
- Implements common climate unit conversions (temperature, precipitation, etc.)
- Temperature: Kelvin ↔ Celsius ↔ Fahrenheit
- Precipitation: mm/day ↔ inches/day ↔ mm/month
- Handles rate conversions (per second ↔ per day)
- Graceful failure with warnings for unsupported conversions

**Parameters**:
- `value` (str): Target units (e.g., "degF", "inches", "mm/month")

**Usage Context**: Ensures data is in appropriate units for analysis and visualization.

---

### Statistical Analysis (Priority 7500)

#### MetricCalc (Priority: 7500)
**File**: `metric_calc.py`  
**Purpose**: Calculate statistical metrics, percentiles, and extreme value analysis  

**How it works**:
- **Basic metrics**: min, max, mean, median calculations
- **Percentile analysis**: User-specified percentile calculations (0-100)
- **Extreme value analysis**: 1-in-X return period calculations using block maxima
- **Adaptive processing**: Memory-aware batch processing for large datasets
- **Vectorized operations**: Efficient computation using xarray/dask
- **Effective Sample Size**: Checks statistical reliability of calculations

**Parameters**:
- `metric` (str): "min", "max", "mean", "median"
- `percentiles` (list): Percentiles to calculate (0-100)
- `dim` (str/list): Dimensions for calculation (default: "time")
- `return_value_years` (list): Return periods for extreme analysis
- `extremes_type` (str): "max" or "min" for extreme analysis
- `check_ess` (bool): Verify effective sample size (default: True)

**Usage Context**: Core component for climate statistics and extreme event analysis.

---

### Final Processing (Priority 9998-9999)

#### UpdateAttributes (Priority: 9998)
**File**: `update_attributes.py`  
**Purpose**: Update dataset attributes with processing provenance  

**How it works**:
- Adds standardized coordinate attributes (lat, lon, time, sim)
- Records all processing steps in dataset attributes
- Updates global attributes with climakitae version info
- Ensures CF-compliant metadata standards
- Maintains processing history for reproducibility

**Parameters**:
- `value`: Optional attribute updates (typically UNSET)

**Usage Context**: Critical for metadata management and dataset provenance.

---

#### Export (Priority: 9999)
**File**: `export.py`  
**Purpose**: Export processed data to various file formats  

**How it works**:
- **NetCDF export**: CF-compliant NetCDF4 files with compression
- **Zarr export**: Cloud-optimized format with local or S3 storage
- **CSV export**: Tabular format for simple datasets
- **Separation options**: Individual files for multi-dataset exports
- **AWS S3 integration**: Direct cloud storage for Zarr files

**Parameters**:
- `filename` (str): Output filename without extension
- `file_format` (str): "NetCDF", "Zarr", or "CSV"
- `mode` (str): "local" or "s3" for Zarr files
- `separated` (bool): Create individual files for multiple datasets
- `export_method` (str): "data", "raw", "calculate", "both", "skip_existing", "None"

**Usage Context**: Final step in processing pipeline for data output and storage.

---

## Processing Pipeline Execution

### Priority-Based Execution Order
1. **Priority 0**: `FilterUnAdjustedModels` - Data quality filtering
2. **Priority 10**: `WarmingLevel` - Time dimension transformation  
3. **Priority 50**: `Concat`, `Template` - Data combination
4. **Priority 100**: `TimeSlice` - Temporal subsetting
5. **Priority 155**: `Localize` - Point-based extraction and bias correction
6. **Priority 200**: `Clip` - Spatial subsetting operations
7. **Priority 750**: `ConvertUnits` - Unit standardization
8. **Priority 7500**: `MetricCalc` - Statistical analysis
9. **Priority 9998**: `UpdateAttributes` - Metadata finalization
10. **Priority 9999**: `Export` - Data output

### Context Management
Each processor updates a shared context dictionary containing:
- Processing steps applied (`_NEW_ATTRS_KEY`)
- Parameter configurations
- Data provenance information
- Error and warning messages

### Data Flow Support
Processors handle multiple data types:
- `xr.Dataset`: Single datasets
- `xr.DataArray`: Single arrays  
- `dict`: Multiple datasets by key
- `list`/`tuple`: Multiple datasets in sequence

## Development Guidelines

### Creating New Processors

1. **Inherit from DataProcessor**:
```python
from climakitae.new_core.processors.abc_data_processor import DataProcessor, register_processor

@register_processor("my_processor", priority=500)
class MyProcessor(DataProcessor):
    def __init__(self, value):
        self.value = value
        self.name = "my_processor"
```

2. **Implement Required Methods**:
```python
def execute(self, result, context):
    # Process the data
    return processed_result

def update_context(self, context):
    # Record processing information
    if _NEW_ATTRS_KEY not in context:
        context[_NEW_ATTRS_KEY] = {}
    context[_NEW_ATTRS_KEY][self.name] = "Processing description"

def set_data_accessor(self, catalog):
    # Optional: Set data access interface
    pass
```

3. **Choose Appropriate Priority**:
   - 0-10: Early filtering and validation
   - 50-100: Data combination and basic transformations
   - 100-500: Spatial and temporal operations
   - 500-1000: Unit conversions and standardization
   - 1000-8000: Analysis and calculations
   - 8000+: Final processing and export

4. **Handle Multiple Data Types**:
```python
def execute(self, result, context):
    match result:
        case dict():
            # Handle dictionary of datasets
            return {key: self._process_single(value) for key, value in result.items()}
        case xr.Dataset() | xr.DataArray():
            # Handle single dataset/array
            return self._process_single(result)
        case list() | tuple():
            # Handle sequence of datasets
            processed = [self._process_single(item) for item in result]
            return type(result)(processed)
```

### Best Practices

1. **Error Handling**: Return data with warnings rather than raising exceptions
2. **Memory Efficiency**: Use lazy evaluation and chunked operations for large datasets
3. **Documentation**: Include comprehensive docstrings with parameter descriptions
4. **Testing**: Write unit tests following the patterns in `tests/new_core/`
5. **Context Updates**: Always record processing steps for reproducibility
6. **Type Hints**: Use proper type annotations for better code clarity

### Integration with ClimateData
Processors are automatically discovered and registered through the import system in `__init__.py`. The `ClimateData` class orchestrates processor execution based on user configuration and data requirements.

For development examples, see `template.py` and existing processor implementations.
