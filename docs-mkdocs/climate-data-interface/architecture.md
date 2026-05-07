# Architecture & Internals

This guide explains the internal architecture of climakitae's `new_core` module for developers and contributors. It covers system design, component responsibilities, key patterns, and how to extend the system.

!!! note "Audience"
    This is the **contributor / extender** reference. If you only want to *use*
    the ClimateData interface, start with [Core Concepts](concepts.md) or the
    [How-To Guides](howto.md).

## On this page

- [System Overview](#system-overview) — pipeline architecture and data flow
- [Core Components](#core-components) — `ClimateData`, `DatasetFactory`, `Dataset`, `DataCatalog`, registries
- [Key Design Patterns](#key-design-patterns) — lazy evaluation, fluent builder, processor priorities, history tracking
- [Extending the System](#extending-the-system) — adding processors, validators, catalogs
- [Threading & Concurrency](#threading-concurrency) — per-thread instances, catalog singleton safety
- [Internal APIs](#internal-apis) — registry inspection helpers
- [Testing New Components](#testing-new-components) — unit / integration patterns
- [Common Pitfalls](#common-pitfalls)
- [Resources](#resources)

---

## System Overview

The `new_core` module uses a **pipeline architecture** with layered responsibilities:

```
┌─────────────────────────────────────────────────────────────┐
│ ClimateData (user_interface.py)                             │
│ Fluent API: .catalog("cadcat").variable("tas").get()        │
└────────────────┬────────────────────────────────────────────┘
                 │ Creates
                 ▼
┌─────────────────────────────────────────────────────────────┐
│ DatasetFactory (dataset_factory.py)                         │
│ - Resolves catalog + validates parameters                   │
│ - Creates configured Dataset object                         │
└────────────────┬────────────────────────────────────────────┘
                 │ Instantiates
                 ▼
┌─────────────────────────────────────────────────────────────┐
│ Dataset.execute() (dataset.py)                              │
│ - Validates query against catalog validators                │
│ - Retrieves data from DataCatalog                           │
│ - Runs processors in priority order                         │
│ - Returns lazy xarray.Dataset                               │
└────────────────┬────────────────────────────────────────────┘
         ┌───────┴───────┬──────────────┬──────────────┐
         │               │              │              │
         ▼               ▼              ▼              ▼
    ┌────────┐   ┌──────────────┐  ┌──────────┐  ┌────────────┐
    │ Query  │   │ Validators   │  │Processors│  │ DataAccess │
    │ Params │   │ (Registry)   │  │(Registry)│  │ (Singleton)│
    └────────┘   └──────────────┘  └──────────┘  └────────────┘
                        │
              ┌─────────┴─────────┐
              │                   │
        ┌─────┴──────┐   ┌────────┴───────┐
        │  Catalog   │   │  Processor     │
        │ Validators │   │  Validators    │
        └────────────┘   └────────────────┘
```

### Data Flow

1. **User calls ClimateData**: Builds fluent query chain (catalog, variable, experiment, etc.)
2. **`.get()` triggers DatasetFactory**: Factory resolves catalog type and creates Dataset instance
3. **Dataset validates**: Runs catalog-specific validator to normalize/validate parameters
4. **Dataset retrieves data**: Queries DataCatalog singleton to fetch data from intake
5. **Processors execute in priority order**: Transform data (clip, time_slice, export, etc.)
6. **Returns lazy xarray**: Dask-backed Dataset, computation deferred until explicit `.compute()`

---

## Core Components

### 1. ClimateData (User Interface)

**File**: `climakitae/new_core/user_interface.py` (~1200 lines)

**Responsibility**: Fluent API entry point for users. All setter methods chain and return `self`.

**Key Methods**:
```python
cd = ClimateData(verbosity=0)
cd.catalog("cadcat")                    # Set catalog
cd.activity_id("WRF")                   # Set downscaling method
cd.variable("t2max")                    # Set variable
cd.experiment_id("ssp245")              # Set scenario
cd.table_id("mon")                      # Set temporal resolution
cd.grid_label("d03")                    # Set spatial resolution
cd.processes({"clip": "Los Angeles"})   # Set processors
cd.show_variable_options()              # Discover available variables
data = cd.get()                         # Execute query, return xarray
```

**Internal Contract**:
- All parameter setters must `return self`
- Query is reset after `.get()` is called
- Parameter values stored internally, passed to DatasetFactory on `.get()`
- Discovery methods (`show_*_options()`) delegate to DatasetFactory

---

### 2. DatasetFactory

**File**: `climakitae/new_core/dataset_factory.py` (~600 lines)

**Responsibility**: Catalog resolution and Dataset construction.

**Key Methods**:
```python
factory = DatasetFactory()

# Resolve catalog by name (with fuzzy matching)
catalog_obj = factory.get_catalog("cadcat")

# Get all valid options for a parameter
variables = factory.get_variable_options("cadcat", "WRF")
institutions = factory.get_institution_id_options("cadcat")

# Create configured Dataset for a query
query = {"catalog": "cadcat", "variable_id": "t2max", ...}
dataset = factory.create_dataset(query)
```

**Internal Logic**:
1. Receives query dict from ClimateData
2. Resolves catalog key (name → catalog type)
3. Selects appropriate catalog validator (e.g., `CadcatParamValidator`)
4. Creates Dataset instance with validator and default processors

**Key Property: Catalog Mapping**
- `cadcat` → main climate data (LOCA2, WRF)
- `renewable energy generation` → wind/solar capacity factors
- `hdp` → historical data platform (weather stations)

---

### 3. Dataset (Pipeline Executor)

**File**: `climakitae/new_core/dataset.py` (~380 lines)

**Responsibility**: Execute the data processing pipeline.

**Key Method**:
```python
result = dataset.execute(query_dict)
```

**Pipeline Stages**:

1. **Validation**: Run catalog-specific validator
   - Normalizes parameter names
   - Validates required parameters
   - Suggests alternatives for typos
   - Returns validated query or None

2. **Data Access**: Query DataCatalog for data
   - Resolves intake catalog
   - Executes intake-esm query
   - Returns xarray.Dataset with dask arrays (lazy)

3. **Processing**: Run registered processors in priority order
   - Each processor receives data + context dict
   - Processors transform data while preserving laziness
   - Processors update context with metadata
   - Execution order determined by priority value

4. **Return**: xarray.Dataset (lazy, ready for `.compute()`)

**Context Dict**: Processors store metadata in `context[_NEW_ATTRS_KEY]`
```python
context = {}
result = dataset.execute(query, context=context)
# context now contains metadata from all processors
```

---

### 4. DataCatalog (Data Access Layer) {#data-access-layer}

**File**: `climakitae/new_core/data_access/data_access.py` (~620 lines)

**Responsibility**: Thread-safe singleton managing all catalog connections.

**Key Properties**:
```python
catalog = DataCatalog()

catalog.data        # intake_esm for main climate data
catalog.boundary    # intake for geographic boundaries
catalog.renewables  # intake_esm for renewable energy
catalog.hdp         # intake_esm for historical data platform
catalog.catalog_df  # Merged DataFrame of all ESM catalogs
catalog.boundaries  # Lazy-loading Boundaries manager
```

**Key Methods**:
```python
# Get data with explicit catalog key (thread-safe)
data = catalog.get_data(query_dict, catalog_key="cadcat")

# Resolve catalog key with fuzzy matching
key = catalog.resolve_catalog_key("cadcat")

# List available boundaries
boundaries = catalog.list_clip_boundaries()

# Get station metadata
stations = catalog.get_stations()
```

**Thread-Safety Pattern**:
```python
# ✅ Thread-safe: Pass catalog_key as parameter
def get_data(query, catalog_key="cadcat"):
    # Lookup is atomic, no state mutation
    return self._catalogs[catalog_key].search(**query).to_dask()

# ❌ Not thread-safe: State stored on singleton
self._current_key = catalog_key  # DON'T DO THIS
```

**Connection Management**:
- Catalogs loaded lazily on first access
- intake-esm catalogs cached in `_catalogs` dict
- Boundaries loaded on-demand via lazy property
- Thread lock ensures atomic initialization

---

### 5. ParameterValidator (Registry Pattern)

**File**: `climakitae/new_core/param_validation/abc_param_validation.py` (~550 lines)

**Responsibility**: Validate and normalize query parameters.

**Two Registry Types**:

#### Catalog Validators
Validate parameters for entire data catalogs (one per catalog type):

```python
@register_catalog_validator("cadcat")
class CadcatParamValidator(ParameterValidator):
    def is_valid_query(self, query):
        # Catalog-specific validation
        if "variable_id" not in query:
            return None  # Invalid
        return self._validate_and_normalize(query)
```

#### Processor Validators
Validate parameters for specific processors:

```python
@register_processor_validator("clip")
def validate_clip(value, **kwargs):
    # Processor-specific validation
    if not isinstance(value, (str, tuple, list)):
        return False
    return True
```

**Validation Flow**:
1. Dataset calls `validator.is_valid_query(query)`
2. Validator checks required parameters
3. Validator searches catalog_df for matching datasets
4. Validator calls processor validators for each processor
5. Returns validated query (normalized keys) or None if invalid

**Key Methods**:
```python
# Load and cache catalog dataframe
validator.load_catalog_df()

# Find matching catalog entries
matches = validator.find_catalog_entries({"variable_id": "t2max"})

# Get default processors for this query
processors = validator.get_default_processors(query)

# Suggest alternatives for typos
suggestions = validator._get_closest_options("tasxxx", "variable_id", n=3)
```

---

### 6. DataProcessor (Registry Pattern)

**File**: `climakitae/new_core/processors/abc_data_processor.py` (~150 lines)

**Responsibility**: Transform data while preserving lazy evaluation.

**Required Methods**:

```python
@register_processor("my_processor", priority=80)
class MyProcessor(DataProcessor):
    def __init__(self, value):
        self.value = value
        self.name = "my_processor"
        self.needs_catalog = False  # True if needs DataCatalog access
        
    def execute(self, result, context) -> xr.Dataset:
        """Transform data. Preserve lazy evaluation with dask."""
        # ✅ CORRECT: Return new object, don't mutate
        return result.sel(lat=slice(33, 35))
        
    def update_context(self, context):
        """Record metadata about processing step."""
        if _NEW_ATTRS_KEY not in context:
            context[_NEW_ATTRS_KEY] = {}
        context[_NEW_ATTRS_KEY]["my_processor"] = "Applied successfully"
        
    def set_data_accessor(self, catalog):
        """Optional: Receive DataCatalog reference if needs_catalog=True."""
        self._catalog = catalog
```

**Priority guidelines (real values used by the shipped processors):**

```
0–10     : Pre-processing (filter_unadjusted_models=0, drop_leap_days=1,
          convert_units=5, warming_level=10) — catalog refinement & GWL subset
50–70    : Combination + correction + spatial
          (concat=50, bias_adjust_model_to_station=60, clip=65,
           convert_to_local_time=70)
150–7500 : Temporal subsetting & metric computation
          (time_slice=150, metric_calc=7500)
9998–9999: Finalization (update_attributes=9998, export=9999)
```

When registering a custom processor, pick a priority that puts it in the
phase that matches its intent. See the [Processors index](./processors/index.md)
for the full registry.

**Key Rules**:  
- **No in-place mutation**: Always return new objects  
- **Preserve laziness**: Don't call `.load()` or `.compute()`  
- **Handle edge cases**: Return data with warnings instead of raising  
- **Update context**: Document what the processor did  

**Registry Access**:
```python
from climakitae.new_core.processors.abc_data_processor import _PROCESSOR_REGISTRY

# Inspect registered processors
for key, (cls, priority) in sorted(_PROCESSOR_REGISTRY.items(), 
                                    key=lambda x: x[1][1]):
    print(f"{key}: {cls.__name__} (priority={priority})")
```

---

## Key Design Patterns

### Understanding Processors: Spatial Subsetting {#spatial-subsetting}

The **Clip processor** is the primary tool for spatial subsetting. It extracts data for specific regions while maintaining data integrity and lazy evaluation.

**Key Characteristics**:

- Supports 5 input modes: named boundaries, points, bounding boxes, weather stations, shapefiles
- Preserves coordinate systems and projections
- Works with lazy dask arrays (no premature loading)
- Returns masked or clipped Dataset depending on mode

**Efficiency Principle**: Always apply spatial subsetting EARLY in your query chain, before aggregation or computation, to minimize data movement.

```python
# ✅ EFFICIENT: Clip first, then aggregate
data = (cd
    .variable("tasmax")
    .processes({
        "clip": "Los Angeles",
        "time_slice": ("2030", "2060")
    })
    .get())
mean_temp = data["tasmax"].mean(dim=["lat", "lon"]).compute()

# ❌ INEFFICIENT: Load all data then subset
data = cd.variable("tasmax").get()
clipped = data.sel(lat=slice(33.5, 35), lon=slice(-119, -117))
mean_temp = clipped["tasmax"].mean().compute()
```

See [Processor: Clip](./processors/clip.md) for detailed API reference.

### Understanding Processors: Bias Correction {#bias-correction}

WRF model output can be bias-corrected using historical weather station observations to improve local accuracy.

**Purpose**: Adjust systematic model bias while preserving projected climate trends using Quantile Delta Mapping (QDM).

**Current Scope**:

- ✅ WRF data only (not LOCA2)
- ✅ Hourly temperature (t2) only
- ✅ HadISD weather stations (~600 globally, ~200 western US)

**When to Use**:

- Local impact assessment where historical accuracy matters
- Building/infrastructure design requiring site-specific bias correction
- When observation-corrected distribution is important

**When NOT to Use**:

- Regional/state-level planning (raw model suitable for large areas)
- LOCA2 data (already bias-corrected during downscaling)
- Other variables/resolutions (not yet supported)

See [Processor: Bias Adjust Model to Station](./processors/bias_adjust_model_to_station.md) for detailed usage.

### Understanding Processors: Data Export Pipeline {#data-export-pipeline}

The **Export processor** writes climate data to disk in multiple formats optimized for different use cases.

**Supported Formats**:

- **NetCDF**: Standard scientific format with CF conventions (default)
- **Zarr**: Cloud-optimized chunked storage for large datasets
- **CSV**: Tabular format for time series and spreadsheet analysis
- **GeoTIFF**: Raster format compatible with GIS software (QGIS, ArcGIS)

**Key Features**:

- Handles gridded datasets, multi-point extractions, and point collections
- Optional location-based filenames (e.g., `data_34.05N_118.25W.nc`)
- S3 cloud storage support (Zarr format only)
- Export method options: `"data"`, `"skip_existing"`, `"none"`
- Automatic format inference or explicit specification

**Efficiency**:

- Export should be the LAST processor (priority 9999)
- Processes build computation graph, export writes results
- For large datasets, prefer Zarr for cloud storage or incremental processing

```python
# Simple NetCDF export
data = (cd
    .variable("tasmax")
    .processes({
        "time_slice": ("2030-01-01", "2060-12-31"),
        "clip": "Los Angeles",
        "export": {
            "filename": "la_temperature",
            "file_format": "NetCDF"
        }
    })
    .get())

# Cloud-optimized Zarr export
data = (cd
    .variable("pr")
    .processes({
        "export": {
            "filename": "precipitation_data",
            "file_format": "Zarr",
            "mode": "s3"  # Store on S3
        }
    })
    .get())
```

See [Processor: Export](./processors/export.md) for complete API reference.

### Understanding Processors: Context Metadata {#context-metadata}

Each processor updates a **context dictionary** to track what transformations were applied.

**Purpose**: Maintain metadata about processing steps for reproducibility and debugging.

**How It Works**:
```python
from climakitae.core.constants import _NEW_ATTRS_KEY

class MyProcessor(DataProcessor):
    def update_context(self, context):
        """Record metadata about this processing step."""
        if _NEW_ATTRS_KEY not in context:
            context[_NEW_ATTRS_KEY] = {}
        context[_NEW_ATTRS_KEY]["my_processor"] = "Applied filter > 100"

# Access processing history
result, context = dataset.execute_with_context(query)
for step, description in context["_new_attributes"].items():
    print(f"{step}: {description}")
```

**Benefits**:

- Track all processing steps applied to data
- Enable reproducible analysis workflows
- Debug unexpected data anomalies
- Document data provenance

---

### 1. Fluent Interface (Method Chaining)

All parameter setters return `self` to enable chaining:

```python
# ✅ CORRECT: Chain multiple setters before .get()
data = (ClimateData()
    .catalog("cadcat")
    .activity_id("WRF")
    .variable("t2max")
    .get())

# ❌ INCORRECT: State resets after .get()
cd = ClimateData()
d1 = cd.variable("tasmax").get()
d2 = cd.variable("pr").get()  # Lost context - create new instance
```

**Implementation**:
```python
def variable(self, value):
    self._variable_id = value
    return self  # Critical: return self
```

### 2. Registry + Decorator Pattern {#registry-pattern}

Components register at import time using decorators:

```python
# Processor registration
@register_processor("clip", priority=65)
class Clip(DataProcessor):
    ...

# Validator registration
@register_catalog_validator("cadcat")
class CadcatValidator(ParameterValidator):
    ...

# Processor parameter validator
@register_processor_validator("warming_level")
def validate_warming_level(value, **kwargs):
    ...
```

**Registry Storage**:
```python
_PROCESSOR_REGISTRY = {}          # {key: (class, priority)}
_CATALOG_VALIDATOR_REGISTRY = {}  # {catalog: validator_class}
_PROCESSOR_VALIDATOR_REGISTRY = {} # {processor_key: validator_fn}
```

**Discovery**:
```python
# List all registered processors
for key in _PROCESSOR_REGISTRY:
    print(key)

# Get processor class and priority
cls, priority = _PROCESSOR_REGISTRY["clip"]
```

### 3. Singleton Pattern (Thread-Safe)

`DataCatalog` uses double-checked locking:

```python
class DataCatalog(dict):
    _instance = UNSET
    _lock = threading.Lock()

    def __new__(cls):
        # Fast path (no lock)
        if cls._instance is not UNSET:
            return cls._instance
        
        # Slow path (with lock) - only for first initialization
        with cls._lock:
            if cls._instance is UNSET:
                cls._instance = super().__new__(cls)
                cls._instance._initialized = False
        return cls._instance
```

**Thread-Safety Contract**:

- Pass mutable state as parameters, not stored on singleton
- Each thread can safely call methods concurrently
- Catalog connections are immutable after initialization

### 4. Lazy Evaluation (Dask Integration) {#lazy-evaluation}

All operations return dask-backed xarray objects:

```python
# ✅ CORRECT: Computation deferred
result = dataset.execute(query)  # Returns lazy xarray.Dataset
final = result.mean(dim='time').compute()  # Triggers computation

# ❌ INCORRECT: Eager loading
result = dataset.execute(query).load()  # Loads entire dataset!
final = result.mean(dim='time')  # Already computed
```

**Why Lazy Evaluation Matters**:

- Climate datasets are huge (100GB+ common)
- `.load()` exhausts memory
- Operations build a computation graph
- `.compute()` only evaluates what's needed
- Subset data spatially BEFORE aggregating to reduce computation

---

## Extending the System

### Adding a New Processor {#add-a-processor-4-step-guide}

**Steps**:

1. **Create processor file** in `climakitae/new_core/processors/`:
   ```python
   # my_processor.py
   from climakitae.new_core.processors.abc_data_processor import (
       DataProcessor, register_processor
   )
   from climakitae.core.constants import _NEW_ATTRS_KEY
   
   @register_processor("my_key", priority=80)
   class MyProcessor(DataProcessor):
       def __init__(self, value):
           self.value = value
           self.name = "my_key"
           self.needs_catalog = False
       
       def execute(self, result, context):
           # Transform data
           return result.where(result > self.value)
       
       def update_context(self, context):
           if _NEW_ATTRS_KEY not in context:
               context[_NEW_ATTRS_KEY] = {}
           context[_NEW_ATTRS_KEY][self.name] = f"Filtered > {self.value}"
       
       def set_data_accessor(self, catalog):
           if self.needs_catalog:
               self._catalog = catalog
   ```

2. **Create validator** (if processor has parameters):
   ```python
   # my_processor_param_validator.py
   from climakitae.new_core.param_validation.abc_param_validation import (
       register_processor_validator
   )
   
   @register_processor_validator("my_key")
   def validate_my_processor(value, **kwargs):
       if not isinstance(value, (int, float)):
           return False
       return True
   ```

3. **Add tests**:
   ```python
   # tests/new_core/processors/test_my_processor.py
   class TestMyProcessor:
       def test_execute_threshold(self):
           processor = MyProcessor(100)
           data = xr.Dataset({"temp": (["x"], [50, 100, 150])})
           result = processor.execute(data, {})
           assert result["temp"].values[0] != result["temp"].values[0]  # NaN
   ```

4. **Import in module**:
   Ensure the processor is imported in `climakitae/new_core/processors/__init__.py`:
   ```python
   from climakitae.new_core.processors.my_processor import MyProcessor
   ```

### Adding a New Catalog Validator

**Steps**:

1. **Create validator** in `climakitae/new_core/param_validation/`:
   ```python
   # my_catalog_param_validator.py
   from climakitae.new_core.param_validation.abc_param_validation import (
       ParameterValidator, register_catalog_validator
   )
   
   @register_catalog_validator("my_catalog")
   class MyCatalogValidator(ParameterValidator):
       def is_valid_query(self, query):
           # Validate required parameters
           if "required_param" not in query:
               return None
           # Validate and normalize
           return self._validate_and_normalize(query)
   ```

2. **Register catalog in DatasetFactory**:
   ```python
   # In dataset_factory.py
   self._catalog_to_validator_map = {
       "cadcat": CadcatParamValidator(),
       "renewable energy generation": RenewablesParamValidator(),
       "my_catalog": MyCatalogValidator(),
   }
   ```

3. **Add tests**:
   ```python
   # tests/new_core/param_validation/test_my_catalog_param_validator.py
   class TestMyCatalogValidator:
       def test_valid_query(self):
           validator = MyCatalogValidator()
           query = {"required_param": "value"}
           result = validator.is_valid_query(query)
           assert result is not None
   ```

### Adding a New Data Catalog

**Steps**:

1. **Add URL constant** in `climakitae/core/paths.py`:
   ```python
   MY_CATALOG_URL = "s3://bucket/path/my-catalog.json"
   ```

2. **Initialize in DataCatalog**:
   ```python
   # In data_access.py __init__()
   self._my_catalog_url = paths.MY_CATALOG_URL
   ```

3. **Add lazy property**:
   ```python
   @property
   def my_catalog(self):
       if not hasattr(self, "_my_catalog"):
           self._my_catalog = intake.open_esm_metastore(
               self._my_catalog_url
           )
       return self._my_catalog
   ```

4. **Register in DatasetFactory**:
   ```python
   self._catalog_keys = {
       "cadcat": "data",
       "my_catalog": "my_catalog",  # Maps to property name
   }
   ```

---

## Threading & Concurrency

### Thread-Safe Data Queries

```python
import concurrent.futures
from climakitae.new_core.user_interface import ClimateData

# ✅ CORRECT: Create new ClimateData instance per thread
def fetch_scenario(scenario):
    cd = ClimateData(verbosity=-1)  # New instance
    return (cd
        .catalog("cadcat")
        .activity_id("WRF")
        .experiment_id(scenario)
        .get())

with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
    futures = {
        executor.submit(fetch_scenario, s): s
        for s in ["historical", "ssp245", "ssp370"]
    }
    results = {futures[f]: f.result() for f in concurrent.futures.as_completed(futures)}

# ❌ INCORRECT: Sharing ClimateData across threads
cd = ClimateData()
with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
    futures = [executor.submit(cd.experiment_id(s).get) for s in scenarios]
    # Race condition: experiment_id() overwrites shared state
```

### DataCatalog Thread Safety

The `DataCatalog` singleton is thread-safe by design:

```python
# ✅ Thread-safe: Multiple threads accessing same catalog
def query_data(scenario):
    catalog = DataCatalog()  # Same instance for all threads
    return catalog.get_data(
        {"variable_id": "t2max", "experiment_id": scenario},
        catalog_key="cadcat"  # Passed as parameter, not stored
    )

# Each thread can safely call without locking
with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
    futures = [executor.submit(query_data, s) for s in scenarios]
    results = [f.result() for f in futures]
```

---

## Internal APIs

### Accessing Registries Programmatically

**Processors**:
```python
from climakitae.new_core.processors.abc_data_processor import _PROCESSOR_REGISTRY

# List all processors in priority order
for key, (cls, priority) in sorted(
    _PROCESSOR_REGISTRY.items(),
    key=lambda x: x[1][1]  # Sort by priority
):
    print(f"{priority:4d}: {key:20s} ({cls.__name__})")
```

**Validators**:
```python
from climakitae.new_core.param_validation.abc_param_validation import (
    _CATALOG_VALIDATOR_REGISTRY, _PROCESSOR_VALIDATOR_REGISTRY
)

# List all catalog validators
for catalog_key, validator_class in _CATALOG_VALIDATOR_REGISTRY.items():
    print(f"Catalog: {catalog_key:30s} → {validator_class.__name__}")

# List all processor validators
for processor_key, validator_fn in _PROCESSOR_VALIDATOR_REGISTRY.items():
    print(f"Processor: {processor_key:30s} → {validator_fn.__name__}")
```

### Context Dictionary

Processors communicate via the `context` dict:

```python
result = dataset.execute(query, context={})

# After execution, context contains metadata:
# {
#     "climate_data_attributes": {
#         "clip": "Clipped to bounding box (34.0, 35.0)",
#         "time_slice": "Subset to 2015-01-01 to 2015-12-31",
#         ...
#     }
# }
```

**Access in processor**:
```python
from climakitae.core.constants import _NEW_ATTRS_KEY

def execute(self, result, context):
    if _NEW_ATTRS_KEY not in context:
        context[_NEW_ATTRS_KEY] = {}
    context[_NEW_ATTRS_KEY]["my_processor"] = "Processed"
    return result
```

---

## Testing New Components

### Test Structure

Tests mirror source structure:

- Source: `climakitae/new_core/processors/clip.py`
- Tests: `tests/new_core/processors/test_clip.py`

### Mocking Strategy

**Patch at import location** (not definition):

```python
# ✅ CORRECT: Patch where imported
@patch("climakitae.new_core.user_interface.DatasetFactory")
def test_init(self, mock_factory):
    ...

# ❌ INCORRECT: Patch at definition
@patch("climakitae.new_core.dataset_factory.DatasetFactory")
```

### Test Data Fixtures

Use `tests/conftest.py` fixtures for realistic datasets:

```python
def test_my_processor(test_data_2022_monthly_45km):
    """Test processor with sample xarray dataset."""
    processor = MyProcessor(value=100)
    result = processor.execute(test_data_2022_monthly_45km, {})
    assert result is not None
```

---

## Common Pitfalls

| Pitfall | Problem | Solution |
|---------|---------|----------|
| **In-place mutation** | Breaks lazy evaluation | Return new objects: `return data.where(...)` not `data.values[:] = ...` |
| **Eager loading** | OOM errors on large datasets | Never call `.load()` or `.compute()` in processors |
| **Shared mutable state** | Race conditions in threads | Pass parameters, don't store on singleton |
| **Query reset after .get()** | Lost context in loops | Create new `ClimateData()` per query |
| **Wrong patch location** | Mocks don't intercept calls | Patch import location: `"module.ClassName"` |
| **Forgetting registration** | Components not discovered | Use `@register_*` decorators at class definition |
| **Context not updated** | No metadata about processing | Update `context[_NEW_ATTRS_KEY]` in processors |
| **Priority conflicts** | Wrong execution order | Pick a priority that places the processor in the correct phase (see [Processors index](./processors/index.md)). Lower numbers run first. |

---

## Resources

- **Registry Pattern**: `climakitae/new_core/processors/abc_data_processor.py`
- **Processor Template**: `climakitae/new_core/processors/template.py`
- **ClimateData**: `climakitae/new_core/user_interface.py`
- **Tests**: `tests/new_core/`
