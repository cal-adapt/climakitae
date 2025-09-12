# ClimakitAE Development Guide for AI Agents

## Repository Overview

**ClimakitAE** is a Python toolkit for climate data analysis and retrieval from the Cal-Adapt Analytics Engine. It provides access to downscaled CMIP6 climate data for California with tools for analysis, visualization, and export.

The project serves climate scientists, researchers, and analysts who need programmatic access to high-resolution climate projections for California, supporting both exploratory analysis and production workflows.

- **Size**: ~55 Python files, ~360k lines of code
- **Language**: Python 3.12
- **Type**: Scientific computing library with dual interface architecture
- **Frameworks**: xarray, dask, geopandas, intake-esm for climate data processing
- **Target Runtime**: Python 3.12+ with conda/mamba or uv environments

## Architecture Patterns

### Dual Interface Pattern

**CRITICAL**: ClimakitAE has two distinct user interfaces:

1. **Legacy Interface** (`climakitae.core.data_interface`)
   - Function-based API with DataParameters class
   - `get_data(param_dict)` pattern
   - Maintained for backward compatibility

2. **New Core Interface** (`climakitae.new_core.user_interface`)
   - Fluent/builder pattern with method chaining
   - `ClimateData().variable("tas").experiment("ssp245").get()` pattern
   - Actively developed, preferred for new features

**Migration Strategy**: The new core is actively developed and preferred for new features. Legacy interface is maintained for backward compatibility.

```python
# Legacy approach
from climakitae.core.data_interface import get_data
data = get_data(param_dict)

# New core approach
from climakitae.new_core.user_interface import ClimateData
data = (ClimateData()
        .catalog("cadcat")
        .variable("tasmax")
        .experiment("ssp245")
        .get())
```

### Factory + Registry Pattern in New Core
The `new_core` module uses extensive factory and registry patterns for extensibility:

- **DatasetFactory**: Creates appropriate dataset objects based on query parameters
- **Processor Registry**: Processors registered via decorators with priority-based execution
- **Validator Registry**: Catalog-specific validators ensure data integrity
- **Data Access Registry**: Different data access strategies per catalog type

```python
# Processor registration example
@register_processor(key="spatial_avg", priority=10)
class SpatialAverageProcessor(DataProcessor):
    def process(self, data: xr.Dataset, **kwargs) -> xr.Dataset:
        return data.mean(dim=['lat', 'lon'])

# Validator registration example  
@register_catalog_validator("cadcat")
class CadcatValidator(ParameterValidator):
    def validate(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        # Catalog-specific validation logic
        return validated_parameters
```

### Singleton Pattern for Data Access
- **DataInterface** (legacy) and **DataCatalog** (new core) are singletons managing persistent connections
- They handle intake-esm catalog connections for climate data access
- Manage cached access to boundaries, weather stations, and warming level lookup tables
- Connection pooling and retry logic for reliable data access

### Lazy Evaluation Pattern
Climate datasets use lazy evaluation throughout:
- xarray.Dataset with dask arrays for memory efficiency
- Processing operations build computation graphs without executing
- `.compute()` or `.execute()` triggers actual data loading and computation

## Environment Setup & Build Instructions

### Environment Setup (REQUIRED)
**Always follow these exact steps in order:**

```bash
# Option 1: Using uv (RECOMMENDED for development)
# First ensure you're in the project root
cd /path/to/climakitae

# Create and activate virtual environment
uv venv
source .venv/bin/activate  # or `. .venv/bin/activate`

# Install dependencies and package
uv sync
pip install -e . --no-deps

# Option 2: Using conda/mamba (production)
# Get conda lock file (contains exact versions)
conda create -n climakitae --file conda-linux-64.lock
conda activate climakitae
pip install -e . --no-deps
```

**CRITICAL**: Always use `pip install -e . --no-deps` after environment setup to install climakitae in development mode without reinstalling dependencies.

### Testing Strategy

```bash
# ALWAYS run basic tests (excludes tests requiring external data)
# Use uv if that's your environment
uv run python -m pytest -n auto -m "not advanced" --no-header -q

# OR use regular pytest if using conda
python -m pytest -n auto -m "not advanced" --no-header -q

# Test markers available:
# - "not advanced": Basic tests (2-3 minutes, 541 tests)
# - "advanced": Tests requiring external data connections
# - "integration": Integration tests

# Run with coverage (for CI validation)
pytest --cov=climakitae --cov-report=xml --cov-branch

# Run specific test modules
python -m pytest tests/new_core/test_user_interface.py -v

# Run with different markers
python -m pytest -m "integration"     # Integration tests
python -m pytest -m "advanced"       # Tests requiring external data

# For uv environments, prepend with uv run
uv run python -m pytest -m "not advanced"
```

**TIMING**: Basic tests take ~2.5 minutes. Advanced tests may take much longer and require network access.

**Test Organization Principles**:
- Tests mirror source code structure in `tests/` directory
- Class-based test organization: `TestClimateDataInit`, `TestParameterValidation`
- Mock external dependencies, test internal logic
- Use fixtures in `conftest.py` for shared test data
- Parameterized tests for multiple scenario coverage

### Code Quality and Formatting

```bash
# Required formatting (CI will fail without this)
black climakitae/ tests/
isort climakitae/ tests/

# Check formatting without changes
black --check .
isort --check-only .

# Type checking (recommended)
mypy climakitae/

# Linting
flake8 climakitae/
```

**WARNING**: isort currently shows many files as incorrectly sorted. This is a known issue. Run `isort .` to fix before committing.

### Documentation

```bash
# Install docs dependencies first
pip install -r docs/requirements.txt

# Build documentation locally
cd docs/
make html

# Serve documentation for preview
make serve-docs

# Clean and rebuild
make clean html
```

## Critical Domain Knowledge

### Climate Data Hierarchy
Climate data is organized by these parameters in order of increasing specificity:

1. **`catalog`** - Data collection source
   - `"cadcat"`: Cal-Adapt's primary LOCA2 downscaled data
   - `"renewables"`: Wind/solar renewable energy data  
   - `"climate"`: CMIP6 global climate model data

2. **`activity_id`** - Downscaling methodology
   - `"WRF"`: Weather Research and Forecasting model
   - `"LOCA2"`: Localized Constructed Analogs version 2
   - `"CMIP6"`: Global climate model output

3. **`institution_id`** - Data producing institution
   - `"CNRM"`: Centre National de Recherches Météorologiques
   - `"DWD"`: Deutscher Wetterdienst
   - `"MOHC"`: Met Office Hadley Centre

4. **`source_id`** - Specific climate model simulation
   - `"CNRM-CM6-1"`, `"ACCESS-CM2"`, `"EC-Earth3"`, etc.

5. **`experiment_id`** - Climate scenario
   - `"historical"`: Historical observations/simulations (1850-2014)
   - `"ssp245"`: Shared Socioeconomic Pathway 2-4.5 (moderate emissions)
   - `"ssp370"`: SSP 3-7.0 (high emissions)
   - `"ssp585"`: SSP 5-8.5 (very high emissions)

6. **`table_id`** - Temporal resolution
   - `"1hr"`: Hourly data
   - `"day"`: Daily data  
   - `"mon"`: Monthly data

7. **`grid_label`** - Spatial resolution (California-specific)
   - `"d01"`: 45km resolution (coarse, state-wide)
   - `"d02"`: 9km resolution (medium, regional)
   - `"d03"`: 3km resolution (fine, local)

8. **`variable_id`** - Climate variable
   - `"tasmax"/"tasmin"`: Daily maximum/minimum temperature (°C)
   - `"pr"`: Precipitation (mm/day)
   - `"huss"`: Specific humidity (kg/kg)
   - `"cf"`: Cloud fraction (0-1)

### Temporal Analysis Approaches

**Time-Based Analysis** (traditional approach):
```python
climate_data.timeframe("2030-01-01", "2060-12-31")
```
- Query data for specific calendar year ranges
- Suitable for planning horizons and policy analysis
- Direct mapping to calendar years

**Warming Level Analysis** (climate science approach):
```python
climate_data.warming_level(2.0)  # 2°C global warming
```
- Query data around specific global temperature increases
- More scientifically meaningful for climate impacts
- Accounts for model-specific warming timing
- May return NaN for simulations that don't reach target warming level

### Processing Pipeline Architecture

The new core uses a configurable processing pipeline:

```python
# Pipeline execution in Dataset.execute()
1. Parameter validation → appropriate ParameterValidator
2. Data catalog access → DataCatalog singleton  
3. Data retrieval → intake-esm query execution
4. Sequential processing → registered DataProcessor classes (by priority)
5. Result caching → xarray.Dataset with dask arrays
6. Return → lazy-evaluated xarray.Dataset
```

**Key Processing Steps**:
- **Spatial subsetting**: Geographic boundary application
- **Temporal subsetting**: Date range or warming level filtering  
- **Variable selection**: Climate variable extraction
- **Aggregation**: Spatial/temporal averaging operations
- **Unit conversion**: Standardize units across datasets

## Project Architecture & Layout

### Key Directory Structure

```
climakitae/
├── core/               # Legacy interface
│   ├── data_interface.py   # Main legacy API
│   ├── data_load.py       # Data loading utilities
│   ├── data_export.py     # Export functions
│   └── boundaries.py      # Geographic boundaries
├── new_core/          # New interface (preferred)
│   ├── user_interface.py  # Main new API  
│   ├── dataset_factory.py # Dataset creation
│   ├── processors/        # Data processing pipeline
│   ├── param_validation/  # Parameter validators
│   └── data_access/       # Data access strategies
├── explore/           # Analysis and visualization tools
├── tools/             # Batch processing and indices
├── util/              # Utilities and helpers
└── data/              # Static data files (boundaries, metadata)

tests/                 # Mirror source structure
├── conftest.py        # Shared test fixtures
├── core/             # Legacy interface tests
├── new_core/         # New interface tests
└── ...               # Module-specific tests
```

### Configuration Files

- `setup.cfg`: Package metadata, dependencies, pytest config
- `requirements.txt`: Pinned dependencies for reproducible installs
- `conda-linux-64.lock`: Exact conda environment specification
- `.coveragerc`: Coverage configuration
- `.vscode/pyproject.toml`: Ruff linting configuration (experimental)

## Integration Points

### External Dependencies and Their Roles

**Core Data Handling**:
- **intake-esm**: Climate data catalog interface and metadata management
- **xarray**: N-dimensional labeled arrays, the primary data structure
- **dask**: Parallel computing and out-of-core operations for large datasets
- **netcdf4**: NetCDF file I/O for climate data storage format

**Geospatial Operations**:
- **geopandas**: Spatial boundary handling and geographic operations
- **cartopy**: Map projections and geospatial plotting
- **pyproj**: Coordinate reference system transformations

**Analysis and Visualization**:
- **pandas**: Tabular data manipulation for time series and metadata
- **numpy**: Numerical computing foundation
- **matplotlib**: Plotting and visualization
- **seaborn**: Statistical data visualization

**Parameter Management**:
- **param**: Parameter validation and type checking in legacy interface
- **pydantic**: Data validation and settings management (future migration target)

### Cross-Component Communication Patterns

**Parameter Flow**:
```
User Input → Validation → Query Building → Data Access → Processing → Output
```

**Key Communication Mechanisms**:
- **`UNSET` constant**: From `climakitae.core.constants`, indicates unset parameters across interfaces
- **Query dictionaries**: Standardized parameter format between components
- **Event system**: Processing pipeline uses event-driven architecture for extensibility
- **Caching layer**: Persistent storage of computed results and metadata

**Data Export Pipeline**:
```python
# Multi-format export support
from climakitae.core.data_export import export

# Export formats: NetCDF (default), CSV, Zarr, GeoTIFF
export(dataset, "output.nc", format="netcdf")
export(dataset, "output.zarr", format="zarr") 
```

### Configuration and Settings

**Data Sources**:
- Station data: `climakitae/data/historical_wx_stations.csv`
- Boundary definitions: `climakitae/data/` (various geospatial files)
- Warming level tables: `climakitae/data/gwl_*.csv`
- Variable metadata: `climakitae/data/variable_descriptions.csv`

**Environment Configuration**:
- Intake catalog endpoints configurable via environment variables
- Dask cluster configuration for parallel processing
- Cache directory settings for persistent storage

## Continuous Integration

### GitHub Workflows

**Main branch** (`.github/workflows/ci-main.yml`):
- Runs on pushes to main
- Black formatting check
- Full test suite with coverage
- Uploads to codecov

**Feature branches** (`.github/workflows/ci-not-main.yml`):
- Runs on pushes to non-main branches
- Black formatting check
- Basic tests only (`-m "not advanced"`)
- Advanced tests run only if PR labeled "Advanced Testing"

### Validation Checklist

Before submitting PRs, ensure:
1. `black climakitae/ tests/` passes
2. `isort climakitae/ tests/` passes  
3. `uv run python -m pytest -n auto -m "not advanced"` passes
4. New tests follow patterns in `tests/` directory
5. Code follows dual interface patterns

## Common Pitfalls and Solutions

### Interface-Specific Issues

**New Core Method Chaining**:
```python
# INCORRECT: Query is reset after .get()
data1 = climate_data.variable("tasmax").get()
data2 = climate_data.variable("pr").get()  # Lost tasmax setting

# CORRECT: Chain before .get() or create new instance
data1 = climate_data.variable("tasmax").get()
data2 = ClimateData().variable("pr").get()
```

**Parameter Validation**:
```python
# INCORRECT: Generic validation
if parameter is None:
    raise ValueError("Parameter required")

# CORRECT: Use catalog-specific validators
validator = get_validator(catalog_type)
validated_params = validator.validate(parameters)
```

### Data Handling Pitfalls

**Grid Label Specifications**:
```python
# Always include resolution information in user-facing messages
grid_help = {
    "d01": "45km resolution (state-wide coverage)",
    "d02": "9km resolution (regional coverage)", 
    "d03": "3km resolution (local/urban coverage)"
}
```

**Warming Level Edge Cases**:
```python
# Handle models that don't reach warming targets
warming_data = climate_data.warming_level(3.0).get()
if warming_data is None or warming_data.isnull().all():
    print("Model simulation does not reach 3°C warming level")
```

**Memory Management**:
```python
# INCORRECT: Loading full dataset into memory
data = dataset.load()  # Can cause OOM errors

# CORRECT: Use lazy evaluation
data = dataset  # Dask arrays, computed on demand
result = data.mean().compute()  # Compute only final result
```

### Testing and Development Pitfalls

**Mock Configuration**:
```python
# INCORRECT: Mock at definition location
@patch('climakitae.core.data_interface.intake')

# CORRECT: Mock at import location  
@patch('climakitae.new_core.user_interface.DatasetFactory')
```

**Test Data Realism**:
```python
# Create realistic test data matching actual structures
test_dataset = xr.Dataset({
    'tasmax': (['time', 'lat', 'lon'], np.random.rand(365, 100, 100)),
    'time': pd.date_range('2020-01-01', periods=365),
    'lat': np.linspace(32.0, 42.0, 100),  # California latitude range
    'lon': np.linspace(-124.0, -114.0, 100)  # California longitude range
})
```

### Error Patterns to Avoid

1. **Memory issues**: Climate datasets are large, use lazy evaluation
2. **Import sorting**: Run `isort .` before committing
3. **Test isolation**: Don't mock internal implementation details
4. **Environment mixing**: Don't mix conda and pip for dependencies
5. **Async operations**: Handle dask lazy operations properly

## Performance Considerations

### Large Dataset Handling
- Use chunked operations for datasets > 1GB
- Prefer spatial subsets before temporal operations
- Leverage dask distributed computing for multi-core processing
- Monitor memory usage with `dask.diagnostics`

### Optimization Strategies
```python
# Efficient spatial subsetting
subset = data.sel(lat=slice(34.0, 36.0), lon=slice(-120.0, -118.0))

# Chunked temporal operations
monthly_avg = data.resample(time='M').mean().compute()

# Parallel processing configuration
import dask
dask.config.set(scheduler='threads', num_workers=4)
```

## Development Workflows

### Making Changes

1. **Always install in development mode**: `pip install -e . --no-deps`
2. **Run tests after changes**: Basic tests take ~2.5 minutes
3. **Format before committing**: `black` and `isort` are required
4. **Add tests**: Mirror source structure in `tests/`
5. **Use new_core for new features**: Legacy core is maintenance-only

### Validation Checklist

Before submitting PRs, ensure:
1. `black climakitae/ tests/` passes
2. `isort climakitae/ tests/` passes  
3. `uv run python -m pytest -n auto -m "not advanced"` passes
4. New tests follow patterns in `tests/` directory
5. Code follows dual interface patterns

## Quick Reference

**Start development**: `source .venv/bin/activate && pip install -e . --no-deps`
**Run tests**: `uv run python -m pytest -n auto -m "not advanced" --no-header -q`  
**Format code**: `black . && isort .`
**Main API entry points**: 
- Legacy: `climakitae.core.data_interface.get_data`
- New: `climakitae.new_core.user_interface.ClimateData`

**Trust these instructions** - they're validated against the actual codebase. Only search for additional information if these instructions are incomplete or incorrect.
