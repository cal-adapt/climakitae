# New Core Architecture

This directory contains a redesigned, modular architecture for climakitae's core data retrieval and processing functionality. The new design provides a clean separation of concerns, improved extensibility, and enhanced maintainability compared to the existing monolithic approach in [`climakitae.core`](../core/).

## Overview

The new architecture is built around a hierarchical class system that separates data representation, dataset management, validation, cataloging, and processing into distinct, focused components. This modular approach enables:

- **Easy expansion to new data sources** (beyond AWS S3 and Pangeo catalogs)
- **Flexible catalog management** for different data formats and storage systems
- **Extensible validation** for diverse parameter types and constraints
- **Pluggable processing pipelines** for complex data transformations
- **Clean separation of concerns** between data access, validation, and processing

To explore how the new architecture functions from a user-perspective you can go open the `example.ipynb` in the primary `climakitae` folder.

## Class Hierarchy

### 1. ClimateData (Top-Level Data Container)

The `ClimateData` class serves as the primary interface for users, representing a complete climate dataset with its associated metadata, spatial/temporal bounds, and processing state.

**Key Responsibilities:**
- Holds the actual xarray Dataset/DataArray with climate data
- Maintains comprehensive metadata (variable info, units, provenance, etc.)
- Tracks spatial and temporal bounds
- Provides user-friendly methods for data access and visualization
- Will eventually integrate with [`climakitaegui`](https://github.com/cal-adapt/climakitaegui) for interactive visualization

**Benefits:**
- **Immutable data representation**: Ensures data integrity throughout processing pipelines
- **Rich metadata preservation**: Maintains full provenance and description information
- **Standardized interface**: Provides consistent API regardless of underlying data source
- **Integration ready**: Will work pretty seamlessly with existing [`climakitae.explore`](../explore/) and [`climakitae.tools`](../tools/) modules

### 2. DatasetFactory (Dataset Creation Orchestrator)

The `DatasetFactory` class orchestrates the creation of `ClimateData` objects by coordinating between catalogs, validators, and processors.

**Key Responsibilities:**
- Validates user parameters using appropriate `ParameterValidator` instances
- Queries `DataCatalog` to locate and access data sources
- Applies `Processors` for subsetting, unit conversion, and transformations
- Assembles final `ClimateData` objects with complete metadata

**Benefits:**
- **Single point of coordination**: Centralizes the complex workflow of data retrieval
- **Flexible parameter handling**: Can accommodate different validation schemes for different data types
- **Extensible processing**: Easy to add new processing steps without breaking existing functionality
- **Error handling**: Provides comprehensive error reporting and recovery strategies

### 3. Dataset (Abstract Base for Data Sources)

The `Dataset` class provides an abstract interface for different types of climate data sources, and gives continued access to data manipulation processes such as clipping, slicing, etc. However, this has not been implemented at the user level yet, but can easily be extended. This class enables agnostic support for diverse input validation, storage systems, file formats, and processing chains.

**Key Responsibilities:**
- Defines standard interface for data access across different storage systems
- Handles authentication and connection management
- Provides metadata extraction and standardization
- Implements data streaming and chunking strategies

**Extensibility Benefits:**
- **Multi-source support**: Easy to add new data sources (Google Cloud, Azure, local files, APIs)
- **Storage agnostic**: Can work with NetCDF, Zarr, HDF5, or custom formats
- **Authentication flexibility**: Can be extended to support different credential and access patterns should the need arise
- **Performance optimization**: Allows source-specific optimizations for data access

### 4. ParameterValidator (Input Validation Engine)

The `ParameterValidator` class provides flexible, extensible validation for user inputs, supporting complex validation rules and cross-parameter dependencies.

**Key Responsibilities:**
- Validates individual parameters (catalog keys mostly for now)
- Provides helpful error messages and suggestions
- Supports dynamic validation based on catalog contents

**Extensibility Benefits:**
- **Custom validation rules**: Easy to add validation for new parameter types
- **Dynamic constraints**: Can adapt validation rules based on available data
- **User-friendly feedback**: Provides clear guidance when parameters are invalid
- **Performance optimization**: Caches validation results to avoid redundant catalog queries

### 5. DataCatalog (Metadata and Discovery Engine)

The `DataCatalog` class manages metadata discovery, search, and access across different data sources and storage systems.

**Key Responsibilities:**
- Maintains searchable metadata indexes for available datasets
- Provides efficient query capabilities across multiple dimensions
- Handles catalog updates and synchronization
- Manages data source registration and discovery

**Extensibility Benefits:**
- **Multiple catalog support**: Can work with Intake-ESM, STAC, custom catalogs
- **Efficient search**: Optimized indexing for fast parameter space exploration
- **Catalog federation**: Can combine metadata from multiple sources
- **Caching strategies**: Intelligent caching of catalog metadata for performance

**Integration with Existing Catalogs:**
- Seamlessly works with existing AWS S3 catalogs used in [`generate_gwl_tables.py`](../util/generate_gwl_tables.py)
- Compatible with Pangeo CMIP6 catalogs
- Will eventually supports the boundary catalogs used in [`boundaries.py`](../core/boundaries.py)

### 6. Processors (Data Transformation Pipeline)

The `Processors` component provides a flexible, composable system for data transformations, subsetting, and post-processing operations.

**Key Responsibilities:**
- Spatial subsetting using boundary geometries from [`boundaries.py`](../core/boundaries.py)
- Temporal slicing and resampling
- Unit conversions using [`unit_conversions.py`](../util/unit_conversions.py)
- Warming Level method approach
- Metadata tracking and modification
- Dataset concatenation (right now it only does it for simulation, but can be easily extneded)
- Custom derived variable calculations (if we have to)

**Modular Design Benefits:**
- **Composable operations**: Chain multiple processors for complex workflows
- **Lazy evaluation**: Processors work with Dask for memory-efficient processing
- **Extensible transformations**: Easy to add new processing steps
- **Performance optimization**: Can optimize processor chains for efficiency

## Architectural Benefits

### 1. Clean Separation of Concerns

Each class has a single, well-defined responsibility:
- `ClimateData`: Data representation and user interface
- `DatasetFactory`: Workflow orchestration
- `Dataset`: Data source abstraction
- `ParameterValidator`: Input validation
- `DataCatalog`: Cloud Data / Metadata management
- `Processors`: Data transformation

This separation makes the codebase easier to understand, test, and maintain.

### 2. Extensibility for New Data Sources

Adding support for new data sources requires only implementing a new `Dataset` subclass:

```python
class GoogleCloudDataset(Dataset):
    def __init__(self, project_id, bucket_name):
        self.client = storage.Client(project=project_id)
        self.bucket = self.client.bucket(bucket_name)
    
    def get_data(self, parameters):
        # Implementation for Google Cloud Storage access
        pass
```

The rest of the system (validation, cataloging, processing) works unchanged.

### 3. Flexible Catalog Management

The modular catalog system can accommodate different metadata formats and sources:

```python
# Combine multiple catalog sources
composite_catalog = CompositeCatalog([
    PangeoCatalog("aws-cesm2-le.json"),
    LocalCatalog("/path/to/local/metadata.json"),
    APICatalog("https://api.example.com/climate-data")
])
```

### 4. Extensible Validation

New validation rules can be added without modifying existing code:

```python
class CustomValidator(ParameterValidator):
    def validate_custom_parameter(self, value):
        # Custom validation logic
        pass
```

### 5. Complex Processing Pipelines

The processor system enables sophisticated data transformation workflows:

```python
# Complex bias correction and downscaling pipeline
pipeline = ProcessorPipeline([
    BiasCorrection(method="quantile_mapping"),
    WarmingLevels([2.0]),
    Clip('SCE')
])
```

## Migration Path

The new architecture is designed to be backward compatible with the existing [`climakitae.core.data_interface`](../core/data_interface.py) API. The [`DataParameters`](../core/data_interface.py) class can be gradually migrated to use the new system internally while maintaining the same user interface.

## Integration with Existing Code

The new core will integrate seamlessly with existing code and position `climakitae` for robust growth while maintaining backward compatibility and improving code maintainability.