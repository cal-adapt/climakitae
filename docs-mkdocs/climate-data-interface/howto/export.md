# Export Data to Files

Save your climate data in multiple formats for external analysis, GIS, or archival.

## Export to NetCDF (Default)

```python
# Simple export during query
data = (cd
    .variable("tasmax")
    .processes({
        "time_slice": ("2015-01-01", "2015-12-31"),
        "export": {
            "filename": "la_temperature_2015",
            "file_format": "NetCDF"
        }
    })
    .get())

# File saved as: la_temperature_2015.nc
# Retrieve as: data["tasmax"].compute()
```

## Export to Zarr (Cloud-Optimized, Scalable)

```python
# Zarr format: excellent for large datasets, cloud-native
data = (cd
    .variable("t2max")
    .activity_id("WRF")
    .institution_id("UCLA")  # Specify WRF producer
    .processes({
        "time_slice": ("2015-01-01", "2050-12-31"),
        "export": {
            "filename": "long_term_temps",
            "file_format": "Zarr"
        }
    })
    .get())

# File saved as: long_term_temps/
# Useful for: large multi-model ensembles, cloud storage (S3)
```

## Export to CSV (Tabular Data)

```python
# CSV export: for time series or point data
data = (cd
    .variable("tasmax")
    .processes({
        "clip": [(34.05, -118.25), (37.77, -122.42)],  # Two points
        "time_slice": ("2015-01-01", "2015-12-31"),
        "export": {
            "filename": "point_timeseries",
            "file_format": "CSV",
            "separated": True,  # Export each point separately
            "location_based_naming": True
        }
    })
    .get())

# Files saved as: point_timeseries_34.05_-118.25.csv, etc.
```

## Export to GeoTIFF (Raster, GIS-compatible)

```python
# GeoTIFF: for single time slice or aggregated spatial rasters
data = (cd
    .variable("tasmax")
    .processes({
        "time_slice": ("2015-01-01", "2015-01-01"),  # Single day
        "export": {
            "filename": "temp_snapshot",
            "file_format": "GeoTIFF"
        }
    })
    .get())

# File saved as: temp_snapshot.tif
# Compatible with: QGIS, ArcGIS, GDAL tools
```

## Export with Compression and Checkpointing

```python
# Skip re-processing if file exists
data = (cd
    .variable("tasmax")
    .processes({
        "time_slice": ("2015-01-01", "2050-12-31"),
        "export": {
            "filename": "long_term",
            "file_format": "Zarr",
            "export_method": "skip_existing"  # Don't re-process if exists
        }
    })
    .get())
```

## Best Practice: Chain Multiple Exports

```python
# Export intermediate results for different use cases
data = (cd
    .variable("tasmax")
    .processes({
        "time_slice": ("2015-01-01", "2050-12-31"),
        "clip": "Los Angeles",
        "export": {  # First export: full temporal data for analysis
            "filename": "la_full_timeseries",
            "file_format": "NetCDF"
        }
    })
    .get())

# Then compute aggregates for sharing
annual_mean = data["tasmax"].resample(time="YS").mean()
annual_mean.to_netcdf("la_annual_means.nc")  # Annual averages
```

---
