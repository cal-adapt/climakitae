# Get Started in 5 Minutes

Query climate data for California in a few lines of code using the `ClimateData` fluent API.

## Installation

```bash
# Using uv (recommended for development)
uv venv
source .venv/bin/activate
uv pip install climakitae

# Or with pip
pip install climakitae
```

## Your First Query

Import and create a `ClimateData` instance:

```python
from climakitae.new_core.user_interface import ClimateData
import matplotlib.pyplot as plt

# Initialize the climate data interface
cd = ClimateData()

# Query WRF temperature data for Los Angeles County, 2015
data = (cd
    .catalog("cadcat")
    .activity_id("WRF")
    .institution_id("UCLA")  # UCLA WRF model (recommended)
    .table_id("mon")
    .grid_label("d03")
    .variable("t2max")
    .processes({
        "time_slice": ("2015-01-01", "2015-12-31"),
        "clip": "Los Angeles"
    })
    .get())

# Plot the result
data["t2max"].isel(sim=0).plot(figsize=(12, 4))
plt.title("Maximum Temperature - Los Angeles 2015")
plt.ylabel("Temperature (K)")
plt.show()
```

## What Just Happened

1. **`catalog("cadcat")`** — Selected Cal-Adapt's primary climate data catalog
2. **`activity_id("WRF")`** — Chose Weather Research & Forecasting dynamical downscaling
3. **`table_id("mon")`** — Selected monthly temporal resolution
4. **`grid_label("d03")`** — Selected 3km spatial resolution (fine-scale, local coverage)
5. **`variable("t2max")`** — Requested daily maximum temperature
6. **`processes()`** — Applied transformations:
   - **`time_slice`** — Subset to calendar year 2015
   - **`clip`** — Subset to Los Angeles County boundary
7. **`.get()`** — Execute the query (returns lazy-loaded xarray Dataset)

### About Temperature Units

WRF data is returned in **Kelvin (K)**, following meteorological conventions. Use the `convert_units` processor to convert to other units:

```python
# Convert to Celsius during the query
data_celsius = (cd
    .catalog("cadcat")
    .activity_id("WRF")
    .institution_id("UCLA")
    .table_id("mon")
    .grid_label("d03")
    .variable("t2max")
    .processes({
        "time_slice": ("2015-01-01", "2015-12-31"),
        "clip": "Los Angeles",
        "convert_units": "degC"  # Convert K → °C
    })
    .get())

# Or convert Celsius to Fahrenheit
data_fahrenheit = (cd
    .catalog("cadcat")
    .activity_id("WRF")
    .institution_id("UCLA")
    .table_id("mon")
    .grid_label("d03")
    .variable("t2max")
    .processes({
        "time_slice": ("2015-01-01", "2015-12-31"),
        "clip": "Los Angeles",
        "convert_units": "degF"  # Convert K → °F
    })
    .get())
```

**Available unit conversions:**
- Temperature: `K` → `degC`, `degF`
- Precipitation: `mm`, `mm/d`, `mm/h` → `inches`, `kg m-2 s-1`
- Wind: `m/s`, `m s-1` → `knots`, `mph`
- Pressure: `Pa`, `hPa` → other pressure units
- See [Unit Conversions](climate-data-interface/processors/convert_units.md) for the complete list

## Next Steps

- **Browse variables**: Use `cd.show_variable_options()` to see all available climate variables
- **Explore boundaries**: Use `cd.show_boundary_options()` to see geographic regions available for clipping
- **Learn processors**: See the [processor reference](climate-data-interface/processors/index.md) for spatial/temporal operations
- **Try warming levels**: Use `.processes({"warming_level": {"warming_levels": [1.5, 2.0]}})` for climate scenario analysis
- **Migrate from legacy**: See [Legacy to ClimateData](migration/legacy-to-climate-data.md) if you're upgrading from the old API

## Troubleshooting

**Q: I got a network error while querying data**  
A: ClimateData accesses data from AWS S3. Check your internet connection. For offline workflows, consider pre-downloading data using export processors.

**Q: How do I work with multiple locations?**  
A: Pass a list of lat/lon tuples to `clip`: `"clip": [(34.05, -118.25), (37.77, -122.42)]`

**Q: Can I export the data to a file?**  
A: Yes! Add `"export": {"filename": "my_data", "file_format": "NetCDF"}` to your processors dict.

For more details, see the [API reference](api/climate-data.md), browse [how-to guides](climate-data-interface/howto.md), or check the [migration guide](migration/legacy-to-climate-data.md) if you're moving from the legacy API.

## See also

- [Cal-Adapt Analytics Engine — About the data](https://analytics.cal-adapt.org/data/about)
- [Cal-Adapt Analytics Engine — Glossary](https://analytics.cal-adapt.org/guidance/glossary) (bias correction, GWLs, downscaling terminology)
- [Cal-Adapt Analytics Engine — Example applications](https://analytics.cal-adapt.org/analytics/example) (featured notebooks)
