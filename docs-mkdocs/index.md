# ClimakitAE: Climate Data Analysis for California

Query, process, and analyze downscaled climate projections for California with
a modern Python API.

ClimakitAE provides programmatic access to:

- **WRF dynamical downscaling** — 3 km / 9 km / 45 km, hourly to monthly
- **LOCA2 statistical downscaling** — bias-corrected CMIP6 at 3 km
- **Flexible processing** — spatial clipping, temporal subsetting, warming-level analysis, unit conversion

Built for climate scientists, environmental analysts, and Python developers.

---

## Quick Start

Install:

```bash
pip install climakitae
```

Run your first query:

```python
from climakitae.new_core.user_interface import ClimateData

cd = ClimateData()

data = (cd
    .catalog("cadcat")
    .activity_id("WRF")
    .institution_id("UCLA")
    .grid_label("d03")
    .table_id("mon")
    .variable("t2max")
    .processes({
        "time_slice": ("2015-01-01", "2015-12-31"),
        "clip": "Los Angeles",
        "convert_units": "degF",
    })
    .get())

data["t2max"].isel(sim=0).mean(dim="time").plot()
```

Full walkthrough: **[Getting Started](getting-started.md)**.

---

## I want to&hellip;

<div class="grid cards" markdown>

-   **Analyze climate data**

    ---

    Recipes for clipping, exporting, warming-level queries, bias adjustment,
    multi-point batches, and derived indices.

    [How-To Guides &rarr;](new-core/howto.md)

-   **Understand the API**

    ---

    Design goals, data hierarchy, the processing pipeline, and per-processor
    parameter reference.

    [Core Concepts &rarr;](new-core/concepts.md) &middot;
    [Processor Reference &rarr;](new-core/processors/index.md)

-   **Migrate from the legacy API**

    ---

    Side-by-side comparison of `get_data()` / `DataParameters` and the new
    `ClimateData` builder.

    [Migration Guide &rarr;](migration/legacy-to-new-core.md)

</div>

---

## Supported data

| Feature | WRF | LOCA2 | Global CM |
|---|---|---|---|
| **Resolution** | 3 / 9 / 45 km | 3 km | Global |
| **Temporal** | hourly, daily, monthly | daily, monthly | monthly |
| **Time range** | 1981–2100 | 1850–2100 | 1850–2100 |
| **Scenarios** | Historical, SSP2-4.5, SSP3-7.0, SSP5-8.5 | Historical, SSP2-4.5, SSP3-7.0, SSP5-8.5 | Historical, SSP2-4.5, SSP3-7.0, SSP5-8.5 |
| **Variables** | `t2max`, `t2min`, `prec`, `u10`, `v10`, &hellip; | `tasmax`, `tasmin`, `pr` | 50+ CMIP6 |

For the full inventory, see the
[Cal-Adapt data catalog](https://analytics.cal-adapt.org/data/catalog).

---

## Related Cal-Adapt resources

ClimakitAE is the Python toolkit underneath the
[Cal-Adapt Analytics Engine](https://analytics.cal-adapt.org/). The companion
website hosts the broader scientific context that complements this API
reference:

- [About climate projections and models](https://analytics.cal-adapt.org/guidance/about_climate_projections_and_models) — GCMs, downscaling, SSPs, global warming levels
- [Glossary](https://analytics.cal-adapt.org/guidance/glossary) — bias correction, GWL, localization, dynamical vs. statistical downscaling
- [Datasets summary](https://analytics.cal-adapt.org/data/about) — WRF + LOCA2 model lists, resolution and time coverage
- [Methods](https://analytics.cal-adapt.org/analytics/methods) — algorithmic details (e.g. the warming-level fetching procedure)
- [Example applications](https://analytics.cal-adapt.org/analytics/example) — featured notebooks and decision-making case studies

---

## License

BSD 3-Clause License — see [LICENSE](https://github.com/cal-adapt/climakitae/blob/main/LICENSE).

[GitHub](https://github.com/cal-adapt/climakitae) &middot;
[PyPI](https://pypi.org/project/climakitae/) &middot;
[Cal-Adapt Analytics Engine](https://analytics.cal-adapt.org/)
