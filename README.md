# Climakitae

[![codecov](https://codecov.io/gh/cal-adapt/climakitae/branch/main/graph/badge.svg)](https://codecov.io/gh/cal-adapt/climakitae)
[![CI](https://github.com/cal-adapt/climakitae/workflows/ci-main/badge.svg)](https://github.com/cal-adapt/climakitae/actions/workflows/ci-main.yml)
[![Documentation Status](https://readthedocs.org/projects/climakitae/badge/?version=latest)](https://climakitae.readthedocs.io/en/latest/?badge=latest)
[![PyPI version](https://badge.fury.io/py/climakitae.svg)](https://badge.fury.io/py/climakitae)
[![Python](https://img.shields.io/badge/python-3.12%20%7C%203.13-blue.svg)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/License-BSD%203--Clause-blue.svg)](https://opensource.org/licenses/BSD-3-Clause)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![DOI:10.5281/zenodo.18111935](https://zenodo.org/badge/DOI/10.5281/zenodo.18111935.svg)](https://doi.org/10.5281/zenodo.18111935)


**A powerful Python toolkit for climate data analysis and retrieval from the Cal-Adapt Analytics Engine (AE).**

Climakitae provides intuitive tools for accessing, analyzing, and visualizing downscaled CMIP6 data, enabling researchers and practitioners to perform comprehensive climate impact assessments for California.

> [!WARNING]
> This package is under active development. APIs may change between versions.

## Key Features

- 🌡️ **Comprehensive Climate Data Access**: Retrieve climate variables from hosted climate models
- 📊 **Downscaled Climate Models**: Access dynamical (WRF) and statistical (LOCA2) downscaling methods  
- 🗺️ **Spatial Analysis Tools**: Built-in support for geographic subsetting and spatial aggregation
- 📈 **Climate Indices**: Calculate heat indices, warming levels, and extreme event metrics
- 🔧 **Flexible Data Export**: Export to NetCDF, CSV, and Zarr
- 📱 **GUI Integration**: Works seamlessly with [climakitaegui](https://github.com/cal-adapt/climakitaegui) for interactive analysis

## About Cal-Adapt

Climakitae is developed as part of the [Cal-Adapt Analytics Engine](https://analytics.cal-adapt.org), a platform for California climate data and tools. Cal-Adapt provides access to cutting-edge climate science to support adaptation planning and decision-making.

## Getting Started

### Installation

Climakitae requires **Python 3.12 or 3.13**. We recommend [`uv`](https://docs.astral.sh/uv/) on Linux/macOS and [`conda`](https://www.anaconda.com/docs/getting-started/miniconda/install) on Windows or when native geospatial dependencies cause trouble.

Quick install of the latest release:

```bash
# with uv
uv pip install climakitae

# with conda
conda create -n climakitae python=3.13 -y
conda activate climakitae
pip install climakitae
```

For editable installs, developer dependencies, and platform-specific tips, see the **[Installation Guide on the wiki](https://github.com/cal-adapt/climakitae/wiki)**.

### Basic Usage

```python
from climakitae.new_core.user_interface import ClimateData

# Retrieve monthly max temperature for Los Angeles in 2015
data = (
    ClimateData()
    .catalog("cadcat")
    .activity_id("WRF")
    .table_id("mon")
    .grid_label("d03")
    .variable("t2max")
    .processes({
        "time_slice": ("2015-01-01", "2015-12-31"),
        "clip": "Los Angeles",
    })
    .get()
)

# Data is returned as a lazy xarray Dataset
print(data)
```

> The legacy `climakitae.core.data_interface.get_data` API is still supported for backward compatibility, but new work should use `ClimateData` from `climakitae.new_core`.

## Documentation

| Resource | Description |
|----------|-------------|
| [**Installation Guide**](https://github.com/cal-adapt/climakitae/wiki) | Casual, power-user, and developer installs |
| [**AE Navigation Guide**](https://github.com/cal-adapt/cae-notebooks/blob/main/AE_navigation_guide.ipynb) | Interactive notebook tutorial |
| [**API Reference**](https://cal-adapt.github.io/climakitae/dev/) | Complete API documentation |
| [**AE Notebooks**](https://github.com/cal-adapt/cae-notebooks) | Sample notebooks and scripts |
| [**Contributing**](https://climakitae.readthedocs.io/en/latest/contribute.html) | Development guidelines |

## Contributing

We welcome contributions! Please see our [contributing guidelines](https://climakitae.readthedocs.io/en/latest/contribute.html) for details on:

- 🐛 Reporting bugs
- 💡 Requesting features
- 🔧 Submitting code changes
- 📖 Improving documentation

For setting up a development environment (editable install, tests, formatters), see the [Installation Guide on the wiki](https://github.com/cal-adapt/climakitae/wiki).

When opening a pull request, please tag at least two project maintainers for review.

## License

This project is licensed under the BSD 3-Clause License - see the [LICENSE](LICENSE) file for details.

## Related Projects

- [**climakitaegui**](https://github.com/cal-adapt/climakitaegui) - Interactive GUI tools for climakitae
- [**cae-notebooks**](https://github.com/cal-adapt/cae-notebooks) - Example notebooks and tutorials

## Support

- 📧 **Email**: [analytics@cal-adapt.org](mailto:analytics@cal-adapt.org)
- 🐛 **Issues**: [GitHub Issues](https://github.com/cal-adapt/climakitae/issues)
- 💬 **Discussions**: [GitHub Discussions](https://github.com/cal-adapt/climakitae/discussions)

---

## Contributors

[![Contributors](https://contrib.rocks/image?repo=cal-adapt/climakitae)](https://github.com/cal-adapt/climakitae/graphs/contributors)
