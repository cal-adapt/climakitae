# Climakitae

[![codecov](https://codecov.io/gh/cal-adapt/climakitae/branch/main/graph/badge.svg)](https://codecov.io/gh/cal-adapt/climakitae)
[![CI](https://github.com/cal-adapt/climakitae/workflows/ci-main/badge.svg)](https://github.com/cal-adapt/climakitae/actions/workflows/ci-main.yml)
[![Documentation Status](https://readthedocs.org/projects/climakitae/badge/?version=latest)](https://climakitae.readthedocs.io/en/latest/?badge=latest)
[![PyPI version](https://badge.fury.io/py/climakitae.svg)](https://badge.fury.io/py/climakitae)
[![Python](https://img.shields.io/badge/python-3.12-blue.svg)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/License-BSD%203--Clause-blue.svg)](https://opensource.org/licenses/BSD-3-Clause)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)


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

### Installation via Conda

#### Prerequisites

- Python 3.12
- [conda / miniconda](https://www.anaconda.com/docs/getting-started/miniconda/install#quickstart-install-instructions)

#### Install 1.4.0 with `conda` on Linux

For additional details on the latest version and step-by-step installation instructions please visit [the wiki](https://github.com/cal-adapt/climakitae/wiki)

```bash
# get the conda lock file from github
curl https://raw.githubusercontent.com/cal-adapt/cae-environments/refs/heads/main/conda-lock/climakitae/1.3.0/conda-linux-64.lock -o conda-linux-64.lock

# create and activate your environment
conda create -n climakitae --file conda-linux-64.lock
conda activate climakitae

# install climakitae
pip install https://github.com/cal-adapt/climakitae/archive/refs/tags/1.4.0.zip
```

### Installation via Pip

#### Prerequisites

- Python 3.12
- pip

#### Install 1.3.0 with `pip` on Linux

For additional details on the latest version and step-by-step installation instructions please visit [the wiki](https://github.com/cal-adapt/climakitae/wiki)

```bash
# get the requirements.txt file from github
curl https://raw.githubusercontent.com/cal-adapt/climakitae/refs/heads/release-1.3.0/requirements.txt -o requirements.txt

# load packages from requirements.txt
pip install -r requirements.txt

# install climakitae
pip install https://github.com/cal-adapt/climakitae/archive/refs/tags/1.4.0.zip
```

### Basic Usage

```python
from climakitae.core.data_interface import get_data

# Retrieve temperature data for California
data = get_data(
    variable="Air Temperature at 2m",
    downscaling_method="Dynamical", 
    resolution="9 km",
    timescale="monthly",
    scenario="SSP 3-7.0",
    cached_area="CA"
)

# Data is returned as an xarray Dataset
print(data)
```

## Documentation

| Resource | Description |
|----------|-------------|
| [**AE Navigation Guide**](https://github.com/cal-adapt/cae-notebooks/blob/main/AE_navigation_guide.ipynb) | Interactive notebook tutorial |
| [**API Reference**](https://climakitae.readthedocs.io/en/latest/) | Complete API documentation |
| [**AE Notebooks**](https://github.com/cal-adapt/cae-notebooks) | Sample notebooks and scripts |
| [**Contributing**](https://climakitae.readthedocs.io/en/latest/contribute.html) | Development guidelines |

## Development Setup

### Prerequisites

- Python 3.12
- [conda / miniconda](https://www.anaconda.com/docs/getting-started/miniconda/install#quickstart-install-instructions)

### Dev Environment Setup (Linux)

```bash
git clone https://github.com/cal-adapt/climakitae.git
cd climakitae
conda create -n climakitae --file conda-linux-64.lock
conda activate climakitae
```

### Running Tests

```bash
# Run basic tests
pytest -m "not advanced"

# Run all tests
pytest

# Run with coverage
pip install pytest-cov
pytest --cov=climakitae --cov-report=html
```

## Contributing

We welcome contributions! Please see our [contributing guidelines](https://climakitae.readthedocs.io/en/latest/contribute.html) for details on:

- 🐛 Reporting bugs
- 💡 Requesting features  
- 🔧 Submitting code changes
- 📖 Improving documentation

### Quick Development Workflow

Open a ⚙️ [code improvement issue](https://github.com/cal-adapt/climakitae/issues/new/choose) describing the feature you'd like to develop.

Then, checkout and setup your branch:
```bash
# Fork the repo and create a feature branch
git checkout -b feature/your-feature-name

# Make your changes and add tests
# ...

# Run tests and linting
pytest
black climakitae/
isort climakitae/

# Submit a pull request
```

When submitting a pull request, please tag at least two project maintainers/developers for review.

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
