# Climakitae

[![CI](https://github.com/cal-adapt/climakitae/workflows/ci/badge.svg)](https://github.com/cal-adapt/climakitae/actions/workflows/ci.yaml)
[![Documentation Status](https://readthedocs.org/projects/climakitae/badge/?version=latest)](https://climakitae.readthedocs.io/en/latest/?badge=latest)
[![PyPI version](https://badge.fury.io/py/climakitae.svg)](https://badge.fury.io/py/climakitae)
[![Python](https://img.shields.io/badge/python-3.11-blue.svg)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/License-BSD%203--Clause-blue.svg)](https://opensource.org/licenses/BSD-3-Clause)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

**A powerful Python toolkit for climate data analysis and retrieval from the Cal-Adapt Analytics Engine.**

Climakitae provides intuitive tools for accessing, analyzing, and visualizing downscaled climate model data, enabling researchers and practitioners to perform comprehensive climate impact assessments for California.

> [!WARNING]
> This package is under active development. APIs may change between versions.

## ✨ Key Features

- 🌡️ **Comprehensive Climate Data Access**: Retrieve temperature, precipitation, and derived climate variables
- 📊 **Downscaled Climate Models**: Access both dynamical (WRF) and statistical (LOCA) downscaling methods  
- 🗺️ **Spatial Analysis Tools**: Built-in support for geographic subsetting and spatial aggregation
- 📈 **Climate Indices**: Calculate heat indices, warming levels, and extreme event metrics
- 🔧 **Flexible Data Export**: Export to NetCDF, CSV, and specialized formats
- 📱 **GUI Integration**: Works seamlessly with [climakitaegui](https://github.com/cal-adapt/climakitaegui) for interactive analysis

## 🚀 Quick Start 

### Installation

```bash
pip install climakitae
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
    scenario="SSP 3-7.0 -- Business as Usual",
    cached_area="CA"
)

# Data is returned as an xarray Dataset
print(data)
```

### Advanced Example

```python
from climakitae.core.data_interface import DataParameters
from climakitae.explore.vulnerability import cava_data

# Configure data parameters
params = DataParameters()
params.variable = "Air Temperature at 2m"
params.scenario_ssp = ["SSP 3-7.0"]
params.resolution = "3 km"

# Perform vulnerability assessment
locations = pd.DataFrame({
    'lat': [37.7749, 34.0522], 
    'lon': [-122.4194, -118.2437]
})

results = cava_data(
    input_locations=locations,
    variable="Air Temperature at 2m",
    approach="warming_level",
    warming_level=2.0,
    metric_calc="max",
    season="summer"
)
```

## 📚 Documentation

| Resource | Description |
|----------|-------------|
| [**Getting Started**](https://github.com/cal-adapt/cae-notebooks/blob/main/getting_started.ipynb) | Interactive notebook tutorial |
| [**API Reference**](https://climakitae.readthedocs.io/en/latest/) | Complete API documentation |
| [**Examples**](examples/) | Sample notebooks and scripts |
| [**Contributing**](https://climakitae.readthedocs.io/en/latest/contribute.html) | Development guidelines |

## 🛠️ Development Setup

### Prerequisites

- Python 3.11
- [uv](https://github.com/astral-sh/uv) (recommended) or conda

### Using uv (Recommended)

```bash
# Install uv
pip install uv

# Clone the repository
git clone https://github.com/cal-adapt/climakitae.git
cd climakitae

# Create environment and install dependencies
uv sync

# Activate environment
source .venv/bin/activate

# Verify installation
python -c "import climakitae; print('Success!')"
```

### Using conda

```bash
# Create environment
conda env create -f environment.yml
conda activate climakitae-tests

# Install in development mode
pip install -e .
```

### macOS Additional Setup

macOS users need additional LLVM dependencies:

```bash
brew install llvm@14
export PATH="/opt/homebrew/opt/llvm@14/bin:$PATH"
export LLVM_CONFIG="/opt/homebrew/opt/llvm@14/bin/llvm-config"
```

### Running Tests

```bash
# Run basic tests
pytest -m "not advanced"

# Run all tests
pytest

# Run with coverage
pytest --cov=climakitae --cov-report=html
```

## 🌍 About Cal-Adapt

Climakitae is developed as part of the [Cal-Adapt Analytics Engine](https://analytics.cal-adapt.org), California's premier platform for climate data and tools. Cal-Adapt provides access to cutting-edge climate science to support adaptation planning and decision-making.

## 🤝 Contributing

We welcome contributions! Please see our [contributing guidelines](https://climakitae.readthedocs.io/en/latest/contribute.html) for details on:

- 🐛 Reporting bugs
- 💡 Requesting features  
- 🔧 Submitting code changes
- 📖 Improving documentation

### Quick Development Workflow

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

## 📄 License

This project is licensed under the BSD 3-Clause License - see the [LICENSE](LICENSE) file for details.

## 🔗 Related Projects

- [**climakitaegui**](https://github.com/cal-adapt/climakitaegui) - Interactive GUI tools for climakitae
- [**cae-notebooks**](https://github.com/cal-adapt/cae-notebooks) - Example notebooks and tutorials

## 📞 Support

- 📧 **Email**: [analytics@cal-adapt.org](mailto:analytics@cal-adapt.org)
- 🐛 **Issues**: [GitHub Issues](https://github.com/cal-adapt/climakitae/issues)
- 💬 **Discussions**: [GitHub Discussions](https://github.com/cal-adapt/climakitae/discussions)

---

## 👥 Contributors

[![Contributors](https://contrib.rocks/image?repo=cal-adapt/climakitae)](https://github.com/cal-adapt/climakitae/graphs/contributors)

*Made with [contrib.rocks](https://contrib.rocks).*
