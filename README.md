

Climakitae
==========
[![CI](https://github.com/cal-adapt/climakitae/workflows/ci/badge.svg)](https://github.com/cal-adapt/climakitae/actions/workflows/ci.yaml)
[![Documentation Status](https://readthedocs.org/projects/climakitae/badge/?version=latest)](https://climakitae.readthedocs.io/en/latest/?badge=latest)
[![PyPI version](https://badge.fury.io/py/climakitae.svg)](https://badge.fury.io/py/climakitae)
[![Python](https://img.shields.io/badge/python-3.11-blue.svg)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/License-BSD%203--Clause-blue.svg)](https://opensource.org/licenses/BSD-3-Clause)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)  

A python toolkit for retrieving and performing scientific analyses with climate data from the [Cal-Adapt Analytics Engine](https://analytics.cal-adapt.org).

**Note:** This package is in active development and should be considered a work in progress. 

Documentation
-------------
Check out the official documentation on ReadTheDocs: https://climakitae.readthedocs.io/en/latest/ 

Installation
------------

Install the latest release with pip.

```
pip install climakitae
```

Basic Usage
-----------

```
# Import functions of interest from climakitae
from climakitae.core.data_interface import (
    get_data_options, 
    get_subsetting_options, 
    get_data
)

# See all the data catalog options as a pandas DataFrame object
get_data_options()

# See all the area subset options for retrieving a spatial subset of the catalog data
get_subsetting_options()

# Retrieve data for a single variable for the state of California
get_data(
    variable = "Precipitation (total)", 
    downscaling_method = "Dynamical", 
    resolution = "9 km", 
    timescale = "monthly", 
    scenario = "SSP 3-7.0 -- Business as Usual",
    cached_area = "CA"
)
```

If you want to use graphic user interfaces to retrieve and view data visualization options (among other features), you'll need to import our sister package `climakitaegui`, which works in tandem with climakitae to produce interactive GUIs. See [climakitaegui](https://github.com/cal-adapt/climakitaegui) for more information on how to use this library. 

Developer Information
-----------

It is strongly recommended that developers use `uv` to manage their packages:
```bash
pip install uv
```

To install a specific branch as a package:
```bash
uv pip install git+https://github.com/cal-adapt/climakitae.git@<BRANCH>
```

## Local Setup

This project uses `conda` or `uv` for local and remote setup.

For demonstration purposes, here are the instructions to follow for setting up a local
environment using `uv`:

### Linux
install `uv`:
```bash
pip install uv
```

install the dependencies for `climakitae`
```bash
uv sync
```

activate your environment:
```bash
source .venv/bin/activate
```

### MacOS
The instructions for MacOS are a little more involved due to some library management.

set up dependencies for llvm:
```zsh
brew install llvm@14
export PATH="/opt/homebrew/opt/llvm@14/bin:$PATH"
export LLVM_CONFIG="/opt/homebrew/opt/llvm@14/bin/llvm-config"
```

install `uv`
```zsh
pip install uv
```

install dependencies for `climakitae`:
```zsh
uv sync
```

activate your virtual environment:
```zsh
source .venv/bin/activate
```

### Testing install environment

You can test your environment works by opening `examples/example_plot_pajaro.ipynb`, setting your kernel to `.venv/bin/python` and running all cells.

Links
-----
* PyPI releases: https://pypi.org/project/climakitae/
* Source code: https://github.com/cal-adapt/climakitae
* Issue tracker: https://github.com/cal-adapt/climakitae/issues

Contributors
------------
[![Contributors](https://contrib.rocks/image?repo=cal-adapt/climakitae)](https://github.com/cal-adapt/climakitae/graphs/contributors)
