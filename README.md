Climakitae
==========
A python toolkit for retrieving and performing scientific analyses with climate data from the [Cal-Adapt Analytics Engine](https://analytics.cal-adapt.org).

**Note:** This package is in active development and should be considered a work in progress. 

Documentation
-------------
Check out the official documentation on ReadTheDocs: https://climakitae.readthedocs.io/en/latest/ 

Installation
------------

Install the latest version in development directly with pip.

```
pip install https://github.com/cal-adapt/climakitae/archive/main.zip
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

Links
-----
* PyPI releases: https://pypi.org/project/climakitae/
* Source code: https://github.com/cal-adapt/climakitae
* Issue tracker: https://github.com/cal-adapt/climakitae/issues

Contributors
------------
[![Contributors](https://contrib.rocks/image?repo=cal-adapt/climakitae)](https://github.com/cal-adapt/climakitae/graphs/contributors)
