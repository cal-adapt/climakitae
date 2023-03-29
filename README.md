Climakitae
==========
A python toolkit for retrieving, visualizing, and performing scientific analyses with data from the [Cal-Adapt Analytics Engine](https://analytics.cal-adapt.org).

**Note:** This package is in active development and should be considered a work in progress. 

Documentation
--------------
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
import climakitae as ck   # Import the package 
app = ck.Application()    # Initialize Application object
app.select()              # Pull up selections GUI to make data settings
data = app.retrieve()     # Retrieve the data from the AWS catalog 
data = app.load(data)     # Read the data into memory 
app.view(data)            # Generate a basic visualization of the data
```

Links
-----
* PyPI releases: https://pypi.org/project/climakitae/
* Source code: https://github.com/cal-adapt/climakitae
* Issue tracker: https://github.com/cal-adapt/climakitae/issues

Contributors
-------------
[![Contributors](https://contrib.rocks/image?repo=cal-adapt/climakitae)](https://github.com/cal-adapt/climakitae/graphs/contributors)
