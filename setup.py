#!/usr/bin/env python
# read the contents of your README file
from pathlib import Path

from setuptools import setup

this_directory = Path(__file__).parent
long_description = (this_directory / "README.md").read_text()

setup(
    name="climakitae",
    # other arguments omitted
    long_description=long_description,
    long_description_content_type="text/markdown",
)
