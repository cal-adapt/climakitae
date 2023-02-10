.. module:: climakitae

Climakitae Documentation
========================


.. note::

   This project is actively in development working towards an alpha release.

Climakitae provides a python toolkit for retrieving, visualizing, and performing 
scientific analyses with data from the Cal-Adapt Analytics Engine. 

.. grid:: 1 1 2 2
    :gutter: 2

    .. grid-item-card:: Cal-Adapt Analytics Engine
        :img-top: _static/cae-logo.svg
        :link: https://analytics.cal-adapt.org/
        :columns: 3

        Cal-Adapt is the larger project through which *climakitae* 
        is being developed.  

    .. grid-item-card:: Getting Started
        :img-top: _static/runner.svg
        :link: https://github.com/cal-adapt/cae-notebooks/blob/main/getting_started.ipynb
        :columns: 3

        An introductory notebook demonstrating how to get started with *climakitae* and 
        the Cal-Adapt Analytics Engine.

    .. grid-item-card::  API Reference
        :img-top: _static/book.svg
        :link: api
        :link-type: doc
        :columns: 3

        A detailed description of the *climakitae* API. 

    .. grid-item-card::  GitHub
        :img-top: _static/github.svg
        :link: https://github.com/cal-adapt/climakitae
        :columns: 3

        Interested in seeing the source code? 
        Check out the repository on GitHub!


*************
Installation
*************
To begin working with *climakitae*, you can install the current version of the package 
directly from pip: :: 

    pip install https://github.com/cal-adapt/climakitae/archive/main.zip

.. toctree::
   :maxdepth: 1
   :hidden:
   :caption: Overview
   
   Getting Started <https://github.com/cal-adapt/cae-notebooks/blob/main/getting_started.ipynb>
   Working with the Data <data>
   climakitae API <climakitae>

.. .. toctree::
..    :maxdepth: 1
..    :hidden:
..    :caption: Toolkits

..    Warming Levels <guide/warming_levels>
..    Meteorological Yeah <guide/meteo_yr>
..    Timeseries Tools <guide/timeseries>
..    Climate Thresholds <guide/thresholds>