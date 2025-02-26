.. _data:

**********************
Working with our Data
**********************
Data from the Cal-Adapt Analytics Engine can be retrieved, subsetted, and 
exported using the *climakitae* library. Visit the `Analytics Engine website <https://analytics.cal-adapt.org/data/>`_ 
to see more information about the various datasets available in our catalog. 


Retrieve and subset the data
#############################
In this section we will detail the various methods to retrieve and subset the catalog data. 

Use the ckg.Select() panel GUI 
*********************************
If you are working in a Jupyter notebook environment, you can view and set your
data and location options in a graphical user interface (GUI) using 
`Climakitaegui <https://climakitaegui.readthedocs.io/>`_. This GUI also 
provides a visual overview of the various datasets available in the 
`AE data catalog <https://analytics.cal-adapt.org/data/>`_. Using this GUI, you
can chose what dataset you'd like to retrieve-- choosing a variable, timeslice,
resolution, etc.-- and the location for which you'd like to retrieve the data.

Install the latest version of climakitaegui if you haven’t: ::

   pip install https://github.com/cal-adapt/climakitaegui/archive/main.zip

Pull up the GUI: ::
   
   import climakitaegui as ckg    # Import the package
   selections = ckg.Select()      # Initialize an Select object 
   selections.show()              # Display the GUI in the notebook 

After using the widgets (buttons, sliders, etc) in the GUI, you can retrieve the data with :py:func:`climakitaegui.Select().retrieve`: ::

   data = selections.retrieve()


Directly modifying the location and selections attributes 
*********************************************************
You can select your desired data and modify the selections using code. This is
trickier than simply using the GUI, but can allow for better reproducability 
of notebooks.

For example, if you want to set the location to the LA Metro demand forecast zone, you would use the 
following code: :: 

   from climakitae.core.data_interface import DataParameters
   selections = DataParameters()
   selections.area_subset = "CA Electricity Demand Forecast Zones"
   selections.cached_area = "LA Metro" 

To compute an area average over that entire region, you can modify the ``area_average`` attribute 
of the  ``selections`` object: :: 

   selections.area_average = "Yes"

To set the the variable to Air Temperature at 2m and retrieve the data in units of degrees Fahrenheit: :: 

   selections.variable = "Air Temperature at 2m" 
   selections.units = "degF"

Similarly, to set the model resolution, timescale, time slice, and scenario: :: 

   selections.scenario_ssp = "SSP 3-7.0"
   selections.scenario_historical = "Historical Climate"
   selections.resolution = "9 km"
   selections.time_slice = (2005, 2025)
   selections.timescale = "hourly"


You must set these attributes using the formatting and naming conventions 
exactly as they appear in the :py:class:`climakitaegui.Select()` GUI.  
For example, you must set ``timescale`` to ``hourly``, not ``Hourly``. Only ``scenario_ssp``, ``scenario_historical``, and ``time_slice`` accept multiple values.

Lastly, you can retrieve the data: :: 

   data = selections.retrieve()


Use a csv config file
**********************
The :py:func:`climakitae.core.data_interface.DataParameters.retrieve()` method can be used to retrieve data from 
a csv configuration file. To retrieve data using the settings in a configuration csv file, set config to the local
filepath of the csv. Depending on the number of rows in the csv, different data types can be returned.
If the csv has one row, a single :py:class:`xarray.DataArray` object will be returned. If the csv has multiple
rows, we assume you want to retrieve **multiple** datasets. Set the function argument ``merge`` to ``False`` to
return a list of :py:class:`xarray.DataArray` objects, or merge to ``True`` (the default value) to return a single :py:class:`xarray.Dataset` object.

The csv file needs to be configured in a particular way in order for the function to properly read it in. 
The row values must match valid options in our data catalog, and the headers of the csv must be labelled 
**exactly** as they are in the following example: 

.. list-table::
   :widths: 5 5 5 5 5 5 5 5 5 5 
   :header-rows: 1

   * - variable
     - units
     - scenario_historical
     - scenario_ssp
     - area_average
     - timescale 
     - resolution
     - time_slice
     - area_subset
     - cached_area
   * - Air Temperature at 2m
     - degF
     - Historical Climate
     - SSP 3-7.0
     - Yes
     - hourly
     - 9 km
     - (2005, 2025)
     - states 
     - CA

Read the data into memory 
###########################
The data is retrieved as lazily loaded Dask arrays until you choose to read the data into 
memory. You'll want to read your data into memory before plotting it, exporting it,
or performing certain computations in order to optimize performance. To read the data 
into memory, use the :py:func:`climakitae.load()` method. ::

   data = selections.retrieve() 
   data = ck.load(data)


Create a quick visualization of the data 
#########################################
Once you've retrieved the data and read it into memory, you can generate a quick visualization 
of the data using the :py:func:`climakitaegui.view()` method. An appropriate visualization
will be automatically generated depending on the dimensionality of the input data. ::

   ckg.view(data)

You can also set the colormap and size of the output visualization using the function arguments; see 
the documentation in the API for more information. 

Export the data 
################
To save data as a file, use the :py:func:`climakitae.export()` method and input your desired

* data to export – an :py:class:`xarray.DataArray` or :py:class:`xarray.Dataset` object, as output by e.g. :py:func:`selections.retrieve()`
* output file name (without file extension)
* file format ("NetCDF", "Zarr", or "CSV")

We recommend NetCDF or Zarr, which suits data and outputs from the Analytics Engine well – they efficiently store large data containing multiple variables and dimensions. Metadata will be retained in these files.

NetCDF or Zarr can be export locally (such as onto the JupyterHUB user partition). Optionally Zarr can be exported to an AWS S3 scratch bucket for storing very large exports.

CSV can also store Analytics Engine data with any number of variables and dimensions. It works best for smaller data with fewer dimensions. The output file will be compressed to ensure efficient storage. Metadata will be preserved in a separate file.

CSV stores data in tabular format. Rows will be indexed by the index coordinate(s) of the DataArray or Dataset (e.g. scenario, simulation, time). Columns will be formed by the data variable(s) and non-index coordinate(s). :: 

   ck.export(data, filename="my_filename", format="NetCDF")
   ck.export(data, filename="my_filename2", format="Zarr")
   ck.export(data, filename="my_filename3", format="Zarr", mode="s3")
   ck.export(data, filename="my_filename4", format="CSV")
