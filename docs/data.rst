.. _data:

**********************
Working with our Data
**********************
Data from the Cal-Adapt Analytics Engine can be retrieved, subsetted, visualized, and 
exported using the *climakitae library*. Visit the `Analytics Engine website <https://analytics.cal-adapt.org/data/>`_ 
to see more information about the various datasets available in our catalog. 


Retrieve and subset the data
#############################
In this section we will detail the various methods to retrieve and subset the catalog data. 

Use the ck.Select() panel GUI 
*********************************
If you are working in a Jupyter notebook environment, you can view and set your data and location 
options in the :py:class:`climakitae.Select()` GUI (graphical user interface). This GUI also provides a visual overview of the various 
datasets available in the AE data catalog. Using this GUI, you can chose what dataset you'd like to 
retrieve-- choosing a variable, timeslice, resolution, etc.-- and the location for which you'd like to 
retrieve the data.::
   
   import climakitae as ck     # Import the package
   selections = ck.Select()    # Initialize an Select object 
   selections.show()           # Display the GUI in the notebook. 

After using the widgets (buttons, sliders, etc) in the GUI, you can retrieve the data with :py:func:`climakitae.Select.retrieve`: ::

   data = selections.retrieve()


Directly modifying the location and selections attributes 
*********************************************************
The :py:class:`climakitae.Select()` object stores the data selections information used to retrieve data. These attributes
can be easily modified in the :py:class:`climakitae.Select()` GUI (see above), but can also be directly
modified in code. This is trickier than simply using the GUI, but can allow for better reproducability of notebooks. 

For example, if you want to set the location to the LA Metro demand forecast zone, you would use the 
following code: :: 

   selections.area_subset = "CA Electricity Demand Forecast Zones"
   selections.cached_area = "LA Metro" 

To compute an area average over that entire region, you can modify the ``area_average`` attribute 
of the  ``selectors`` object: :: 

   selections.area_average = "Yes"

To set the the variable to Air Temperature at 2m and retrive the data in units of degrees Fahrenheit : :: 

   selections.variable = "Air Temperature at 2m" 
   selections.units = "degF"

Similarly, to set the model resolution, timescale, time slice, and scenario: :: 

   selections.scenario_ssp = "SSP 3-7.0 -- Business as Usual"
   selections.scenario_historical = "Historical Climate"
   selections.resoltion = "9 km"
   selections.time_slice = (2005, 2025)
   selections.timescale = "hourly"


You must set these attributes using the formatting and naming conventions 
exactly as they appear in the :py:class:`climakitae.Select()` GUI.  
For example, you must set ``timescale`` to ``hourly``, not ``Hourly``.

Lastly, you'll need to retrive the data: :: 

   data = selections.retrieve()


Use a csv config file
**********************
The :py:func:`climakitae.core.DataParams.retrieve()` method can be used to retrieve data from 
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
     - SSP 3-7.0 -- Business as Usual
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
of the data using the :py:func:`climakitae.view()` method. An appropriate visualization
will be automatically generated depending on the dimensionality of the input data. ::

   ck.view(data)

You can also set the colormap and size of the output visualization using the function arguments; see 
the documentation in the API for more information. 

Export the data 
################
To export your final data (which should be an :py:class:`xarray.DataArray` object), first create the export
object using :py:class:`climakitae.Export()`. Then the filetype you want to export the data to using the
:py:func:`climakitae.Export().export_as()` dropdown menu. This will allow you to choose between three
options: NetCDF, CSV, and GeoTIFF. ::

   export = ck.Export()
   export.export_as()

We recommend exporting the data to NetCDF, which will work with any number of variables and dimensions. 
CSV and GeoTIFF can only be used for datasets with a single variable.
CSV works best for up to 2-dimensional data (e.g., lon x lat), and will be compressed and exported 
with a separate metadata file. 
For GeoTIFF exports, metadata will be accessible as "tags" in the tif file. 
GeoTIFF can accept 3 dimensions total:

* Horizontal dimensions; i.e. x and y (required)
* The third dimension is flexible and will be a "band" in the file: time, simulation, or scenario could go here

After selecting your desired output filetype, input the data you want to export and the 
desired filename (excluding the file extension) as arguments to the 
:py:func:`climakitae.Export().export_dataset()` function. :: 

   export.export_dataset(data, "my_filename")