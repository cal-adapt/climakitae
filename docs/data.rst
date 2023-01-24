.. _data:

**********************
Working with our Data
**********************
Data from the Analytics Engine AWS data bucket can be retrieved, subsetted, visualized, and 
exported using the climakitae library. 

This section will tell the users how to access, subset, and export the data. 
It will also give an overview of how to get the closest gricell to a coordinate pair, and describe other methods (bias-correction? I can't remember exactly) for doing similar computations. 

Retrieve and subset the data
#############################
In this section we will detail the various methods to retrieve and subset the catalog data. 

Use the app.select() panel GUI 
*********************************
If you are working in a Jupyter notebook environment, you can view and set your data and location 
options in the :py:func:`climakitae.Application.select`: GUI (graphical user interface). This GUI also provides a visual overview of the various 
datasets available in the AE data catalog. Using this GUI, you can chose what dataset you'd like to 
retrieve-- chosing a variable, timeslice, resolution, etc.-- and the location for which you'd like to 
retrieve the data.::
   
   import climakitae as ck  # Import the package
   app = ck.Application     # Initialize an Application object 
   app.select()             # Display the GUI in the notebook. 

After using the widgets (buttons, sliders, etc) in the GUI, you can retrieve the data with :py:func:`climakitae.Application.retrieve`: ::

   data = app.retrieve()


Directly modifying the location and selections attributes 
*********************************************************
The :py:class:`climakitae.Application` object has two attributes-- selections and location-- that contain 
information about the user's selections. These attributes can be easily modified in the 
:py:func:`climakitae.Application.select` GUI (see above), but can also be directly modified in code. This 
is trickier than simply using the GUI, but can allow for better reproducability of notebooks. 

For example, if you want to set the location to the LA Metro demand forecast zone, you would use the 
following code: :: 

   app.location.area_subset = "CA Electricity Demand Forecast Zones"
   app.location.cached_area = "LA Metro" 

To compute an area average over that entire region, you can modify the ``area_average`` attribute 
of the  ``selectors`` object: :: 

   app.selections.area_average = "Yes"

To set the the variable to Air Temperature at 2m and retrive the data in units of degrees Fahrenheit : :: 

   app.selections.variable = "Air Temperature at 2m" 
   app.selections.units = "degF"

Similarly, to set the model resolution, timescale, time slice, and scenario: :: 

   app.selections.scenario_ssp = "SSP 3-7.0 -- Business as Usual"
   app.selections.scenario_historical = "Historical Climate"
   app.selections.resoltion = "9 km"
   app.selections.time_slice = (2005, 2025)
   app.selections.timescale = "hourly"


You must set these attributes using the formatting and naming conventions 
exactly as they appear in the :py:func:`climakitae.Application.select` GUI.  
For example, you must set ``timescale`` to ``hourly``, not ``Hourly``.

Lastly, you'll need to retrive the data: :: 

   data = app.retrieve()


Use a csv config file
**********************
The :py:func:`climakitae.Application.retrieve_from_csv` method can be used to retrieve data from 
a csv configuration file. It just takes the filepath to the csv as an argument. Depending on the number of 
rows in the csv, different datatypes can be returned. If the csv has one row, a single :py:class:`xarray.DataArray`
object will be returned. If the csv has multiple rows, we assume you want to retrieve **multiple** datasets. 
Set the function argument ``merge`` to ``False`` to return a list of :py:class:`xarray.DataArray` objects, or 
merge to ``True`` (the default value) to return a single :py:class:`xarray.Dataset` object.

The csv file needs to be configured in a particular way in order for the function to properly read it in. 
The row values must match valid options in our data catalog. An example table is provided below. 
The headers of the csv must be **exactly** as they are in the following example: 

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
The data is retrieved as lazily loaded Dask arrays until you chose to read the data into 
memory. You'll want to read your data into memory before plotting it, exporting it,
or performing certain computations in order to optimize performance. To read the data 
into memory, use the :py:func:`climakitae.Application.load` method. ::

   data = app.retrieve() 
   data = app.load(data)


Create a quick visualization of the data 
#########################################
Once you've retrieved the data and read it into memory, you can generate a quick visualization 
of the data using the :py:func:`climakitae.Application.view` method. An appropriate visualization
will be automatically generated depending on the dimensionality of the input data. ::

   app.view(data)

You can also set the colormap and size of the output visualization using the function arguments; see 
the documentation in the API for more information. 

Export the data 
################
To export the data, first chose the filetype you want to export the data to using the 
:py:func:`climakitae.Application.export_as` dropdown menu. This will allow you to choose 
between three options: netcdf, csv, and geotiff.::

   app.export_as() 

After selecting your desired output filetype, input the data you want to export and the 
desired filename (excluding the file extension) as arguments to the 
:py:func:`climakitae.Application.export_dataset` function. :: 

   export_dataset(data, "my_filename")

**Note:** This data export functionality will only work within a notebook environment. 