.. _data:

Data
========================
This section will tell the users how to access, subset, and export the data. 
It will also give an overview of how to get the closest gricell to a coordinate pair, and describe other methods (bias-correction? I can't remember exactly) for doing similar computations. 

Accessing the data 
-------------------
#. app.select(), followed by app.retrieve 
#. modifying app.selections and app.location 
#. app.retrieve_from_csv() 
#. directly using the intake catalog 

Subsetting the data
--------------------
#. Directly with app.select(), followed by app.retrieve(), or 
#. Using the Boundaries class 

Get the closest gridcell
-------------------------
#. How to just get the closest gridcell to a coordinate pair 
#. Our recommended fancier, more scientifically sound methods (need to discuss with Beth/Owen on this)

Exporting the data 
-------------------
Perhaps Beth can help with this section 

A note on xarray 
-----------------
* Information about xarray since the entire package is built around xarray DataArray/Dataset objects 
