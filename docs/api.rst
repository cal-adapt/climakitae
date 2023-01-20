.. currentmodule:: climakitae

climakitae API 
================

Auto-generated descriptions of useful functions and classes. 


Core functions
-------------------
Core functionality of the library. 
These functions are intended to be used after initializing a climakitae.Application object. 
For example, climakitae.Application.select can be easily accessed in the following way: 

.. code-block:: bash
    
    import climakitae as ck 
    app = ck.Application()
    app.select()

.. autosummary::
   :toctree: generated/

   Application
   Application.select
   Application.retrieve
   Application.retrieve_from_csv
   Application.load
   Application.view
   Application.export_as 
   Application.export_dataset

Meteorological Year 
-------------------
Helper functions for analyses involving average and severe meteorological years. 
Functions help with data retrieval, comutation of the meteorological year, and plotting of the data. 

.. autosummary::
   :toctree: generated/

   meteo_yr.retrieve_meteo_yr_data
   meteo_yr.compute_amy 
   meteo_yr.compute_severe_yr
   meteo_yr.compute_mean_monthly_meteo_yr
   meteo_yr.meteo_yr_heatmap
   meteo_yr.meteo_yr_heatmap_static
   meteo_yr.lineplot_from_amy_data


Misc
-------------------
Other uncatecorized functions that may be useful to users. 

.. autosummary::
   :toctree: generated/

   utils.get_closest_gridcell
   derive_variables.compute_hdd_cdd