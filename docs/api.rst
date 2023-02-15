.. currentmodule:: climakitae

.. _api:

climakitae API 
================

Auto-generated descriptions of useful functions and classes. 


Core Functions
----------------
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
   Application.retrieve_meteo_yr_data
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

   meteo_yr.compute_amy 
   meteo_yr.compute_severe_yr
   meteo_yr.compute_mean_monthly_meteo_yr
   meteo_yr.meteo_yr_heatmap
   meteo_yr.meteo_yr_heatmap_static
   meteo_yr.lineplot_from_amy_data

Warming Levels  
-------------------
Helper functions for performing analyses using climate warming levels. 

.. autosummary::
   :toctree: generated/
   
   warming_levels.get_anomaly_data

Threshold Tools 
----------------
Helper functions for thresholds-related analyses.

.. autosummary::
   :toctree: generated/

   threshold_tools.get_ams
   threshold_tools.get_ks_stat
   threshold_tools.get_return_value
   threshold_tools.get_return_prob
   threshold_tools.get_return_period


Timeseries Tools
-----------------
Helper functions for working with timeseries data. Documentation in progess. 

Model Uncertainty
-------------------
Helper functions for performing analyses related to assessing uncertainty quantification in climate models.

.. autosummary::
   :toctree: generated/

   uncertain_utils.CmipOpt
   uncertain_utils.grab_temp_data
   uncertain_utils.cmip_annual
   uncertain_utils.cmip_annual
   uncertain_utils.compute_vmin_vmax
   
Misc
-----
Other uncatecorized functions that may be useful to users. 

.. autosummary::
   :toctree: generated/

   utils.get_closest_gridcell
   derive_variables.compute_hdd_cdd