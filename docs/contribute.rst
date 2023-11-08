**********************
Contribution Guidelines
**********************

This is an open-source project, and all contributions, including new feature development, documentation, bug reports, bug fixes, and other ideas are welcome. 

The Analytics Engine consists of two separate open-source repositories (hosted on GitHub) that you may contribute to.

1. The `climakitae <https://github.com/cal-adapt/climakitae>`_ Python library:

	Building primarily on `xarray <https://docs.xarray.dev/en/stable/>`_ and the whole `Pangeo 	<https://pangeo.io/>`_ ecosystem, climakitae primarily offers tools that are:

	* Meant to be run inside a Jupyter notebook (see below)
	* Specific to the data for the `State of California <https://analytics.cal-adapt.org/data/>`_ that has 	been downscaled based on CMIP6 climate models.

2. The `cae-notebooks <https://github.com/cal-adapt/cae-notebooks>`_ repository 

	`Jupyter <https://jupyter.org/>`_ notebooks in Python which: 

	* Highlight a particular climakitae tool, or
	* Do a specific analysis for an example use-case or application of the data

We welcome contributions to either repository.

Welcomed contributions could involve a performance improvement, something which makes any of the tools more generally applicable, or adds a feature to handle a commonly-requested operation (without duplicating anything that already exists in the Pangeo ecosystem).

Notebooks should be applications of general interest to our user community, which use the tools in climakitae whenever such a tool exists, and do not have dependencies outside of the `Pangeo libraries <https://github.com/pangeo-data/pangeo-docker-images/blob/master/pangeo-notebook/packages.txt>`_.

If you have an idea for a contribution, please employ the following procedure:

1. Reach out to analytics@cal-adapt.org with a summary of what you are proposing to contribute to make sure that it is complementary to features we are already working on. 
2. Submit a pull-request to the repository in question for members of our team to review.
