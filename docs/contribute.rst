***********************
Contribution Guidelines
***********************

This is an open-source project, and all contributions, including new feature development, documentation, bug reports, bug fixes, and other ideas are welcome. 

The Analytics Engine consists of two separate open-source repositories (hosted on GitHub) that you may contribute to.

1. The `climakitae <https://github.com/cal-adapt/climakitae>`_ Python library:

	Building primarily on `xarray <https://docs.xarray.dev/en/stable/>`_ and the whole `Pangeo 	<https://pangeo.io/>`_ ecosystem, climakitae primarily offers tools that are:

	* Meant to be run inside a Jupyter notebook (see below)
	* Specific to the downscaled CMIP6 climate model data produced for the `State of California <https://analytics.cal-adapt.org/data/>`_ .

2. The `cae-notebooks <https://github.com/cal-adapt/cae-notebooks>`_ repository 

	`Jupyter <https://jupyter.org/>`_ notebooks in Python which: 

	* Highlight a particular climakitae tool, or
	* Perform a specific analysis for an example use-case or application of the data

We welcome contributions to either repository.

Welcomed contributions could involve a performance improvement, something which makes any of the tools more generally applicable, or adds a feature to handle a commonly-requested operation (without duplicating anything that already exists in the Pangeo ecosystem).

Contributed notebooks should be applications of general interest to our user community, which use the tools in climakitae whenever such a tool exists, and do not have dependencies outside of the `Pangeo libraries <https://github.com/pangeo-data/pangeo-docker-images/blob/master/pangeo-notebook/packages.txt>`_.

If you have an idea for contributing to either climakitae or cae-notebooks, please employ the following procedure:

1. Reach out to analytics@cal-adapt.org with a summary of what you are proposing to contribute to make sure that it is complementary to features we are already working on. 
2. Submit a pull-request to the repository in question for members of our team to review.

	* For climakitae, new functions and features should be sufficiently documented via docstrings
	* For contributed functionality, we recommend either integrating new code into our existing scripts (e.g., ``util/utils.py``) or introducing new code into a new .py script in either the ``/tools`` or ``/utils`` folders
	* Contributed notebooks to cae-notebooks should reside in a new folder housed in the ``/collaborative`` folder. We recommend utilizing a descriptive lowercase name for your contributed notebook. Contributed notebooks are requested to follow our notebook structure, consisting of:

		* Brief markdown overview/purpose of notebook, including a section titled "Intended Application" with a brief numbered list of the key take-away points a user will learn from the notebook
		* Step 0: package imports
		* Step 1: data retrieval and modifications if necessary
		* Step 3+: analysis and optional figures
		* Concluding Step: export of data object(s)
