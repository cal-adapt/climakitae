# For developers: how to build and develop the docs

You'll only need to do steps 1 and 3 **once** to build the environment. Then, just activate the environment (step 2) and run steps 3-4 to build the docs locally. 

1. If you haven't already, build the conda environment for this project: ``conda env create -f environment.yml``
2. Next, activate the conda environment for this project: ``conda activate climakitae-tests``
3. In the conda environment, install the following packages through pip: ``pip install sphinx-book-theme sphinx-design nbsphinx ``
4. Serialize RST to HTML and start a web server (locally): ``make serve-docs`` 
5. To see the locally served docs: http://localhost:8000/