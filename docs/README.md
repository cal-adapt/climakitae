# For developers: how to build the docs

You'll only need to do steps 1 and 3 **once** to build the environment. Then, just activate the environment (step 2) and run steps 4-5 to build the docs locally. 

1. If you haven't already, build the conda environment for this project: ``conda env create -f environment.yml``
2. Next, activate the conda environment for this project: ``conda activate climakitae-tests``
3. In the conda environment, install the following packages through pip: ``pip install sphinx-book-theme==0.3.3 sphinx-design==0.3.0``
4. Serialize RST to HTML and start a web server (locally): ``make serve-docs`` **NOTE:** this needs to be run from the root `climakitae` directory, not `climakitae/docs`
5. To see the locally served docs: http://localhost:8000/
6. Push changes to the file `docs/climakitae.rst`

## If you're working on a windows device 
You may encounter issues when trying to build the docs due to path issues.<br><br> If you encounter this error message when building the docs:<br> 
``Makefile error make (e=2): The system cannot find the file specified``<br><br>
We were able to do the following to solve the issue (based off [this](https://stackoverflow.com/questions/33674973/makefile-error-make-e-2-the-system-cannot-find-the-file-specified) stack overflow question):<br>
Add the ``<git-installation-directory>/usr/bin`` directory to your PATH variable too. This basically adds the rest of the linux-like commands that come with the "GIT bash" to your environment. After applying this, the Makefile should run normally again


## If you are interested in other free SVG vectors
Click [here](https://www.svgrepo.com/) for other vectors to include. Vectors should be kept in the `_static/` folder.
