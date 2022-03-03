PKGNAME = climakitae
PYTHON ?= python

check:
	$(PYTHON) -m unittest discover -v

clean:
	rm -rf build dist *.egg-info
	find $(PKGNAME) -iname '*.py[co]' -delete

deps:
	pip install -r requirements.txt

dist: clean
	$(PYTHON) setup.py sdist

html:
	rm -f docs/$(PKGNAME).rst docs/modules.rst
	sphinx-apidoc -o docs $(PKGNAME)
	@$(MAKE) -C docs html

serve-docs: html
	cd docs/_build/html && python -m http.server
