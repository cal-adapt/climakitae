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
