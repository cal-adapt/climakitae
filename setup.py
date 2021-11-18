#!/usr/bin/env python

from setuptools import setup, find_packages

with open('README.rst') as fp:
    _description = f.read()

setup(name='climakit',
      version='0.0.1',
      description='Climate data analysis toolkit',
      long_description=_description,
      author='Cal-Adapt Analytics Engine Team',
      author_email='analytics@cal-adapt.org',
      url='https://github.com/cal-adapt/climakit',
      license='BSD',
      packages=find_packages(exclude=('tests', 'docs')))
