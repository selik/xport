#!/usr/bin/env python3
# -*- encoding: utf-8 -*-
"""
To upload to PyPI:

    $ python setup.py sdist
    $ python setup.py bdist_wheel --universal
    $ twine upload dist/*

"""
# Community Packages
from setuptools import setup

# Most arguments for ``setup`` should be written in ``setup.cfg``.
# https://setuptools.readthedocs.io/en/latest/setuptools.html#using-a-src-layout
setup()
