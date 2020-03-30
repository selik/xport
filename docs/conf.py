"""
Configuration file for the Sphinx documentation builder.
"""
# http://www.sphinx-doc.org/en/master/config

# -- Path setup --------------------------------------------------------

# Standard Library
import os
import sys

# Xport Modules
from xport import __version__  # noqa: E402

sys.path.insert(0, os.path.abspath('../src'))

# -- Project information -----------------------------------------------

project = 'Xport'
copyright = '2020 Michael Selik'
author = 'Michael Selik'

version = str(__version__)  # The short X.Y version
release = str(__version__)  # The full version, including a/b/rc tags

# -- General configuration ---------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.doctest',
    'sphinx.ext.todo',
    'sphinx.ext.mathjax',
    'sphinx.ext.viewcode',
]

# Add any paths that contain templates here, relative to this directory.
templates_path = ['_templates']

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files. This pattern also
# affects html_static_path and html_extra_path.
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']

# -- Options for HTML output -------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation
# for a list of builtin themes.
html_theme = 'alabaster'

# Add any paths that contain custom static files (such as style sheets)
# here, relative to this directory. They are copied after the builtin
# static files, so a file named "default.css" will overwrite the builtin
# "default.css".
html_static_path = ['_static']
