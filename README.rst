########################################################################
  Xport
########################################################################

.. sphinx-page-start

Read and write SAS Transport files (``*.xpt``).

SAS uses a handful of archaic file formats: XPORT/XPT, CPORT, SAS7BDAT.
If someone publishes their data in one of those formats, this Python
package will help you convert the data into a more useful format.  If
someone, like the FDA, asks you for an XPT file, this package can write
it for you.


What's it for?
==============

XPORT is the binary file format used by a bunch of `United States
government agencies`_ for publishing data sets. It made a lot of sense
if you were trying to read data files on your IBM mainframe back in
1988.

The official `SAS specification for XPORT`_ is relatively
straightforward. The hardest part is converting IBM-format floating
point to IEEE-format, which the specification explains in detail.

There was an `update to the XPT specification`_ for SAS v8 and above.
This module *has not yet been updated* to work with the new version.
However, if you're using SAS v8+, you're probably not using XPT
format. The changes to the format appear to be trivial changes to the
metadata, but this module's current error-checking will raise a
``ParseError``. If you'd like an update for v8, please let me know by
`submitting an issue`_.

.. _United States government agencies: https://www.google.com/search?q=site:.gov+xpt+file

.. _SAS specification for XPORT: http://support.sas.com/techsup/technote/ts140.pdf

.. _update to the XPT specification: https://support.sas.com/techsup/technote/ts140_2.pdf

.. _submitting an issue: https://github.com/selik/xport/issues/new



Installation
============

Grab the latest stable version from PyPI.::

.. code:: bash

    $ python -m pip install xport



Reading XPT
===========

This module follows the common pattern of providing ``load`` and
``loads`` functions for reading data from a SAS file format.

.. code:: python

    with open('example.xpt', 'rb') as f:
        library = xport.load(f)


The ``library`` in this example is structured as a collection of
datasets, which in turn are collections of columns, mapping column names
to lists of values.  This is similar to the `Pandas`_ dataframe concept.
Each value will be either a ``str`` or ``float``, as specified by the
XPT file metadata.  Note that since XPT files are in an unusual binary
format, you should open them using mode ``'rb'``.

Each ``Member`` dataset of a ``Library`` has a handful of metadata
attributes:

* ``Member.name`` -- The name of the dataset.

* ``Member.labels`` -- Descriptions of each variable.

* ``Member.formats`` -- Display formats of each variable.


.. _Pandas: http://pandas.pydata.org/



You can also use the ``xport`` module as a command-line tool to convert an XPT
file to CSV (comma-separated values) file.

.. code:: bash

    $ python -m xport example.xpt > example.csv


Writing XPT
===========

The ``xport`` package follows the common pattern of providing ``dump``
and ``dumps`` functions for writing data to a SAS file format.

.. code:: python

    columns = {
        'numbers': [1, 3.14, 42],
        'text': ['life', 'universe', 'everything'],
    }
    with open('answers.xpt', 'wb') as f:
        xport.dump(columns, f)


Because column names are restricted to 8 characters, you may wish to
specify column labels as well, which are restricted to 40 characters for
SAS V5 Transport files.

.. code:: python

    columns = {
        'numbers': [1, 3.14, 42],
        'text': ['life', 'universe', 'everything'],
    }
    labels = {
        'numbers': 'All numbers are converted to float',
        'text': 'All text is encoded by ISO-8859-1',
    }
    with open('answers.xpt', 'wb') as f:
        xport.dump(columns, f, name='dataset1', labels=labels)


Feature requests
================

I'm happy to fix bugs, improve the interface, or make the module
faster. Just `submit an issue`_ and I'll take a look.

.. _submit an issue: https://github.com/selik/xport/issues/new



Contributing
============

This project is configured to be developed in a Conda environment.::

.. code:: bash

    $ git clone git@github.com:selik/xport.git
    $ cd xport
    $ make install  # Install into a Conda environment
    $ conda activate xport
    $ make install-html  # Build the docs website


Authors
=======

Original version by `Jack Cushman`_, 2012.
Major revision by Michael Selik, 2016.

.. _Jack Cushman: https://github.com/jcushman
