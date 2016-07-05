========
Xport
========

Python reader for SAS XPORT data transport files (``*.xpt``).


What's it for?
==============

XPORT is the binary file format used by a bunch of `United States government
agencies`_ for publishing data sets. It made a lot of sense if you were trying
to read data files on your IBM mainframe back in 1988.

The official `SAS specification for XPORT`_ is relatively straightforward.
The hardest part is converting IBM-format floating point to IEEE-format,
which the specification explains in detail.


.. _United States government agencies: https://www.google.com/search?q=site:.gov+xpt+file

.. _SAS specification for XPORT: http://support.sas.com/techsup/technote/ts140.html



How do I use it?
================

This module mimics the ``csv`` module of the standard library

.. code:: python

    import xport
    with open('example.xpt', 'rb') as f:
        for row in xport.reader(f):
            print row

Each ``row`` will be a namedtuple, with an attribute for each field in the
dataset. Values in the row will be either a unicode string or a float, as
specified by the XPT file metadata. Note that since XPT files are in an
unusual binary format, you should open them using mode ``'rb'``.

You can also use the ``xport`` module as a command-line tool to convert an XPT
file to CSV (comma-separated values).::

    $ python -m xport example.xpt > example.csv


The ``reader`` object also has a handful of metadata:

* ``reader.fields`` -- Names of the fields in each observation.

* ``reader.version`` -- SAS version number used to create the XPT file.

* ``reader.os`` -- Operating system used to create the XPT file.

* ``reader.created`` -- Date and time that the XPT file was created.

* ``reader.modified`` -- Date and time that the XPT file was last modified.



Random access to records
========================

If you want to access specific records, you should either consume the reader in
a ``list`` or use one of ``itertools``
recipes_ for quickly consuming and throwing away unncessary elements.

.. code:: python

    # Collect all the records in a list for random access
    rows = list(xport.reader(f)))

    # Select only record 42
    from itertools import islice
    row = next(islice(xport.reader(f), 42, None))

    # Select only the last 42 records
    from collections import deque
    rows = deque(xport.reader(f), maxlen=42)

.. _recipes: https://docs.python.org/2/library/itertools.html#recipes



Recent changes
==============

* Improved the API.

* Fixed handling of NaNs.

* Fixed piping the file from ``stdin`` in Python 3.


Authors
=======

Original version by `Jack Cushman`_, 2012.
Major revision by Michael Selik, 2016.

.. _Jack Cushman: https://github.com/jcushman

