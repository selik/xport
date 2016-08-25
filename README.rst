========
Xport
========

Python reader for SAS XPORT data transport files (``*.xpt``).



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
``ValueError``.

.. _United States government agencies: https://www.google.com/search?q=site:.gov+xpt+file

.. _SAS specification for XPORT: http://support.sas.com/techsup/technote/ts140.pdf

.. _update to the XPT specification: https://support.sas.com/techsup/technote/ts140_2.pdf



Reading XPT
===========

This module mimics the ``csv`` module of the standard library for
iterating over rows.

.. code:: python

    import xport
    with open('example.xpt', 'rb') as f:
        for row in xport.reader(f):
            print row

Each ``row`` will be a namedtuple, with an attribute for each field in the
dataset. Values in the row will be either a unicode string or a float, as
specified by the XPT file metadata. Note that since XPT files are in an
unusual binary format, you should open them using mode ``'rb'``.



This module also provides ``load`` and ``loads`` functions, similar to
modules like ``json`` and ``pickle``, for reading the XPT all at once
into a list of rows.

.. code:: python

    import xport
    with open('example.xpt', 'rb') as f:
        rows = load(f)



For convenient conversion to a `NumPy`_ array or `Pandas`_ dataframe,
you can use ``to_numpy`` and ``to_dataframe``.

.. code:: python

    a = xport.to_numpy('example.xpt')
    df = xport.to_dataframe('example.xpt')

.. _NumPy: http://www.numpy.org/

.. _Pandas: http://pandas.pydata.org/



The ``reader`` object has a handful of metadata attributes:

* ``reader.fields`` -- Names of the fields in each observation.

* ``reader.version`` -- SAS version number used to create the XPT file.

* ``reader.os`` -- Operating system used to create the XPT file.

* ``reader.created`` -- Date and time that the XPT file was created.

* ``reader.modified`` -- Date and time that the XPT file was last modified.



You can also use the ``xport`` module as a command-line tool to convert an XPT
file to CSV (comma-separated values) file.::

    $ python -m xport example.xpt > example.csv



If you want to access specific records, you should use the ``load``
function to gather the rows in a list or use one of ``itertools``
recipes_ for quickly consuming and throwing away unncessary elements.

.. code:: python

    # Collect all the records in a list for random access
    rows = load(f)

    # Select only record 42
    from itertools import islice
    row = next(islice(xport.reader(f), 42, None))

    # Select only the last 42 records
    from collections import deque
    rows = deque(xport.reader(f), maxlen=42)

.. _recipes: https://docs.python.org/2/library/itertools.html#recipes



Writing XPT
===========

This module mimics the ``json`` and ``pickle`` standard library
modules in providing ``dump`` and ``dumps`` functions to transform
Python objects into XPT file format.

.. code:: python

    columns = {'numbers': [1, 3.14, 42], 'text': ['life', 'universe', 'everything']}
    with open('answers.xpt', 'wb') as f:
        dump(f, columns)



If you have unlabeled rows, one way to convert them to labeled columns
is to assign labels as whole numbers starting from 0.

.. code:: python

    rows = [('a', 1), ('b', 2)]
    columns = {str(label): column for label, column in enumerate(zip(*rows))}

    with open('example.xpt', 'wb') as f:
        dump(f, columns)



Column labels are restricted to 40 characters. Column names are
restricted to 8 characters and will be automatically created based on
the column label -- the first 8 characters, non-alphabet characters
replaced with underscores, padded to 8 characters if necessary. All
text strings, including column labels, will be converted to bytes
using the ISO-8859-1 encoding. Any byte strings will not be changed
and may create invalid XPT files if they were encoded inappropriately.

Unfortunately, writing XPT files cannot cleanly mimic the ``csv``
module, because we must examine all rows before writing any rows to
correctly write the XPT file headers.



Recent changes
==============

* Added capability to write XPT files

* Added ``load`` and ``loads`` functions to match the new ``dump`` and
  ``dumps`` functions


Authors
=======

Original version by `Jack Cushman`_, 2012.
Major revision by Michael Selik, 2016.

.. _Jack Cushman: https://github.com/jcushman

