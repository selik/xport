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
``ParseError``. If you'd like an update for v8, please let me know by
`submitting an issue`_.

.. _United States government agencies: https://www.google.com/search?q=site:.gov+xpt+file

.. _SAS specification for XPORT: http://support.sas.com/techsup/technote/ts140.pdf

.. _update to the XPT specification: https://support.sas.com/techsup/technote/ts140_2.pdf

.. _submitting an issue: https://github.com/selik/xport/issues/new



Reading XPT
===========

This module mimics the ``csv`` module of the standard library,
providing ``Reader`` and ``DictReader`` classes. Note that
``xport.Reader`` is capitalized, unlike ``csv.reader``.

.. code:: python

    with open('example.xpt', 'rb') as f:
        for row in xport.Reader(f):
            print row



Values in the row will be either a unicode string or a float, as
specified by the XPT file metadata. Note that since XPT files are in
an unusual binary format, you should open them using mode ``'rb'``.



For convenience, you can also use the ``NamedTupleReader`` to get each
row as a namedtuple, with an attribute for each field in the dataset.



The module also provides a handful of utility functions for reading
the whole XPT file and loading the rows into a Python data structure.
The ``to_rows`` function will simply return a list of rows. The
``to_columns`` function will return the data as columns rather than
rows. The columns will be an ``OrderedDict`` mapping the column labels
as strings to the column values as lists of either strings or floats.

.. code:: python

    with open('example.xpt', 'rb') as f:
        columns = xport.to_columns(f)



For convenient conversion to a `NumPy`_ array or `Pandas`_ dataframe,
you can use ``to_numpy`` and ``to_dataframe``.

.. code:: python

    with open('example.xpt', 'rb') as f:
        a = xport.to_numpy(f)

    with open('example.xpt', 'rb') as f:
        df = xport.to_dataframe(f)

.. _NumPy: http://www.numpy.org/

.. _Pandas: http://pandas.pydata.org/



The ``Reader`` object has a handful of metadata attributes:

* ``Reader.fields`` -- Names of the fields in each observation.

* ``Reader.version`` -- SAS version number used to create the XPT file.

* ``Reader.os`` -- Operating system used to create the XPT file.

* ``Reader.created`` -- Date and time that the XPT file was created.

* ``Reader.modified`` -- Date and time that the XPT file was last modified.



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
    row = next(islice(xport.Reader(f), 42, None))

    # Select only the last 42 records
    from collections import deque
    rows = deque(xport.Reader(f), maxlen=42)

.. _recipes: https://docs.python.org/2/library/itertools.html#recipes



Writing XPT
===========

The ``from_columns`` function will write an XPT file from a mapping of
labels (as string) to columns (as iterable) or an iterable of (label,
column) pairs.

.. code:: python

    # a mapping of labels to columns
    mapping = {'numbers': [1, 3.14, 42],
               'text': ['life', 'universe', 'everything']}

    with open('answers.xpt', 'wb') as f:
        xport.from_columns(mapping, f)



Column labels are restricted to 40 characters. Column names are
restricted to 8 characters and will be automatically created based on
the column label -- the first 8 characters, non-alphabet characters
replaced with underscores, padded to 8 characters if necessary. All
text strings, including column labels, will be converted to bytes
using the ISO-8859-1 encoding.

Unfortunately, writing XPT files cannot cleanly mimic the ``csv``
module, because we must examine all rows before writing any rows to
correctly write the XPT file headers.



The ``to_rows`` function expects an iterable of iterables, like a list
of tuples. In this case, the column labels have not been specified and
will automatically be assigned as 'x0', 'x1', 'x2', ..., 'xM'.

.. code:: python

    rows = [('a', 1), ('b', 2)]

    with open('example.xpt', 'wb') as f:
        xport.from_rows(rows, f)



To specify the column labels for ``to_rows``, each row can be a
mapping (such as a ``dict``) of the column labels to that row's
values. Each row should have the same keys. Passing in rows as
namedtuples, or any instance of a ``tuple`` that has a ``._fields``
attribute, will set the column labels to the attribute names of the
first row.

.. code:: python

    rows = [{'letters': 'a', 'numbers': 1},
            {'letters': 'b', 'numbers': 2}]

    with open('example.xpt', 'wb') as f:
        xport.from_rows(rows, f)






Recent changes
==============

* Switched from ``load``/``dump`` with mode flags to ``to_rows``,
  ``to_columns``, ``from_rows`` and ``from_columns``.


Authors
=======

Original version by `Jack Cushman`_, 2012.
Major revision by Michael Selik, 2016.

.. _Jack Cushman: https://github.com/jcushman
