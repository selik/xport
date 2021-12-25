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
``ValueError``. If you'd like an update for v8, please let me know by
`submitting an issue`_.

.. _United States government agencies: https://www.google.com/search?q=site:.gov+xpt+file

.. _SAS specification for XPORT: http://support.sas.com/techsup/technote/ts140.pdf

.. _update to the XPT specification: https://support.sas.com/techsup/technote/ts140_2.pdf

.. _submitting an issue: https://github.com/selik/xport/issues/new



Installation
============

This project requires Python v3.7+.  Grab the latest stable version from
PyPI.

.. code:: bash

    $ python -m pip install --upgrade xport



Reading XPT
===========

This module follows the common pattern of providing ``load`` and
``loads`` functions for reading data from a SAS file format.

.. code:: python

    import xport.v56

    with open('example.xpt', 'rb') as f:
        library = xport.v56.load(f)


The XPT decoders, ``xport.load`` and ``xport.loads``, return a
``xport.Library``, which is a mapping (``dict``-like) of
``xport.Dataset``s.  The ``xport.Dataset``` is a subclass of
``pandas.DataFrame`` with SAS metadata attributes (name, label, etc.).
The columns of a ``xport.Dataset`` are ``xport.Variable`` types, which
are subclasses of ``pandas.Series`` with SAS metadata (name, label,
format, etc.).

If you're not familiar with `Pandas`_'s dataframes, it's easy to think
of them as a dictionary of columns, mapping variable names to variable
data.

The SAS Transport (XPORT) format only supports two kinds of data.  Each
value is either numeric or character, so ``xport.load`` decodes the
values as either ``str`` or ``float``.

Note that since XPT files are in an unusual binary format, you should
open them using mode ``'rb'``.

.. _Pandas: http://pandas.pydata.org/


You can also use the ``xport`` module as a command-line tool to convert
an XPT file to CSV (comma-separated values) file.  The ``xport``
executable is a friendly alias for ``python -m xport``. Caution: if this command-line does not work with the lastest version, it should be working with version 2.0.2. To get this version, we can either download the files from this `link`_ or simply type the following command line your bash terminal: ``pip install xport==2.0.2``.

.. _link: https://pypi.org/project/xport/2.0.2/#files

.. code:: bash

    $ xport example.xpt > example.csv


Writing XPT
===========

The ``xport`` package follows the common pattern of providing ``dump``
and ``dumps`` functions for writing data to a SAS file format.

.. code:: python

    import xport
    import xport.v56

    ds = xport.Dataset()
    with open('example.xpt', 'wb') as f:
        xport.v56.dump(ds, f)


Because the ``xport.Dataset`` is an extension of ``pandas.DataFrame``,
you can create datasets in a variety of ways, converting easily from a
dataframe to a dataset.

.. code:: python

    import pandas as pd
    import xport
    import xport.v56

    df = pandas.DataFrame({'NUMBERS': [1, 2], 'TEXT': ['a', 'b']})
    ds = xport.Dataset(df, name='MAX8CHRS', label='Up to 40!')
    with open('example.xpt', 'wb') as f:
        xport.v56.dump(ds, f)


SAS Transport v5 restricts variable names to 8 characters (with a
strange preference for uppercase) and labels to 40 characters.  If you
want the relative comfort of SAS Transport v8's limit of 246 characters,
please `make an enhancement request`_.


It's likely that most people will be using Pandas_ dataframes for the
bulk of their analysis work, and will want to convert to XPT at the
very end of their process.

.. code:: python

    import pandas as pd
    import xport
    import xport.v56

    df = pd.DataFrame({
        'alpha': [10, 20, 30],
        'beta': ['x', 'y', 'z'],
    })

    ...  # Analysis work ...

    ds = xport.Dataset(df, name='DATA', label='Wonderful data')

    # SAS variable names are limited to 8 characters.  As with Pandas
    # dataframes, you must change the name on the dataset rather than
    # the column directly.
    ds = ds.rename(columns={k: k.upper()[:8] for k in ds})

    # Other SAS metadata can be set on the columns themselves.
    for k, v in ds.items():
        v.label = k.title()
        if v.dtype == 'object':
            v.format = '$CHAR20.'
        else:
            v.format = '10.2'

    # Libraries can have multiple datasets.
    library = xport.Library({'DATA': ds})

    with open('example.xpt', 'wb') as f:
        xport.v56.dump(library, f)


Feature requests
================

I'm happy to fix bugs, improve the interface, or make the module
faster.  Just `submit an issue`_ and I'll take a look.  If you work for
a corporation or well-funded non-profit, please consider a sponsorship_.

.. _make an enhancement request: https://github.com/selik/xport/issues/new
.. _submit an issue: https://github.com/selik/xport/issues/new
.. _sponsorship: https://github.com/sponsors/selik


Thanks
======

Current and past sponsors include:

|ProtocolFirst|

.. |ProtocolFirst| image:: docs/_static/protocolfirst.png
   :alt: Protocol First
   :target: https://www.protocolfirst.com


Contributing
============

This project is configured to be developed in a Conda environment.

.. code:: bash

    $ git clone git@github.com:selik/xport.git
    $ cd xport
    $ make install          # Install into a Conda environment
    $ conda activate xport  # Activate the Conda environment
    $ make install-html     # Build the docs website


Authors
=======

Original version by `Jack Cushman`_, 2012.

Major revisions by `Michael Selik`_, 2016 and 2020.

Minor revisions by `Alfred Chan`_, 2020.

Minor revisions by `Derek Croote`_, 2021.

.. _Jack Cushman: https://github.com/jcushman

.. _Michael Selik: https://github.com/selik

.. _Alfred Chan: https://github.com/alfred-b-chan

.. _Derek Croote: https://github.com/dcroote
