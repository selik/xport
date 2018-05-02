#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
Read SAS XPORT/XPT files.

Copyright (c) 2016 Michael Selik.
Inspired by Jack Cushman's original 2012 version.
'''

from io import BytesIO
from . import reading
from . import writing

__version__ = (3, 0, 0)

__all__ = [
    'load',
    'loads',
    'dump',
    'dumps',
]

def load(fp):
    '''
    Deserialize ``fp`` (a ``.read()``-supporting file-like object
    containing a XPORT document) to a Python object.
    '''
    reader = reading.Reader(fp)
    keys = reader.fields
    columns = {k: [] for k in keys}
    for row in reader:
        for key, value in zip(keys, row):
            columns[key].append(value)
    return columns

def loads(s):
    '''
    Deserialize ``s`` (a ``bytes`` instance containing an XPORT
    document) to a Python object.
    '''
    return load(BytesIO(s))

def dump(columns, fp):
    '''
    Serialize ``columns`` as a XPORT formatted bytes stream to ``fp`` (a
    ``.write()``-supporting file-like object).
    '''
    return writing.from_columns(columns, fp)

def dumps(columns):
    '''
    Serialize ``columns`` to a JSON formatted ``bytes`` object.
    '''
    fp = BytesIO()
    dump(columns, fp)
    fp.seek(0)
    return fp.read()
