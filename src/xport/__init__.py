"""
Read and write SAS XPORT/XPT-format files.
"""

# Standard Library
import itertools
import logging
import pathlib

# Xport Modules
import xport.v56
from xport.v56 import Library, Member, dump, dumps

from .__about__ import __version__  # noqa: F401

LOG = logging.getLogger(__name__)

__all__ = [
    'Library',
    'Member',
    'load',
    'loads',
    'dump',
    'dumps',
]

formats = {
    '.xpt': [xport.v56],
}


def load(fp):
    """
    Deserialize a SAS file format document::

        with open('example.xpt', 'rb') as f:
            data = load(f)
    """
    modules = formats[pathlib.Path(fp.name).suffix]
    for fmt in modules:
        try:
            return fmt.load(fp)
        except Exception:
            LOG.exception('Could not parse file with %s', fmt.__name__)
    raise ValueError('Could not parse file')


def loads(s):
    """
    Deserialize a SAS file format document from a string::

        with open('example.xpt', 'rb') as f:
            bytestring = f.read()
        data = loads(bytestring)
    """
    modules = itertools.chain.from_iterable(formats.values())
    for fmt in modules:
        try:
            return fmt.loads(s)
        except Exception:
            LOG.exception('Could not parse file with %s', fmt.__name__)
    raise ValueError('Could not parse file')
