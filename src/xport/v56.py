"""
Read and write the SAS XPORT/XPT file format from SAS Version 5 or 6.

The SAS V5 Transport File format, also called XPORT, or simply XPT, ...
"""
# Standard Library
import re
from collections.abc import Mapping
from io import BytesIO

__all__ = [
    'Library',
    'Member',
    'load',
    'loads',
    'dump',
    'dumps',
]


class Member(Mapping):
    """
    A SAS dataset as a collection of columns.
    """
    def __init__(self, columns, name=None, labels=None, formats=None):
        """
        Initialize a new member dataset of a SAS data library.
        """
        raise NotImplementedError

    @property
    def name(self):
        """
        Dataset name.
        """
        return self._name

    @property
    def labels(self):
        """
        Column descriptions.
        """

    @property
    def formats(self):
        """
        Column display formats.
        """

    def __getitem__(self, key):
        """
        Get a column from the dataset.
        """

    def __iter__(self, key):
        """
        Get an iterator of column names.
        """

    def __len__(self):
        """
        Get the number of columns in the dataset.
        """


class Library(Mapping):
    """
    A collection of datasets from a SAS file.
    """
    def __getitem__(self, key):
        """
        Get a member from the library.
        """
        return Member(self, name=key)

    def __iter__(self):
        """
        Get an iterator of member names.
        """
        return iter(self.members)

    def __len__(self):
        """
        Get the number of members in the library.
        """
        return len(self.members)


def load(fp):
    """
    Deserialize a SAS V5 transport file format document::

        with open('example.xpt', 'rb') as f:
            data = load(f)
    """
    raise NotImplementedError()


def loads(s):
    """
    Deserialize a SAS V5 transport file format document from a string::

        with open('example.xpt', 'rb') as f:
            bytestring = f.read()
        data = loads(bytestring)
    """
    fp = BytesIO(s)
    return load(fp)


def dump(columns, fp, name=None, labels=None, formats=None):
    """
    Serialize a SAS V5 transport file format document::

        data = {
            'a': [1, 2],
            'b': [3, 4],
        }
        with open('example.xpt', 'wb') as f:
            dump(data, f)
    """
    names = {key: key.encode('ISO-8859-1').ljust(8) for key in columns}
    for key, name in names.items():
        if len(name) > 8:
            raise ValueError(f'Encoded column name for {key} exceeds 8 bytes')
        if not re.match(r'[A-Za-z_ ]{8}$', name):
            raise ValueError(f'Invalid character in column name {key}')
    raise NotImplementedError()


def dumps(columns, name=None, labels=None, formats=None):
    """
    Serialize a SAS V5 transport file format document to a string::

        data = {
            'a': [1, 2],
            'b': [3, 4],
        }
        bytestring = dumps(data)
        with open('example.xpt', 'wb') as f:
            f.write(bytestring)
    """
    fp = BytesIO()
    dump(columns, fp)
    fp.seek(0)
    return fp.read()
