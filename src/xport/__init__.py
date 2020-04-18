"""
Read and write SAS XPORT/XPT-format files.
"""

# Standard Library
import enum
import logging
import re
import warnings
from collections.abc import MutableMapping
from datetime import datetime

# Community Packages
import pandas as pd

from .__about__ import __version__  # noqa: F401

LOG = logging.getLogger(__name__)

__all__ = [
    'Library',
    'Member',
]


class VariableType(enum.IntEnum):
    """
    SAS variables can be either Numeric or Character type.
    """
    NUMERIC = 1
    CHARACTER = 2


class FormatAlignment(enum.IntEnum):
    """
    SAS formats are either left- or right-aligned.
    """
    LEFT = 0
    RIGHT = 1


class Informat:
    """
    SAS variable informat.
    """

    patterns = {
        'Character': re.compile(r'^\$(?P<name>[A-Z0-9]*?)(?P<w>\d+)\.$', re.IGNORECASE),
        'Numeric': re.compile(r'^(?P<name>[A-Z0-9]*?)(?P<w>\d+)\.(?P<d>)?$', re.IGNORECASE),
        'Date/Time': re.compile(r'^(?P<name>[A-Z0-9]*?)(?P<w>\d+)\.$', re.IGNORECASE),
    }

    def __init__(self, spec):
        """
        Create an informat from a text specification.
        """
        for vtype, pattern in self.patterns.items():
            mo = pattern.fullmatch(spec)
            if mo is not None:
                break
        else:
            raise ValueError(f'Unrecognized informat {spec}')
        self._name = mo['name'].upper()
        if len(self._name) > 8:
            raise ValueError(f'Format name {self._name} is longer than 8 characters')
        self._length = int(mo.group('w'))
        try:
            self._decimals = int(mo.group('d'))
        except IndexError:
            self._decimals = None

    @property
    def name(self):
        """The name of the format."""  # noqa: D401
        return self._name

    @property
    def length(self):
        """The width value of the format: ``INFORMATw.``."""  # noqa: D401
        return self._length

    @property
    def decimals(self):
        """The ``d`` value of numeric formats: ``INFORMATw.d``."""  # noqa: D401
        # The documentation states that ``d`` optionally specifies the
        # power of 10 by which to divide numeric input.  If the data
        # contain decimal points, the ``d``` value is ignored.
        return self._decimals


class Format(Informat):
    """
    SAS variable format.
    """

    def __init__(self, name, justify=FormatAlignment.LEFT):
        """
        Create a format from a text specification.
        """
        super().__init__(name)
        self.justify = justify


@pd.api.extensions.register_series_accessor('sas')
class Variable:
    """
    SAS variable metadata.
    """

    def __init__(self, series):
        """
        Initialize the Pandas Series accessor.
        """
        self._series = series
        if series.name is not None:
            self.name = self.label = series.name
        else:
            self.name = self.label = ''
        self.format = None
        self.iformat = None

    @property
    def name(self):
        """
        SAS variable name.
        """
        return self._name.decode('ascii')

    @name.setter
    def name(self, value):
        value = value.encode('ascii')
        if len(value) > 32:
            raise ValueError(f'SAS variable name {value} is longer than 32 characters.')
        if len(value) > 8:
            warnings.warn(
                f'SAS variable name {value} is longer than 8 characters.  SAS'
                'Version 5 or 6 Transport (XPORT) Format limits names to 8 characters.'
            )
        self._name = value

    @property
    def label(self):
        """
        SAS variable label.
        """
        return self._label.decode('ascii')

    @label.setter
    def label(self, value):
        value = value.encode('ascii')
        if len(value) > 256:
            raise ValueError(f'SAS variable label {value} is longer than 256 characters.')
        if len(value) > 40:
            warnings.warn(
                f'SAS variable label {value} is longer than 40 characters.  SAS'
                'Version 5 or 6 Transport (XPORT) Format limits labels to 40 characters.'
            )
        self._label = value

    @property
    def type(self):
        """
        SAS variable type, either numeric or character.
        """
        dtype = self._series.dtype
        if dtype.kind in {'f', 'i'}:
            return VariableType.NUMERIC
        elif dtype.kind == 'O':
            return VariableType.CHARACTER
        raise TypeError(f'{type(self._series).__name__}.dtype {dtype} not supported')

    @property
    def length(self):
        """
        SAS variable maximum length in bytes.
        """
        if self.type == VariableType.NUMERIC:
            return 8
        fact = self._series.str.len().max()
        jure = getattr(self, '_length', None)
        if jure:
            if jure < fact:
                raise ValueError('Maximum string length greater than SAS variable length')
            return jure
        return fact

    @length.setter
    def length(self, value):
        if self.type == VariableType.NUMERIC and value != 8:
            raise NotImplementedError('Numeric variables must be length 8')
        fact = self._series.str.len().max()
        if value < fact:
            raise ValueError(f'Maximum string length greater than {value}')
        self._length = value

    @property
    def number(self):
        """
        Index of the SAS variable within the dataset.
        """
        raise NotImplementedError('Check ``DataFrame.sas.contents`` instead')

    @property
    def position(self):
        """
        Byte-index of the SAS variable field within an observation.
        """
        raise NotImplementedError('Check ``DataFrame.sas.contents`` instead')

    @property
    def format(self):
        """
        SAS variable format.
        """
        return self._format

    @format.setter
    def format(self, value):
        if value is None:
            self._format = None
        else:
            self._format = Format(value)

    @property
    def iformat(self):
        """
        SAS variable informat.
        """
        return self._iformat

    @iformat.setter
    def iformat(self, value):
        if value is None:
            self._format = None
        else:
            self._iformat = Informat(value)


@pd.api.extensions.register_dataframe_accessor('sas')
class Contents:
    """
    SAS dataset metadata.
    """

    def __init__(self, dataframe):
        """
        Initialize the Pandas DataFrame accessor.
        """
        self._df = dataframe
        self.name = self.label = ''
        self.type = 'DATA'
        self.modified = self.created = datetime.now()
        self.os = ''
        self.version = ''

    @property
    def contents(self):
        """
        Variable metadata, such as label, format, number, and position.
        """
        # TODO: Can this dataframe be read-only?
        df = pd.DataFrame({
            'Variable': [c.sas.name for k, c in self._df.items()],
            'Type': [c.sas.type for k, c in self._df.items()],
            'Length': [c.sas.length for k, c in self._df.items()],
            'Format': [str(c.sas.format) for k, c in self._df.items()],
            'Informat': [str(c.sas.informat) for k, c in self._df.items()],
            'Label': [c.sas.label for k, c in self._df.items()],
        })
        df.index.name = '#'
        df['Position'] = df['Length'].cumsum()
        return df

    @property
    def name(self):
        """
        Dataset name.
        """
        return self._name.decode('ascii')

    @name.setter
    def name(self, value):
        value = value.encode('ascii')
        if len(value) > 8:
            raise ValueError(f'Dataset name {value} is longer than 8 characters')
        self._name = value

    @property
    def label(self):
        """
        Dataset label.
        """
        return self._label.decode('ascii')

    @label.setter
    def label(self, value):
        value = value.encode('ascii')
        if len(value) > 40:
            raise ValueError(f'Dataset label {value} is longer than 40 characters')
        self._label = value

    @property
    def type(self):
        """
        Dataset type.
        """
        return self._type.decode('ascii')

    @type.setter
    def type(self, value):
        value = value.encode('ascii')
        if len(value) > 8:
            raise ValueError(f'Dataset type {value} is longer than 8 characters')
        self._type = value

    @property
    def created(self):
        """
        Dataset created.
        """
        return self._created

    @created.setter
    def created(self, value):
        if not isinstance(value, datetime):
            raise TypeError(f'Expected datetime, not {type(value).__name__}')
        if not (datetime(1900, 1, 1) < value < datetime(2100, 1, 1)):
            raise ValueError('Datetime must be in 1900s or 2000s')
        self._created = value

    @property
    def modified(self):
        """
        Dataset modified.
        """
        return self._modified

    @modified.setter
    def modified(self, value):
        if not isinstance(value, datetime):
            raise TypeError(f'Expected datetime, not {type(value).__name__}')
        if not (datetime(1900, 1, 1) < value < datetime(2100, 1, 1)):
            raise ValueError('Datetime must be in 1900s or 2000s')
        self._modified = value

    @property
    def os(self):
        """
        OS used to create the dataset.
        """
        return self._os.decode('ascii')

    @os.setter
    def os(self, value):
        value = value.encode('ascii')
        if len(value) > 8:
            raise ValueError(f'OS name {value} is longer than 8 characters')
        self._os = value

    @property
    def version(self):
        """
        SAS version used to create the dataset.
        """
        return self._version.decode('ascii')

    @version.setter
    def version(self, value):
        value = value.encode('ascii')
        if len(value) > 8:
            raise ValueError(f'SAS version {value} is longer than 8 characters')
        self._version = value


class Member(pd.DataFrame):
    """
    SAS library member.
    """


class Library(MutableMapping):
    """
    Collection of datasets from a SAS file.
    """

    def __init__(self, members, created=None, modified=None, os='', version=''):
        """
        Initialize a SAS data library.
        """
        self._members = {}
        for name, dataframe in members.items():
            if dataframe.sas.name is None:
                raise ValueError('')
            self[name] = dataframe

        if created is None:
            self.created = datetime.now()
        else:
            self.created = created

        if modified is None:
            self.modified = self.created
        else:
            self.modified = modified

        self.os = os
        self.version = version

    @property
    def version(self):
        """
        SAS version used to create the dataset library.
        """
        return self._version.decode('ascii')

    @version.setter
    def version(self, value):
        value = value.encode('ascii')
        if len(value) > 8:
            raise ValueError('SAS version must be <= 8 characters')
        self._version = value

    @property
    def os(self):
        """
        Operating system used to create the dataset library.
        """
        return self._os.decode('ascii')

    @os.setter
    def os(self, value):
        value = value.encode('ascii')
        if len(value) > 8:
            raise ValueError('SAS os must be <= 8 characters')
        self._os = value

    def __repr__(self):
        """
        REPL-format string.
        """
        fmt = '<{cls} members={members}>'
        return fmt.format(cls=type(self).__name__, members=list(self))

    def __getitem__(self, name):
        """
        Get a member dataset.
        """
        return self._members[name]

    def __setitem__(self, name, dataframe):
        """
        Insert or update a member in the library.
        """
        if dataframe.sas.name is None:
            dataframe.sas.name = name
            LOG.debug(f'Set dataframe.sas.name to {name!r}')
        elif name != dataframe.sas.name:
            msg = 'Library member name {a} must match dataset name {b}'
            raise ValueError(msg.format(a=name, b=dataframe.sas.name))
        self._members[name] = dataframe

    def __delitem__(self, name):
        """
        Remove a member datset.
        """
        del self._members[name]

    def __iter__(self):
        """
        Get an iterator of dataset names.
        """
        return iter(self._members)

    def __len__(self):
        """
        Get the number of datasets in the library.
        """
        return len(self._members)

    def __eq__(self, other):
        """
        Compare equality.
        """
        same_keys = set(self) == set(other)
        same_values = all((self[k] == other[k]).all(axis=None) for k in self)
        return same_keys and same_values
