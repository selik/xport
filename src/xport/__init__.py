"""
Read and write SAS XPORT/XPT-format files.
"""

# Standard Library
import enum
import logging
import re
import struct
import warnings
from collections.abc import Mapping, MutableMapping
from datetime import datetime
from io import StringIO

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

    templates = {
        VariableType.CHARACTER: '${name}{w}.',
        VariableType.NUMERIC: '{name}{w}.{d}',
        # 'Date/Time': '{name}{w}.',
    }

    patterns = {
        VariableType.CHARACTER: r'^\$(?P<name>[A-Z0-9]*?)(?P<w>\d+)\.$',
        VariableType.NUMERIC: r'^(?P<name>[A-Z0-9]*?)(?P<w>\d+)\.(?P<d>\d*)$',
        # 'Date/Time': r'^(?P<name>[A-Z0-9]*?)(?P<w>\d+)\.$',
    }
    patterns = {k: re.compile(v, re.IGNORECASE) for k, v in patterns.items()}

    #    char8 niform;      /* NAME OF INPUT FORMAT                   */
    #    short nifl;         /* INFORMAT LENGTH ATTRIBUTE              */
    #    short nifd;        /* INFORMAT NUMBER OF DECIMALS            */
    byte_structure = '>8shh'

    def __init__(self, name='', length=None, decimals=None, vtype=None):
        """
        Initialize an input format.
        """
        self._name = name
        self._length = length
        self._decimals = decimals
        self._vtype = vtype

    def __str__(self):
        """
        Pleasant display value.
        """
        try:
            fmt = self.templates[self.vtype]
        except KeyError:
            return f'<{type(self).__name__}> ' + ', '.join([
                f'name={self.name!r}',
                f'length={self.length!r}',
                f'decimals={self.decimals!r}',
                f'vtype={self.vtype!r}',
            ])
        if self.vtype == VariableType.CHARACTER:
            return fmt.format(name=self.name, w=self.length)
        decimals = self.decimals if self.decimals is not None else ''
        return fmt.format(name=self.name, w=self.length, d=decimals)

    def __repr__(self):
        """
        REPL-format string.
        """
        return f'{type(self).__name__}({str(self)!r})'

    def __bytes__(self):
        """
        XPORT-format byte string.
        """
        fmt = self.byte_structure
        name = self.name.encode('ascii')
        if len(name) > 8:
            raise ValueError('ASCII-encoded {name!r} longer than 8 bytes')
        length = self.length if self.length is not None else 0
        decimals = self.decimals if self.decimals is not None else 0
        return struct.pack(fmt, name, length, decimals)

    @classmethod
    def unpack(cls, bytestring):
        """
        Create an informat from an XPORT-format bytestring.
        """
        LOG.debug(f'Unpacking informat from {bytestring}')
        fmt = cls.byte_structure
        return cls.from_struct_tokens(*struct.unpack(fmt, bytestring))

    @classmethod
    def from_struct_tokens(cls, name, length, decimals):
        """
        Create an informat from unpacked struct tokens.
        """
        LOG.debug(f'Creating informat from struct tokens {(name, length, decimals)}')
        # TODO: Determine vtype when unpacking.  Perhaps there's a list
        #       of format names we can grab from SAS documentation.
        #       That wouldn't solve the edge case of a numeric vtype
        #       with no decimals specification and no name, which would
        #       be indistinguishable from a character vtype.
        name = name.strip(b'\x00').decode('ascii').strip()
        return cls(name=name, length=length, decimals=decimals, vtype=None)

    @classmethod
    def from_spec(cls, spec, *args, **kwds):
        """
        Create an informat from a text specification.
        """
        LOG.debug(f'Parsing informat specification {spec}')
        for vtype, pattern in cls.patterns.items():
            mo = pattern.fullmatch(spec)
            if mo is not None:
                break
        else:
            raise ValueError(f'Unrecognized informat {spec}')
        name = mo['name'].upper()
        if len(name) > 8:
            raise ValueError(f'Format name {name} is longer than 8 characters')
        length = int(mo.group('w'))
        try:
            decimals = mo.group('d')
        except IndexError:
            decimals = None
        else:
            if decimals != '':
                decimals = int(decimals)
            else:
                decimals = None
        return cls(*args, name=name, length=length, decimals=decimals, vtype=vtype, **kwds)

    @property
    def vtype(self):
        """Variable type."""
        return self._vtype

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

    def __eq__(self, other):
        """Equality."""
        return (
            self.vtype == other.vtype and self.name == other.name and self.length == other.length
            and self.decimals == other.decimals
        )


class Format(Informat):
    """
    SAS variable format.
    """

    #    char8 nform;       /* NAME OF FORMAT                         */
    #    short nfl;          /* FORMAT FIELD LENGTH OR 0               */
    #    short nfd;         /* FORMAT NUMBER OF DECIMALS              */
    #    short nfj;         /* 0=LEFT JUSTIFICATION, 1=RIGHT JUST     */
    byte_structure = '>8shhh'

    def __init__(self, *args, justify=FormatAlignment.LEFT, **kwds):
        """
        Initialize a SAS variable format.
        """
        super().__init__(*args, **kwds)
        self._justify = justify

    def __bytes__(self):
        """
        XPORT-format byte string.
        """
        # TODO: It'd be nice to avoid copy-pasting code from parent.
        fmt = self.byte_structure
        name = self.name.encode('ascii')
        if len(name) > 8:
            raise ValueError('ASCII-encoded {name!r} longer than 8 bytes')
        length = self.length if self.length is not None else 0
        decimals = self.decimals if self.decimals is not None else 0
        return struct.pack(fmt, name, length, decimals, self.justify)

    @classmethod
    def from_spec(cls, spec, justify=FormatAlignment.LEFT):
        """
        Create a format from a text specification.
        """
        return super().from_spec(spec=spec, justify=justify)

    @property
    def justify(self):
        """
        Left- or right-alignment.
        """
        return self._justify

    def __eq__(self, other):
        """Equality."""
        return super().__eq__(other) and self.justify == other.justify


# The Pandas documentation suggests avoiding inheritance, but their
# other options for extending ``Series`` objects fall flat, because so
# many Pandas methods return new instances.  To carry variable metadata
# over to the new instances, we need to overrride the constructors.
# https://pandas.pydata.org/pandas-docs/stable/development/extending.html


class Variable(pd.Series):
    """
    SAS variable.

    ``Variable`` extends Pandas' ``Series``, adding SAS metadata.
    """

    # Register metadata with Pandas' convoluted attribute accessors.
    _metadata = [
        '_sas_name',
        '_sas_label',
        '_sas_format',
        '_sas_iformat',
        # '_sas_variable_type',
        '_sas_variable_number',
        '_sas_variable_position',
        '_sas_variable_length',
    ]

    def __init__(self, *args, **kwds):
        """
        Initialize SAS variable metadata.
        """
        super().__init__(*args, **kwds)
        self.sas_variable_type  # Force dtype validation.

    @property
    def _constructor(self):
        """
        Construct an instance with the same dimensions as the original.
        """
        return Variable

    @property
    def _constructor_expanddim(self):
        """
        Construct an instance with an extra dimension.

        For example, transforming a series into a dataframe.
        """
        raise NotImplementedError("Can't copy SAS variable metadata to dataframe")

    @property
    def sas_name(self):
        """
        SAS variable name.
        """
        try:
            return self._sas_name
        except AttributeError:
            return self.name

    @sas_name.setter
    def sas_name(self, value):
        bytestring = value.encode('ascii')
        if len(bytestring) > 32:
            raise ValueError(f'ASCII-encoded name {bytestring} longer than 32 characters')
        if len(bytestring) > 8:
            warnings.warn(
                f'ASCII-encoded name {bytestring} is longer than 8 characters.  SAS'
                'Version 5 or 6 Transport (XPORT) Format limits names to 8 characters.'
            )
        self._sas_name = value

    @property
    def sas_label(self):
        """
        SAS variable label.
        """
        try:
            return self._sas_label
        except AttributeError:
            return ''

    @sas_label.setter
    def sas_label(self, value):
        bytestring = value.encode('ascii')
        if len(bytestring) > 256:
            raise ValueError(f'ASCII-encoded label {bytestring} is longer than 256 characters')
        if len(bytestring) > 40:
            warnings.warn(
                f'ASCII-encoded label {bytestring} is longer than 40 characters.  SAS'
                'Version 5 or 6 Transport (XPORT) Format limits labels to 40 characters.'
            )
        self._sas_label = value

    @property
    def sas_variable_type(self):
        """
        SAS variable type, either numeric or character.
        """
        if self.dtype.kind in {'f', 'i'}:
            return VariableType.NUMERIC
        elif self.dtype.kind == 'O':
            return VariableType.CHARACTER
        elif self.dtype.kind == 'b':
            # We'll encode Boolean columns as 1 if True else 0.
            return VariableType.NUMERIC
        raise TypeError(f'{type(self).__name__}.dtype {self.dtype} not supported')

    @property
    def sas_variable_length(self):
        """
        SAS variable maximum length in bytes.
        """
        if self.sas_variable_type == VariableType.NUMERIC:
            return 8
        fact = self.str.len().max() if not self.empty else 0
        jure = getattr(self, '_sas_variable_length', None)
        if jure is not None:
            if jure < fact:
                raise ValueError('Maximum string length greater than SAS variable length')
            return jure
        return fact

    @sas_variable_length.setter
    def sas_variable_length(self, value):
        if self.sas_variable_type == VariableType.NUMERIC and value != 8:
            raise NotImplementedError('Numeric variables must be length 8')
        fact = self.str.len().max() if not self.empty else 0
        if value < fact:
            raise ValueError(f'Maximum string length greater than {value}')
        self._sas_variable_length = value

    @property
    def sas_variable_number(self):
        """
        Index of the SAS variable within the dataset.
        """
        return getattr(self, '_sas_variable_number', None)

    @sas_variable_number.setter
    def sas_variable_number(self, value):
        if value < 1 or value % 1:
            raise ValueError(f'Variable number {value} should be a natural number')
        self._sas_variable_number = int(value)

    @property
    def sas_variable_position(self):
        """
        Byte-index of the SAS variable field within an observation.
        """
        return getattr(self, '_sas_variable_position', None)

    @sas_variable_position.setter
    def sas_variable_position(self, value):
        if value < 0 or value % 1:
            raise ValueError(f'Variable position {value} should be a whole number')
        self._sas_variable_position = int(value)

    @property
    def sas_format(self):
        """
        SAS variable format.
        """
        return getattr(self, '_sas_format', None)

    @sas_format.setter
    def sas_format(self, value):
        if value is None:
            self._sas_format = None
        else:
            self._sas_format = Format.from_spec(value)

    @property
    def sas_iformat(self):
        """
        SAS variable informat.
        """
        return getattr(self, '_sas_iformat', None)

    @sas_iformat.setter
    def sas_iformat(self, value):
        if value is None:
            self._sas_iformat = None
        else:
            self._sas_iformat = Informat.from_spec(value)


class Dataset(pd.DataFrame):
    """
    SAS data set.

    ``Dataset`` extends Pandas' ``DataFrame``, adding SAS metadata.
    """

    _metadata = [
        '_sas_name',
        '_sas_label',
        '_sas_dataset_type',
        '_sas_dataset_created',
        '_sas_dataset_modified',
        '_sas_os',
        '_sas_version',
    ]

    def __init__(self, *args, sas_name='', sas_label='', **kwds):
        """
        Fix column order to match variable order if possible.
        """
        super().__init__(*args, **kwds)
        self._sas_name = sas_name
        self._sas_label = sas_label

        if 'data' in kwds:
            data = kwds['data']
        elif args:
            data = args[0]
        else:
            data = None
        if isinstance(data, Mapping):
            p = 0
            for i, (k, v) in enumerate(data.items(), 1):
                if isinstance(v, Variable):
                    cpy = self[k]
                    # Bypass attribute validation.
                    cpy._sas_name = v.sas_name
                    cpy._sas_label = v.sas_label
                    cpy._sas_format = v.sas_format
                    cpy._sas_iformat = v.sas_iformat
                    if v.sas_variable_number is None:
                        cpy._sas_variable_number = i
                    else:
                        cpy._sas_variable_number = v.sas_variable_number
                    if v.sas_variable_position is None:
                        cpy._sas_variable_position = p
                    else:
                        cpy._sas_variable_position = v.sas_variable_position
                    cpy._sas_variable_length = v.sas_variable_length
                    p += v.sas_variable_length
        LOG.debug('Dataset variables\n%s', self.sas_variables)
        LOG.debug('Contents\n%s', self.infos())

    @property
    def _constructor(self):
        """
        Construct an instance with the same dimensions as the original.
        """
        return Dataset

    @property
    def _constructor_sliced(self):
        """
        Construct an instance with one less dimension.

        For example, slicing a single column from a dataframe.
        """
        return Variable

    @property
    def sas_variables(self):
        """
        Variable metadata, such as label, format, number, and position.
        """
        # TODO: Can this dataframe be read-only?
        # TODO: Does this unnecessarily make bunches of copies of the data?
        df = pd.DataFrame({
            'Variable': [v.sas_name for k, v in self.items()],
            'Type': [v.sas_variable_type for k, v in self.items()],
            'Length': [v.sas_variable_length for k, v in self.items()],
            'Position': [v.sas_variable_position for k, v in self.items()],
            'Format': [str(v.sas_format) for k, v in self.items()],
            'Informat': [str(v.sas_iformat) for k, v in self.items()],
            'Label': [v.sas_label for k, v in self.items()],
        })
        df.index = order = [v.sas_variable_number for k, v in self.items()]
        df.index.name = '#'
        if order != list(range(1, len(df) + 1)):
            warnings.warn(f"SAS variable numbers {order} don't match column order")
        if not df.empty and (df['Length'].cumsum() != df['Position']).any():
            warnings.warn(f"SAS variable positions don't match order and length")
        return df

    def infos(self):
        """
        Like ``DataFrame.info`` but returns a string.
        """
        buf = StringIO()
        self.info(buf=buf)
        buf.seek(0)
        return buf.read()

    @property
    def sas_name(self):
        """
        Dataset name.
        """
        return self._sas_name

    @sas_name.setter
    def sas_name(self, value):
        bytestring = value.encode('ascii')
        if len(bytestring) > 8:
            raise ValueError(f'ASCII-encoded name {bytestring} longer than 8 characters')
        self._sas_name = value

    @property
    def sas_label(self):
        """
        Dataset label.
        """
        return self._sas_label

    @sas_label.setter
    def sas_label(self, value):
        bytestring = value.encode('ascii')
        if len(bytestring) > 40:
            raise ValueError(f'ASCII-encoded label {bytestring} longer than 40 characters')
        self._sas_label = value

    @property
    def sas_dataset_type(self):
        """
        Dataset type.
        """
        return self._sas_dataset_type

    @sas_dataset_type.setter
    def sas_dataset_type(self, value):
        bytestring = value.encode('ascii')
        if len(bytestring) > 8:
            raise ValueError(f'ASCII-encoded type {bytestring} longer than 8 characters')
        self._sas_dataset_type = value

    @property
    def sas_dataset_created(self):
        """
        Dataset created.
        """
        return self._sas_dataset_created

    @sas_dataset_created.setter
    def sas_dataset_created(self, value):
        if not isinstance(value, datetime):
            raise TypeError(f'Expected datetime, not {type(value).__name__}')
        if not (datetime(1900, 1, 1) < value < datetime(2100, 1, 1)):
            raise ValueError('Datetime must be in 1900s or 2000s')
        self._sas_dataset_created = value

    @property
    def sas_dataset_modified(self):
        """
        Dataset modified.
        """
        return self._sas_dataset_modified

    @sas_dataset_modified.setter
    def sas_dataset_modified(self, value):
        if not isinstance(value, datetime):
            raise TypeError(f'Expected datetime, not {type(value).__name__}')
        if not (datetime(1900, 1, 1) < value < datetime(2100, 1, 1)):
            raise ValueError('Datetime must be in 1900s or 2000s')
        self._sas_dataset_modified = value

    @property
    def sas_os(self):
        """
        OS used to create the dataset.
        """
        return self._sas_os

    @sas_os.setter
    def sas_os(self, value):
        bytestring = value.encode('ascii')
        if len(bytestring) > 8:
            raise ValueError(f'ASCII-encoded OS name {bytestring} longer than 8 characters')
        self._sas_os = value

    @property
    def sas_version(self):
        """
        SAS version used to create the dataset.
        """
        return self._sas_version

    @sas_version.setter
    def sas_version(self, value):
        bytestring = value.encode('ascii')
        if len(bytestring) > 8:
            raise ValueError(f'ASCII-encoded SAS version {bytestring} longer than 8 characters')
        self._sas_version = value


class Library(MutableMapping):
    """
    Collection of datasets from a SAS file.
    """

    def __init__(self, members=(), created=None, modified=None, os='', version=''):
        """
        Initialize a SAS data library.
        """
        self._members = {}
        if isinstance(members, Mapping):
            for name, dataset in members.items():
                self[name] = dataset  # Use __setitem__ to validate metadata.
        else:
            for dataset in members:
                if dataset.sas_name in self:
                    warnings.warn(f'More than one dataset named {dataset.sas_name!r}')
                self[dataset.sas_name] = dataset

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
        return self._version

    @version.setter
    def version(self, value):
        bytestring = value.encode('ascii')
        if len(bytestring) > 8:
            raise ValueError('ASCII-encoded {bytestring} longer than 8 characters')
        self._version = value

    @property
    def os(self):
        """
        Operating system used to create the dataset library.
        """
        return self._os

    @os.setter
    def os(self, value):
        bytestring = value.encode('ascii')
        if len(bytestring) > 8:
            raise ValueError('ASCII-encoded {bytestring} longer than 8 characters')
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

    def __setitem__(self, name, dataset):
        """
        Insert or update a member in the library.
        """
        if dataset.sas_name is None:
            dataset.sas_name = name
            LOG.debug(f'Set dataset SAS name to {name!r}')
        elif name != dataset.sas_name:
            msg = 'Library member name {a} must match dataset SAS name {b}'
            raise ValueError(msg.format(a=name, b=dataset.sas_name))
        self._members[name] = dataset

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
