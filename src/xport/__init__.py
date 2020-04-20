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

    pattern = re.compile(
        r'^(?P<name>\$?[A-Z0-9]*?)(?P<w>\d+)\.(?P<d>\d+)?$',
        re.IGNORECASE,
    )

    #    char8 niform;      /* NAME OF INPUT FORMAT                   */
    #    short nifl;         /* INFORMAT LENGTH ATTRIBUTE              */
    #    short nifd;        /* INFORMAT NUMBER OF DECIMALS            */
    byte_structure = '>8shh'

    def __init__(self, name='', length=0, decimals=0):
        """
        Initialize an input format.
        """
        self._name = name
        self._length = length
        self._decimals = decimals

    def __str__(self):
        """
        Pleasant display value.
        """
        if not (self.name or self.length or self.decimals):
            return ''
        decimals = self.decimals if self.decimals else ''
        return f'{self.name}{self.length}.{decimals}'

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
        name = self.name.encode('ascii').ljust(8)
        if len(name) > 8:
            raise ValueError('ASCII-encoded {name!r} longer than 8 bytes')
        return struct.pack(fmt, name, self.length, self.decimals)

    @classmethod
    def unpack(cls, bytestring):
        """
        Create an informat from an XPORT-format bytestring.
        """
        fmt = cls.byte_structure
        return cls.from_struct_tokens(*struct.unpack(fmt, bytestring))

    @classmethod
    def from_struct_tokens(cls, name, length, decimals):
        """
        Create an informat from unpacked struct tokens.
        """
        name = name.strip(b'\x00').decode('ascii').strip()
        return cls(name=name, length=length, decimals=decimals)

    @classmethod
    def from_spec(cls, spec, *args, **kwds):
        """
        Create an informat from a text specification.
        """
        mo = cls.pattern.fullmatch(spec)
        if mo is None:
            raise ValueError(f'Invalid informat {spec}')
        name = mo['name'].upper()
        bytestring = name.encode('ascii')
        if len(bytestring) > 8:
            raise ValueError(f'ASCII-encoded name {bytestring} longer than 8 characters')
        length = int(mo.group('w'))
        try:
            decimals = int(mo.group('d'))
        except (TypeError, IndexError):
            decimals = 0
        return cls(*args, name=name, length=length, decimals=decimals, **kwds)

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
        if not isinstance(other, Informat):
            raise TypeError(f"Can't compare {type(self).__name__} with {type(other).__name__}")
        attributes = [
            'name',
            'length',
            'decimals',
        ]
        return all(getattr(self, a) == getattr(other, a) for a in attributes)


class Format(Informat):
    """
    SAS variable format.
    """

    #    char8 nform;       /* NAME OF FORMAT                         */
    #    short nfl;          /* FORMAT FIELD LENGTH OR 0               */
    #    short nfd;         /* FORMAT NUMBER OF DECIMALS              */
    #    short nfj;         /* 0=LEFT JUSTIFICATION, 1=RIGHT JUST     */
    byte_structure = '>8shhh'

    def __init__(self, name='', length=0, decimals=0, justify=FormatAlignment.LEFT):
        """
        Initialize a SAS variable format.
        """
        super().__init__(name, length, decimals)
        self._justify = justify

    def __bytes__(self):
        """
        XPORT-format byte string.
        """
        # TODO: It'd be nice to avoid copy-pasting code from parent.
        fmt = self.byte_structure
        name = self.name.encode('ascii').ljust(8)
        if len(name) > 8:
            raise ValueError('ASCII-encoded {name!r} longer than 8 bytes')
        length = self.length if self.length is not None else 0
        decimals = self.decimals if self.decimals is not None else 0
        return struct.pack(fmt, name, length, decimals, self.justify)

    @classmethod
    def from_struct_tokens(cls, name, length, decimals, justify):
        """
        Create a format from unpacked struct tokens.
        """
        form = super().from_struct_tokens(name, length, decimals)
        form._justify = justify
        return form

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

    _metadata = [
        'label',
        'width',
        'vtype',
        '_format',
        '_informat',
    ]

    def copy_metadata(self, other):
        """
        Copy metadata from another Variable.
        """
        if isinstance(other, Variable):
            for name in self._metadata:
                value = getattr(self, name, None)
                if value is None:
                    value = getattr(other, name, None)
                object.__setattr__(self, name, value)

    def __repr__(self):
        """REPL-format."""
        return f'<{super().__repr__()} label={self.label!r}>'

    def __init__(
        self,
        data=None,
        index=None,
        dtype=None,
        name=None,
        copy=False,
        fastpath=False,
        label=None,
        vtype=None,
        width=None,
        format=None,
        informat=None,
        **kwds,
    ):
        """
        Initialize SAS variable metadata.
        """
        metadata = {
            'label': label,
            'vtype': vtype,
            'width': width,
            'format': format,
            'informat': informat,
        }
        super().__init__(data, index, dtype, name, copy, fastpath, **kwds)
        for name, value in metadata.items():
            if value is not None:
                setattr(self, name, value)
        self.copy_metadata(data)
        for name, value in metadata.items():
            setattr(self, name, getattr(self, name, value))

    def __finalize__(self, other, method=None, **kwds):
        """
        Extend Series finalize to handle more methods.
        """
        self = super().__finalize__(other, method, **kwds)
        if method == 'concat':
            first, *rest = other.objs
            source = first
        else:
            source = other
        self.copy_metadata(source)
        return self

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
    def format(self):
        """
        SAS variable format.
        """
        return self._format

    @format.setter
    def format(self, value):
        if value is None:
            self._format = None
        elif isinstance(value, Format):
            self._format = value
        else:
            self._format = Format.from_spec(value)

        if self.format and self.format.name.startswith('$'):
            self.vtype = VariableType.CHARACTER
        elif self.format and (self.format.name or self.format.decimals):
            self.vtype = VariableType.NUMERIC

    @property
    def informat(self):
        """
        SAS variable informat.
        """
        return self._informat

    @informat.setter
    def informat(self, value):
        if value is None:
            self._informat = None
        elif isinstance(value, Informat):
            self._informat = value
        else:
            self._informat = Informat.from_spec(value)

        if self.informat and self.informat.name.startswith('$'):
            self.vtype = VariableType.CHARACTER
        elif self.informat and (self.informat.name or self.informat.decimals):
            self.vtype = VariableType.NUMERIC


class Dataset(pd.DataFrame):
    """
    SAS data set.

    ``Dataset`` extends Pandas' ``DataFrame``, adding SAS metadata.
    """

    _metadata = [
        'name',
        'label',
        'dataset_type',
        'created',
        'modified',
        'sas_os',
        'sas_version',
        # TODO: Consider including dataset type: {'DATA', 'VIEW', ''}.
    ]

    def copy_metadata(self, other):
        """
        Copy metadata from a Dataset or mapping of Variables.
        """
        if isinstance(other, Dataset):
            for name in self._metadata:
                object.__setattr__(self, name, getattr(other, name, None))
        if isinstance(other, (Dataset, Mapping)):
            for k, v in self.items():
                try:
                    v.copy_metadata(other[k])
                except KeyError:
                    continue

    def __repr__(self):
        """REPL-format."""
        return '\n'.join([
            f'{type(self).__name__} (Name: {self.name})',
            str(self.contents),
            super().__repr__()
        ])

    def __init__(
        self,
        data=None,
        index=None,
        columns=None,
        dtype=None,
        copy=False,
        name=None,
        label=None,
        dataset_type=None,
        created=None,
        modified=None,
        sas_os=None,
        sas_version=None,
        **kwds,
    ):
        """
        Initialize SAS dataset metadata.
        """
        metadata = {
            'name': name,
            'label': label,
            'created': created,
            'modified': modified,
            'sas_os': sas_os,
            'sas_version': sas_version,
            'dataset_type': dataset_type,
        }
        super().__init__(data=data, index=index, columns=columns, dtype=dtype, copy=copy, **kwds)
        for name, value in metadata.items():
            if value is not None:
                setattr(self, name, value)
        self.copy_metadata(data)
        for name, value in metadata.items():
            setattr(self, name, getattr(self, name, value))

    def __finalize__(self, other, method=None, **kwds):
        """
        Propagate metadata to a copy.
        """
        # TODO: Is the call to super redundant?
        self = super().__finalize__(other, method, **kwds)
        if method == 'concat':
            first, *rest = other.objs
            source = first
        elif method == 'merge':
            source = other.left
        else:
            source = other
        self.copy_metadata(source)
        LOG.debug(f'{type(self).__name__}.__finalize__\n{self!r}')
        return self

    def __setitem__(self, key, value):
        """
        When inserting/updating a column, we must copy metadata.
        """
        # TODO: There are probably other ways Pandas adds columns to
        #       a DataFrame.  We need to copy metadata in those, too.
        old = self.iloc[:0].copy()
        super().__setitem__(key, value)
        if isinstance(value, Variable):
            self[key].copy_metadata(value)
        for k, v in old.items():
            if k != key:
                self[k].copy_metadata(v)

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
    def contents(self):
        """
        Variable metadata, such as label, format, number, and position.
        """
        df = pd.DataFrame({
            'Variable': v.name,
            'Type': v.vtype.name.title() if v.vtype is not None else '',
            'Length': v.width,
            'Format': str(v.format) if v.format is not None else '',
            'Informat': str(v.informat) if v.informat is not None else '',
            'Label': v.label if v.label is not None else '',
        } for k, v in self.items())
        if df.empty:
            return df
        df.index = df.index + 1
        df.index.name = '#'
        # BUG: Pandas Series.cumsum() seems to fail on its new Int type;
        #      thus we have a redundant conversion to Int64Dtype.
        df['Position'] = df['Length'].cumsum().astype(pd.Int64Dtype())
        df['Length'] = df['Length'].fillna(pd.NA).astype(pd.Int64Dtype())
        df.loc[1, 'Position'] = 0
        return df

    def infos(self):
        """
        Like ``DataFrame.info`` but returns a string.
        """
        buf = StringIO()
        self.info(buf=buf)
        buf.seek(0)
        return buf.read()


class Library(MutableMapping):
    """
    Collection of datasets from a SAS file.
    """

    def __init__(self, members=(), created=None, modified=None, sas_os='', sas_version=''):
        """
        Initialize a SAS data library.
        """
        self.created = created
        self.modified = modified
        self.sas_os = sas_os
        self.sas_version = sas_version

        self._members = {}
        if isinstance(members, Library):
            self._members = members._members
            self.created = members.created
            self.modified = members.modified
            self.sas_os = members.sas_os
            self.sas_version = members.sas_version
        elif isinstance(members, Mapping):
            for name, dataset in members.items():
                self[name] = dataset  # Use __setitem__ to validate metadata.
        else:
            for dataset in members:
                if dataset.name in self:
                    warnings.warn(f'More than one dataset named {dataset.name!r}')
                self[dataset.name] = dataset

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
        if not isinstance(dataset, Dataset):
            dataset = Dataset(dataset, name=name)
        elif dataset.name is None and name is not None:
            dataset.name = name
            warnings.warn(f'Set dataset name to {name!r}')
        elif name != dataset.name:
            raise ValueError(f'Library member name {name} must match dataset name {dataset.name}')
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


def from_columns(mapping, f):
    """

    """
    # Avoid circular import problems.
    import xport.v56
    warnings.warn('Please use ``xport.v56.dump`` in the future', DeprecationWarning)
    library = xport.v56.Library([xport.v56.Member(mapping)])
    f.write(bytes(library))


def from_rows(iterable, f):
    """

    """
    # Avoid circular import problems.
    import xport.v56
    warnings.warn('Please use ``xport.v56.dump`` in the future', DeprecationWarning)
    df = pd.DataFrame(iterable)
    library = xport.v56.Library([xport.v56.Member(df)])
    f.write(bytes(library))
