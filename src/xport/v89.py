"""
Read and write the SAS XPORT/XPT file format from SAS Version 8 or 9.

The SAS V8 Transport File format, also called XPORT, or simply XPT, ...
"""

# Member names may now be up to 32 characters.
# Variable names may now be up to 32 characters.
# Variable labels may now be up to 256 characters.
# Character fields may now exceed 200 characters.
# Format fields may now exceed 8 characters (v9).

# Standard Library
import dataclasses
import logging
import re
import struct

# Xport Modules
import xport.v56

LOG = logging.getLogger(__name__)


class Library(xport.v56.Library):
    """
    Collection of datasets from a SAS Version 8 or 9 Transport file.
    """

    template = xport.v56.Library.template.replace(b'LIBRARY', b'LIBV8  ')
    pattern = re.compile(
        xport.v56.Library.pattern.pattern.replace(b'LIBRARY', b'LIBV8  '),
        re.DOTALL,
    )

    @classmethod
    def from_bytes(cls, bytestring):
        """
        Construct a ``Library`` from an XPORT-format bytes string.
        """
        return super().from_bytes(bytestring, MemberHeader=MemberHeader, Member=Member)


class MemberHeader(xport.v56.MemberHeader):
    """
    Dataset metadata from a SAS Version 8 or 9 Transport (XPORT) file.
    """

    pattern = re.compile(
        rb'HEADER RECORD\*{7}MEMBV8  HEADER RECORD\!{7}0{17}160{8}(?P<descriptor_size>140)  '
        rb'HEADER RECORD\*{7}DSCPTV8 HEADER RECORD\!{7}0{30}  '
        rb'SAS {5}(?P<name>.{32})SASDATA (?P<version>.{8})(?P<os>.{8})(?P<created>.{16})'
        rb'(?P<modified>.{16}) {16}(?P<label>.{40})(?P<type>    DATA|    VIEW| {8})'
        rb'HEADER RECORD\*{7}NAMSTV8 HEADER RECORD\!{7}0{6}(?P<n_variables>.{4})0{20}  '
        rb'(?P<namestrs>.*?)'
        rb'HEADER RECORD\*{7}LABELV(?P<v>8|9) HEADER RECORD\!{7}(?P<n_labels>.{5}) {27}'
        rb'(?P<labels>.*?)'
        rb'HEADER RECORD\*{7}OBSV8   HEADER RECORD\!{7}0{30}  ', re.DOTALL
    )

    @classmethod
    def from_bytes(cls, bytestring):
        """
        Construct a ``MemberHeader`` from an XPORT-format byte string.

        The Transport Version 8 format allows long variable names and labels.
        The Transport Version 9 format allows long variable format and informat
        descriptions.
        """
        self = super().from_bytes(bytestring)
        match = cls.pattern.search(bytestring)
        v9 = match['v'] == b'9'
        n = int(match['n_labels'].strip())
        data = match['labels']
        namestrs = {n.number: n for n in self.namestrs.values()}
        for _ in range(n):
            number, name_length, label_length = struct.unpack('>hhh', data[:6])
            i = (10 if v9 else 6) + name_length
            j = i + label_length
            namestrs[number].name = data[6:i].decode('ISO-8859-1')
            namestrs[number].label = data[i:j].decode('ISO-8859-1')
            if v9:
                format_length, informat_length = struct.unpack('>hh', data[6:10])
            data = data[j:]
            if v9:
                i = format_length
                j = i + informat_length
                namestrs[number].format = xport.Format.from_spec(data[:i].decode('ISO-8859-1'))
                namestrs[number].informat = xport.Informat.from_spec(
                    data[i:j].decode('ISO-8859-1')
                )
                data = data[j:]
        if set(data) != {ord(b' ')}:
            raise ValueError(f'Expected only padding, got {data}')
        return self


class Member(xport.v56.Member):
    """
    A dataset; a member of a dataset library.
    """

    @classmethod
    def from_bytes(cls, bytestring):
        """
        Parse a ``Member`` from an XPORT-format bytes string.
        """
        return super().from_bytes(bytestring, MemberHeader=MemberHeader)


@dataclasses.dataclass
class Namestr(xport.v56.Namestr):
    """
    Variable metadata from a SAS Version 8 or 9 Transport (XPORT) file.
    """

    vtype: xport.VariableType
    length: int
    number: int
    name: str
    label: str
    format: xport.Format
    informat: xport.Informat
    position: int
    longname: str
    label_length: int

    # The C structure definition for namestr records in the v5 format
    # ends with 52 unused bytes.  The v8 format replaces that with:
    #
    #   char longname[32]   /* long name for Version 8-style */
    #   short lablen        /* length of label */
    #   char rest[18]       /* remaining fields are irrelevant */

    fmts = {
        140: '>hhhh8s40s8shhh2s8shhl32sh18s',
        # v5/6 had a 136-length option, but that is not supported in v8/9.
    }

    @classmethod
    def from_bytes(cls, bytestring: bytes):
        """
        Construct a ``Namestr`` from an XPORT-format byte string.
        """
        v56 = super().from_bytes(bytestring)
        fmt = cls.fmts[len(bytestring)]
        tokens = struct.unpack(fmt, bytestring)
        return cls(
            vtype=v56.vtype,
            length=v56.length,
            number=v56.number,
            name=v56.name,
            label=v56.label,
            format=v56.format,
            informat=v56.informat,
            position=v56.position,
            longname=tokens[-3],
            label_length=tokens[-2],
        )


def load(fp):
    """
    Deserialize a dataset library from a SAS Transport v8 (XPT) file.

        >>> with open('test/data/example.v8xpt', 'rb') as file:
        ...     library = load(file)
    """
    try:
        bytestring = fp.read()
    except UnicodeDecodeError:
        raise TypeError(f'Expected a BufferedReader in bytes-mode, got {type(fp).__name__}')
    return loads(bytestring)


def loads(bytestring):
    """
    Deserialize a dataset library from an XPORT-format string.

        >>> with open('test/data/example.v8xpt', 'rb') as file:
        ...     bytestring = file.read()
        >>> library = loads(bytestring)
    """
    return Library.from_bytes(bytestring)


def dump(library, fp):
    """
    Serialize a dataset library to a SAS Transport v8/9 (XPORT) file.

        >>> library = Library()
        >>> with open('test/data/doctest.v8xpt', 'wb') as file:
        ...     dump(library, file)

    The input ``library`` can be either an ``xport.Library``, an
    ``xport.Dataset`` collection, or a single ``pandas.DataFrame``.

        >>> ds = xport.Dataset(name='EMPTY')
        >>> with open('test/data/doctest.v8xpt', 'wb') as file:
        ...     dump(ds, file)
    """
    fp.write(dumps(library))


def dumps(library):
    """
    Serialize a dataset library to a string in XPORT format.

        >>> library = Library()
        >>> bytestring = dumps(library)
    """
    return bytes(Library(library))
