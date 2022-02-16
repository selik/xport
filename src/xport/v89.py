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
import logging
import re
import struct
import warnings
from datetime import datetime

# Xport Modules
import xport.v56
from xport.v56 import _encoding, strftime, text_encode

__all__ = [
    'load',
    'loads',
    'dump',
    'dumps',
    '_encoding',
]

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

    def _bytes(self):
        return super()._bytes(Member=Member)


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
        rb'(HEADER RECORD\*{7}LABELV(?P<v>8|9) HEADER RECORD\!{7}(?P<n_labels>.{5}) {27}'
        rb'(?P<labels>.*?))?'
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
        self = super().from_bytes(bytestring, Namestr=Namestr)
        match = cls.pattern.search(bytestring)
        v9 = match['v'] == b'9'
        n = int((match['n_labels'] or '0').strip())
        data = match['labels']
        namestrs = {n.number: n for n in self.namestrs.values()}
        for _ in range(n):
            number, name_length, label_length = struct.unpack('>hhh', data[:6])
            i = (10 if v9 else 6) + name_length
            j = i + label_length
            namestrs[number].name = data[6:i].decode(xport.v56.TEXT_METADATA_ENCODING)
            namestrs[number].label = data[i:j].decode(xport.v56.TEXT_METADATA_ENCODING)
            if v9:
                format_length, informat_length = struct.unpack('>hh', data[6:10])
            data = data[j:]
            if v9:
                i = format_length
                j = i + informat_length
                namestrs[number].format = xport.Format.from_spec(data[:i].decode('ascii'))
                namestrs[number].informat = xport.Informat.from_spec(data[i:j].decode('ascii'))
                data = data[j:]
        if data and set(data) != {ord(b' ')}:
            raise ValueError(f'Expected only padding, got {data}')
        return self

    @classmethod
    def from_dataset(cls, dataset: xport.Dataset):
        """
        Construct a ``MemberHeader`` from an ``xport.Dataset``.
        """
        # The example "v8xpt" file I generated with Stata enumerated variables
        # starting from 0, which differs from the v5 xpt files I've seen.  This
        # may be a flaw in Stata's implementation that I've copied here.
        return super().from_dataset(dataset, Namestr=Namestr, variable_enumeration_start=0)

    template = f'''\
HEADER RECORD{'*' * 7}MEMBV8  HEADER RECORD{'!' * 7}{'0' * 17}16{'0' * 8}140  \
HEADER RECORD{'*' * 7}DSCPTV8 HEADER RECORD{'!' * 7}{'0' * 30}  \
SAS     %(name)32bSASDATA %(version)8b%(os)8b%(created)16b\
%(modified)16b{' ' * 16}%(label)40b%(type)8b\
HEADER RECORD{'*' * 7}NAMSTV8 HEADER RECORD{'!' * 7}{'0' * 6}\
%(n_variables)04d{'0' * 20}  \
%(namestrs)b\
%(labels)b\
HEADER RECORD{'*' * 7}OBSV8   HEADER RECORD{'!' * 7}{'0' * 30}  \
'''.encode('ascii')

    def __bytes__(self):
        """
        Encode in XPORT format.
        """
        namestrs = b''.join(map(bytes, self.values()))
        if len(namestrs) % 80:
            namestrs += b' ' * (80 - len(namestrs) % 80)

        # TODO: Handle long format and informat names for Transport v9.
        labels = b''
        n_labels = 0
        triggers = {'label': 40}  # , 'format': 8, 'informat': 8}
        for namestr in self.values():
            strings = {
                'name': namestr.name,
                'label': namestr.label if namestr.label is not None else '',
            }
            strings = {k: v.encode(xport.v56.TEXT_METADATA_ENCODING) for k, v in strings.items()}
            if any(len(strings[k]) > l for k, l in triggers.items()):
                strings = list(strings.values())
                fmt = '>hhh' + ''.join(f'{len(s)}s' for s in strings)
                labels += struct.pack(fmt, namestr.number, *map(len, strings), *strings)
                n_labels += 1
        if labels:
            labels = f'''\
HEADER RECORD{'*' * 7}LABELV8 HEADER RECORD{'!' * 7}%(n_labels)5s{' ' * 27}\
'''.encode('ascii') % {
                b'n_labels': str(n_labels or '').ljust(5).encode(xport.v56.TEXT_METADATA_ENCODING),
            } + labels
        if len(labels) % 80:
            labels += b' ' * (80 - len(labels) % 80)
        return self.template % {
            b'name': text_encode(self, 'name', 32),
            b'label': text_encode(self, 'dataset_label', 40),
            b'type': text_encode(self, 'dataset_type', 8),
            b'n_variables': len(self),
            b'os': text_encode(self, 'sas_os', 8),
            b'version': text_encode(self, 'sas_version', 8),
            b'created': strftime(self.created if self.created else datetime.now()),
            b'modified': strftime(self.modified if self.modified else datetime.now()),
            b'namestrs': namestrs,
            b'labels': labels,
        }


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

    def _bytes(self):
        return super()._bytes(MemberHeader=MemberHeader)


class Namestr(xport.v56.Namestr):
    """
    Variable metadata from a SAS Version 8 or 9 Transport (XPORT) file.
    """

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
        longname = tokens[-3].strip(b'\x00').decode(xport.v56.TEXT_METADATA_ENCODING
                                                    ).rstrip() or v56.name
        if v56.name not in longname:
            warnings.warn(f'Short name {v56.name} not included in {longname}')
        # TODO: What should I do with the label length?  ``tokens[-2]``
        return cls(
            vtype=v56.vtype,
            length=v56.length,
            number=v56.number,
            name=longname,
            label=v56.label,
            format=v56.format,
            informat=v56.informat,
            position=v56.position,
        )

    def __bytes__(self):
        """
        Encode in XPORT format.
        """
        LOG.debug(f'Encode {type(self).__name__}')
        fmt = self.fmts[140]

        longname = text_encode(self, 'name', 32)
        shortname = longname[:8]

        longlabel = (self.label if self.label is not None else '').encode(
            xport.v56.TEXT_METADATA_ENCODING
        )
        if len(longlabel) > 256:
            raise ValueError('ASCII-encoded label {longlabel} exceeds 256 characters')
        shortlabel = longlabel[:40].ljust(40)

        # TODO: What's the right way to handle long format names for v9?
        format_name = self.format.name.encode('ascii')[:8]
        informat_name = self.informat.name.encode('ascii')[:8]

        if self.number is None:
            raise ValueError('Variable number not assigned')
        if self.position is None:
            raise ValueError('Variable position not assigned')

        return struct.pack(
            fmt,
            self.vtype,
            0,  # "Hash" of name, always 0.
            self.length,
            self.number,
            shortname,
            shortlabel,
            format_name.ljust(8),
            self.format.length,
            self.format.decimals,
            self.format.justify,
            b'',  # Unused
            informat_name.ljust(8),
            self.informat.length,
            self.informat.decimals,
            self.position,
            longname,
            len(longlabel),
            b'',  # Padding
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
