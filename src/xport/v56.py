"""
Read and write the SAS XPORT/XPT file format from SAS Version 5 or 6.

The SAS V5 Transport File format, also called XPORT, or simply XPT, ...
"""

# All "records" are 80 bytes long, padded if necessary.
# Character data are ASCII-encoded.
# Integer data are IBM-style integer format.
# Floating point data are IBM-style double format.

# Standard Library
import json
import logging
import math
import re
import struct
import warnings
from collections.abc import Mapping
from datetime import datetime

# Community Packages
import pandas as pd

__all__ = [
    'load',
    'loads',
    'dump',
    'dumps',
]

LOG = logging.getLogger(__name__)


class Overflow(ArithmeticError):
    """Number too large to express."""


class Underflow(ArithmeticError):
    """Number too small to express, rounds to zero."""


def strptime(timestring):
    """
    Parse a datetime from an XPT format string.

    All text in an XPT document are ASCII-encoded.  This function
    expects a bytes string in the "ddMMMyy:hh:mm:ss" format.  For
    example, ``b'16FEB11:10:07:55'``.  Note that XPT supports only
    2-digit years, which are expected to be either 1900s or 2000s.
    """
    text = timestring.decode('ascii')
    return datetime.strptime(text, '%d%b%y:%H:%M:%S')


def ibm_to_ieee(ibm: bytes) -> float:
    """
    Convert IBM-format floating point (bytes) to IEEE 754 64-bit (float).
    """
    # IBM mainframe:    sign * 0.mantissa * 16 ** (exponent - 64)
    # Python uses IEEE: sign * 1.mantissa * 2 ** (exponent - 1023)

    # Pad-out to 8 bytes if necessary. We expect 2 to 8 bytes, but
    # there's no need to check; bizarre sizes will cause a struct
    # module unpack error.
    ibm = ibm.ljust(8, b'\x00')

    # parse the 64 bits of IBM float as one 8-byte unsigned long long
    ulong, = struct.unpack('>Q', ibm)

    # IBM: 1-bit sign, 7-bits exponent, 56-bits mantissa
    sign = ulong & 0x8000000000000000
    exponent = (ulong & 0x7f00000000000000) >> 56
    mantissa = ulong & 0x00ffffffffffffff

    if mantissa == 0:
        if ibm[0:1] == b'\x00':
            return 0.0
        elif ibm[0:1] == b'\x80':
            return -0.0
        elif ibm[0:1] in b'_.ABCDEFGHIJKLMNOPQRSTUVWXYZ':
            return float('nan')
        else:
            raise ValueError('Neither zero nor NaN: %r' % ibm)

    # IBM-format exponent is base 16, so the mantissa can have up to 3
    # leading zero-bits in the binary mantissa. IEEE format exponent
    # is base 2, so we don't need any leading zero-bits and will shift
    # accordingly. This is one of the criticisms of IBM-format, its
    # wobbling precision.
    if ulong & 0x0080000000000000:
        shift = 3
    elif ulong & 0x0040000000000000:
        shift = 2
    elif ulong & 0x0020000000000000:
        shift = 1
    else:
        shift = 0
    mantissa >>= shift

    # clear the 1 bit to the left of the binary point
    # this is implicit in IEEE specification
    mantissa &= 0xffefffffffffffff

    # IBM exponent is excess 64, but we subtract 65, because of the
    # implicit 1 left of the radix point for the IEEE mantissa
    exponent -= 65
    # IBM exponent is base 16, IEEE is base 2, so we multiply by 4
    exponent <<= 2
    # IEEE exponent is excess 1023, but we also increment for each
    # right-shift when aligning the mantissa's first 1-bit
    exponent += shift + 1023

    # IEEE: 1-bit sign, 11-bits exponent, 52-bits mantissa
    # We didn't shift the sign bit, so it's already in the right spot
    ieee = sign | (exponent << 52) | mantissa
    return struct.unpack(">d", struct.pack(">Q", ieee))[0]


def ieee_to_ibm(ieee):
    """
    Convert Python floating point numbers to IBM-format (bytes).
    """
    # Python uses IEEE: sign * 1.mantissa * 2 ** (exponent - 1023)
    # IBM mainframe:    sign * 0.mantissa * 16 ** (exponent - 64)

    if ieee == 0.0:
        return b'\x00' * 8
    if ieee is None or math.isnan(ieee):
        return b'_' + b'\x00' * 7
    if math.isinf(ieee):
        raise NotImplementedError('Cannot convert infinity')

    bits = struct.pack('>d', ieee)
    ulong, = struct.unpack('>Q', bits)

    sign = (ulong & (1 << 63)) >> 63  # 1-bit     sign
    exponent = ((ulong & (0x7ff << 52)) >> 52) - 1023  # 11-bits   exponent
    mantissa = ulong & 0x000fffffffffffff  # 52-bits   mantissa/significand

    if exponent > 248:
        msg = 'Cannot store magnitude more than ~ 16 ** 63 as IBM-format'
        raise Overflow(msg)
    if exponent < -260:
        msg = 'Cannot store magnitude less than ~ 16 ** -65 as IBM-format'
        raise Underflow(msg)

    # IEEE mantissa has an implicit 1 left of the radix:    1.significand
    # IBM mantissa has an implicit 0 left of the radix:     0.significand
    # We must bitwise-or the implicit 1.mmm into the mantissa
    # later we will increment the exponent to account for this change
    mantissa = 0x0010000000000000 | mantissa

    # IEEE exponents are for base 2:    mantissa * 2 ** exponent
    # IBM exponents are for base 16:    mantissa * 16 ** exponent
    # We must divide the exponent by 4, since 16 ** x == 2 ** (4 * x)
    quotient, remainder = divmod(exponent, 4)
    exponent = quotient

    # We don't want to lose information;
    # the remainder from the divided exponent adjusts the mantissa
    mantissa <<= remainder

    # Increment exponent, because of earlier adjustment to mantissa
    # this corresponds to the 1.mantissa vs 0.mantissa implicit bit
    exponent += 1

    # IBM exponents are excess 64
    exponent += 64

    # IBM has 1-bit sign, 7-bits exponent, and 56-bits mantissa.
    # We must shift the sign and exponent into their places.
    sign <<= 63
    exponent <<= 56

    # We lose some precision, but who said floats were perfect?
    return struct.pack('>Q', sign | exponent | mantissa)


class Namestr(dict):
    """
    Variable metadata from a SAS XPORT file member.
    """

    # Here is the C structure definition for the namestr record:
    #
    # struct NAMESTR {
    #    short ntype;       /* VARIABLE TYPE: 1=NUMERIC, 2=CHAR       */
    #    short nhfun;       /* HASH OF NNAME (always 0)               */
    #    short nlng;        /* LENGTH OF VARIABLE IN OBSERVATION      */
    #    short nvar0;       /* VARNUM                                 */
    #    char8 nname;       /* NAME OF VARIABLE                       */
    #    char40 nlabel;     /* LABEL OF VARIABLE                      */
    #    char8 nform;       /* NAME OF FORMAT                         */
    #    short nfl;         /* FORMAT FIELD LENGTH OR 0               */
    #    short nfd;         /* FORMAT NUMBER OF DECIMALS              */
    #    short nfj;         /* 0=LEFT JUSTIFICATION, 1=RIGHT JUST     */
    #    char nfill[2];     /* (UNUSED, FOR ALIGNMENT AND FUTURE)     */
    #    char8 niform;      /* NAME OF INPUT FORMAT                   */
    #    short nifl;        /* INFORMAT LENGTH ATTRIBUTE              */
    #    short nifd;        /* INFORMAT NUMBER OF DECIMALS            */
    #    long npos;         /* POSITION OF VALUE IN OBSERVATION       */
    #    char rest[52];     /* remaining fields are irrelevant        */
    #    };
    #
    # Note that the length given in the last 4 bytes of the member
    # header record indicates the actual number of bytes for the NAMESTR
    # structure. The size of the structure listed above is 140 bytes.
    # Under VAX/VMS, the size will be 136 bytes, meaning that the 'rest'
    # variable may be truncated.

    fmts = {140: '>hhhh8s40s8shhh2s8shhl52s', 136: '>hhhh8s40s8shhh2s8shhl48s'}

    @classmethod
    def unpack(cls, bytestring):
        """
        Parse a namestr from a byte string.
        """
        LOG.debug(f'Unpacking namestr from {bytestring}')
        size = len(bytestring)
        if size == 136:
            warnings.warn('File written on VAX/VMS, module behavior not tested')
        fmt = cls.fmts[size]
        tokens = struct.unpack(fmt, bytestring)
        info = {
            'name': tokens[4].strip(b'\x00').decode('ascii').rstrip(),
            'label': tokens[5].strip(b'\x00').decode('ascii').rstrip(),
            'type': {
                1: 'Numeric',
                2: 'Character'
            }[tokens[0]],
            'number': tokens[3],
            'position': tokens[14],
            'length': tokens[2],
            'format': {
                'name': tokens[6].strip(b'\x00').decode('ascii').strip(),
                'length': tokens[7],
                'decimals': tokens[8],
                'justify': {
                    0: 'left',
                    1: 'right'
                }[tokens[9]],
            },
            'iformat': {
                'name': tokens[11].strip(b'\x00').decode('ascii').strip(),
                'length': tokens[12],
                'decimals': tokens[13],
            },
        }
        LOG.debug(f'Unpacked namestr {json.dumps(info, indent=2)}')
        return Namestr(**info)

    def __bytes__(self):
        """
        XPORT-format bytes string.
        """
        fmt = self.fmts[140]
        return struct.pack(
            fmt,
            1 if self.get('type') == 'Numeric' else 2,
            0,  # "Hash" of name, always 0.
            self['length'],
            self['number'],
            self['name'].encode('ascii')[:8],
            self.get('label', '').encode('ascii')[:40],
            b'',  # TODO: format name
            0,  # TODO: format field length
            0,  # TODO: format n decimals
            0,  # TODO: 0 if self.format.justify == 'left' else 1
            b'',  # Unused
            b'',  # TODO: iformat name
            0,  # TODO: iformat length
            0,  # TODO: iformat n decimals
            self['position'],
            b'',  # Padding
        )


class Observation:
    """
    A record from a SAS XPORT library member.
    """

    # TODO: Change these class methods to Member instance methods.
    #       Then delete this class.

    # 9. Data records
    #    Data records are streamed in the same way that namestrs are.
    #    There is ASCII blank padding at the end of the last record if
    #    necessary. There is no special trailing record.

    decoders = {
        'Numeric': ibm_to_ieee,
        'Character': lambda s: s.strip(b'\x00').decode('ascii').rstrip()
    }

    @classmethod
    def parser(cls, namestrs):
        """
        Make an observation parser for a collection of namestrs.
        """
        names = [v['name'] for v in namestrs]
        sizes = [v['length'] for v in namestrs]
        types = [v['type'] for v in namestrs]
        parsers = [cls.decoders[t] for t in types]
        fmt = ''.join(f'{x}s' for x in sizes)
        stride = sum(sizes)

        def finditer(bytestring):
            """
            Parse observations from a byte string.
            """
            if stride == 0:
                return
            sentinel = b' ' * stride
            for i in range(0, len(bytestring), stride):
                chunk = bytestring[i:i + stride]
                LOG.debug(f'Parsing observation from {chunk}')
                if len(chunk) != stride or chunk == sentinel:
                    LOG.debug(f'End padding {chunk}')
                    break
                tokens = struct.unpack(fmt, chunk)
                obs = {n: f(b) for n, f, b in zip(names, parsers, tokens)}
                LOG.debug(f'Parsed observation {json.dumps(obs, indent=2)}')
                yield obs

        return finditer

    @classmethod
    def formatter(cls, namestrs):
        """
        Make an observation formatter for a collection of namestrs.
        """
        # names = [v['name'] for v in namestrs]
        sizes = [v['length'] for v in namestrs]
        fmt = ''.join(f'{x}s' for x in sizes)

        def bytes_(t):
            """
            Convert an observation (tuple) to a bytes string.
            """
            return struct.pack(fmt, *t)

        return bytes_


class Member(Mapping):
    """
    A SAS dataset as a collection of columns.
    """

    # 4. Member header records
    #    Both of these records occur for every member in the file.
    #
    #    HEADER RECORD*******MEMBER HEADER RECORD!!!!!!!
    #    000000000000000001600000000140
    #    HEADER RECORD*******DSCRPTR HEADER RECORD!!!!!!!
    #    000000000000000000000000000000
    #
    #    Note the 0140 that appears in the member header record above.
    #    This value specifies the size of the variable descriptor
    #    (NAMESTR) record that is described later in this document. On
    #    the VAX/VMS operating system, the value will be 0136 instead of
    #    0140. This means that the descriptor will be only 136 bytes
    #    instead of 140.
    #
    # 5. Member header data ... as C structure:
    #
    #       struct REAL_HEADER {
    #          char sas_symbol[8];      /* "SAS"                    */
    #          char sas_dsname[8];      /* dataset name             */
    #          char sasdata[8];         /* "SASDATA"                */
    #          char sasver[8];          /* version of SAS used      */
    #          char sas_osname[8];      /* operating system used    */
    #          char blanks[24];
    #          char sas_create[16];     /* datetime created         */
    #          };
    #
    #    The second header record as C structure:
    #
    #       struct SECOND_HEADER {
    #          char dtmod[16];            /* date modified           */
    #          char padding[16];
    #          char dslabel[40];          /* dataset label          */
    #          char dstype[8]             /* dataset type           */
    #          };
    #
    # 6. Namestr header record
    #    One for each member
    #
    #       HEADER RECORD*******NAMESTR HEADER RECORD!!!!!!!
    #       000000xxxx0000000000000 0000000
    #
    #    In this header record, xxxx is the number of variables in the
    #    data set, displayed with blank-padded numeric characters. For
    #    example, for 2 variables, xxxx=0002.
    #
    # 7. Namestr records
    #    Each namestr field is 140 bytes long, but the fields are
    #    streamed together and broken in 80-byte pieces. If the last
    #    byte of the last namestr field does not fall in the last byte
    #    of the 80-byte record, the record is padded with ASCII blanks
    #    to 80 bytes.
    #
    # 8. Observation header
    #
    #       HEADER RECORD*******OBS HEADER RECORD!!!!!!!
    #       000000000000000000000000 000000

    template = f'''\
HEADER RECORD{'*' * 7}MEMBER  HEADER RECORD{'!' * 7}{'0' * 17}16{'0' * 8}140  \
HEADER RECORD{'*' * 7}DSCRPTR HEADER RECORD{'!' * 7}{'0' * 30}  \
SAS     %(name)8bSASDATA %(version)8b%(os)8b{' ' * 24}%(created)16b\
%(modified)16b{' ' * 16}%(label)40b%(type)8b\
HEADER RECORD{'*' * 7}NAMESTR HEADER RECORD{'!' * 7}{'0' * 6}\
%(n_variables)04d{'0' * 20}  \
%(namestrs)b\
HEADER RECORD{'*' * 7}OBS     HEADER RECORD{'!' * 7}{'0' * 30}  \
%(observations)b\
'''.encode('ascii')

    pattern = re.compile(
        # Header line 1
        rb'HEADER RECORD\*{7}MEMBER  HEADER RECORD\!{7}0{17}'
        rb'160{8}(?P<descriptor_size>140|136)  '
        # Header line 2
        rb'HEADER RECORD\*{7}DSCRPTR HEADER RECORD\!{7}0{30} {2}'
        # Header line 3
        rb'SAS {5}(?P<name>.{8})SASDATA '
        rb'(?P<version>.{8})(?P<os>.{8}) {24}(?P<created>.{16})'
        # Header line 4
        rb'(?P<modified>.{16}) {16}'
        rb'(?P<label>.{40})(?P<type>    DATA|    VIEW| {8})'
        # Namestrs
        rb'HEADER RECORD\*{7}NAMESTR HEADER RECORD\!{7}0{6}'
        rb'(?P<n_variables>.{4})0{20} {2}'
        rb'(?P<namestrs>.*?)'
        rb'HEADER RECORD\*{7}OBS {5}HEADER RECORD\!{7}0{30} {2}',
        # Observations ... until EOF or another Member.
        re.DOTALL,
    )

    def __init__(
        self,
        observations=None,
        namestrs=None,
        name=None,
        label=None,
        type=None,
        created=None,
        modified=None,
        os=None,
        sas_version=None
    ):
        """
        Initialize a SAS data library member.
        """
        if observations is None:
            observations = pd.DataFrame()
        if namestrs is None:
            namestrs = {}
            for variable in observations:
                namestrs[variable] = Namestr(name=variable, )
        for i, (variable, namestr) in enumerate(namestrs.items(), 1):
            column = observations[variable]
            namestr.setdefault('name', variable)
            namestr.setdefault('label', '')
            if column.dtype == 'object':
                namestr.setdefault('type', 'Character')
                namestr.setdefault('length', column.str.len().max())
            else:
                namestr.setdefault('type', 'Numeric')
                namestr.setdefault('length', 8)
            namestr.setdefault('number', i)
        i = 0
        for namestr in sorted(namestrs.values(), key=lambda n: n['number']):
            namestr.setdefault('position', i)
            i += namestr['length']
        if name is None:
            name = ''
        if label is None:
            label = name
        if type is None:
            type = 'DATA'
        if created is None:
            created = datetime.now()
        if modified is None:
            modified = created
        if os is None:
            os = ''
        if sas_version is None:
            sas_version = ''

        if type == 'VIEW':
            raise NotImplementedError('SAS data views are not supported')

        self.name = name
        self.label = label
        self.type = type
        self.created = created
        self.modified = modified
        self.os = os
        self.sas_version = sas_version
        self.namestrs = namestrs
        self.observations = observations

    @classmethod
    def finditer(cls, bytestring, Namestr=Namestr):
        """
        Parse SAS library members from a byte string in XPORT format.
        """
        LOG.debug('Searching for library members ...')
        matches = cls.pattern.finditer(bytestring)
        try:
            mo = next(matches)
        except StopIteration:
            warnings.warn('No library members found')
            LOG.debug(f'Byte string begins with\n{bytestring[:80 * 6]}')
            return
        while True:
            observations_index = mo.end(0)
            namestrs = []
            stride = int(mo['descriptor_size'])
            for i in range(0, len(mo['namestrs']), stride):
                b = mo['namestrs'][i:i + stride]
                if len(b) == stride:
                    namestrs.append(Namestr.unpack(b))
            if len(namestrs) != int(mo['n_variables']):
                raise ValueError(f'Expected {mo["n_variables"]}, got {len(namestrs)}')
            obs_finditer = Observation.parser(namestrs)
            kwds = dict(
                name=mo['name'].strip(b'\x00').decode('ascii').strip(),
                label=mo['label'].strip(b'\x00').decode('ascii').strip(),
                type=mo['type'].strip(b'\x00').decode('ascii').strip(),
                os=mo['os'].strip(b'\x00').decode('ascii').strip(),
                sas_version=tuple(int(s) for s in mo['version'].strip().split(b'.') if s),
                created=strptime(mo['created']),
                modified=strptime(mo['modified']),
                namestrs={n['name']: n
                          for n in namestrs},
            )
            LOG.debug(f'Found library member {kwds["name"]!r}')
            try:
                mo = next(matches)
            except StopIteration:
                break
            chunk = bytestring[observations_index:mo.start(0)]
            df = pd.DataFrame(obs_finditer(chunk), columns=list(kwds['namestrs']))
            yield Member(observations=df, **kwds)
        # Last chunk goes until EOF
        chunk = bytestring[observations_index:]
        df = pd.DataFrame(obs_finditer(chunk), columns=list(kwds['namestrs']))
        yield Member(observations=df, **kwds)

    def __repr__(self):
        """
        REPL-format string.
        """
        fmt = '<{cls} columns={keys}>'
        return fmt.format(
            cls=type(self).__name__,
            keys=list(self),
        )

    def __bytes__(self):
        """
        XPORT-format bytes string.
        """
        bytes_ = Observation.formatter(self.namestrs.values())
        df = self.observations.copy()
        for k, dtype in df.dtypes.iteritems():
            if dtype == 'object':
                df[k] = df[k].str.encode('ascii')
            else:
                df[k] = df[k].map(ieee_to_ibm)
        observations = b''.join(bytes_(t) for t in df.itertuples(index=False, name=None))
        if len(observations) % 80:
            observations += b' ' * (80 - len(observations) % 80)
        namestrs = b''.join(bytes(n) for n in self.namestrs.values())
        if len(namestrs) % 80:
            namestrs += b' ' * (80 - len(namestrs) % 80)
        return self.template % {
            b'name': self.name.encode('ascii')[:8],
            b'label': self.label.encode('ascii')[:8],
            b'type': self.type.encode('ascii')[:8],
            b'n_variables': len(self.namestrs),
            b'os': self.os.encode('ascii')[:8],
            b'version': '.'.join(self.sas_version).encode('ascii')[:8],
            b'created': self.created.strftime('%y%b%d:%H:%M:%S').upper().encode('ascii'),
            b'modified': self.modified.strftime('%y%b%d:%H:%M:%S').upper().encode('ascii'),
            b'namestrs': namestrs,
            b'observations': observations,
        }

    def __getitem__(self, name):
        """
        Get a column by name.
        """
        return self.observations[name]

    def __iter__(self):
        """
        Get iterator of column names.
        """
        return iter(self.observations)

    def __len__(self):
        """
        Get the number of columns in the library member.
        """
        return len(self.observations)

    def __eq__(self, other):
        """
        Compare equality.
        """
        if self.observations.empty:
            return (other.observations.empty and set(self) == set(other))
        return (self.observations == other.observations).all(axis=None)


class Library(Mapping):
    """
    A collection of datasets from a SAS file.
    """

    # 1. The first header record:
    #
    #   HEADER RECORD*******LIBRARY HEADER RECORD!!!!!!!
    #   000000000000000000000000000000
    #
    # 2. The first real header record ... as a C structure:
    #
    #   struct REAL_HEADER {
    #      char sas_symbol[2][8];       /* "SAS", twice             */
    #      char saslib[8];              /* "SASLIB"                 */
    #      char sasver[8];              /* version of SAS used      */
    #      char sas_os[8];              /* operating system used    */
    #      char blanks[24];
    #      char sas_create[16];         /* datetime created         */
    #      };
    #
    # 3. Second real header record
    #
    #       ddMMMyy:hh:mm:ss
    #
    #    In this record, the string is the datetime modified. Most
    #    often, the datetime created and datetime modified will always
    #    be the same. Pad with ASCII blanks to 80 bytes. Note that only
    #    a 2-digit year appears. If any program needs to read in this
    #    2-digit year, be prepared to deal with dates in the 1900s or
    #    the 2000s.

    template = f'''\
HEADER RECORD{'*' * 7}LIBRARY HEADER RECORD{'!' * 7}{'0' * 30}  \
SAS     SAS     SASLIB  \
%(version)8b%(os)8b{' ' * 24}%(created)16b\
%(modified)16b{' ' * 64}\
%(members)b\
'''.encode('ascii')

    pattern = re.compile(
        rb'HEADER RECORD\*{7}LIBRARY HEADER RECORD\!{7}0{30} {2}'
        rb'SAS {5}SAS {5}SASLIB {2}'
        rb'(?P<version>.{8})(?P<os>.{8}) {24}(?P<created>.{16})'
        rb'(?P<modified>.{16}) {64}'
        rb'(?P<members>.*)',
        re.DOTALL,
    )

    def __init__(self, members=None, created=None, modified=None, os=None, sas_version=None):
        """
        Initialize a SAS data library.
        """
        if members is None:
            members = {}
        if created is None:
            created = datetime.now()
        if modified is None:
            modified = created
        if sas_version is None:
            sas_version = ''
        if os is None:
            os = ''
        self.created = created
        self.modified = modified
        self.os = os
        self.sas_version = sas_version
        self.members = members
        for name, member in members.items():
            if not member.name:
                member.name = name

    @classmethod
    def match(cls, bytestring, Member=Member):
        """
        Parse a SAS XPORT document from a byte string.
        """
        mo = cls.pattern.match(bytestring)
        if mo is None:
            LOG.error('Failed to match byte string with header\n%s', bytestring[:80 * 4])
            raise ValueError('Document does not match SAS Version 5 or 6 Transport (XPORT) format')
        return Library(
            members={m.name: m
                     for m in Member.finditer(mo['members'])},
            os=mo['os'].strip(b'\x00').decode('ascii').strip(),
            sas_version=tuple(int(s) for s in mo['version'].strip().split(b'.') if s),
            created=strptime(mo['created']),
            modified=strptime(mo['modified']),
        )

    def __repr__(self):
        """
        REPL-format string.
        """
        fmt = '<{cls} members={members}>'
        return fmt.format(
            cls=type(self).__name__,
            members={k: repr(v)
                     for k, v in self.items()},
        )

    def __bytes__(self):
        """
        XPORT-format bytes string.
        """
        return self.template % {
            b'version': '.'.join(self.sas_version).encode('ascii')[:8],
            b'os': self.os.encode('ascii')[:8],
            b'created': self.created.strftime('%y%b%d:%H:%M:%S').upper().encode('ascii'),
            b'modified': self.modified.strftime('%y%b%d:%H:%M:%S').upper().encode('ascii'),
            b'members': b''.join(bytes(member) for member in self.values()),
        }

    def __getitem__(self, name):
        """
        Get a member by name.
        """
        return self.members[name]

    def __iter__(self):
        """
        Get iterator of member names.
        """
        return iter(self.members)

    def __len__(self):
        """
        Get the number of members in the library.
        """
        return len(self.members)

    def __eq__(self, other):
        """
        Compare equality.
        """
        return (set(self) == set(other) and all(self[k] == other[k] for k in self))


def load(fp):
    """
    Deserialize a SAS V5 transport file format document::

        with open('example.xpt', 'rb') as f:
            data = load(f)
    """
    return loads(fp.read())


def loads(s):
    """
    Deserialize a SAS V5 transport file format document from a string::

        with open('example.xpt', 'rb') as f:
            bytestring = f.read()
        data = loads(bytestring)
    """
    return Library.match(s)


def dump(library, fp):
    """
    Serialize a SAS V5 transport file format document::

        with open('example.xpt', 'wb') as f:
            dump(library, f)
    """
    fp.write(dumps(library))


def dumps(library):
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
    return bytes(library)
