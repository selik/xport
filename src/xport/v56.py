"""
Read and write the SAS XPORT/XPT file format from SAS Version 5 or 6.

The SAS V5 Transport File format, also called XPORT, or simply XPT, ...
"""

# All "records" are 80 bytes long, padded if necessary.
# Character data are ASCII-encoded.
# Integer data are IBM-style integer format.
# Floating point data are IBM-style double format.

# Standard Library
import itertools
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

# Xport Modules
import xport

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


def strftime(dt):
    """
    Convert a datetime to an XPT format byte string.
    """
    return dt.strftime('%d%b%y:%H:%M:%S').upper().encode('ascii')


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


class Namestr(xport.Variable):
    """
    Variable metadata from a SAS Version 5 or 6 Transport (XPORT) file.
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

    fmts = {
        140: '>hhhh8s40s8shhh2s8shhl52s',
        136: '>hhhh8s40s8shhh2s8shhl48s',
    }

    @classmethod
    def unpack(cls, bytestring):
        """
        Parse a namestr from a byte string.
        """
        # LOG.debug(f'Unpacking namestr from {bytestring}')
        size = len(bytestring)
        if size == 136:
            warnings.warn('File written on VAX/VMS, module behavior not tested')
        fmt = cls.fmts[size]
        tokens = struct.unpack(fmt, bytestring)
        vtype = xport.VariableType(tokens[0])
        v = cls(
            name=tokens[4].strip(b'\x00').decode('ascii').rstrip(),
            dtype='float' if vtype == xport.VariableType.NUMERIC else 'string',
        )
        v.sas_label = tokens[5].strip(b'\x00').decode('ascii').rstrip()
        v._sas_format = xport.Format.from_struct_tokens(*tokens[6:10])
        v._sas_iformat = xport.Informat.from_struct_tokens(*tokens[11:14])
        v.sas_variable_number = tokens[3]
        v.sas_variable_position = tokens[14]
        v.sas_variable_length = tokens[2]
        LOG.debug(f'Parsed namestr {v}')
        return v

    def __bytes__(self):
        """
        XPORT-format bytes string.
        """
        fmt = self.fmts[140]
        name = self.sas_name.encode('ascii')
        if len(name) > 8:
            raise ValueError('Name {name} longer than 8 characters')
        label = self.sas_label.encode('ascii')
        if len(label) > 40:
            raise ValueError(f'Label {label} longer than 40 characters')
        return struct.pack(
            fmt,
            self.sas_variable_type,
            0,  # "Hash" of name, always 0.
            self.sas_variable_length,
            self.sas_variable_number,
            name.ljust(8),
            label.ljust(40),
            self.sas_format.name.encode('ascii').ljust(8),
            self.sas_format.length,
            self.sas_format.decimals,
            self.sas_format.justify,
            b'',  # Unused
            self.sas_iformat.name.encode('ascii').ljust(8),
            self.sas_iformat.length,
            self.sas_iformat.decimals,
            self.sas_variable_position,
            b'',  # Padding
        )


class Member(xport.Dataset):
    """
    SAS library member from a SAS Version 5 or 6 Transport file.
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

    # 9. Data records
    #    Data records are streamed in the same way that namestrs are.
    #    There is ASCII blank padding at the end of the last record if
    #    necessary. There is no special trailing record.

    decoders = {
        xport.VariableType.NUMERIC: ibm_to_ieee,
        xport.VariableType.CHARACTER: lambda s: s.strip(b'\x00').decode('ascii').rstrip()
    }

    @classmethod
    def finditer(cls, bytestring):
        """
        Parse SAS library members from a byte string.
        """
        LOG.debug('Searching for library members ...')
        matches = list(cls.pattern.finditer(bytestring))

        if not matches:
            warnings.warn('No library members found')
            lines = [bytestring[i * 80:(i + 1) * 80] for i in range(6)]
            LOG.debug(f'Byte string begins with' + '\n%s' * len(lines), *lines)
            return

        headers = []
        for mo in matches:
            variables = []
            LOG.info(f'Found library member {mo["name"]!r}')
            stride = int(mo['descriptor_size'])
            for i in range(0, len(mo['namestrs']), stride):
                b = mo['namestrs'][i:i + stride]
                if len(b) == stride:
                    variables.append(Namestr.unpack(b))
            if len(variables) != int(mo['n_variables']):
                raise ValueError(f'Expected {mo["n_variables"]}, got {len(variables)}')
            data = {v.sas_name: v for v in sorted(variables, key=lambda v: v.sas_variable_number)}
            h = cls(data)
            h.sas_name = mo['name'].strip(b'\x00').decode('ascii').strip()
            h.sas_label = mo['label'].strip(b'\x00').decode('ascii').strip()
            h.sas_dataset_type = mo['type'].strip(b'\x00').decode('ascii').strip()
            h.sas_os = mo['os'].strip(b'\x00').decode('ascii').strip()
            h.sas_version = mo['version'].strip().decode('ascii')
            h.sas_dataset_created = strptime(mo['created'])
            h.sas_dataset_modified = strptime(mo['modified'])
            LOG.debug(f'Parsed member header {h.sas_name} with {len(variables)} variables')
            headers.append(h)

        mview = memoryview(bytestring)
        ends = [mo.end(0) for mo in matches]
        starts = [mo.start(0) for mo in matches]
        chunks = (mview[i:j] for i, j in zip(ends, starts[1:] + [None]))
        for h, chunk in zip(headers, chunks):
            yield h

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

    def encoder(self):
        """
        Make an observation encoder.
        """
        # Try to avoid keeping a reference to a Variable in the closure.
        variables = [v for k, v in self.items()]
        sizes = [v.sas_variable_length for v in variables]
        vtypes = [v.sas_variable_type for v in variables]
        fmt = ''.join(f'{x}s' for x in sizes)
        converters = []
        for vtype, x in zip(vtypes, sizes):
            if vtype == xport.VariableType.NUMERIC:
                converters.append(ieee_to_ibm)
            else:
                converters.append(lambda s: s.encode('ascii').ljust(x))

        def bytes_(t):
            """Convert an observation (tuple) to a byte string."""
            g = (f(v) for f, v in zip(converters, t))
            return struct.pack(fmt, *g)

        return bytes_

    def __bytes__(self):
        """
        XPORT-format bytes string.
        """
        self.update_variable_number_and_position()

        namestrs = b''.join(bytes(xport.v56.Namestr(v)) for k, v in self.items())
        if len(namestrs) % 80:
            namestrs += b' ' * (80 - len(namestrs) % 80)

        observations = self.itertuples(index=False, name=None)
        observations = b''.join(self.encoder()(t) for t in observations)
        if len(observations) % 80:
            observations += b' ' * (80 - len(observations) % 80)

        return self.template % {
            b'name': self.sas_name.encode('ascii')[:8].ljust(8),
            b'label': self.sas_label.encode('ascii')[:40].ljust(40),
            b'type': self.sas_dataset_type.encode('ascii')[:8].ljust(8),
            b'n_variables': len(self.columns),
            b'os': self.sas_os.encode('ascii')[:8].ljust(8),
            b'version': self.sas_version.encode('ascii')[:8].ljust(8),
            b'created': strftime(self.sas_dataset_created),
            b'modified': strftime(self.sas_dataset_modified),
            b'namestrs': namestrs,
            b'observations': observations,
        }

        # df = self.data
        # variables = [df[k].sas for k in df]
        # names = [v.name for v in variables]
        # parsers = [self.decoders[v.type] for v in variables]
        # sizes = [v.length for v in variables]
        # fmt = ''.join(f'{x}s' for x in sizes)
        # stride = sum(sizes)
        # LOG.info(f'Observation struct fmt {fmt}')

        # if stride == 0:
        #     return
        # sentinel = b' ' * stride
        # for i in range(0, len(mview), stride):
        #     chunk = mview[i:i + stride]
        #     LOG.debug(f'Parsing observation from {bytes(chunk)}')
        #     if len(chunk) != stride or chunk == sentinel:
        #         LOG.debug(f'End padding {chunk}')
        #         break
        #     tokens = struct.unpack(fmt, chunk)
        #     obs = {n: f(b) for n, f, b in zip(names, parsers, tokens)}
        #     LOG.debug(f'Parsed observation {json.dumps(obs, indent=2)}')
        #     yield obs

        # for h, chunk in zip(headers, chunks):
        #     rows = pd.DataFrame(h.parse_observations(chunk))
        #     h.data = pd.concat([h.data, rows])
        #     # BUG: Argh! Every time we change columns, we lose the series accessors.
        #     h.data.sas = h
        #     LOG.info('Parsed dataframe for %s\n%s', h.name, h.info())
        #     yield h

    # def __bytes__(self):
    #     """
    #     XPORT-format bytes string.
    #     """
    #     bytes_ = Observation.formatter(self.namestrs.values())
    #     df = self.observations.copy()
    #     for k, dtype in df.dtypes.iteritems():
    #         if dtype == 'object':
    #             df[k] = df[k].str.encode('ascii')
    #         else:
    #             df[k] = df[k].map(ieee_to_ibm)
    #     observations = b''.join(bytes_(t) for t in df.itertuples(index=False, name=None))
    #     if len(observations) % 80:
    #         observations += b' ' * (80 - len(observations) % 80)


class Library(xport.Library):
    """
    Collection of datasets from a SAS Version 5 or 6 Transport file.
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

    pattern = re.compile(
        rb'HEADER RECORD\*{7}LIBRARY HEADER RECORD\!{7}0{30} {2}'
        rb'SAS {5}SAS {5}SASLIB {2}'
        rb'(?P<version>.{8})(?P<os>.{8}) {24}(?P<created>.{16})'
        rb'(?P<modified>.{16}) {64}'
        rb'(?P<members>.*)',
        re.DOTALL,
    )

    @classmethod
    def match(cls, bytestring):
        """
        Parse a SAS XPORT document from a byte string.
        """
        mview = memoryview(bytestring)
        lines = [mview[i * 80:(i + 1) * 80] for i in range(4)]
        LOG.debug(f'Document begins with' + '\n%s' * len(lines), *lines)
        mo = cls.pattern.match(mview)
        if mo is None:
            raise ValueError('Document does not match SAS Version 5 or 6 Transport (XPORT) format')
        members = {header.name: header.data for header in MemberHeader.finditer(mo['members'])}
        return Library(
            members=members,
            created=strptime(mo['created']),
            modified=strptime(mo['modified']),
            os=mo['os'].strip(b'\x00').decode('ascii').strip(),
            version=mo['version'].strip(b'\x00').decode('ascii').strip(),
        )


#     template = f'''\
# HEADER RECORD{'*' * 7}LIBRARY HEADER RECORD{'!' * 7}{'0' * 30}  \
# SAS     SAS     SASLIB  \
# %(version)8b%(os)8b{' ' * 24}%(created)16b\
# %(modified)16b{' ' * 64}\
# %(members)b\
# '''.encode('ascii')

#     def __bytes__(self):
#         """
#         XPORT-format bytes string.
#         """
#         return self.template % {
#             b'version': '.'.join(self.sas_version).encode('ascii')[:8],
#             b'os': self.os.encode('ascii')[:8],
#             b'created': self.created.strftime('%y%b%d:%H:%M:%S').upper().encode('ascii'),
#             b'modified': self.modified.strftime('%y%b%d:%H:%M:%S').upper().encode('ascii'),
#             b'members': b''.join(bytes(member) for member in self.values()),
#         }


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
