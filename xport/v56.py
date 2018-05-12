#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
Read SAS XPORT/XPT files from SAS Version 5 or 6.
'''

from collections import namedtuple
from collections.abc import Mapping, Sequence
from datetime import datetime
from io import BytesIO
from itertools import accumulate
import os, re, struct



########################################################################
## Reading XPT                                                        ##
########################################################################

# All "records" are 80 bytes long, padded if necessary.
# Character data are ASCII format.
# Integer data are IBM-style integer format.
# Floating point data are IBM-style double format.



def strptime(timestring):
    '''
    Parse date from XPT formatted string (ex. '16FEB11:10:07:55')
    '''
    text = timestring.decode('ascii')
    return datetime.strptime(text, '%d%b%y:%H:%M:%S')



def ibm_to_ieee(ibm):
    '''
    Translate IBM-format floating point numbers (as bytes) to IEEE 754
    64-bit floating point format (as Python float).
    '''
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



class Library(Mapping):
    '''

    Parse metadata -- name, type, date created, etc. -- for all
    members of ``fp`` (a file-like object with ``.read()`` and
    ``.seek()`` methods, containing a SAS Version 5 or 6 XPORT document)

    '''

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

    LINE = 80

    header_re = re.compile(
        rb'HEADER RECORD\*{7}LIBRARY HEADER RECORD\!{7}0{30}  '
        rb'SAS     SAS     SASLIB  '
        rb'(?P<version>.{8})(?P<os>.{8}) {24}(?P<created>.{16})'
        rb'(?P<modified>.{16}) {64}'
    )

    @staticmethod
    def match(header):
        '''
        Parse the 3-line (240-byte) header of a SAS XPORT file.
        '''
        mo = Library.header_re.match(header)
        if mo is None:
            raise ValueError(f'Not a SAS Version 5 or 6 XPORT file')
        return {
            'created': strptime(mo['created']),
            'modified': strptime(mo['modified']),
            'sas_version': float(mo['version']),
            'os_version': mo['os'].decode().strip(),
        }


    def __init__(self, fp):
        '''
        Find the names and locations of each dataset member in the file.
        '''
        self.fp = fp
        try:
            header = fp.read(3 * Library.LINE)
        except UnicodeDecodeError:
            raise TypeError(f'Expected bytes reader, got {type(fp).__name__}.'
                            ' Opened files must be in "rb" mode.')
        info = self.match(header)
        self.name = os.path.basename(fp.name)
        self.created = info['created']
        self.modified = info['modified']
        self.sas_version = info['sas_version']
        self.os_version = info['os_version']

        fp.seek(0)
        lines = iter(lambda: fp.read(Library.LINE), b'')
        matches = map(Member.line1_match, lines)
        positions = [i * Library.LINE for i, mo in enumerate(matches) if mo]
        eof = fp.tell()

        headers = []
        for i in positions:
            fp.seek(i)
            headers.append(fp.read(4 * Library.LINE))
        self.members = [Member.header_match(s)['name'] for s in headers]

        starts = positions
        stops = positions[1:] + [eof]
        self.spans = {k: (i, j) for k, i, j in zip(self.members, starts, stops)}


    def __repr__(self):
        '''
        '''
        cls = type(self).__name__
        return f'<{cls} name={self.name!r} members={self.members!r}>'


    def __getitem__(self, key):
        '''
        Get a member from the library.
        '''
        return Member(self, name=key)


    def __iter__(self):
        '''
        Iterator of member names.
        '''
        return iter(self.members)


    def __len__(self):
        '''
        Number of members in the library.
        '''
        return len(self.members)



class Member(Sequence):
    '''
    '''

    # 4. Member header records
    #    Both of these records occur for every member in the file.
    #
    #   HEADER RECORD*******MEMBER HEADER RECORD!!!!!!!
    #   000000000000000001600000000140
    #   HEADER RECORD*******DSCRPTR HEADER RECORD!!!!!!!
    #   000000000000000000000000000000
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
    #   The second header record as C structure:
    #
    #       struct SECOND_HEADER {
    #          char dtmod[16];            /* date modified          */
    #          char padding[16];
    #          char dslabel[40];          /* dataset label          */
    #          char dstype[8]             /* dataset type           */
    #          };

    line1_re = re.compile(
        rb'HEADER RECORD\*{7}'
        rb'MEMBER  HEADER RECORD\!{7}0{17}'
        rb'160{8}(?P<descriptor_size>140|136)  '
    )

    header_re = re.compile(
        # line 1
        rb'HEADER RECORD\*{7}MEMBER  HEADER RECORD\!{7}0{17}'
        rb'160{8}(?P<descriptor_size>140|136)  '
        # line 2
        rb'HEADER RECORD\*{7}DSCRPTR HEADER RECORD\!{7}0{30}  '
        # line 3
        rb'SAS     (?P<name>.{8})SASDATA '
        rb'(?P<version>.{8})(?P<os>.{8}) {24}(?P<created>.{16})'
        # line 4
        rb'(?P<modified>.{16}) {16}'
        rb'(?P<label>.{40})(?P<type>.{8})'
    )

    @classmethod
    def line1_match(cls, line):
        '''
        Parse the 1st line (80 bytes) of a library member header.
        '''
        return cls.line1_re.match(line)

    @classmethod
    def header_match(cls, header):
        '''
        Parse the 4-line (320-byte) library member header.
        '''
        mo = cls.header_re.match(header)
        if mo is None:
            msg = f'Expected {cls.header_re.pattern!r}, got {header!r}'
            raise ValueError(msg)
        return {
            'name': mo['name'].decode().strip(),
            'label': mo['label'].decode().strip(),
            'type': mo['type'].decode().strip(),
            'created': strptime(mo['created']),
            'modified': strptime(mo['modified']),
            'sas_version': float(mo['version']),
            'os_version': mo['os'].decode().strip(),
            'namestr_size': mo['descriptor_size'],
        }

    def __init__(self, library, name=None, index=0):
        '''

        Read member metadata -- label, variables, created date, etc. --
        specified by either name or index within the library.

        '''
        self.library = library
        if name is None:
            name = library.members[index]
        
        start, stop = library.spans[name]
        library.fp.seek(start)
        info = self.header_match(library.fp.read(4 * library.LINE))
        self.name = info['name']
        self.label = info['label']
        self.type = info['type']
        self.created = info['created']
        self.modified = info['modified']
        self.sas_version = info['sas_version']
        self.os_version = info['os_version']

        NAMESTR = int(info['namestr_size'])
        self.variables = {v.name: v for v in Variable.readall(self, NAMESTR)}
        self._observations = Observations(self)


    def __getitem__(self, index):
        '''
        '''
        return self._observations[index]


    def __len__(self):
        '''

        '''
        return len(self._observations)


    def __iter__(self):
        '''

        '''
        return iter(self._observations)


    def __repr__(self):
        '''
        '''
        cls = type(self).__name__
        return f'<{cls} name={self.name!r} variables={list(self.variables)}>'



class Variable(namedtuple('Variable',
    'name label type number position length format iformat')):
    '''
    Variable metadata from a SAS XPORT file member.
    '''

    # 6. Namestr header record
    #    One for each member
    #
    #   HEADER RECORD*******NAMESTR HEADER RECORD!!!!!!!
    #   000000xxxx0000000000000 0000000
    #
    #    In this header record, xxxx is the number of variables in the
    #    data set, displayed with blank-padded numeric characters. For
    #    example, for 2 variables, xxxx=0002.

    header_re = re.compile(
        rb'HEADER RECORD\*{7}NAMESTR HEADER RECORD\!{7}0{6}'
        rb'(?P<n_variables>.{4})0{20}'
    )

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
        136: '>hhhh8s40s8shhh2s8shhl48s'
    }


    @classmethod
    def header_match(cls, data):
        '''
        Parse a member namestrs header (1 line, 80 bytes).
        '''
        mo = cls.header_re.match(data)
        return int(mo['n_variables'])


    @classmethod
    def unpack(cls, data):
        '''

        Parse namestrs -- variable metadata -- for one member of an XPORT
        file, given the metadata of that member.

        '''
        size = len(data)
        if size == 136:
            warnings.warn('File written on VAX/VMS, module behavior unknown')
        fmt = cls.fmts[size]
        tokens = struct.unpack(fmt, data)
        return {
            'name': tokens[4].decode().rstrip(),
            'label': tokens[5].decode().rstrip(),
            'type': {1: 'numeric', 2: 'text'}[tokens[0]],
            'number': tokens[3],
            'position': tokens[14],
            'length': tokens[2],
            'format': {
                'name': tokens[6].decode().strip(),
                'length': tokens[7],
                'decimals': tokens[8],
                'justify': {0: 'left', 1: 'right'}[tokens[9]],
            },
            'iformat': {
                'name': tokens[11].decode().strip(),
                'length': tokens[12],
                'decimals': tokens[13],
            },
        }


    @classmethod
    def readall(cls, member, size):
        '''
        Parse variable metadata for a XPORT file member.
        '''
        fp = member.library.fp
        LINE = member.library.LINE

        n = cls.header_match(fp.read(LINE))
        namestrs = [fp.read(size) for i in range(n)]

        # Each namestr field is 140 bytes long, but the fields are
        # streamed together and broken in 80-byte pieces. If the last
        # byte of the last namestr field does not fall in the last byte
        # of the 80-byte record, the record is padded with ASCII blanks
        # to 80 bytes.

        remainder = n * size % LINE
        if remainder:
            padding = 80 - remainder
            fp.read(padding)

        info = [cls.unpack(s) for s in namestrs]
        for d in info:
            d['format'] = Format(**d['format'])
            d['iformat'] = InputFormat(**d['iformat'])
        return [Variable(**d) for d in info]



Format = namedtuple('Format', 'name length decimals justify')
InputFormat = namedtuple('InputFormat', 'name length decimals')



class Observations(Sequence):
    '''
    '''

    # 8. Observation header
    #
    #   HEADER RECORD*******OBS     HEADER RECORD!!!!!!!
    #   000000000000000000000000000000

    header_re = re.compile(
        rb'HEADER RECORD\*{7}OBS     HEADER RECORD\!{7}0{10}'
    )

    parsers = {
        'numeric': ibm_to_ieee,
        'text': lambda s: s.decode('ISO-8859-1').rstrip()
    }

    def __init__(self, member):
        '''
        '''
        self.fp = member.library.fp
        LINE = member.library.LINE

        line = self.fp.read(LINE)
        mo = self.header_re.match(line)
        if mo is None:
            msg = f'Expected {self.header_re.pattern!r}, got {line!r}.'
            raise RuntimeError(msg)

        names = list(member.variables)
        sizes = [v.length for v in member.variables.values()]
        types = [v.type for v in member.variables.values()]
        self.fmt = ''.join(f'{x}s' for x in sizes)
        self.parsers = [self.parsers[t] for t in types]

        start, stop = member.library.spans[member.name]
        start = self.fp.tell()
        step = sum(sizes)
        self.range = range(start, stop, step)
        self.sentinel = b' ' * step


    def __getitem__(self, index):
        '''
        Get an observation from the data set member.
        '''
        i = self.range[index]
        self.fp.seek(i)
        chunk = self.fp.read(self.range.step)
        if chunk == self.sentinel:
            raise IndexError(index)
        tokens = struct.unpack(self.fmt, chunk)
        return tuple(f(s) for f, s in zip(self.parsers, tokens))


    def __len__(self):
        '''
        Number of observations in the data set member.
        '''
        return len(self.range)


    def __iter__(self):
        '''
        Iterator of observations in the data set member.
        '''
        self.fp.seek(self.range.start)
        for i in self.range:
            chunk = self.fp.read(self.range.step)
            if chunk == self.sentinel:
                break
            tokens = struct.unpack(self.fmt, chunk)
            yield tuple(f(s) for f, s in zip(self.parsers, tokens))



########################################################################
## Writing XPT                                                        ##
########################################################################

# Unfortunately, writing XPT files cannot use the same API as the CSV
# module, because XPT requires every row to be the same size in bytes,
# regardless of the data stored in that particular row. This means
# that if there are any text columns, we must find the row with the
# longest possible text string for that column and size the entire
# column for that maximum length.


# Names cannot exceed 8 characters
# Labels cannot exceed 40 characters
# Character fields may not exceed 200 characters.



def strftime(dt):
    '''
    Format XPT datetime string (bytes) from Python datetime object
    '''
    return dt.strftime('%d%b%y:%H:%M:%S').upper().encode('ISO-8859-1')


def ieee_to_ibm(ieee):
    '''
    Translate Python floating point numbers to IBM-format (as bytes).
    '''
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

    sign = (ulong & (1 << 63)) >> 63                    # 1-bit     sign
    exponent = ((ulong & (0x7ff << 52)) >> 52) - 1023   # 11-bits   exponent
    mantissa = ulong & 0x000fffffffffffff               # 52-bits   mantissa/significand

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


def dump(columns, fp):
    '''
    Serialize ``columns`` as a XPORT formatted bytes stream to ``fp`` (a
    ``.write()``-supporting file-like object).
    '''
    # return writing.from_columns(columns, fp)
    raise NotImplementedError('not yet')


def dumps(columns):
    '''
    Serialize ``columns`` to a JSON formatted ``bytes`` object.
    '''
    fp = BytesIO()
    dump(columns, fp)
    fp.seek(0)
    return fp.read()
