# -*- coding: utf-8 -*-
'''
Write SAS XPORT/XPT files.
'''

from __future__ import division, print_function
from collections import namedtuple
from datetime import datetime
from functools import partial
import math
import struct
import warnings

from .reading import Variable

# Unfortunately, writing XPT files cannot use the same API as the CSV
# module, because XPT requires every row to be the same size in bytes,
# regardless of the data stored in that particular row. This means
# that if there are any text columns, we must find the row with the
# longest possible text string for that column and size the entire
# column for that maximum length.

from collections import OrderedDict
try:
    from collections.abc import Mapping
except ImportError:
    from collections import Mapping

from itertools import tee
try:
    from itertools import zip_longest
except ImportError:
    from itertools import izip_longest as zip_longest

from numbers import Number
import platform
import re



class Overflow(ArithmeticError):
    'Number too large to express'

class Underflow(ArithmeticError):
    'Number too small to express, rounds to zero'



def format_date(dt):
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



def from_columns(mapping, fp):
    '''
    Write columns to the open file opbject ``fp`` in XPT-format.

    The mapping should be of column names to equal-length sequences.

    Column labels are restricted to 40 characters. The XPT format also
    requires a separate column "name" that is restricted to 8
    characters. This name will be automatically created based on the
    column label -- the first 8 characters, non-alphabet characters
    replaced with underscores, padded to 8 characters if necessary.
    All text strings, including column labels, will be converted to
    bytes using the ISO-8859-1 encoding.
    '''
    if not mapping:
        msg = 'must have at least one column, got {!r}'
        raise ValueError(msg.format(mapping))
    if not all(mapping.values()):
        raise ValueError('all columns must have at least one element')

    if fp.tell():
        warnings.warn('not writing to beginning of file', stacklevel=2)

    # make a copy to avoid accidentally mutating the passed-in data
    columns = OrderedDict((k, list(v)) for k, v in mapping.items())


    ### headers ###

    sas_version = b'6.06    ' # the version in the SAS XPT specification I read
    os_version = platform.system().encode('ISO-8859-1')[:8].ljust(8)
    created = format_date(datetime.now())

    # 1st real header record
    fp.write(b'HEADER RECORD*******LIBRARY HEADER RECORD!!!!!!!'
             b'000000000000000000000000000000  '
             b'SAS     '
             b'SAS     '
             b'SASLIB  '
             + sas_version
             + os_version
             + b' ' * 24
             + created)

    # 2nd real header record
    fp.write(created + 64 * b' ')

    # Member header record
    fp.write(b'HEADER RECORD*******MEMBER  HEADER RECORD!!!!!!!'
             b'000000000000000001600000000140  '
             b'HEADER RECORD*******DSCRPTR HEADER RECORD!!!!!!!'
             b'000000000000000000000000000000  ')

    # Member header data
    fp.write(b'SAS     '
             b'dataset ' # dataset name
             b'SASDATA '
             + sas_version
             + os_version
             + 24 * b' '
             + created)

    # 2nd member header record
    fp.write(created
             + 16 * b' '
             + b'dataset'.ljust(40) # datset label
             + b'table'.ljust(8)) # dataset type (What is this?)


    ### field metatdata ###

    fields = OrderedDict()
    position = 0
    for label, column in columns.items():
        label = label.encode('ISO-8859-1')

        # name must be exactly 8 bytes and usually is alphanumeric
        name = b'_'.join(re.findall(b'[A-Za-z0-9_]+', label))[:8].ljust(8)

        numeric = all(isinstance(value, Number) for value in column if value is not None)

        # encode as bytes
        if numeric:
            column[:] = [ieee_to_ibm(value) for value in column]
        else:
            for i, value in enumerate(column):
                if value is None:
                    column[i] = b''
                elif isinstance(value, float) and math.isnan(value):
                    column[i] = b''
                elif isinstance(value, str):
                    column[i] = value.encode('ISO-8859-1')
                else:
                    column[i] = str(value).encode('ISO-8859-1')

        # standardize the size of the values
        size = max(map(len, column))
        if not numeric:
            column[:] = [s.ljust(size) for s in column]

        fields[label] = Variable(name, numeric, position, size)

        # increment position for next field, after recording current position
        position += size

    # Namestr header record
    fp.write(b'HEADER RECORD*******NAMESTR HEADER RECORD!!!!!!!'
             b'000000'
             + str(len(fields)).zfill(4).encode('ISO-8859-1')
             + b'00000000000000000000  ')

    # Namestrs, one for each column
    for i, (label, (name, numeric, position, size)) in enumerate(fields.items()):
        fmt = '>hhhh8s40s8shhh2s8shhl52s'
        data = (1 if numeric else 2, # variable type
                0, # hash of name, always 0
                size,
                i,
                name,
                label[:40].ljust(40),
                b'        ', # name of format
                0, # format field length or 0
                0, # format number of decimals
                0, # 0 for left justified, 1 for right justified
                b'\x00\x00', # two unused bytes
                b'        ', # name of input format
                0, # informat length
                0, # informat number of decimals
                position,
                b'\x00' * 52, # spec says "remaining fields are irrelevant"
                )
        fp.write(struct.pack(fmt, *data))

    # blank padding after the last namestr to ensure 80 bytes per line
    remainder = len(fields) * 140 % 80
    if remainder:
        fp.write(b' ' * (80 - remainder))


    ### data ###

    # Observation header
    fp.write(b'HEADER RECORD*******OBS     HEADER RECORD!!!!!!!'
             b'000000000000000000000000000000  ')

    # Data records
    for row in zip(*columns.values()):
        for cell in row:
            fp.write(cell)

    # blank padding after the last record to ensure 80 bytes per line
    position = fp.tell()
    remainder = position % 80
    if remainder:
        fp.write(b' ' * (80 - remainder))

    fp.flush()



def from_rows(rows, fp):
    '''
    Write rows to the open file object ``fp`` in XPT-format.

    In this case, ``rows`` should be an iterable of iterables, such as
    a list of tuples. If the rows are mappings or namedtuples (or any
    instance of a tuple that has a ``._fields`` attribute), the column
    labels will be inferred from the keys or attributes of the first
    row.

    Column labels are restricted to 40 characters. The XPT format also
    requires a separate column "name" that is restricted to 8
    characters. This name will be automatically created based on the
    column label -- the first 8 characters, non-alphabet characters
    replaced with underscores, padded to 8 characters if necessary.
    All text strings, including column labels, will be converted to
    bytes using the ISO-8859-1 encoding.
    '''
    if not rows:
        msg = 'must have at least one row, got {!r}'
        raise ValueError(msg.format(rows))

    it = iter(rows)
    rows, duplicate = tee(it)
    firstrow = next(duplicate)

    if isinstance(firstrow, Mapping):
        labels = list(firstrow.keys())
        rows = (mapping.values() for mapping in rows)
    elif isinstance(firstrow, tuple) and hasattr(firstrow, '_fields'):
        labels = firstrow._fields
    else:
        labels = ['x%d' % i for i, cell in enumerate(firstrow)]

    columns = OrderedDict(zip(labels, zip_longest(*rows)))
    return from_columns(columns, fp)



def from_dataframe(dataframe, fp):
    '''
    Write a Pandas Dataframe to an open file-like object, ``fp``, in
    XPT-format.
    '''
    mapping = OrderedDict((label, list(dataframe[label])) for label in dataframe.columns)
    return from_columns(mapping, fp)


