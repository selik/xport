#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
Read SAS XPORT/XPT files.

Copyright (c) 2016 Michael Selik.
Inspired by Jack Cushman's original 2012 version.
'''

from __future__ import division, print_function
from collections import namedtuple
from datetime import datetime
from functools import partial
import math
import struct


__version__ = (1, 1, 2)

__all__ = ['Reader', 'DictReader',
           'load', 'loads',
           'dump', 'dumps']



# All "records" are 80 bytes long, padded if necessary.
# Character data are ASCII format.
# Integer data are IBM-style integer format.
# Floating point data are IBM-style double format.



######################################################################
### Reading XPT                                                   ####
######################################################################

class ParseError(ValueError):
    '''
    Bytes did not match expected format
    '''
    def __init__(self, message, expected, got):
        message += ' -- expected {!r}, got {!r}'.format(expected, got)
        super(ParseError, self).__init__(message)
        self.expected = expected
        self.got = got



Variable = namedtuple('Variable', 'name numeric position size')



def parse_date(timestring):
    '''
    Parse date from XPT formatted string (ex. '16FEB11:10:07:55')
    '''
    text = timestring.decode('ascii')
    return datetime.strptime(text, '%d%b%y:%H:%M:%S')



def ibm_to_ieee(ibm):
    '''
    Translate IBM-format floating point numbers (as bytes) to IEEE float.
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



def match(pattern, data, topic=''):
    if data != pattern:
        raise ParseError(message=topic, expected=pattern, got=data)
    return True



class Reader(object):
    '''
    Deserialize ``self._fp`` (a ``.read()``-supporting file-like
    object containing an XPT document) to a Python object.

    The returned object is an iterator.
    Each iteration returns an observation from the XPT file.

        with open('example.xpt', 'rb') as f:
            for row in xport.reader(f):
                process(row)
    '''

    def __init__(self, fp):
        self._fp = fp
        try:
            version, os, created, modified = self._read_header()
            self.version = version
            self.os = os
            self.created = created
            self.modified = modified

            namestr_size = self._read_member_header()[-1]
            nvars = self._read_namestr_header()
            self._variables = self._read_namestr_records(nvars, namestr_size)

            self._read_observations_header()
        except UnicodeDecodeError:
            msg = 'Expected a stream of bytes, got {stream}'
            raise TypeError(msg.format(stream=fp))


    @property
    def fields(self):
        return tuple(v.name for v in self._variables)


    @property
    def _row_size(self):
        return sum(v.length for v in self._variables)


    def __iter__(self):
        for obs in self._read_observations(self._variables):
            yield obs


    def _read_header(self):
        match_ = partial(match, topic='header')

        # --- line 1 -------------
        fmt = '>48s32s'
        raw = self._fp.read(80)
        tokens = tuple(t.rstrip() for t in struct.unpack(fmt, raw))

        prefix, padding = tokens

        try:
            match_(b'HEADER RECORD*******LIBRARY HEADER RECORD!!!!!!!', prefix)
            match_(b'0' * 30, padding)
        except ParseError:
            line = prefix + padding
            if line[:74] == b' '.join([b'**COMPRESSED**'] * 5):
                msg = '{name!r} is a CPORT file, not XPORT'
                raise NotImplementedError(msg.format(name=self._fp.name))
            else:
                raise

        # --- line 2 -------------
        fmt = '>8s8s8s8s8s24s16s'
        raw = self._fp.read(80)
        tokens = tuple(t.rstrip() for t in struct.unpack(fmt, raw))

        prefix = tokens[:3]
        version, os, _, created = tokens[3:]

        match_((b'SAS', b'SAS', b'SASLIB'), prefix)

        version = tuple(int(s) for s in version.split(b'.'))
        created = parse_date(created)

        # --- line 3 -------------
        fmt = '>16s64s'
        raw = self._fp.read(80)
        tokens = tuple(t.rstrip() for t in struct.unpack(fmt, raw))

        modified, _ = tokens

        modified = parse_date(modified)

        # ------------------------
        return version, os, created, modified


    def _read_member_header(self):
        match_ = partial(match, topic='member header')

        # --- line 1 -------------
        fmt = '>48s26s4s2s'
        raw = self._fp.read(80)
        tokens = tuple(t.rstrip() for t in struct.unpack(fmt, raw))

        prefix, _, namestr_size, _ = tokens


        match_(b'HEADER RECORD*******MEMBER  HEADER RECORD!!!!!!!', prefix)

        namestr_size = int(namestr_size)

        # --- line 2 -------------
        fmt = '>48s32s'
        raw = self._fp.read(80)
        tokens = tuple(t.rstrip() for t in struct.unpack(fmt, raw))

        prefix, _ = tokens

        match_(b'HEADER RECORD*******DSCRPTR HEADER RECORD!!!!!!!', prefix)

        # --- line 3 -------------
        fmt = '>8s8s8s8s8s24s16s'
        raw = self._fp.read(80)
        tokens = tuple(t.rstrip() for t in struct.unpack(fmt, raw))

        prefix, dsname, sasdata, version, os, _, created = tokens

        match_(b'SAS', prefix)
        match_(b'SASDATA', sasdata)

        version = tuple(map(int, version.rstrip().split(b'.')))
        created = parse_date(created)

        # --- line 4 -------------
        fmt = '>16s16s40s8s'
        raw = self._fp.read(80)
        tokens = tuple(t.rstrip() for t in struct.unpack(fmt, raw))

        modified, _, dslabel, dstype = tokens

        modified = parse_date(modified)

        # ------------------------
        return (dsname, dstype, dslabel,
                version, os,
                created, modified,
                namestr_size)


    def _read_namestr_header(self):
        match_ = partial(match, topic='member header')

        # --- line 1 -------------
        fmt = '>48s6s4s22s'
        raw = self._fp.read(80)
        tokens = tuple(t.rstrip() for t in struct.unpack(fmt, raw))

        prefix, _, number_of_variables, _ = tokens

        match_(b'HEADER RECORD*******NAMESTR HEADER RECORD!!!!!!!', prefix)

        # ------------------------
        return int(number_of_variables)


    def _read_namestr_record(self, size):
        if size == 140:
            fmt = '>hhhh8s40s8shhh2s8shhl52s'
        else:
            assert size == 136
            fmt = '>hhhh8s40s8shhh2s8shhl48s'
        raw = self._fp.read(size)
        chunks = struct.unpack(fmt, raw)
        tokens = [t.rstrip() if isinstance(t, str) else t for t in chunks]

        is_numeric, _, length, number, name, label = tokens[:6]
        format_data = tokens[6:-2]
        position = tokens[-2]

        name = name.decode('ascii').rstrip()
        # try to make the name a valid nameduple field
        # must be a valid identifier that does not start with underscore
        if name.isnumeric():
            name = 'x' + name

        is_numeric = True if is_numeric == 1 else False

        if is_numeric and (length < 2 or length > 8):
            msg = 'Numerics must be floating points, 2 to 8 bytes long, not %r'
            raise NotImplementedError(msg % length)

        return Variable(name, is_numeric, position, length)


    def _read_namestr_records(self, n, size):
        variables = [self._read_namestr_record(size) for i in range(n)]
        spillover = n * size % 80
        if spillover != 0:
            padding = 80 - spillover
            self._fp.read(padding)
        return variables


    def _read_observations_header(self):
        match_ = partial(match, topic='observations header')

        # --- line 1 -------------
        fmt = '>48s32s'
        raw = self._fp.read(80)
        tokens = tuple(t.rstrip() for t in struct.unpack(fmt, raw))

        prefix, _ = tokens

        match_(b'HEADER RECORD*******OBS     HEADER RECORD!!!!!!!', prefix)


    def _read_observations(self, variables):
        Row = namedtuple('Row', [v.name for v in variables])

        blocksize = sum(v.size for v in variables)
        padding = b' '
        sentinel = padding * blocksize

        count = 0
        while True:
            block = self._fp.read(blocksize)
            if len(block) < blocksize:
                if block and set(block) != set(padding):
                    raise ParseError('incomplete record', sentinel, block)
                remainder = count * blocksize % 80
                if remainder:
                    match(80 - remainder, len(block), 'end-of-file padding')
                break
            elif block == sentinel:
                rest = self._fp.read()
                if rest and set(rest) != set(padding):
                    raise NotImplementedError('Cannot read multiple members.')
                match(80 - (count * blocksize % 80), blocksize + len(rest),
                      'end-of-file padding')
                break

            count += 1
            yield Row._make(self._parse_observation(block, variables))


    def _parse_observation(self, block, variables):
        '''
        Parse values from an XPT-formatted observation/row.
        '''
        for v in variables:
            chunk = block[v.position:v.position + v.size]
            if v.numeric:
                yield ibm_to_ieee(chunk)
            else:
                yield chunk.rstrip().decode('ISO-8859-1')



class DictReader(object):

    def __init__(self, fp):
        self.reader = Reader(fp)

    def __iter__(self):
        return (row._asdict() for row in self.reader)



def to_rows(fp):
    '''
    Read a file in XPT-format and return rows.

    Deserialize ``fp`` (a ``.read()``-supporting file-like object
    containing an XPT document) to a list of rows. As XPT files are
    encoded in their own special format, the ``fp`` object must be in
    bytes-mode. ``Row`` objects will be namedtuples with attributes
    parsed from the XPT metadata.
    '''
    return list(Reader(fp))



def to_columns(fp):
    '''
    Read a file in XPT-format and return columns as a dict of lists.

    Deserialize ``fp`` (a ``.read()``-supporting file-like object
    containing an XPT document) to a list of rows. As XPT files are
    encoded in their own special format, the ``fp`` object must be in
    bytes-mode.
    '''
    reader = Reader(fp)
    return dict(zip(reader.fields, zip(*reader)))



def to_numpy(fp):
    '''
    Read a file in SAS XPT format and return a NumPy array.

    Deserialize ``fp`` (a ``.read()``-supporting file-like object
    containing an XPT document) to a list of rows. As XPT files are
    encoded in their own special format, the ``fp`` object must be in
    bytes-mode.
    '''
    import numpy as np
    return np.vstack(Reader(fp))



def to_dataframe(fp):
    '''
    Read a file in SAS XPT format and return a Pandas DataFrame.

    Deserialize ``fp`` (a ``.read()``-supporting file-like object
    containing an XPT document) to a list of rows. As XPT files are
    encoded in their own special format, the ``fp`` object must be in
    bytes-mode.
    '''
    import pandas as pd
    reader = Reader(fp)
    return pd.DataFrame(iter(reader), columns=reader.fields)



######################################################################
### Writing XPT                                                   ####
######################################################################

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
             + b'                        '
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
    mapping = OrderedDict((label, list(df[label])) for label in dataframe.columns)
    return from_columns(mapping, fp)



######################################################################
### Main                                                          ####
######################################################################

import argparse
import sys



def parse_args(*args, **kwargs):
    if sys.version_info < (3, 0):
        stdin = sys.stdin
    else:
        stdin = sys.stdin.buffer

    parser = argparse.ArgumentParser(description='Read SAS XPORT/XPT files.')
    parser.add_argument('input',
                        type=argparse.FileType('rb'),
                        nargs='?',
                        default=stdin,
                        help='XPORT/XPT file to read, defaults to stdin')
    return parser.parse_args(*args, **kwargs)



if __name__ == '__main__':
    args = parse_args()
    with args.input:
        reader = Reader(args.input)
        print(','.join(reader.fields))
        for row in reader:
            print(','.join(map(str, row)))



