# -*- coding: utf-8 -*-
'''
Read SAS XPORT/XPT files.
'''

from __future__ import division, print_function
from collections import namedtuple
from datetime import datetime
from functools import partial
import math
import struct
import warnings

# All "records" are 80 bytes long, padded if necessary.
# Character data are ASCII format.
# Integer data are IBM-style integer format.
# Floating point data are IBM-style double format.


class ParseError(ValueError):
    '''
    Bytes did not match expected format
    '''
    def __init__(self, message, expected, got):
        message += f' -- {expected!r}, {got!r}'
        super().__init__(message)
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
                print(row)
    '''

    def __init__(self, fp):
        if fp.tell():
            warnings.warn('not starting from beginning of file', stacklevel=2)

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
        blocksize = sum(v.size for v in variables)
        padding = b' '
        sentinel = padding * blocksize

        count = 0
        while True:
            block = self._fp.read(blocksize)
            if len(block) < blocksize:
                if not set(block) <= set(padding):
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
            yield tuple(self._parse_observation(block, variables))


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


class DictReader(Reader):

    def __iter__(self):
        names = self.fields
        for row in super(DictReader, self).__iter__():
            yield dict(zip(names, row))



class NamedTupleReader(Reader):

    def __iter__(self):
        Row = namedtuple('Row', self.fields)
        for row in super(NamedTupleReader, self).__iter__():
            yield Row._make(row)


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


