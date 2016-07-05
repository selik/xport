'''
Read SAS XPORT/XPT files.

Copyright (c) 2016 Michael Selik.
Inspired by Jack Cushman's original 2012 version.
'''

from __future__ import division, print_function
import argparse
import collections
import contextlib
from datetime import datetime
from functools import partial
import struct
import sys


__all__ = ['reader', 'DictReader']



# All "records" are 80 bytes long, padded if necessary.
# Character data are ASCII format.
# Integer data are IBM-style integer format.
# Floating point data are IBM-style double format.



Variable = collections.namedtuple('Variable', 'name numeric position size')



def parse_date(timestring):
    '''
    Parse date from XPT formatted string (ex. "16FEB11:10:07:55")
    '''
    text = timestring.decode('ascii')
    return datetime.strptime(text, "%d%b%y:%H:%M:%S")



def ibm_to_ieee(ibm):
    '''
    Translate IBM-format floating point numbers (as bytes) to IEEE float.
    '''
    # pad-out to 8 bytes if necessary
    size = len(ibm)
    if size != 8:
        assert 2 <= size <= 8, 'Expected 2 to 8 bytes, not %r' % size
        ibm += b'\x00' * (8 - size)

    # parse the 64 bits of IBM float as one 8-byte unsigned long long
    ulong = struct.unpack('>Q', ibm)[0]
    # drop 1 bit for sign and 7 bits for exponent
    ieee = ulong & 0x00ffffffffffffff

    if ieee == 0:
        if ibm[0:1] == b'\x00':
            return 0.0
        elif ibm[0:1] in '_.ABCDEFGHIJKLMNOPQRSTUVWXYZ':
            return float('nan')
        else:
            raise ValueError('Neither zero nor NaN: %r' % ibm)

    # some junk about fraction bits
    shift = 0
    if ulong & 0x0080000000000000:
        shift = 3
    elif ulong & 0x0040000000000000:
        shift = 2
    elif ulong & 0x0020000000000000:
        shift = 1
    ieee >>= shift

    # clear the 1 bit to the left of the binary point
    ieee &= 0xffefffffffffffff

    # set the sign bit
    sign = ulong & 0x8000000000000000
    ieee |= sign

    # fix the exponent
    exponent = (ulong & 0x7f00000000000000) >> (24 + 32)
    exponent -= 65
    exponent <<= 2
    exponent += shift + 1023
    exponent <<= (20 + 32)
    ieee |= exponent

    return struct.unpack(">d", struct.pack(">Q", ieee))[0]



def _read_header(fp):
    # --- line 1 -------------
    fmt = '>48s32s'
    raw = fp.read(80)
    tokens = tuple(t.rstrip() for t in struct.unpack(fmt, raw))

    prefix, padding = tokens

    if prefix != b'HEADER RECORD*******LIBRARY HEADER RECORD!!!!!!!':
        raise ValueError('Invalid header: %r' % prefix)
    if padding != b'0' * 30:
        raise ValueError('Invalid header: %r' % padding)

    # --- line 2 -------------
    fmt = '>8s8s8s8s8s24s16s'
    raw = fp.read(80)
    tokens = tuple(t.rstrip() for t in struct.unpack(fmt, raw))

    prefix = tokens[:3]
    version, os, _, created = tokens[3:]

    if prefix != (b'SAS', b'SAS', b'SASLIB'):
        raise ValueError('Invalid header: %r' % prefix)

    version = tuple(int(s) for s in version.split(b'.'))
    created = parse_date(created)

    # --- line 3 -------------
    fmt = '>16s64s'
    raw = fp.read(80)
    tokens = tuple(t.rstrip() for t in struct.unpack(fmt, raw))

    modified, _ = tokens

    modified = parse_date(modified)

    # ------------------------
    return version, os, created, modified



def _read_member_header(fp):
    # --- line 1 -------------
    fmt = '>48s26s4s2s'
    raw = fp.read(80)
    tokens = tuple(t.rstrip() for t in struct.unpack(fmt, raw))

    prefix, _, namestr_size, _ = tokens

    if prefix != b'HEADER RECORD*******MEMBER  HEADER RECORD!!!!!!!':
        raise ValueError('Invalid header: %r' % prefix)

    namestr_size = int(namestr_size)

    # --- line 2 -------------
    fmt = '>48s32s'
    raw = fp.read(80)
    tokens = tuple(t.rstrip() for t in struct.unpack(fmt, raw))

    prefix, _ = tokens

    if prefix != b'HEADER RECORD*******DSCRPTR HEADER RECORD!!!!!!!':
        raise ValueError('Invalid header: %r' % prefix)

    # --- line 3 -------------
    fmt = '>8s8s8s8s8s24s16s'
    raw = fp.read(80)
    tokens = tuple(t.rstrip() for t in struct.unpack(fmt, raw))

    prefix, dsname, sasdata, version, os, _, created = tokens
    
    if prefix != b'SAS':
        raise ValueError('Invalid header: %r' % prefix)
    if sasdata != b'SASDATA':
        raise ValueError('Invalid header: %r' % prefix)
    
    version = tuple(map(int, version.rstrip().split(b'.')))
    created = parse_date(created)

    # --- line 4 -------------
    fmt = '>16s16s40s8s'
    raw = fp.read(80)
    tokens = tuple(t.rstrip() for t in struct.unpack(fmt, raw))

    modified, _, dslabel, dstype = tokens

    modified = parse_date(modified)

    # ------------------------
    return (dsname, dstype, dslabel,
            version, os,
            created, modified,
            namestr_size)



def _read_namestr_header(fp):
    # --- line 1 -------------
    fmt = '>48s6s4s22s'
    raw = fp.read(80)
    tokens = tuple(t.rstrip() for t in struct.unpack(fmt, raw))

    prefix, _, number_of_variables, _ = tokens

    if prefix != b'HEADER RECORD*******NAMESTR HEADER RECORD!!!!!!!':
        raise ValueError('Invalid header: %r' % prefix)

    # ------------------------        
    return int(number_of_variables)



def _read_namestr_record(fp, size):
    if size == 140:
        fmt = '>hhhh8s40s8shhh2s8shhl52s'
    else:
        assert size == 136
        fmt = '>hhhh8s40s8shhh2s8shhl48s'
    raw = fp.read(size)
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



def _read_namestr_records(fp, n, size):
    variables = [_read_namestr_record(fp, size) for i in range(n)]
    spillover = n * size % 80
    if spillover != 0:
        padding = 80 - spillover
        fp.read(padding)
    return variables



def _read_observations_header(fp):
    # --- line 1 -------------
    fmt = '>48s32s'
    raw = fp.read(80)
    tokens = tuple(t.rstrip() for t in struct.unpack(fmt, raw))

    prefix, _ = tokens

    if prefix != b'HEADER RECORD*******OBS     HEADER RECORD!!!!!!!':
        raise ValueError('Invalid header: %r' % prefix)



def _parse_field(raw, variable):
    if variable.numeric:
        return ibm_to_ieee(raw)
    return raw.rstrip().decode('ISO-8859-1')



def _read_observations(fp, variables):
    Row = collections.namedtuple('Row', [v.name for v in variables])

    blocksize = sum(v.size for v in variables)
    padding = b' '
    sentinel = padding * blocksize

    # at the end of the file, the last block should be all padding
    # in Python 3, looping over a bytes object gives integers, not bytes
    # therefore, instead of ``all(c = padding for c in block)``
    # we must write ``len(block) == block.count(padding)``

    count = 0
    while True:
        block = fp.read(blocksize)
        if len(block) < blocksize:
            if not len(block) == block.count(padding):
                raise ValueError('Incomplete record, {!r}'.format(block))
            remainder = count * blocksize % 80
            if remainder and len(block) != 80 - remainder:
                raise ValueError('Insufficient padding at end of file')
            break
        elif block == sentinel:
            rest = fp.read()
            if not len(rest) == rest.count(padding):
                raise NotImplementedError('Cannot read multiple members.')
            if blocksize + len(rest) != 80 - (count * blocksize % 80):
                raise ValueError('Incorrect padding at end of file')
            break

        count += 1
        chunks = [block[v.position : v.position + v.size] for v in variables]
        yield Row._make(_parse_field(raw, v) for raw, v in zip(chunks, variables))



class reader(object):
    '''
    Deserialize ``fp`` (a ``.read()``-supporting file-like object containing
    an XPT document) to a Python object.

    The returned object is an iterator.
    Each iteration returns an observation from the XPT file.

        with open('example.xpt', 'rb') as f:
            for row in xport.reader(f):
                process(row)
    '''

    def __init__(self, fp):
        self._fp = fp
        
        version, os, created, modified = _read_header(fp)
        self.version = version
        self.os = os
        self.created = created
        self.modified = modified

        namestr_size = _read_member_header(fp)[-1]
        nvars = _read_namestr_header(fp)

        self._variables = _read_namestr_records(fp, nvars, namestr_size)

        _read_observations_header(fp)


    @property
    def fields(self):
        return tuple(v.name for v in self._variables)


    @property
    def _row_size(self):
        return sum(v.length for v in self._variables)


    def __iter__(self):
        for obs in _read_observations(self._fp, self._variables):
            yield obs



class DictReader(object):

    def __init__(self, fp):
        self.reader = reader(fp)

    def __iter__(self):
        return (row._asdict() for row in self.reader)



def to_numpy(filename):
    '''
    Read a file in SAS XPT format and return a NumPy array.
    '''
    import numpy as np
    with open(filename, 'rb') as f:
        return np.vstack(reader(f))



def to_dataframe(filename):
    '''
    Read a file in SAS XPT format and return a Pandas DataFrame.
    '''
    import pandas as pd
    with open(filename, 'rb') as f:
        xptfile = reader(f)
        return pd.DataFrame(list(xptfile), columns=xptfile.fields)



def parse_args(*args, **kwargs):
    if sys.version_info < (3, 0):
        stdin = sys.stdin
    else:
        stdin = sys.stdin.detach()

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
        xpt = reader(args.input)
        print(','.join(xpt.fields))
        for row in xpt:
            print(','.join(map(str, row)))



