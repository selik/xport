from __future__ import print_function
import logging
import argparse
import sys
import struct
from collections import namedtuple
from datetime import datetime
from itertools import repeat, takewhile, imap
import csv


class XptError(Exception):
    '''Base error for this module'''
    pass


class XptNotMember(XptError):
    '''Tried to read something that is not an XPT member'''
    pass


class XportReader(object):
    def __init__(self, f):
        '''
        Yields observations as dictionaries.
        Params: f, an opened XPT file
        '''
        self.file = f
        self.header, self.members = read_xpt(f)

    def __iter__(self):
        return self.iterobs()

    def iterobs(self):
        for m in self.members:
            for o in m.obs:
                yield o

    def __enter__(self):
        return self

    def __exit__(self, type, value, traceback):
        try:
            self.file.close()
        except:
            logging.warning('Failed to close file: {0}'.format(self.file))


def read_xpt(f):
    header = read_header(f)

    # TODO: Fix detection of next member in file

    is_member = lambda x: x is not None
    members = takewhile(is_member, imap(read_member, repeat(f)))
    
    XPT = namedtuple('XPT', 'header members')
    return XPT(header, members)


def read_header(f):
    num_lines = 3
    line_size = 80
    info = f.read(num_lines * line_size)

    fmt = '>48s32s 8s8s8s8s8s24s16s 16s64s'
    line1_fields = 'prefix1 padding1'
    line2_fields = 'symbol dsname data version os padding2 created'
    line3_fields = 'modified padding3'
    fields = ' '.join((line1_fields, line2_fields, line3_fields))
    Header = namedtuple('Header', fields)
    try:
        header = Header._make(struct.unpack(fmt, info))
    except struct_error:
        raise XptError('Could not unpack header: {0}'.format(info))

    if header.prefix1 != 'HEADER RECORD*******LIBRARY HEADER RECORD!!!!!!!':
        raise XptError('Expected header, found:{0}'.format(header))

    return header


def read_member(f):
    try:
        header = read_member_header(f)
    except XptNotMember, e:
        logging.info('Tried to read next member. {0}'.format(e))
        return None
    except Exception, e:
        raise e

    varlist = [var for var in read_namestrs(f, header)]
    obs = read_obs(f, varlist)

    Member = namedtuple('Member', 'header varlist obs')
    return Member(header, varlist, obs)


def read_member_header(f):
    num_lines = 4
    line_size = 80
    header_size = num_lines * line_size
    info = f.read(header_size)

    if len(info) < header_size:
        raise XptNotMember('Member header not found: "{0}"'.format(info))
        # raise StopIteration()

    fmt = '>48s26s4s2s 48s32s 8s8s8s8s8s24s16s 16s16s40s8s'
    line1_fields = 'prefix1 padding1 namestr_size padding1a'
    line2_fields = 'prefix2 padding2'
    line3_fields = 'symbol dsname data version os padding3 created'
    line4_fields = 'modified padding4 dslabel dstype'
    fields = ' '.join((line1_fields, line2_fields, line3_fields, line4_fields))
    Header = namedtuple('MemberHeader', fields)
    try:
        header = Header._make(struct.unpack(fmt, info))
    except struct_error:
        raise XptError('Could not unpack member header: {0}'.format(info))
    
    if header.prefix1 != 'HEADER RECORD*******MEMBER  HEADER RECORD!!!!!!!':
        raise XptNotMember('Expected member header, found:{0}'.format(header))
    if header.prefix2 != 'HEADER RECORD*******DSCRPTR HEADER RECORD!!!!!!!':
        raise XptNotMember('Expected member header, found:{0}'.format(header))
    
    return header


def read_namestrs(f, member_header):
    '''
    Here is the C structure definition for the namestr record:

      struct NAMESTR {
        short   ntype;              /* VARIABLE TYPE: 1=NUMERIC, 2=CHAR    */
        short   nhfun;              /* HASH OF NNAME (always 0)            */
        short   nlng;               /* LENGTH OF VARIABLE IN OBSERVATION   */
        short   nvar0;              /* VARNUM                              */
        char8   nname;              /* NAME OF VARIABLE                    */
        char40  nlabel;             /* LABEL OF VARIABLE                   */
      
        char8   nform;              /* NAME OF FORMAT                      */
        short   nfl;                /* FORMAT FIELD LENGTH OR 0            */
        short   nfd;                /* FORMAT NUMBER OF DECIMALS           */
        short   nfj;                /* 0=LEFT JUSTIFICATION, 1=RIGHT JUST  */
        char    nfill[2];           /* (UNUSED, FOR ALIGNMENT AND FUTURE)  */
        char8   niform;             /* NAME OF INPUT FORMAT                */
        short   nifl;               /* INFORMAT LENGTH ATTRIBUTE           */
        short   nifd;               /* INFORMAT NUMBER OF DECIMALS         */
        long    npos;               /* POSITION OF VALUE IN OBSERVATION    */
        char    rest[52];           /* remaining fields are irrelevant     */
        };
    '''

    header = read_namestrs_header(f)

    namestr_size = int(member_header.namestr_size)
    num_vars = int(header.num_vars)

    fmt = '>hhhh8s40s8shhh2s8shhl52s'
    fields = ['type', 'hash', 'length', 'id', 'name', 'label', 'format',
              'size', 'decimals', 'just', 'fill', 'iformat', 'ilength',
              'idecimals', 'position', 'junk']
    Var = namedtuple('Var', fields)

    for i in range(num_vars):
        info = f.read(namestr_size)
        try:
            yield Var._make(struct.unpack(fmt, info))
        except struct_error:
            raise XptError('Could not unpack var: {0}'.format(info))

    line_size = 80
    info_size = num_vars * namestr_size
    remainder = info_size % line_size
    padding_size = 0 if remainder == 0 else line_size - remainder
    padding = f.read(padding_size)


def read_namestrs_header(f):
    num_lines = 1
    line_size = 80
    info = f.read(num_lines * line_size)

    fmt = '>48s6s4s22s'
    fields = 'prefix padding1 num_vars padding2'
    Header = namedtuple('NamestrsHeader', fields)
    try:
        header = Header._make(struct.unpack(fmt, info))
    except struct_error:
        raise XptError('Could not unpack namestr header: {0}'.format(info))

    if header.prefix != 'HEADER RECORD*******NAMESTR HEADER RECORD!!!!!!!':
        raise XptError('Expected namestrs header, found:{0}'.format(header))

    return header


def read_obs(f, varlist):
    header = read_obs_header(f)

    info_size = reduce(lambda x, y: x + int(y.length), varlist, 0)
    is_ob = (lambda x: x is not None and len(x) == info_size)

    # TODO: detect new member/dataset

    for info in takewhile(is_ob, imap(f.read, repeat(info_size))):
        yield parse_ob(info, varlist)


def parse_ob(info, varlist):
    o = {}
    for var in sorted(varlist, key = lambda x: x.position):
        key, val = parse_var(info, var)
        o[key] = val
    return o

def parse_var(info, var):
    key = var.name.strip()
    val = None

    var_str = info[var.position:var.position + var.length]

    if var_str[0] == '.':
        # Missing
        val = None
    elif var.type == 2:
        # Unicode characters
        val = unicode(var_str.strip(), 'ISO-8859-1')
    elif var.type == 1:
        # IBM float
        val = ibm_to_ieee(var_str)
        # Note that the namestr may say 'decimals = 0'
        # Even when the value is not an integer
    else:
        raise XptError('Unknown variable type: {0}'.format(var))

    return key, val


def ibm_to_ieee(ibm):
    '''
    /**************************************************************/
    /* Translate IBM format floating point numbers into IEEE */
    /* format floating point numbers. */
    /* */
    /* IEEE format: */
    /* */
    /* 6 5 0 */
    /* 3 1 0 */
    /* */
    /* SEEEEEEEEEEEMMMM ............ MMMM */
    /* */
    /* Sign bit, 11 bits exponent, 52 bit fraction. Exponent is */
    /* excess 1023. The fraction is multiplied by a power of 2 of */
    /* the actual exponent. Normalized floating point numbers are */
    /* represented with the binary point immediately to the left */
    /* of the fraction with an implied "1" to the left of the */
    /* binary point. */
    /* */
    /* IBM format: */
    /* */
    /* 6 5 0 */
    /* 3 1 0 */
    /* */
    /* SEEEEEEEMMMM ......... MMMM */
    /* */
    /* Sign bit, 7 bit exponent, 56 bit fraction. Exponent is */
    /* excess 64. The fraction is multiplied bya power of 16 of */
    /* the actual exponent. Normalized floating point numbers are */
    /* represented with the radix point immediately to the left of*/
    /* the high order hex fraction digit. */
    /* */
    /* How do you translate from IBM format to IEEE? */
    /* */
    /* Translating back to ieee format from ibm is easier than */
    /* going the other way. You lose at most, 3 bits of fraction, */
    /* but nothing can be done about that. The only tricky parts */
    /* are setting up the correct binary exponent from the ibm */
    /* hex exponent, and removing the implicit "1" bit of the ieee*/
    /* fraction (see vzctdbl). We must shift down the high order */
    /* nibble of the ibm fraction until it is 1. This is the */
    /* implicit 1. The bit is then cleared and the exponent */
    /* adjusted by the number of positions shifted. A more */
    /* thorough discussion is in vzctdbl.c. */

    /* Get the first half of the ibm number without the exponent */
       /* into the ieee number */
       ieee1 = xport1 & 0x00ffffff;
     /* get the second half of the ibm number into the second half */
       /* of the ieee number . If both halves were 0. then just */
       /* return since the ieee number is zero. */
       if ((!(ieee2 = xport2)) && !xport1)
       return;
     /* The fraction bit to the left of the binary point in the */
       /* ieee format was set and the number was shifted 0, 1, 2, or */
       /* 3 places. This will tell us how to adjust the ibm exponent */
       /* to be a power of 2 ieee exponent and how to shift the */
       /* fraction bits to restore the correct magnitude. */
     if ((nib = (int)xport1) & 0x00800000)
       shift = 3;
       else
       if (nib & 0x00400000)
       shift = 2;
       else
       if (nib & 0x00200000)
       shift = 1;
       else
       shift = 0;
     if (shift)
       {
       /* shift the ieee number down the correct number of places*/
       /* then set the second half of the ieee number to be the */
       /* second half of the ibm number shifted appropriately, */
       /* ored with the bits from the first half that would have */
       /* been shifted in if we could shift a double. All we are */
       /* worried about are the low order 3 bits of the first */
       /* half since we're only shifting by 1, 2, or 3. */
       ieee1 >>= shift;
       ieee2 = (xport2 >> shift) |
       ((xport1 & 0x00000007) << (29 + (3 - shift)));
       }
     /* clear the 1 bit to the left of the binary point */
       ieee1 &= 0xffefffff;
     /* set the exponent of the ieee number to be the actual */
       /* exponent plus the shift count + 1023. Or this into the */
       /* first half of the ieee number. The ibm exponent is excess */
       /* 64 but is adjusted by 65 since during conversion to ibm */
       /* format the exponent is incremented by 1 and the fraction */
       /* bits left 4 positions to the right of the radix point. */
       ieee1 |=
       (((((long)(*temp & 0x7f) - 65) << 2) + shift + 1023) << 20) |
       (xport1 & 0x80000000);
     REVERSE(&ieee1,sizeof(unsigned long)); 
       memcpy(ieee,((char *)&ieee1)+sizeof(unsigned long)-4,4);
       REVERSE(&ieee2,sizeof(unsigned long)); 
       memcpy(ieee+4,((char *)&ieee2)+sizeof(unsigned long)-4,4);
       return;
    }
    '''
    ieee = None

    # parse the 64 bits of IBM float as one 8-byte unsigned long long
    ulong = struct.unpack('>Q', ibm)[0]
    # drop 1 bit for sign and 7 bits for exponent
    ieee = ulong & 0x00ffffffffffffff

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


def read_obs_header(f):
    info = f.read(80)
    
    fmt = '>48s32s'
    fields = 'prefix padding'
    Header = namedtuple('ObsHeader', fields)
    try:
        header = Header._make(struct.unpack(fmt, info))
    except struct_error:
        raise XptError('Could not unpack obs header: {0}'.format(info))

    if header.prefix != 'HEADER RECORD*******OBS     HEADER RECORD!!!!!!!':
        raise XptError('Expected observations header, found:{0}'.format(header))

    return header


def parse_date(datestr):
    '''
    Param: date in XPT format (ex. "16FEB11:10:07:55")
    Returns: python datetime obj
    '''
    return datetime.strptime(datestr, "%d%b%y:%H:%M:%S")


if __name__ == '__main__':
    # Parse args
    parser = argparse.ArgumentParser(
        description = 'Reads XPT format. Writes CSV format.')
    parser.add_argument('-i',
                        type=argparse.FileType('rb'),
                        default=sys.stdin)
    parser.add_argument('-o',
                        type=argparse.FileType('w'),
                        default=sys.stdout)
    args = parser.parse_args()

    # Write CSV
    writer = None
    with XportReader(args.i) as reader:
        for obs in reader:
            if not writer:
                writer = csv.DictWriter(args.o, obs.keys())
                writer.writeheader()
            try:
                writer.writerow(obs)
            except IOError, e:
                import errno
                if e.errno == errno.EPIPE:
                    sys.exit()
                raise
