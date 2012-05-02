from datetime import datetime
import struct

def parse_date(datestr):
    """ Given a date in xport format, return Python date. """
    return datetime.strptime(datestr, "%d%b%y:%H:%M:%S") # e.g. "16FEB11:10:07:55"

def _split_line(s, parts):
    """
        s: fixed-length string to split
        parts:  list of (name, length) pairs used to break up string
                name '_' will be filtered from output.

        result: dict of name:contents of string at given location.
    """
    out = {}
    start = 0
    for name, length in parts:
        out[name] = s[start:start+length].strip()
        start += length
    del out['_']
    return out


def parse_float(bitstring):
    """
        Given IBM-style float stored as string, return Python float.

        This is adapted from the following C code in the spec. The adaptation may not be correct, or optimal.

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
    """

    xport1, xport2 = struct.unpack('>II', bitstring)

    # Start by setting first half of ieee number to first half of IBM number sans exponent
    ieee1 = xport1 & 0x00ffffff
    # get the second half of the ibm number into the second half of the ieee number
    ieee2 = xport2

    # If both halves were 0. then just return since the ieee number is zero.
    if not ieee1 and not ieee2:
        return 0.0
        # The fraction bit to the left of the binary point in the ieee format was set and the number was shifted 0, 1, 2, or
    # 3 places. This will tell us how to adjust the ibm exponent to be a power of 2 ieee exponent and how to shift
    # the fraction bits to restore the correct magnitude.
    if xport1 & 0x00800000:
        shift = 3
    elif xport1 & 0x00400000:
        shift = 2
    elif xport1 & 0x00200000:
        shift = 1
    else:
        shift = 0

    if shift:
        # shift the ieee number down the correct number of places then set the second half of the ieee number to be the
        # second half of the ibm number shifted appropriately, ored with the bits from the first half that would have
        # been shifted in if we could shift a double. All we are worried about are the low order 3 bits of the first
        # half since we're only shifting by 1, 2, or 3.
        ieee1 >>= shift
        ieee2 = (xport2 >> shift) | ((xport1 & 0x00000007) << (29 + (3 - shift)))

    # clear the 1 bit to the left of the binary point
    ieee1 &= 0xffefffff

    # set the exponent of the ieee number to be the actual exponent plus the shift count + 1023. Or this into the
    # first half of the ieee number. The ibm exponent is excess 64 but is adjusted by 65 since during conversion to ibm
    # format the exponent is incremented by 1 and the fraction bits left 4 positions to the right of the radix point.
    # (had to add >> 24 because C treats & 0x7f as 0x7f000000 and Python doesn't)
    ieee1 |= ((((((xport1 >> 24) & 0x7f) - 65) << 2) + shift + 1023) << 20) | (xport1 & 0x80000000)

    # Python doesn't limit to 4 bytes like we need it to ...
    ieee1 &= 0xffffffff
    ieee2 &= 0xffffffff

    return struct.unpack(">d", struct.pack(">II", ieee1, ieee2))[0]

class XportReader(object):

    def __init__(self, file, encoding='ISO-8859-1'):
        self.encoding = encoding # is this ever anything else that ISO-8859-1 ??
        self.loadfile(file)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.opened_file:
            try:
                self.file.close()
            except:
                pass

    def _get_row(self):
        return self.file.read(80)

    def loadfile(self, file):
        """
            Open file, seek to start, verify that it is an xport file, set:
                self.file = file
                self.file_info
                self.member_info
                self.fields

            From spec, format of fields is as follows (items with stars renamed for clarity):

              struct NAMESTR {
                short   ntype;              /* VARIABLE TYPE: 1=NUMERIC, 2=CHAR    */
                short   nhfun;              /* HASH OF NNAME (always 0)            */
            *   short   field_length;               /* LENGTH OF VARIABLE IN OBSERVATION   */
                short   nvar0;              /* VARNUM                              */
            *   char8   name;              /* NAME OF VARIABLE                    */
            *   char40  label;             /* LABEL OF VARIABLE                   */

                char8   nform;              /* NAME OF FORMAT                      */
                short   nfl;                /* FORMAT FIELD LENGTH OR 0            */
            *   short   num_decimals;                /* FORMAT NUMBER OF DECIMALS           */
                short   nfj;                /* 0=LEFT JUSTIFICATION, 1=RIGHT JUST  */
                char    nfill[2];           /* (UNUSED, FOR ALIGNMENT AND FUTURE)  */
                char8   niform;             /* NAME OF INPUT FORMAT                */
                short   nifl;               /* INFORMAT LENGTH ATTRIBUTE           */
                short   nifd;               /* INFORMAT NUMBER OF DECIMALS         */
                long    npos;               /* POSITION OF VALUE IN OBSERVATION    */
                char    rest[52];           /* remaining fields are irrelevant     */
                };
        """
        self.opened_file = False
        try:
            file = open(file, 'rb')
            self.opened_file = True
        except TypeError:
            try:
                file.seek(0)
            except AttributeError:
                raise TypeError("File should be a string-like or file-like object.")
        self.file = file

        # read file header
        line1 = self._get_row()
        if not line1 == "HEADER RECORD*******LIBRARY HEADER RECORD!!!!!!!000000000000000000000000000000  ":
            raise Exception("Header record is not an XPORT file.")

        line2 = self._get_row()
        file_info = _split_line(line2, [ ['prefix',24], ['version',8], ['OS',8], ['_',24], ['created',16]])
        if file_info['prefix'] != "SAS     SAS     SASLIB":
            raise Exception("Header record has invalid prefix.")
        file_info['created'] = parse_date(file_info['created'])
        self.file_info = file_info

        line3 = self._get_row()
        file_info['modified'] = parse_date(line3[:16])

        # read member header
        header1 = self._get_row()
        header2 = self._get_row()
        if  not header1.startswith("HEADER RECORD*******MEMBER  HEADER RECORD!!!!!!!000000000000000001600000000")\
        or not header2 == "HEADER RECORD*******DSCRPTR HEADER RECORD!!!!!!!000000000000000000000000000000  ":
            raise Exception("Member header not found.")
        fieldnamelength = int(header1[-5:-2]) # usually 140, could be 135

        # member info
        member_info = _split_line(self._get_row(), [['prefix',8],['set_name',8],['sasdata',8],['version',8],['OS',8],['_',24],['created',16]])
        member_info.update( _split_line(self._get_row(), [['modified',16],['_',16],['label',40],['type',8]]))
        member_info['modified'] = parse_date(member_info['modified'])
        member_info['created'] = parse_date(member_info['created'])
        self.member_info = member_info

        # read field names
        types = {1:'numeric', 2:'char'}
        fieldcount = int(self._get_row()[54:58])
        datalength = fieldnamelength*fieldcount
        if datalength%80: # round up to nearest 80
            datalength += 80 - datalength%80
        fielddata = file.read(datalength)
        fields = []
        obs_length = 0
        while len(fielddata)>=fieldnamelength:
            field, fielddata = (fielddata[:fieldnamelength], fielddata[fieldnamelength:]) # pull data for this field from start of fielddata
            field = field.ljust(140) # rest at end gets ignored, so if field is short, pad out to match struct pattern below
            fieldstruct = struct.unpack('>hhhh8s40s8shhh2s8shhl52s', field)
            field = dict(zip(['ntype','nhfun','field_length','nvar0','name','label','nform','nfl','num_decimals','nfj','nfill','niform','nifl','nifd','npos','_'],fieldstruct))
            del field['_']
            field['ntype'] = types[field['ntype']]
            if field['ntype']=='numeric' and field['field_length'] != 8:
                raise TypeError("Oops -- only 8-byte floats are currently implemented. Can't read field %s." % field)
            for k, v in field.items():
                try:
                    field[k] = v.strip()
                except AttributeError:
                    pass
            obs_length += field['field_length']
            fields += [field]

        if not self._get_row() == "HEADER RECORD*******OBS     HEADER RECORD!!!!!!!000000000000000000000000000000  ":
            raise Exception("Observation header not found.")

        self.fields = fields
        self.record_length = obs_length
        self.record_start = self.file.tell()

    def record_count(self):
        """
            Get number of records in file.
            This is maybe suboptimal because we have to seek to the end of the file.
        """
        self.file.seek(0,2)
        total_records_length = self.file.tell() - self.record_start
        return total_records_length / self.record_length

    def __iter__(self):
        return self

    def next(self):
        s = self.file.read(self.record_length)
        if not s or len(s) < self.record_length:
            raise StopIteration()
        obs = {}
        for field in self.fields:
            bytes = field['field_length']
            field_str, s = (s[:bytes], s[bytes:]) # pull data for this field from start of obs data
            if field['ntype'] == 'char':
                field_val = unicode(field_str.strip(), self.encoding)
            else:
                field_val = parse_float(field_str)
                if field['num_decimals'] == 0:
                    field_val = int(field_val)
            obs[field['name']] = field_val
        return obs



if __name__ == "__main__":
    import sys
    with XportReader(sys.argv[1]) as reader:
        for obj in reader:
            print obj
