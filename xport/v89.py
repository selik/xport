#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
Read SAS XPORT/XPT files from SAS Version 8 or 9.
'''

# Labels may now exceed 40 characters
# Character fields may now exceed 200 characters.

from xport.v56 import Library, Member, Variable, Observations
import re



class Library(Library):
    '''
    '''

    header_re = re.compile(
        rb'HEADER RECORD\*{7}LIBV8   HEADER RECORD\!{7}0{30}  '
        rb'SAS     SAS     SASLIB  '
        rb'(?P<version>.{8})(?P<os>.{8}) {24}(?P<created>.{16})'
        rb'(?P<modified>.{16}) {64}'
    )

           

class Member(Member):
    '''
    '''

    line1_re = re.compile(
        rb'HEADER RECORD\*{7}MEMBV8  HEADER RECORD\!{7}0{17}160{8}140  '
    )

    header_re = re.compile(
        # line 1
        rb'HEADER RECORD\*{7}MEMBV8  HEADER RECORD\!{7}0{17}160{8}140  '
        # line 2
        rb'HEADER RECORD\*{7}DSCPTV8 HEADER RECORD\!{7}0{30}  '
        # line 3
        rb'SAS     (?P<name>.{8})SASDATA '
        rb'(?P<version>.{8})(?P<os>.{8}) {24}(?P<created>.{16})'
        # line 4
        rb'(?P<modified>.{16}) {16}'
        rb'(?P<label>.{40})(?P<type>.{8})'
    )


class Variable(Variable):
    '''
    '''
    # Names now have a maximum of 32 characters
    # using struct element 15 instead of 4 (0-indexed)

    header_re = re.compile(
        rb'HEADER RECORD\*{7}NAMSTV8 HEADER RECORD\!{7}0{6}'
        rb'(?P<n_variables>.{6})0{18}'
    )

    # struct NAMESTR {
    #     short ntype;          /* VARIABLE TYPE: 1=NUMERIC, 2=CHAR */ 
    #     short nhfun;          /* HASH OF NNAME (always 0) */
    #     short nlng;           /* LENGTH OF VARIABLE IN OBSERVATION */ 
    #     short nvar0;          /* VARNUM */
    #     char8 nname;          /* NAME OF VARIABLE */
    #     char40 nlabel;        /* LABEL OF VARIABLE */
    #     char8 nform;          /* NAME OF FORMAT */
    #     short nfl;            /* FORMAT FIELD LENGTH OR 0 */
    #     short nfd;            /* FORMAT NUMBER OF DECIMALS */
    #     short nfj;            /* 0=LEFT JUSTIFICATION, 1=RIGHT JUST */ 
    #     char nfill[2];        /* (UNUSED, FOR ALIGNMENT AND FUTURE) */
    #     char8 niform;         /* NAME OF INPUT FORMAT */
    #     short nifl;           /* INFORMAT LENGTH ATTRIBUTE */ 
    #     short nifd;           /* INFORMAT NUMBER OF DECIMALS */
    #     long npos;            /* POSITION OF VALUE IN OBSERVATION */ 
    #     char longname[32];    /* long name for Version 8-style */
    #     short lablen;         /* length of label */
    #     char rest[18];        /* remaining fields are irrelevant */
    # };

    fmts = {
        140: '>hhhh8s40s8shhh2s8shhl32sh18s',
    }

    @classmethod
    def unpack(cls, data):
        '''

        Parse namestrs -- variable metadata -- for one member of an XPORT
        file, given the metadata of that member.

        '''
        fmt = cls.fmts[140]
        tokens = struct.unpack(fmt, data)
        return {
            'name': tokens[15].decode().rstrip(),
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



class Observations(Observations):
    '''
    '''

    header_re = re.compile(
        rb'HEADER RECORD\*{7}OBSV8   HEADER RECORD\!{7}0{10}'
    )



class Labels:
    '''

    '''

    # If you have any labels that exceed 40 characters, they can be placed
    # in this section. The label records section starts with this header:
    # HEADER RECORD*******LABELV8 HEADER RECORD!!!!!!!nnnnn where nnnnn is
    # the number of variables for which long labels will be defined. Each
    # label is defined using the following: aabbccd.....e..... where
    # aa=variable number bb=length of name cc=length of label d....=name in
    # bb bytes e....=label in cc bytes For example, variable number 1 named
    # x with the 43-byte label 'a very long label for x is given right here'
    # would be provided as a stream of 6 bytes in hex '00010001002B'X
    # followed by the ASCII characters. xa very long label for x is given
    # right here These are streamed together. The last label descriptor is
    # followed by ASCII blanks ('20'X) to an 80-byte boundary. If you have
    # any format or informat names that exceed 8 characters, regardless of
    # the label length, a different form of label record header is used:
    # HEADER RECORD*******LABELV9 HEADER RECORD!!!!!!!nnnnn where nnnnn is
    # the number of variables for which long format names and any labels
    # will be defined. Each label is defined using the following:
    # aabbccddeef.....g.....h.....i..... where aa=variable number bb=length
    # of name in bytes cc=length of label in bytes dd=length of format
    # description in bytes ee=length of informat description in bytes
    # f.....=text for variable name g.....=text for variable label
    # h.....=text for format description i.....=text of informat description
    # Note: The FORMAT and INFORMAT descriptions are in the form used in a
    # FORMAT or INFORMAT statement. For example, my_long_fmt.,
    # my_long_fmt8., my_long_fmt8.2. The text values are streamed together
    # and no characters appear for attributes with a length of 0 bytes.

    header_re = re.compile(
        rb'HEADER RECORD\*{7}LABELV(?P<version>8|9) HEADER RECORD\!{7}'
        rb'(?P<n>.{5}) {5}'
    )

    # TODO: Parse long labels.











