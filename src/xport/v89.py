"""
Read and write the SAS XPORT/XPT file format from SAS Version 8 or 9.

The SAS V8 Transport File format, also called XPORT, or simply XPT, ...
"""

# Member names may now be up to 32 characters.
# Variable names may now be up to 32 characters.
# Variable labels may now be up to 256 characters.
# Character fields may now exceed 200 characters. SAS 字符长度最大 32767
# Format fields may now exceed 8 characters (v9).

# Standard Library
import logging
import warnings
import struct
import xport
import pandas as pd
from io import BytesIO
from .v56 import Member, Library, MemberHeader, Namestr, strptime, strftime, ibm_to_ieee, ieee_to_ibm
import re
code_type = 'ISO-8859-1'
# code_type = 'GBK'

LOG = logging.getLogger(__name__)


class NamestrV8(Namestr):
    """
    long npos; 后与v5版本不同 name 最长32
    char longname[32]; /* long name for Version 8-style *//* 版本 8 样式的长名称 */ 32
    short lablen; /* length of label */                   /* 标签长度 */ 2bytes
    char rest[18]; /* remaining fields are irrelevant */  /* 其余字段不相关 */
    """

    fmts = {
        140: '>hhhh8s40s8shhh2s8shhl32sh18s',  # 0-17 len = 18
        136: '>hhhh8s40s8shhh2s8shhl32sh14s',
    }

    @classmethod
    def from_bytes(cls, bytestring: bytes):
        """
        Construct a ``Namestr`` from an XPORT-format byte string.
        """
        LOG.debug(f'Decode {type(cls).__name__}')
        # dtype='float' if vtype == xport.VariableType.NUMERIC else 'string'
        size = len(bytestring)
        if size == 136:
            warnings.warn(
                'File written on VAX/VMS, module behavior not tested')
        fmt = cls.fmts[size]
        tokens = struct.unpack(fmt, bytestring)
        return cls(
            vtype=xport.VariableType(tokens[0]),
            length=tokens[2],
            number=tokens[3],
            name=tokens[4].strip(b'\x00').decode(code_type).rstrip(),
            # V8 longname[32]
            label=tokens[15].strip(b'\x00').decode(code_type).rstrip(),
            format=xport.Format.from_struct_tokens(*tokens[6:10]),
            informat=xport.Informat.from_struct_tokens(*tokens[11:14]),
            position=tokens[14],
        )


class NamestrLablelv:
    """
    Names 前置现在bytes里面包含其他内容需要判断后再调用.
    变量标签超过40个字符(btyes)
    HEADER RECORD*******LABELV8 HEADER RECORD!!!!!!!nnnnn                   
    超过 8 个字符的格式或信息名称，无论标签长度如何
    HEADER RECORD*******LABELV9 HEADER RECORD!!!!!!!nnnnn                   
    """
    pattern = re.compile(
        rb'(?P<namestrs>.*?)'
        rb'HEADER RECORD\*{7}(?P<label_v>.{7}) HEADER RECORD\!{7}'
        rb'(?P<label_num>.{5}) {27}', re.DOTALL
    )

    @classmethod
    def from_bytes(cls, bytestring, stride=None):
        mview = memoryview(bytestring)
        nlmv = cls.pattern.match(mview)
        if nlmv:
            # LABELV8 变量标签超过40 LABELV9超过 8 个字符的格式或信息名称
            label_v = nlmv['label_v']
            label_num = int(nlmv['label_num'].decode(code_type).strip())  # 变量数
            # 匹配到需要内容的长度 - 总长度 剩余的为labelv 的内容
            lavlev89bytes = mview[nlmv.regs[0][1]:nlmv.endpos]
            if label_v == b'LABELV8':
                name_label_sets = lavlev8(lavlev89bytes, label_num)
            if label_v == b'LABELV9':
                name_label_sets = lavlev9(lavlev89bytes, label_num)
            namestrsbytes = nlmv['namestrs']
        else:
            namestrsbytes = bytestring

        def chunks(stride):
            for i in range(0, len(namestrsbytes), stride):
                chunk = namestrsbytes[i:i + stride]
                if len(chunk) == stride:
                    yield chunk
        # print([NamestrLablelv.from_bytes(b) for b in chunks()])
        namestrs = [NamestrV8.from_bytes(b) for b in chunks(stride)]
        if nlmv:
            new_namestrs = []
            for n in namestrs:
                nlfi = name_label_sets.get(str(n.number))
                if nlfi:  # 这里如果没有超过的部分不会出现在labelv8里面
                    n.name = nlfi['name']
                    n.label = nlfi['label']
                    if label_v == b'LABELV9':
                        n.format = xport.Format(
                            name=nlfi['format'], length=n.format.length, decimals=n.format.decimals)
                        n.informat = xport.Informat(
                            name=nlfi['informat'], length=n.informat.length, decimals=n.informat.decimals)
                new_namestrs.append(n)
            namestrs = new_namestrs
        # name_label_sets = name_label_sets  # 考虑导出内容的形式
        return namestrs


class MemberHeaderV8(MemberHeader):
    """
    Dataset metadata from a SAS Version 8 Transport (XPORT) file.
    v8格式 dataset 类扩展
    HEADER RECORD*******OBSV8   HEADER RECORD!!!!!!!              1                 
    1 或者2 代表rows 长度
    """
    pattern = re.compile(
        # Header line 1
        rb'HEADER RECORD\*{7}MEMBV8  HEADER RECORD\!{7}0{17}'
        rb'160{8}(?P<descriptor_size>140|136)  '
        # Header line 2
        rb'HEADER RECORD\*{7}DSCPTV8 HEADER RECORD\!{7}0{30} {2}'
        # Header line 3
        rb'SAS {5}(?P<name>.{32})SASDATA '
        rb'(?P<version>.{8})(?P<os>.{8})(?P<created>.{16})'
        # Header line 4
        rb'(?P<modified>.{16}) {16}'
        rb'(?P<label>.{40})(?P<type>    DATA|    VIEW| {8})'
        # Namestrs
        rb'HEADER RECORD\*{7}NAMSTV8 HEADER RECORD\!{7}0{6}'
        rb'(?P<n_variables>.{4})0{20} {2}'
        # NamestrLablelv
        rb'(?P<namestrs>.*?)'
        rb'HEADER RECORD\*{7}OBSV8 {3}HEADER RECORD\!{7}'
        rb'(?P<obs_rows>.{15}).{17}',
        # Observations ... until EOF or another Member.
        re.DOTALL,
    )

    def __init__(self, name, label, dataset_type, created,
                 modified, sas_os, sas_version, obs_rows, namestrs=()):
        """ 增加 obs_rows参数"""
        super().__init__(
            name=name,
            label=label,
            dataset_type=dataset_type,
            created=created,
            modified=modified,
            sas_os=sas_os,
            sas_version=sas_version,
            namestrs=namestrs,
        )
        self.obs_rows = obs_rows

    @classmethod
    def from_bytes(cls, bytestring: bytes):
        """
        Construct a ``MemberHeader`` from an XPORT-format byte string.
        """
        LOG.debug(f'Decode {type(cls).__name__}')
        mo = cls.pattern.search(bytestring)
        if mo is None:
            raise ValueError('No member header found')
        namestrlablelvbytes = mo['namestrs']
        stride = int(mo['descriptor_size'])
        namestrs = NamestrLablelv.from_bytes(namestrlablelvbytes, stride=stride)
        n = int(mo['n_variables'])
        if len(namestrs) != n:
            raise ValueError(f'Expected {n}, got {len(namestrs)}')
        self = cls(
            name=mo['name'].strip(b'\x00').decode(code_type).strip(),
            label=mo['label'].strip(b'\x00').decode(code_type).strip(),
            dataset_type=mo['type'].strip(
                b'\x00').decode(code_type).strip(),
            sas_os=mo['os'].strip(b'\x00').decode(code_type).strip(),
            sas_version=mo['version'].strip().decode(code_type),
            created=strptime(mo['created']),
            modified=strptime(mo['modified']),
            obs_rows=int(mo['obs_rows'].strip(b'\x00')),
            namestrs=namestrs,
        )
        return self


class MemberV8(Member):
    """
    Dataset from a SAS Version 8 Transport (XPORT) file.
    """

    @classmethod
    def from_bytes(cls, bytestring, pattern=MemberHeaderV8.pattern,):
        """
        Decode the first ``Member`` from an XPORT-format byte string.
        """
        LOG.debug(f'Decode {type(cls).__name__}')
        mview = memoryview(bytestring)
        matches = pattern.finditer(mview)

        try:
            mo = next(matches)
        except StopIteration:
            raise ValueError('No member header found')
        i = mo.end(0)

        try:
            mo = next(matches)
        except StopIteration:
            j = None
        else:
            j = mo.start(0)

        header = xport.v89.MemberHeaderV8.from_bytes(mview[:i])
        observations = xport.v56.Observations.from_bytes(mview[i:j], header)

        # This awkwardness works around Pandas subclasses misbehaving.
        # ``DataFrame.append`` discards subclass attributes.  Lame.
        head = cls.from_header(header)
        data = Member(pd.DataFrame.from_records(observations, columns=list(header)))
        data.copy_metadata(head)
        LOG.info(f'Decoded XPORT dataset {data.name!r}')
        LOG.debug('%s', data)
        return data


class LibraryV8(Library):
    """
    Collection of datasets from a SAS Version 8 Transport file.
    # HEADER RECORD*******LIBV8   HEADER RECORD!!!!!!!000000000000000000000000000000  
    """
    pattern = re.compile(
        rb'HEADER RECORD\*{7}LIBV8   HEADER RECORD\!{7}0{30} {2}'
        rb'SAS {5}SAS {5}SASLIB {2}'
        rb'(?P<version>.{8})(?P<os>.{8}) {24}(?P<created>.{16})'
        rb'(?P<modified>.{16}) {64}'
        rb'(?P<members>.*)',
        re.DOTALL,
    )

    @classmethod
    def from_bytes(cls, bytestring, member_header_re=MemberHeaderV8.pattern):
        """
        Parse a SAS XPORT document from a byte string.
        """
        LOG.debug(f'Decoding {cls.__name__} from {len(bytestring)} bytes')
        mview = memoryview(bytestring)
        mo = cls.pattern.match(mview)
        if mo is None:
            lines = [mview[i * 80:(i + 1) * 80] for i in range(8)]
            LOG.error(f'Document begins with' + '\n%s' * len(lines), *lines)
            raise ValueError(
                'Document does not match SAS Version 5 or 6 or 8 Transport (XPORT) format')

        matches = member_header_re.finditer(mview)
        indices = [m.start(0) for m in matches] + [None]
        chunks = (mview[i:j] for i, j in zip(indices, indices[1:]))
        self = Library(
            members=map(MemberV8.from_bytes, chunks),
            created=strptime(mo['created']),
            modified=strptime(mo['modified']),
            sas_os=mo['os'].strip(b'\x00').decode(code_type).strip(),
            sas_version=mo['version'].strip(
                b'\x00').decode(code_type).strip(),
        )
        LOG.info(f'Decoded {self}')
        return self


def lavlev8(lavlev8bytes, label_num):
    """
    a a b b c c d.....e.....
    00010001002B X (ASCII)
    aa = 0001  # 变量号码
    bb = 0001  # 变量名称长度
    cc = 002B  # 变量标签长度

    where
        aa=variable number
        bb=length of name
        cc=length of label
        d. =name in bb bytes
        e. =label in cc bytes

    """
    name_label_sets = {}
    for i in range(label_num):
        aa, bb, cc = (int.from_bytes(lavlev8bytes[0:2], byteorder='big'),
                      int.from_bytes(lavlev8bytes[2:4], byteorder='big'),
                      int.from_bytes(lavlev8bytes[4:6], byteorder='big'),)
        totalLength = 6 + bb + cc
        # 得到变量名称
        d = lavlev8bytes[6:6+bb].tobytes().strip(b'\x00').decode(code_type).strip()
        e = lavlev8bytes[6+bb:totalLength].tobytes().strip(b'\x00').decode(code_type).strip()  # 变量标签
        lavlev8bytes = lavlev8bytes[totalLength:lavlev8bytes.shape[0]]
        name_label_sets = dict(**name_label_sets, **{str(aa):  {'lengthname': bb,
                                                                'lengthlabel': cc,
                                                                'name': d,
                                                                'label': e}})
    return name_label_sets


def lavlev9(lavlev9bytes, label_num):
    """
    超过 8 个字符的格式或信息名称，无论标签长度如何
    a a b b c c d d e e f.....g.....h.....i.....
    0001000B003E000C0001
    aa = 0001 # 变量号码
    bb = 000B # 变量名称长度
    cc = 003E # 变量标签长度
    dd = 000C # 变量描述格式长度
    ee = 0001 # 变量描述信息格式长度

    where
        aa=variable number
        bb=length of name in bytes
        cc=length of label in bytes
        dd=length of format description in bytes
        ee=length of informat description in bytes
        f.= text for variable name
        g. =text for variable label
        h. = text for format description
        i. = text of informat description
    """
    name_label_sets = {}
    for i in range(label_num):
        aa, bb, cc, dd, ee = (
            int.from_bytes(lavlev9bytes[0:2], byteorder='big'),
            int.from_bytes(lavlev9bytes[2:4], byteorder='big'),
            int.from_bytes(lavlev9bytes[4:6], byteorder='big'),
            int.from_bytes(lavlev9bytes[6:8], byteorder='big'),
            int.from_bytes(lavlev9bytes[8:10], byteorder='big'),)
        totalLength = 10 + bb + cc + dd + ee
        # 得到变量名称
        f = lavlev9bytes[10:10+bb].tobytes().strip(b'\x00').decode(code_type).strip()
        g = lavlev9bytes[10+bb:10+bb+cc].tobytes().strip(b'\x00').decode(code_type).strip()  # 变量标签
        h = lavlev9bytes[10+bb+cc:10+bb+cc+dd].tobytes().strip(b'\x00').decode(code_type).strip()
        i = lavlev9bytes[10+bb+cc+dd:totalLength].tobytes().strip(b'\x00').decode(code_type).strip()
        lavlev9bytes = lavlev9bytes[totalLength:lavlev9bytes.shape[0]]
        name_label_sets = dict(**name_label_sets, **{str(aa):  {'lengthname': bb,
                                                                'lengthlabel': cc,
                                                                'lengthformat': dd,
                                                                'lengthinformat': ee,
                                                                'name': f,
                                                                'label': g,
                                                                'format': h,
                                                                'informat': i}})
    return name_label_sets


def load(fp, encoding='ISO-8859-1'):
    """
    Deserialize a SAS V8 transport file format document.

        with open('example.v8xpt', 'rb') as f:
            data = load(f)
    """
    try:
        bytestring = fp.read()
    except UnicodeDecodeError:
        raise TypeError(
            f'Expected a BufferedReader in bytes-mode, got {type(fp).__name__}')
    return loads(bytestring)


def loads(bytestring):
    """
    Deserialize a SAS V8 transport file format document from a string.

        with open('example.xpt', 'rb') as f:
            bytestring = f.read()
        data = loads(bytestring)
    """
    return LibraryV8.from_bytes(bytestring)


def dump(columns, fp, name=None, labels=None, formats=None):
    """
    Serialize a SAS V8 transport file format document.

        data = {
            'a': [1, 2],
            'b': [3, 4],
        }
        with open('example.xpt', 'wb') as f:
            dump(data, f)
    """
    raise NotImplementedError()


def dumps(columns, name=None, labels=None, formats=None):
    """
    Serialize a SAS V8 transport file format document to a string.

        data = {
            'a': [1, 2],
            'b': [3, 4],
        }
        bytestring = dumps(data)
        with open('example.xpt', 'wb') as f:
            f.write(bytestring)
    """
    fp = BytesIO()
    dump(columns, fp)
    fp.seek(0)
    return fp.read()
