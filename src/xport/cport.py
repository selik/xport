"""
Read and write the CPORT file format.

CPORT-format is a compressed XPORT format.
"""

# Standard Library
from io import BytesIO


def load(fp):
    """
    Deserialize a SAS compressed transport file format document.

        with open('example.cpt', 'rb') as f:
            data = load(f)
    """
    raise NotImplementedError()


def loads(s):
    """
    Deserialize a SAS compressed transport file from a string.

        with open('example.cpt', 'rb') as f:
            bytestring = f.read()
        data = loads(bytestring)
    """
    fp = BytesIO(s)
    return load(fp)


def dump(columns, fp, name=None, labels=None, formats=None):
    """
    Serialize a SAS compressed transport file format document.

        data = {
            'a': [1, 2],
            'b': [3, 4],
        }
        with open('example.cpt', 'wb') as f:
            dump(data, f)
    """
    raise NotImplementedError()


def dumps(columns, name=None, labels=None, formats=None):
    """
    Serialize a SAS compressed transport file to a string.

        data = {
            'a': [1, 2],
            'b': [3, 4],
        }
        bytestring = dumps(data)
        with open('example.cpt', 'wb') as f:
            f.write(bytestring)
    """
    fp = BytesIO()
    dump(columns, fp)
    fp.seek(0)
    return fp.read()
