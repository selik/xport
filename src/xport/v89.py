"""
Read and write the SAS XPORT/XPT file format from SAS Version 8 or 9.

The SAS V8 Transport File format, also called XPORT, or simply XPT, ...
"""

# Member names may now be up to 32 characters.
# Variable names may now be up to 32 characters.
# Variable labels may now be up to 256 characters.
# Character fields may now exceed 200 characters.
# Format fields may now exceed 8 characters (v9).

# Standard Library
from io import BytesIO


def load(fp):
    """
    Deserialize a SAS V8 transport file format document.

        with open('example.xpt', 'rb') as f:
            data = load(f)
    """
    raise NotImplementedError()


def loads(s):
    """
    Deserialize a SAS V8 transport file format document from a string.

        with open('example.xpt', 'rb') as f:
            bytestring = f.read()
        data = loads(bytestring)
    """
    fp = BytesIO(s)
    return load(fp)


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
