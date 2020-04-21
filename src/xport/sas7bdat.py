"""
Read and write the SAS7BDAT file format.
"""

# Standard Library
from io import BytesIO


def load(fp):
    """
    Deserialize a SAS7 binary data format document.

        with open('example.sas7bdat', 'rb') as f:
            data = load(f)
    """
    raise NotImplementedError()


def loads(s):
    """
    Deserialize a SAS7 binary data file from a string.

        with open('example.sas7bdat', 'rb') as f:
            bytestring = f.read()
        data = loads(bytestring)
    """
    fp = BytesIO(s)
    return load(fp)


def dump(columns, fp, name=None, labels=None, formats=None):
    """
    Serialize a SAS7 binary data file format document.

        data = {
            'a': [1, 2],
            'b': [3, 4],
        }
        with open('example.sas7bdat', 'wb') as f:
            dump(data, f)
    """
    raise NotImplementedError()


def dumps(columns, name=None, labels=None, formats=None):
    """
    Serialize a SAS7 binary data file to a string.

        data = {
            'a': [1, 2],
            'b': [3, 4],
        }
        bytestring = dumps(data)
        with open('example.sas7bdat', 'wb') as f:
            f.write(bytestring)
    """
    fp = BytesIO()
    dump(columns, fp)
    fp.seek(0)
    return fp.read()
