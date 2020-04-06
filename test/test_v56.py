"""
Tests for XPT format from SAS versions 5 and 6.
"""

# Standard Library
import math
import random

# Community Packages
import pytest

# Xport Modules
import xport.v56


@pytest.fixture(scope='function')
def dataset():
    """
    Create a 4-column, 6-row dataset with numbers and text.
    """
    return xport.v56.Member(
        columns={
            'VIT_STAT': ['ALIVE'] * 3 + ['DEAD'] * 3,
            'ECON': ['POOR', 'NOT', 'UNK'] * 2,
            'COUNT': [1216, 1761, 2517, 254, 60, 137],
            'TEMP': [98.6, 95.4, 86.7, 93.4, 103.5, 56.7],
        },
        labels={
            'VIT_STAT': 'Vital status',
            'ECON': 'Economic status',
            'COUNT': 'Count',
            'TEMP': 'Temperature',
        },
        formats={
            'VIT_STAT': '$5',
            'ECON': '$CHAR4',
            'COUNT': 'comma8.0',
            'TEMP': '8.1',
        },
        dataset_name='ECON',
        dataset_label='Blank-padded dataset label',
    )


@pytest.fixture(scope='function')
def bytestring():
    """
    Create the same dataset in SAS V5 Transport format.
    """
    return b'''\
HEADER RECORD*******LIBRARY HEADER RECORD!!!!!!!000000000000000000000000000000  \
SAS     SAS     SASLIB  9.3     W32_7PRO                        13NOV15:10:35:08\
13NOV15:10:35:08                                                                \
HEADER RECORD*******MEMBER  HEADER RECORD!!!!!!!000000000000000001600000000140  \
HEADER RECORD*******DSCRPTR HEADER RECORD!!!!!!!000000000000000000000000000000  \
SAS     ECON    SASDATA 9.3     W32_7PRO                        13NOV15:10:35:08\
13NOV15:10:35:08                Blank-padded dataset label                      \
HEADER RECORD*******NAMESTR HEADER RECORD!!!!!!!000000000400000000000000000000  \
\x00\x02\x00\x00\x00\x08\x00\x01VIT_STATVital status                            \
$w      \x00\x05\x00\x00\x00\x00\x00\x00        \x00\x00\x00\x00\x00\x00\x00\x00\
\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\
\x00\x02\x00\x00\x00\x08\x00\x02ECON    Economic status                         \
$CHARw  \x00\x04\x00\x00\x00\x01\x00\x00        \x00\x00\x00\x00\x00\x00\x00\x08\
\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\
\x00\x01\x00\x00\x00\x08\x00\x03COUNT   Count                                   \
COMMAw.d\x00\x08\x00\x00\x00\x00\x00\x00        \x00\x00\x00\x00\x00\x00\x00\x10\
\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\
\x00\x01\x00\x00\x00\x08\x00\x04TEMP    Temperature                             \
w.d     \x00\x08\x00\x01\x00\x00\x00\x00        \x00\x00\x00\x00\x00\x00\x00\x18\
\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\
HEADER RECORD*******OBS     HEADER RECORD!!!!!!!000000000000000000000000000000  \
ALIVE   POOR    CL\x00\x00\x00\x00\x00\x00Bb\x99\x99\x99\x99\x99\x98\
ALIVE   NOT     Cn\x10\x00\x00\x00\x00\x00B_fffffh\
ALIVE   UNK     C\x9dP\x00\x00\x00\x00\x00BV\xb333334\
DEAD    POOR    B\xfe\x00\x00\x00\x00\x00\x00B]fffffh\
DEAD    NOT     B<\x00\x00\x00\x00\x00\x00Bg\x80\x00\x00\x00\x00\x00\
DEAD    UNK     B\x89\x00\x00\x00\x00\x00\x00B8\xb333334\
                                                \
'''


def test_basic_loads(dataset, bytestring):
    """
    Verify reading dataset columns, name, labels, and formats.
    """
    library = xport.v56.loads(bytestring)
    member = next(iter(library.values()))
    assert dataset == member


def test_dumps_with_name_labels_and_formats(dataset, bytestring):
    """
    Verify writing dataset columns, name, labels, and formats.
    """
    assert bytestring == xport.v56.dumps(
        columns=dict(dataset),
        labels=dataset.labels,
        formats=dataset.formats,
        dataset_name=dataset.dataset_name,
        dataset_label=dataset.dataset_label,
    )


def test_dumps_numeric_type_conversion():
    """
    Verify numeric types convert to float when writing.
    """
    bytestring = xport.v56.dumps({'a': 1})
    dataset = xport.v56.loads(bytestring)
    assert isinstance(dataset['a'][0], float)


def test_dumps_text_type_conversion():
    """
    Verify text types are converted when writing.
    """
    # This test is interesting because b'\xff' is not valid Unicode.
    # https://en.wikipedia.org/wiki/ISO/IEC_8859-1
    b = b'\xff'
    s = b.decode('ISO-8859-1')
    assert xport.v56.dumps({'a': s}) == xport.v56.dumps({'a': b})


def test_dumps_invalid_types():
    """
    Verify non-numeric, non-text data will raise an error.
    """
    with pytest.raises(TypeError):
        xport.v56.dumps({'a': []})


def test_dumps_name_and_label_length_validation():
    """
    Verify variable and dataset name and label length.
    """
    # Names must be <= 8 characters.
    # Labels must be <= 40 characters.
    # SAS v8 Transport Files allow longer labels.
    with pytest.raises(ValueError):
        xport.v56.dumps({'a': 1}, daset_name='a' * 9)
    with pytest.raises(ValueError):
        xport.v56.dumps({'a': 1}, daset_label='a' * 41)
    with pytest.raises(ValueError):
        xport.v56.dumps({'a' * 9: 1})
    with pytest.raises(ValueError):
        xport.v56.dumps({'a': 1}, labels={'a': 'a' * 41})


def test_float_round_trip():
    """
    Verify a variety of random floats convert correctly.
    """
    random.seed(42)
    n = 10
    columns = {
        'near-zero': [random.uniform(-1e-6, 1e6) for i in range(n)],
        'large': [random.lognormvariate(1e300, 1) for i in range(n)],
    }
    blob = xport.v56.dumps(columns)
    library = xport.v56.loads(blob)
    member = next(iter(library.values()))
    for key, originals in columns.items():
        for a, b in zip(originals, member[key]):
            assert math.is_close(a, b, rel_tol=False)
