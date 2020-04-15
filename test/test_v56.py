"""
Tests for XPT format from SAS versions 5 and 6.
"""

# Standard Library
from datetime import datetime

# Community Packages
import numpy as np
import pandas as pd
import pytest

# Xport Modules
import xport.v56


@pytest.fixture(scope='function')
def library_member():
    """
    Create a 4-column, 6-row dataset with numbers and text.
    """
    return xport.v56.Member(
        observations=pd.DataFrame({
            'VIT_STAT': ['ALIVE'] * 3 + ['DEAD'] * 3,
            'ECON': ['POOR', 'NOT', 'UNK'] * 2,
            'COUNT': [1216, 1761, 2517, 254, 60, 137],
            'TEMP': [98.6, 95.4, 86.7, 93.4, 103.5, 56.7],
        }),
        name='ECON',
        label='Blank-padded dataset label',
        created=datetime(2015, 11, 13, 10, 35, 8),
        modified=datetime(2015, 11, 13, 10, 35, 8),
        os='W32_7PRO',
        sas_version=(9, 3),
        namestrs={
            'VIT_STAT': {
                'label': 'Vital status',
                'format': {
                    'name': '$5'
                },
            },
            'ECON': {
                'label': 'Economic status',
                'format': {
                    'name': '$CHAR4'
                },
            },
            'COUNT': {
                'label': 'Count',
                'format': {
                    'name': 'comma8.0'
                },
            },
            'TEMP': {
                'label': 'Temperature',
                'format': {
                    'name': '8.1'
                },
            },
        },
    )


@pytest.fixture(scope='function')
def library_bytestring():
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


def test_basic_loads(library_member, library_bytestring):
    """
    Verify reading dataset columns, name, labels, and formats.
    """
    library = xport.v56.loads(library_bytestring)
    assert library[library_member.name] == library_member


def test_empty_library():
    """
    Verify dumps/loads of an empty library.
    """
    library = xport.v56.Library()
    b = bytes(library)
    with pytest.warns(UserWarning, match=r'No library members found'):
        assert xport.v56.Library.match(b) == library


def test_empty_member():
    """
    Verify dumps/loads of a library with an empty member.
    """
    library = xport.v56.Library(members={
        'x': xport.v56.Member(),
    })
    b = bytes(library)
    assert xport.v56.Library.match(b) == library


def test_no_observations():
    """
    Verify dumps/loads of 1 member, 1 variable, 0 observations.
    """
    library = xport.v56.Library(
        members={
            'x': xport.v56.Member(observations=pd.DataFrame({
                'a': [],
            })),
        }
    )
    b = bytes(library)
    assert xport.v56.Library.match(b) == library


def test_dumps_numeric_type_conversion():
    """
    Verify numeric types convert to float when writing.
    """
    library = xport.v56.Library(
        members={
            'x': xport.v56.Member(observations=pd.DataFrame({
                'a': [1],
            }), ),
        },
    )
    bytestring = xport.v56.dumps(library)
    output = xport.v56.loads(bytestring)
    assert output['x']['a'].dtype.name == 'float64'


@pytest.mark.skip('XPORT requires ASCII, not ISO-8859-1')
def test_dumps_text_type_conversion():
    """
    Verify text types are converted when writing.
    """
    # This test is interesting because b'\xff' is not valid Unicode.
    # https://en.wikipedia.org/wiki/ISO/IEC_8859-1
    b = b'\xff'
    s = b.decode('ISO-8859-1')
    library_b = xport.v56.Library(
        members={
            'x': xport.v56.Member(observations=pd.DataFrame({
                'a': [b],
            }), ),
        },
    )
    library_s = xport.v56.Library(
        members={
            'x': xport.v56.Member(observations=pd.DataFrame({
                'a': [s],
            }), ),
        },
    )
    assert xport.v56.dumps(library_s) == xport.v56.dumps(library_b)


@pytest.mark.skip('Debugging')
def test_dumps_invalid_types():
    """
    Verify non-numeric, non-text data will raise an error.
    """
    library = xport.v56.Library(
        members={
            'x': xport.v56.Member(observations=pd.DataFrame({'a': [object()]})),
        }
    )
    with pytest.raises(TypeError):
        xport.v56.dumps(library)
    # TODO: Investigate why it raises struct.error instead of TypeError.


@pytest.mark.skip('Length validation not yet implemented.')
def test_dumps_name_and_label_length_validation():
    """
    Verify variable and dataset name and label length.
    """
    # Names must be <= 8 characters.
    # Labels must be <= 40 characters.
    # SAS v8 Transport Files allow longer labels.
    with pytest.raises(ValueError):
        xport.v56.dumps(
            xport.v56.Library(
                members={
                    'x' * 9: xport.v56.Member(observations=pd.DataFrame({'a': []})),
                }
            )
        )
    with pytest.raises(ValueError):
        xport.v56.dumps(
            xport.v56.Library(
                members={
                    'x': xport.v56.Member(observations=pd.DataFrame({'a' * 9: []})),
                }
            )
        )
    # TODO: Test label length error checking.


def test_troublesome_text():
    """
    Some text patterns have been trouble in the past.
    """
    trouble = [
        "'<>",
    ]
    for issue in trouble:
        b = xport.v56.dumps(
            xport.v56.Library(
                members={
                    'x': xport.v56.Member(observations=pd.DataFrame({'a': trouble})),
                }
            )
        )
        dataset = xport.v56.loads(b)
        assert (dataset['x']['a'] == issue).all()


def test_overflow():
    """
    Some values are too large for IBM-format.
    """
    library = xport.v56.Library(
        members={
            'x': xport.v56.Member(observations=pd.DataFrame({'a': [np.finfo('float64').max]})),
        }
    )
    with pytest.raises(xport.v56.Overflow):
        xport.v56.dumps(library)


@pytest.mark.skip('Epsilon does not cause Underflow')
def test_underflow():
    """
    Some values are too small for IBM-format.
    """
    library = xport.v56.Library(
        members={
            'x': xport.v56.Member(observations=pd.DataFrame({'a': [np.finfo('float64').eps]})),
        }
    )
    with pytest.raises(xport.v56.Underflow):
        xport.v56.dumps(library)


def test_float_round_trip():
    """
    Verify a variety of random floats convert correctly.
    """
    np.random.seed(42)
    n = 10
    df = pd.DataFrame({
        'tiny': [np.random.uniform(-1e-6, 1e6) for i in range(n)],
        'large': [np.random.uniform(16**61, 16**62) for i in range(n)],
    })
    library = xport.v56.Library(members={
        'x': xport.v56.Member(observations=df),
    })
    b = xport.v56.dumps(library)
    approx = xport.v56.loads(b)
    for key, originals in library['x'].items():
        assert np.isclose(originals, approx['x'][key]).all()


@pytest.mark.skip('not implemented')
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
