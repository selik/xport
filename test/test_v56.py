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
import xport
import xport.v56


@pytest.fixture(scope='module')
def library():
    """
    Create a 4-column, 6-row dataset with numbers and text.
    """
    df = xport.Dataset(
        data={
            'VIT_STAT': ['ALIVE'] * 3 + ['DEAD'] * 3,
            'ECON': ['POOR', 'NOT', 'UNK'] * 2,
            'COUNT': [1216, 1761, 2517, 254, 60, 137],
            'TEMP': [98.6, 95.4, 86.7, 93.4, 103.5, 56.7],
        },
        sas_name='ECON',
        sas_label='Blank-padded dataset label',
    )
    df.sas_dataet_type = ''
    df.sas_dataet_created = df.sas_dataet_modified = datetime(2015, 11, 13, 10, 35, 8)
    df.sas_os = 'W32_7PRO'
    df.sas_version = '9.3'

    df['VIT_STAT'].sas_label = 'Vital status'
    df['VIT_STAT'].sas_format = '$5.'
    df['VIT_STAT'].sas_variable_length = 8
    df['ECON'].sas_label = 'Economic status'
    df['ECON'].sas_format = '$CHAR4.'
    df['ECON'].sas_variable_length = 8
    df['COUNT'].sas_label = 'Count'
    df['COUNT'].sas_format = 'comma8.0'
    df['TEMP'].sas_label = 'Temperature'
    df['TEMP'].sas_format = '8.1'

    return xport.Library(members={'ECON': df})


@pytest.fixture(scope='module')
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
        \x00\x05\x00\x00\x00\x00\x00\x00        \x00\x00\x00\x00\x00\x00\x00\x00\
\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\
\x00\x02\x00\x00\x00\x08\x00\x02ECON    Economic status                         \
CHAR    \x00\x04\x00\x00\x00\x01\x00\x00        \x00\x00\x00\x00\x00\x00\x00\x08\
\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\
\x00\x01\x00\x00\x00\x08\x00\x03COUNT   Count                                   \
COMMA   \x00\x08\x00\x00\x00\x00\x00\x00        \x00\x00\x00\x00\x00\x00\x00\x10\
\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\
\x00\x01\x00\x00\x00\x08\x00\x04TEMP    Temperature                             \
        \x00\x08\x00\x01\x00\x00\x00\x00        \x00\x00\x00\x00\x00\x00\x00\x18\
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


@pytest.fixture(scope='module')
def variable(library):
    """
    Example variable.
    """
    return library['ECON']['VIT_STAT']


@pytest.fixture(scope='module')
def namestr_bytestring(library_bytestring):
    """
    Example namestr bytestring.
    """
    index = 80 * 8
    return library_bytestring[index:index + 140]


class TestNamestr:

    def test_unpack(self, variable, namestr_bytestring):
        v = variable
        parsed = xport.v56.Namestr.unpack(namestr_bytestring)
        assert v.sas_name == parsed.sas_name
        assert v.sas_label == parsed.sas_label
        assert v.sas_variable_type == parsed.sas_variable_type
        assert v.sas_format == parsed.sas_format
        assert v.sas_iformat == parsed.sas_iformat
        assert v.sas_variable_number == parsed.sas_variable_number
        assert v.sas_variable_position == parsed.sas_variable_position
        assert v.sas_variable_length == parsed.sas_variable_length

    def test_pack(self, variable, namestr_bytestring):
        v = xport.v56.Namestr(variable)
        b = bytes(v)
        assert b == namestr_bytestring


# class TestMemberHeader:

#     def test_parse(self, library, library_bytestring):
#         h = library['ECON'].sas
#         parsed = next(xport.v56.MemberHeader.finditer(library_bytestring))
#         # assert h == parsed
#         assert h.name == parsed.name
#         assert h.label == parsed.label
#         assert h.type == parsed.type
#         assert h.created == parsed.created
#         assert h.modified == parsed.modified
#         assert h.os == parsed.os
#         assert h.version == parsed.version
#         left = [h.data[k].sas for k in h.data]
#         right = [parsed.data[k].sas for k in parsed.data]
#         for a, b in zip(left, right):
#             assert a.name == b.name
#             assert a.label == b.label
#             assert a.type == b.type
#             # assert a.format == b.format
#             # assert a.iformat == b.iformat
#             assert a.length == b.length
#             assert a.number == b.number
#             assert a.position == b.position

# def test_parse_observations(self, library, library_bytestring):
#     df = library['ECON']
#     h = next(xport.v56.MemberHeader.finditer(library_bytestring))
#     parsed = h.data
#     assert df.columns == parsed.columns
#     assert df.index == parsed.index
#     assert (df == parsed).all(axis=None)

# def test_basic_loads(library, library_bytestring):
#     """
#     Verify reading dataset columns, name, labels, and formats.
#     """
#     expected = library
#     got = xport.v56.loads(library_bytestring)
#     assert expected == got

# def test_empty_library():
#     """
#     Verify dumps/loads of an empty library.
#     """
#     library = xport.v56.Library()
#     b = bytes(library)
#     with pytest.warns(UserWarning, match=r'No library members found'):
#         assert xport.v56.Library.match(b) == library

# def test_empty_member():
#     """
#     Verify dumps/loads of a library with an empty member.
#     """
#     library = xport.v56.Library(members={
#         'x': xport.v56.Member(),
#     })
#     b = bytes(library)
#     assert xport.v56.Library.match(b) == library

# def test_no_observations():
#     """
#     Verify dumps/loads of 1 member, 1 variable, 0 observations.
#     """
#     library = xport.v56.Library(members={'x': xport.v56.Member({'a': []})})
#     b = bytes(library)
#     assert xport.v56.Library.match(b) == library

# def test_dumps_numeric_type_conversion():
#     """
#     Verify numeric types convert to float when writing.
#     """
#     library = xport.v56.Library(members={'x': xport.v56.Member({'a': [1]})})
#     bytestring = xport.v56.dumps(library)
#     output = xport.v56.loads(bytestring)
#     assert output['x']['a'].dtype.name == 'float64'

# @pytest.mark.skip('XPORT requires ASCII, not ISO-8859-1')
# def test_dumps_text_type_conversion():
#     """
#     Verify text types are converted when writing.
#     """
#     # This test is interesting because b'\xff' is not valid Unicode.
#     # https://en.wikipedia.org/wiki/ISO/IEC_8859-1
#     b = b'\xff'
#     s = b.decode('ISO-8859-1')
#     library_b = xport.v56.Library(members={'x': xport.v56.Member({'a': [b]})})
#     library_s = xport.v56.Library(members={'x': xport.v56.Member({'a': [s]})})
#     assert xport.v56.dumps(library_s) == xport.v56.dumps(library_b)

# @pytest.mark.skip('Debugging')
# def test_dumps_invalid_types():
#     """
#     Verify non-numeric, non-text data will raise an error.
#     """
#     library = xport.v56.Library(members={'x': xport.v56.Member({'a': [object()]})})
#     with pytest.raises(TypeError):
#         xport.v56.dumps(library)
#     # TODO: Investigate why it raises struct.error instead of TypeError.

# def test_dumps_name_and_label_length_validation():
#     """
#     Verify variable and dataset name and label length.
#     """
#     # Names must be <= 8 characters.
#     # Labels must be <= 40 characters.
#     # SAS v8 Transport Files allow longer labels.
#     with pytest.raises(ValueError):
#         xport.v56.dumps(
#             xport.v56.Library(
#                 members={
#                     'x' * 9: xport.v56.Member(observations=pd.DataFrame({'a': []})),
#                 }
#             )
#         )
#     with pytest.raises(ValueError):
#         xport.v56.dumps(
#             xport.v56.Library(
#                 members={
#                     'x': xport.v56.Member(observations=pd.DataFrame({'a' * 9: []})),
#                 }
#             )
#         )
#     # TODO: Test label length error checking.

# def test_troublesome_text():
#     """
#     Some text patterns have been trouble in the past.
#     """
#     trouble = [
#         "'<>",
#     ]
#     for issue in trouble:
#         b = xport.v56.dumps(
#             xport.v56.Library(
#                 members={
#                     'x': xport.v56.Member(observations=pd.DataFrame({'a': trouble})),
#                 }
#             )
#         )
#         dataset = xport.v56.loads(b)
#         assert (dataset['x']['a'] == issue).all()

# def test_overflow():
#     """
#     Some values are too large for IBM-format.
#     """
#     library = xport.v56.Library(
#         members={
#             'x': xport.v56.Member(observations=pd.DataFrame({'a': [np.finfo('float64').max]})),
#         }
#     )
#     with pytest.raises(xport.v56.Overflow):
#         xport.v56.dumps(library)

# @pytest.mark.skip('Epsilon does not cause Underflow')
# def test_underflow():
#     """
#     Some values are too small for IBM-format.
#     """
#     library = xport.v56.Library(
#         members={
#             'x': xport.v56.Member(observations=pd.DataFrame({'a': [np.finfo('float64').eps]})),
#         }
#     )
#     with pytest.raises(xport.v56.Underflow):
#         xport.v56.dumps(library)

# def test_float_round_trip():
#     """
#     Verify a variety of random floats convert correctly.
#     """
#     np.random.seed(42)
#     n = 10
#     df = pd.DataFrame({
#         'tiny': [np.random.uniform(-1e-6, 1e6) for i in range(n)],
#         'large': [np.random.uniform(16**61, 16**62) for i in range(n)],
#     })
#     library = xport.v56.Library(members={
#         'x': xport.v56.Member(observations=df),
#     })
#     b = xport.v56.dumps(library)
#     approx = xport.v56.loads(b)
#     for key, originals in library['x'].items():
#         assert np.isclose(originals, approx['x'][key]).all()

# @pytest.mark.skip('not implemented')
# def test_dumps_with_name_labels_and_formats(dataset, bytestring):
#     """
#     Verify writing dataset columns, name, labels, and formats.
#     """
#     assert bytestring == xport.v56.dumps(
#         columns=dict(dataset),
#         labels=dataset.labels,
#         formats=dataset.formats,
#         dataset_name=dataset.dataset_name,
#         dataset_label=dataset.dataset_label,
#     )
