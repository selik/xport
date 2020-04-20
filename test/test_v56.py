"""
Tests for XPT format from SAS versions 5 and 6.
"""

# Standard Library
from datetime import datetime

# Community Packages
import pytest

# Xport Modules
import xport
import xport.v56


@pytest.fixture(scope='module')
def library():
    """
    Create a 4-column, 6-row dataset with numbers and text.
    """
    ds = xport.Dataset(
        data={
            'VIT_STAT': ['ALIVE'] * 3 + ['DEAD'] * 3,
            'ECON': ['POOR', 'NOT', 'UNK'] * 2,
            'COUNT': [1216, 1761, 2517, 254, 60, 137],
            'TEMP': [98.6, 95.4, 86.7, 93.4, 103.5, 56.7],
        },
        name='ECON',
        label='Blank-padded dataset label',
        dataset_type='',
    )
    ds.created = ds.modified = datetime(2015, 11, 13, 10, 35, 8)
    ds.sas_os = 'W32_7PRO'
    ds.sas_version = '9.3'
    ds['VIT_STAT'].label = 'Vital status'
    ds['VIT_STAT'].format = '$5.'
    ds['VIT_STAT'].informat = xport.Informat()
    ds['VIT_STAT'].width = 8
    ds['ECON'].label = 'Economic status'
    ds['ECON'].format = xport.Format('$CHAR', 4, 0, xport.FormatAlignment.RIGHT)
    ds['ECON'].informat = xport.Informat()
    ds['ECON'].width = 8
    ds['COUNT'].label = 'Count'
    ds['COUNT'].format = 'comma8.0'
    ds['COUNT'].informat = xport.Informat()
    ds['COUNT'].width = 8
    ds['TEMP'].label = 'Temperature'
    ds['TEMP'].format = '8.1'
    ds['TEMP'].informat = xport.Informat()
    ds['TEMP'].width = 8
    return xport.Library(
        members=[ds],
        created=ds.created,
        modified=ds.modified,
        sas_os=ds.sas_os,
        sas_version=ds.sas_version,
    )


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
$       \x00\x05\x00\x00\x00\x00\x00\x00        \x00\x00\x00\x00\x00\x00\x00\x00\
\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\
\x00\x02\x00\x00\x00\x08\x00\x02ECON    Economic status                         \
$CHAR   \x00\x04\x00\x00\x00\x01\x00\x00        \x00\x00\x00\x00\x00\x00\x00\x08\
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
def library_header_bytestring(library_bytestring):
    """
    First 3 lines of the file.
    """
    return library_bytestring[:80 * 3]


@pytest.fixture(scope='module')
def dataset_bytestring(library_bytestring, library_header_bytestring):
    """
    Everything after the library header.
    """
    return library_bytestring[len(library_header_bytestring):]


@pytest.fixture(scope='module')
def member_header_bytestring(dataset_bytestring):
    """
    First 5 lines of the member dataset, plus namestrs and a final line.
    """
    index = 80 * 5 + 140 * 4 + 80 * 1
    return dataset_bytestring[:index]


@pytest.fixture(scope='module')
def observations_bytestring(dataset_bytestring, member_header_bytestring):
    """
    Observations are streamed in the file after the member header.
    """
    return dataset_bytestring[len(member_header_bytestring):]


@pytest.fixture(scope='module')
def namestr_bytestring(member_header_bytestring):
    """
    Namestrs start after 5 header lines and are 140 bytes long.
    """
    index = 80 * 5
    return member_header_bytestring[index:index + 140]


@pytest.fixture(scope='module')
def variable(library):
    """
    Example variable.
    """
    return library['ECON']['VIT_STAT']


@pytest.fixture(scope='module')
def dataset(library):
    """
    Example dataset.
    """
    return library['ECON']


class TestNamestr:

    def test_decode(self, variable, namestr_bytestring):
        v = xport.v56.Namestr.from_variable(variable, number=1, position=0)
        b = xport.v56.Namestr.from_bytes(namestr_bytestring)
        assert b == v

    def test_encode(self, variable, namestr_bytestring):
        namestr = xport.v56.Namestr.from_variable(variable, number=1, position=0)
        assert bytes(namestr) == namestr_bytestring


class TestMemberHeader:

    def test_decode(self, dataset, member_header_bytestring):
        d = xport.v56.MemberHeader.from_dataset(dataset)
        b = xport.v56.MemberHeader.from_bytes(member_header_bytestring)
        assert b == d

    def test_encode(self, dataset, member_header_bytestring):
        d = xport.v56.MemberHeader.from_dataset(dataset)
        assert bytes(d) == member_header_bytestring


class TestObservations:

    def test_decode(self, dataset, observations_bytestring):
        header = xport.v56.MemberHeader.from_dataset(dataset)
        namestrs = header.namestrs
        obs = xport.v56.Observations.from_bytes(observations_bytestring, namestrs)
        for got, expected in zip(obs, dataset.itertuples(index=False)):
            assert got == expected

    def test_encode(self, dataset, observations_bytestring):
        obs = xport.v56.Observations.from_dataset(dataset)
        i = 0
        for chunk in obs.to_bytes():
            assert chunk == observations_bytestring[i:i + len(chunk)], f'Index {i}'
            i += len(chunk)
        obs = xport.v56.Observations.from_dataset(dataset)
        assert bytes(obs) == observations_bytestring


class TestMember:

    def test_decode(self, dataset, dataset_bytestring):
        member = xport.v56.Member.from_bytes(dataset_bytestring)
        assert (member == dataset).all(axis=None)
        for name in dataset._metadata:
            assert getattr(member, name) == getattr(dataset, name), name
        for k, v in dataset.items():
            u = member[k]
            for name in v._metadata:
                assert getattr(u, name) == getattr(v, name), name

    def test_encode(self, dataset, dataset_bytestring):
        ds = xport.v56.Member(dataset)
        assert bytes(ds) == dataset_bytestring


class TestLibrary:

    def test_decode(self, library, library_bytestring):
        got = xport.v56.Library.from_bytes(library_bytestring)
        assert got == library

    def test_encode(self, library, library_bytestring):
        assert bytes(xport.v56.Library(library)) == library_bytestring

    def test_empty_library(self):
        """
        Verify dumps/loads of an empty library.
        """
        empty = xport.v56.Library()
        bytestring = bytes(empty)
        xport.v56.Library.from_bytes(bytestring)

    def test_empty_member(self):
        """
        Verify dumps/loads of a library with an empty member.
        """
        empty = xport.v56.Library([xport.v56.Member()])
        bytestring = bytes(empty)
        xport.v56.Library.from_bytes(bytestring)

    def test_no_observations(self):
        """
        Verify dumps/loads of 1 member, 1 variable, 0 observations.
        """
        library = xport.v56.Library({'x': xport.v56.Member({'a': []})})
        bytestring = bytes(library)
        assert xport.v56.Library.from_bytes(bytestring)


class TestEncode:
    """
    Verify various XPORT-encode features.
    """

    def test_dumps_numeric_type_conversion(self):
        """
        Verify numeric types convert to float when writing.
        """
        library = xport.v56.Library({'x': xport.v56.Member({'a': [1]})})
        bytestring = xport.v56.dumps(library)
        output = xport.v56.loads(bytestring)
        assert output['x']['a'].dtype.name == 'float64'


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

# def test_variable_numbers(self):
#     """
#     Verify enforcement of Variable numbers matching Dataset order.
#     """
#     v = xport.Variable(name='test_variable_numbers', dtype='float')
#     v.sas_variable_number = 10
#     with pytest.warns(UserWarning, match=r'SAS variable numbers'):
#         xport.Dataset({v.sas_name: v})

# def test_variable_positions(self):
#     """
#     Verify enforcement of Variable positions matching Dataset order.
#     """
#     v = xport.Variable(name='test_variable_positions', dtype='float')
#     v.sas_variable_position = 10
#     with pytest.warns(UserWarning, match=r'SAS variable positions'):
#         xport.Dataset({v.sas_name: v})

# def test_sas_name(self):
#     df = xport.Dataset()
#     df.sas_name = value = 'EXAMPLE1'
#     assert df.sas_name == value
#     with pytest.raises(ValueError):
#         df.sas_name = 'a' * 9
#     with pytest.raises(UnicodeEncodeError):
#         df.sas_name = '\N{snowman}'
#     with pytest.raises((TypeError, AttributeError)):
#         df.sas_name = 0

# def test_sas_label(self):
#     df = xport.Dataset()
#     df.sas_label = value = 'Example label'
#     assert df.sas_label == value
#     with pytest.raises(ValueError):
#         df.sas_label = 'a' * 41

# def test_sas_dataset_type(self):
#     df = xport.Dataset()
#     df.sas_dataset_type = value = 'DATA'
#     assert df.sas_dataset_type == value
#     with pytest.raises(ValueError):
#         df.sas_dataset_type = 'a' * 9

# def test_sas_dataset_created(self):
#     df = xport.Dataset()
#     df.sas_dataset_created = value = datetime.now()
#     assert df.sas_dataset_created == value
#     with pytest.raises(ValueError):
#         df.sas_dataset_created = datetime(1800, 1, 1)
#     with pytest.raises((TypeError, AttributeError)):
#         df.sas_dataset_created = '2000-Jan-01'

# def test_sas_dataset_modified(self):
#     df = xport.Dataset()
#     df.sas_dataset_modified = value = datetime(1920, 1, 1)
#     assert df.sas_dataset_modified == value
#     with pytest.raises(ValueError):
#         df.sas_dataset_modified = datetime(2100, 1, 1)
#     with pytest.raises(TypeError):
#         df.sas_dataset_modified = 1

# def test_sas_os(self):
#     df = xport.Dataset()
#     df.sas_os = value = 'MAC10.15'
#     assert df.sas_os == value
#     with pytest.raises(ValueError):
#         df.sas_os = 'a' * 9

# def test_sas_version(self):
#     df = xport.Dataset()
#     df.sas_os = value = '9.3'
#     assert df.sas_os == value
#     with pytest.raises(ValueError):
#         df.sas_version = 'a' * 9
