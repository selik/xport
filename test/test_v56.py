"""
Tests for XPT format from SAS versions 5 and 6.
"""

# Standard Library
import math
import string
from datetime import datetime

# Community Packages
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
        with pytest.warns(UserWarning, match=r'Converting column dtypes'):
            assert bytes(ds) == dataset_bytestring


class TestLibrary:

    def test_decode(self, library, library_bytestring):
        got = xport.v56.Library.from_bytes(library_bytestring)
        assert got == library

    def test_encode(self, library, library_bytestring):
        with pytest.warns(UserWarning, match=r'Converting column dtypes'):
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
        empty = xport.v56.Library(xport.v56.Member())
        bytestring = bytes(empty)
        xport.v56.Library.from_bytes(bytestring)

    def test_dataframe(self):
        lib = xport.Library(pd.DataFrame({'a': [1]}))
        with pytest.warns(UserWarning, match=r'Converting column dtypes'):
            result = xport.v56.loads(xport.v56.dumps(lib))
        assert (result[''] == lib[None]).all(axis=None)

    def test_no_observations(self):
        """
        Verify dumps/loads of 1 member, 1 variable, 0 observations.
        """
        with pytest.warns(UserWarning, match=r'Set dataset name'):
            library = xport.v56.Library({'x': xport.v56.Member({'a': []})})
        bytestring = bytes(library)
        assert xport.v56.Library.from_bytes(bytestring)


class TestIEEEtoIBM:

    def roundtrip(self, n):
        ibm = xport.v56.ieee_to_ibm(n)
        ieee = xport.v56.ibm_to_ieee(ibm)
        return round(ieee, 9)

    def test_overflow(self):
        with pytest.raises(xport.v56.Overflow):
            xport.v56.ieee_to_ibm(16**63)

    def test_underflow(self):
        with pytest.raises(xport.v56.Underflow):
            xport.v56.ieee_to_ibm(16**-66)

    def test_nan(self):
        n = float('nan')
        assert math.isnan(self.roundtrip(n))

    def test_special_missing_values(self):
        for c in '_' + string.ascii_uppercase:
            n = getattr(xport.NaN, c)
            assert chr(bytes(n)[0]) == c
            assert math.isnan(self.roundtrip(n))

    def test_zero(self):
        assert self.roundtrip(0) == 0

    def test_small_magnitude_integers(self):
        for i in range(-1000, 1000):
            assert self.roundtrip(i) == i

    def test_small_magnitude_floats(self):
        for i in range(-10, 10):
            i /= 1000
            assert self.roundtrip(i) == i

    def test_large_magnitude_floats(self):
        n = int(1e9)
        for i in range(n, n + 100):
            assert self.roundtrip(i) == i

    def test_large_magnitude_floats_with_fraction(self):
        offset = 1e9
        for i in range(100):
            i /= 1e9
            x = i + offset
            assert self.roundtrip(x) == x

    def test_very_small_magnitude_floats(self):
        for i in range(-10, 10):
            i /= 1e6
            assert self.roundtrip(i) == i


class TestEncode:
    """
    Verify various XPORT-encode features.
    """

    def dump_and_load(self, library):
        bytestring = xport.v56.dumps(library)
        return xport.v56.loads(bytestring)

    def test_numeric_type_conversion(self):
        """
        Verify numeric types convert to float when writing.
        """
        numerics = [
            1,
            True,
        ]
        for x in numerics:
            with pytest.warns(UserWarning, match=r'Converting column dtypes'):
                library = xport.Library({'A': xport.Dataset({'x': [x]})})
                output = self.dump_and_load(library)
                assert output['A']['x'].dtype.name == 'float64'
                assert output['A']['x'].iloc[0] == 1.0

    def test_text_null(self):
        # https://github.com/selik/xport/issues/44
        df = pd.DataFrame({
            'a': pd.Series([None], dtype='string'),
            'b': [0],  # Avoid issue #46 by including a numeric column.
        })
        library = self.dump_and_load(df)
        assert list(next(iter(library.values()))['a']) == ['']

    @pytest.mark.skip("Pandas coerces everything to a string")
    def test_invalid_types(self):
        """
        Verify invalid types raise errors on write.
        """
        invalid = [
            b'\x00',
            object(),
            (1, 2, 3),
        ]
        for bad in invalid:
            with pytest.warns(UserWarning, match=r'Converting column dtypes'):
                with pytest.raises(TypeError):
                    library = xport.Library(xport.Dataset({'a': [bad]}))
                    xport.v56.dumps(library)

    def test_invalid_values(self):
        invalid = [
            '\N{snowman}',
        ]
        for bad in invalid:
            library = xport.Library(xport.Dataset({'a': [bad]}))
            with pytest.raises(ValueError):
                with pytest.warns(UserWarning, match=r'Converting column dtypes'):
                    xport.v56.dumps(library)

    def test_dumps_name_and_label_length_validation(self):
        """
        Verify variable and dataset name and label length.
        """
        # Names must be <= 8 characters.
        # Labels must be <= 40 characters.
        # SAS v8 Transport Files allow longer labels.
        invalid = [
            xport.Library(xport.Dataset(), sas_version='a' * 9),
            xport.Library(xport.Dataset(name='a' * 9)),
            xport.Library(xport.Dataset(label='a' * 41)),
            xport.Library(xport.Dataset({'a' * 9: [1.0]})),
            xport.Library(xport.Dataset({'a': xport.Variable([1.0], label='a' * 41)})),
        ]
        for bad_metadata in invalid:
            with pytest.raises(ValueError):
                xport.v56.dumps(bad_metadata)

    def test_troublesome_text(self):
        """
        Some text patterns have been trouble in the past.
        """
        trouble = xport.Variable(["'<>"], dtype='string')
        dataset = xport.Dataset({'a': trouble}, name='trouble')
        library = xport.Library(dataset)
        with pytest.warns(UserWarning, match=r'Converting column dtypes'):
            assert self.dump_and_load(library) == library

    def test_dataset_created(self):
        invalid = datetime(1800, 1, 1)
        ds = xport.Dataset(created=invalid)
        assert ds.created == invalid
        with pytest.raises(ValueError):
            xport.v56.dumps(ds)
        with pytest.raises((TypeError, AttributeError)):
            ds = xport.Dataset(created='2000-Jan-01')
            xport.v56.dumps(ds)

    def test_dataset_modified(self):
        invalid = datetime(2100, 1, 1)
        ds = xport.Dataset(modified=invalid)
        assert ds.modified == invalid
        with pytest.raises(ValueError):
            xport.v56.dumps(ds)
        with pytest.raises((TypeError, AttributeError)):
            ds = xport.Dataset(modified=1)
            xport.v56.dumps(ds)


class TestTextEncoding:
    """
    Validate writing and reading different text encodings.
    """

    def test_default(self):
        """
        Test handling non-ASCII Windows-1252 characters.
        """
        enye = '\u00f1'
        enye.encode('Windows-1252')
        with pytest.raises(UnicodeEncodeError):
            enye.encode('ascii')

        # Metadata defaults to ASCII
        with pytest.raises(UnicodeEncodeError):
            xport.v56.dumps(xport.Dataset(name=enye))
        # TODO: Ensure failure to load a non-ASCII dataset name.

        # Data defaults to Windows-1252
        example = xport.Dataset({'v': xport.Variable([enye])}, name='d')
        bytestring = xport.v56.dumps(example)
        library = xport.v56.loads(bytestring)
        assert library['d']['v'][0] == enye

    def test_ascii(self):
        """
        Test rejecting non-ASCII characters.
        """
        enye = '\u00f1'
        with pytest.raises(UnicodeEncodeError):
            enye.encode('ascii')

        # We're only testing the data, not metadata.
        enye.encode(xport.v56.TEXT_DATA_ENCODING)
        example = xport.Dataset({'v': xport.Variable([enye])})
        with pytest.raises(UnicodeEncodeError):
            with xport.v56._encoding(data='ascii'):
                xport.v56.dumps(example)
        bytestring = xport.v56.dumps(example)
        with pytest.raises(UnicodeDecodeError):
            with xport.v56._encoding(data='ascii'):
                xport.v56.loads(bytestring)

    def test_unicode(self):
        """
        Test encoding and decoding with UTF-8.
        """
        comet = '\u2604'
        with pytest.raises(UnicodeEncodeError):
            comet.encode('ascii')

        # Data
        example = xport.Dataset({'v': xport.Variable([comet])})
        with pytest.raises(UnicodeEncodeError):
            xport.v56.dumps(example)
        with xport.v56._encoding(data='utf-8'):
            bytestring = xport.v56.dumps(example)
            library = xport.v56.loads(bytestring)
        assert library['']['v'][0] == comet

        # Metadata
        example = xport.Dataset(name=comet)
        with xport.v56._encoding(metadata='utf-8'):
            bytestring = xport.v56.dumps(example)
            library = xport.v56.loads(bytestring)
        assert comet in library


class TestEncodeLabels:
    """
    Validate writing data set and variable labels.
    """

    def test_dataset(self):
        """
        Data set label, no variable label.
        """
        example = xport.Dataset(
            {'a': xport.Variable(dtype=object)},
            name='TEST',
            label='This is a test',
        )
        bytestring = xport.v56.dumps(example)
        library = xport.v56.loads(bytestring)
        assert example.label == library['TEST'].label == 'This is a test'

    def test_dataset_deprecated(self):
        """
        The ``Dataset.label`` attribute is now ``Dataset.label``.
        """
        with pytest.warns(DeprecationWarning, match=r'dataset_label'):
            example = xport.Dataset(name='TEST', dataset_label='This is a test')
        bytestring = xport.v56.dumps(example)
        library = xport.v56.loads(bytestring)
        assert example.label == library['TEST'].label == 'This is a test'

    def test_variable(self):
        """
        Only a variable lable, no data set label.
        """
        example = xport.Dataset({'a': xport.Variable(label='b', dtype=object)}, name='TEST')
        bytestring = xport.v56.dumps(example)
        library = xport.v56.loads(bytestring)
        assert example.label is None
        assert library['TEST'].label == ''
        assert example['a'].label == library['TEST']['a'].label == 'b'

    def test_both(self):
        """
        Both data set label and variable label.
        """
        example = xport.Dataset(
            data={'a': xport.Variable(label='b', dtype=object)},
            name='TEST',
            label='Test',
        )
        bytestring = xport.v56.dumps(example)
        library = xport.v56.loads(bytestring)
        assert example.label == library['TEST'].label == 'Test'
        assert example['a'].label == library['TEST']['a'].label == 'b'

    def test_neither(self):
        """
        Neither data set label nor variable label.
        """
        example = xport.Dataset({'a': xport.Variable(dtype=object)}, name='TEST')
        bytestring = xport.v56.dumps(example)
        library = xport.v56.loads(bytestring)
        assert example.label is None
        assert library['TEST'].label == ''
        assert example['a'].label is None
        assert library['TEST']['a'].label == ''
