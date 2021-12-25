"""
Tests for XPT format from SAS versions 8 and 9.
"""

# Standard Library
import datetime

# Community Packages
import pytest

# Xport Modules
import xport.v89


@pytest.fixture()
def library_bytestring():
    """
    Dataset library encoded in SAS Transport Version 8/9 format.
    """
    return b'''\
HEADER RECORD*******LIBV8   HEADER RECORD!!!!!!!000000000000000000000000000000  \
SAS     SAS     SASLIB  6.06    bsd4.2                          11NOV21:22:33:22\
11NOV21:22:33:22                                                                \
HEADER RECORD*******MEMBV8  HEADER RECORD!!!!!!!000000000000000001600000000140  \
HEADER RECORD*******DSCPTV8 HEADER RECORD!!!!!!!000000000000000000000000000000  \
SAS     DATASET                         SASDATA 6.06    bsd4.2  11NOV21:22:33:22\
11NOV21:22:33:22                                                                \
HEADER RECORD*******NAMSTV8 HEADER RECORD!!!!!!!000000000600000000000000000000  \
\x00\x01\x00\x00\x00\x08\x00\x00Float   Floating Point                          BEST    \x00\x00\x00\x00\x00\x01\x00\x00BEST    \x00\x00\x00\x00\x00\x00\x00\x00Float                           \x00\x0e\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x01\x00\x00\x00\x08\x00\x01Double  Double Precision                        BEST    \x00\x00\x00\x00\x00\x01\x00\x00BEST    \x00\x00\x00\x00\x00\x00\x00\x08Double                          \x00\x10\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x01\x00\x00\x00\x08\x00\x02Long    Long Integer                            BEST    \x00\x00\x00\x00\x00\x01\x00\x00BEST    \x00\x00\x00\x00\x00\x00\x00\x10Long                            \x00\x0c\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x01\x00\x00\x00\x08\x00\x03Int     Integer                                 BEST    \x00\x00\x00\x00\x00\x01\x00\x00BEST    \x00\x00\x00\x00\x00\x00\x00\x18Int                             \x00\x07\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x01\x00\x00\x00\x08\x00\x04Byte    Byte. This is a very long label, probablBEST    \x00\x00\x00\x00\x00\x01\x00\x00BEST    \x00\x00\x00\x00\x00\x00\x00 Byte                            \x00P\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x02\x00\x00\x00\x03\x00\x05Str     String                                  $3      \x00\x00\x00\x00\x00\x00\x00\x00$3      \x00\x00\x00\x00\x00\x00\x00(Str                             \x00\x06\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00                                        \
HEADER RECORD*******LABELV8 HEADER RECORD!!!!!!!1                               \
\x00\x04\x00\x04\x00PByteByte. This is a very long label, probably longer than 40 characters, maybe even.                                                                      \
HEADER RECORD*******OBSV8   HEADER RECORD!!!!!!!000000000000000000000000000000  \x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00a  A\x10\x00\x00\x00\x00\x00\x00A\x10\x00\x00\x00\x00\x00\x00A\x10\x00\x00\x00\x00\x00\x00A\x10\x00\x00\x00\x00\x00\x00A\x10\x00\x00\x00\x00\x00\x001  A\x11\x99\x99\xa0\x00\x00\x00A\x11\x99\x99\x99\x99\x99\x9aA \x00\x00\x00\x00\x00\x00A \x00\x00\x00\x00\x00\x00A \x00\x00\x00\x00\x00\x00\xe2\x98\x83.\x00\x00\x00\x00\x00\x00\x00.\x00\x00\x00\x00\x00\x00\x00.\x00\x00\x00\x00\x00\x00\x00.\x00\x00\x00\x00\x00\x00\x00.\x00\x00\x00\x00\x00\x00\x00                                                                       \
'''  # E501: line too long


@pytest.fixture(scope='module')
def library():
    """
    Create a ``Library`` that matches the ``library_bytestring`` fixture.
    """
    ds = xport.Dataset(
        data={
            'Float': [0, 1, 9227469 / 8388608, float('nan')],
            'Double': [0, 1, 1.1, float('nan')],
            'Long': [0, 1, 2, None],
            'Int': [0, 1, 2, None],
            'Byte': [0, 1, 2, None],
            'Str': ['a', '1', '\N{snowman}'.encode().decode('ISO-8859-1'), ''],
        },
        name='DATASET',
        dataset_label='',
        dataset_type='',
    )
    ds.created = ds.modified = datetime.datetime(2021, 11, 11, 22, 33, 22)
    ds.sas_os = 'bsd4.2'
    ds.sas_version = '6.06'
    for name, variable in ds.items():
        variable.width = 8
        variable.format = 'BEST0.'
        variable.informat = 'BEST0.'
    ds['Float'].label = 'Floating Point'
    ds['Double'].label = 'Double Precision'
    ds['Long'].label = 'Long Integer'
    ds['Int'].label = 'Integer'
    ds['Byte'].label = (
        'Byte. This is a very long label, probably longer than 40 characters, maybe even.'
    )
    ds['Str'].label = 'String'
    ds['Str'].width = 3
    ds['Str'].format = '$30.'
    ds['Str'].informat = '$30.'
    return xport.Library(
        members=[ds],
        created=ds.created,
        modified=ds.modified,
        sas_os=ds.sas_os,
        sas_version=ds.sas_version,
    )


class TestLibrary:
    """
    Test SAS Transport v8/9 features.
    """

    def test_decode_labels(self, library, library_bytestring):
        """Test decoding long variable names and labels."""
        got = xport.v89.loads(library_bytestring)
        assert len(got['DATASET']['Byte'].label) > 40
        assert got['DATASET']['Byte'].label == library['DATASET']['Byte'].label
        assert ((got['DATASET'] == library['DATASET'])
                | (got['DATASET'].isna() & library['DATASET'].isna())).all(axis=None)

    @pytest.mark.skip("Not yet implemented")
    def test_encode_labels(self, library, library_bytestring):
        """Test encoding long variable names and labels."""
        got = xport.v89.dumps(library)
        assert got == library_bytestring

    @pytest.mark.skip("We need verified example data.")
    def test_decode_formats(self):
        """Test decoding long format descriptions."""
        assert False

    @pytest.mark.skip("We need verified example data.")
    def test_encode_formats(self):
        """Test encoding long format descriptions."""
        assert False
