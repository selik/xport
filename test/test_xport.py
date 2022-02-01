"""
Tests for the core interface.
"""

# Standard Library
import math
import string
from io import BytesIO

# Community Packages
import pandas as pd
import pytest

# Xport Modules
import xport


class TestNaN:
    """
    Test special missing values.
    """

    def test_names(self):
        for c in '_' + string.ascii_uppercase:
            assert getattr(xport.NaN, c)

    def test_values(self):
        for c in '_' + string.ascii_uppercase:
            assert math.isnan(getattr(xport.NaN, c))


class TestInformat:
    """
    Verify parsing and display of input formats.
    """

    def test_spec_character(self):
        """
        Verify parsing from an informat specification.
        """
        assert xport.Informat.from_spec('$3.') == xport.Informat('$', 3, 0)
        assert xport.Informat.from_spec('$CHAR10.') == xport.Informat('$CHAR', 10, 0)

    def test_spec_numeric(self):
        """
        Verify parsing from an informat specification.
        """
        assert xport.Informat.from_spec('10.2') == xport.Informat('', 10, 2)
        assert xport.Informat.from_spec('dollar26.') == xport.Informat('DOLLAR', 26, 0)

    def test_struct_unpack(self):
        """
        Verify construction from struct tokens.
        """
        specs = [
            '$CHAR10.',
            'DOLLAR26.',
        ]
        for spec in specs:
            form = xport.Informat.from_spec(spec)
            b = bytes(form)
            cpy = xport.Informat.unpack(b)
            assert form.name == cpy.name
            assert form.length == cpy.length
            assert (form.decimals == cpy.decimals or form.decimals is None and cpy.decimals == 0)

    def test_display(self):
        """
        Verify string representation.
        """
        specs = [
            '$CHAR10.',
            'DOLLAR26.',
        ]
        for s in specs:
            iform = xport.Informat.from_spec(s)
            assert str(iform) == s


class TestFormat:
    """
    Verify parsing and display of variable formats.
    """

    def test_spec_character(self):
        """
        Verify parsing from an informat specification.
        """
        just = xport.FormatAlignment.LEFT
        assert xport.Format.from_spec('$3.', just) == xport.Format('$', 3, 0, just)
        just = xport.FormatAlignment.RIGHT
        assert xport.Format.from_spec('$CHAR10.', just) == xport.Format('$CHAR', 10, 0, just)

    def test_spec_numeric(self):
        """
        Verify parsing from an informat specification.
        """
        assert xport.Format.from_spec('10.2') == xport.Format('', 10, 2)
        assert xport.Format.from_spec('dollar26.') == xport.Format('DOLLAR', 26, 0)

    def test_display(self):
        """
        Verify string representation.
        """
        specs = [
            '$CHAR10.',
            'DOLLAR26.',
        ]
        for s in specs:
            iform = xport.Format.from_spec(s)
            assert str(iform) == s


class TestVariableMetadata:
    """
    Verify set/get and validation of variable metadata.
    """

    @staticmethod
    def compare_metadata(got, expected):
        for name in expected._metadata:
            assert getattr(got, name) == getattr(expected, name)
        assert got.vtype == expected.vtype

    def test_init(self):
        """
        Verify initialization.
        """
        v = xport.Variable(dtype='float')
        for name in v._metadata:
            getattr(v, name)  # Does not raise an error.

    def test_copy_metadata(self):
        """
        Verify ``Series`` methods that copy will keep SAS metadata.
        """
        v = xport.Variable(
            name='A',
            label='Alpha',
            format='$CHAR4.',
            dtype='string',
        )
        self.compare_metadata(v.copy(), v)
        self.compare_metadata(v.append(xport.Variable(['1'])), v)

    def test_format(self):
        v = xport.Variable(dtype='object')
        v.format = value = '$CHAR10.'
        assert v.format == xport.Format.from_spec(value)
        with pytest.raises(ValueError):
            v.format = ''
        with pytest.raises(ValueError):
            v.format = '$abcdefghi1.'

    def test_informat(self):
        v = xport.Variable(dtype='object')
        v.informat = value = '10.2'
        assert v.informat == xport.Informat.from_spec(value)
        with pytest.raises(ValueError):
            v.informat = '1.2.3'

    @pytest.mark.skip('Not implemented')
    def test_vtype(self):
        character = ['string', 'object']
        numeric = ['float', 'int', 'bool']
        invalid = ['datetime64[ns]']
        for dtype in character:
            v = xport.Variable(dtype=dtype)
            assert v.vtype == xport.VariableType.CHARACTER
        for dtype in numeric:
            v = xport.Variable(dtype=dtype)
            assert v.vtype == xport.VariableType.NUMERIC
        for dtype in invalid:
            with pytest.raises(TypeError):
                xport.Variable(dtype=dtype)


class TestDatasetMetadata:
    """
    Verify set/get and validation of dataset metadata.
    """

    @staticmethod
    def compare_metadata(got, expected):
        for name in expected._metadata:
            assert getattr(got, name) == getattr(expected, name)
        for k, v in expected.items():
            TestVariableMetadata.compare_metadata(got[k], v)
        assert (got.contents == expected.contents).all(axis=None)

    def test_init(self):
        """
        Verify initialization.
        """
        v = xport.Dataset()
        for name in v._metadata:
            getattr(v, name)  # Does not raise an error.

    def test_copy_metadata(self):
        """
        Verify ``DataFrame`` methods that copy will keep SAS metadata.
        """
        ds = xport.Dataset(
            data={
                'a': [1],
                'b': xport.Variable(['x'], label='Beta')
            },
            name='EXAMPLE',
            label='Example',
        )
        self.compare_metadata(ds.copy(), ds)
        self.compare_metadata(
            ds.append(pd.DataFrame({
                'a': [2],
                'b': ['y'],
            })),
            ds,
        )
        self.compare_metadata(pd.concat([ds, ds]), ds)

    def test_contents(self):
        """
        Verify variables metadata summary.
        """
        ds = xport.Dataset(
            data={
                'a': [1],
                'b': xport.Variable(['x'], label='Beta'),
                'c': [None],
            },
            name='EXAMPLE',
            label='Example',
        )
        ds['a'].vtype = xport.VariableType.NUMERIC
        ds['b'].vtype = xport.VariableType.CHARACTER
        got = ds.contents
        assert list(got.index) == [1, 2, 3]
        assert list(got['Label']) == ['', 'Beta', '']
        assert list(got['Type']) == ['Numeric', 'Character', '']


class TestLibrary:
    """
    Verify get/set of dataset library attributes and members.
    """

    def test_create_empty(self):
        xport.Library()

    def test_create_from_mapping(self):
        with pytest.warns(UserWarning, match=r'Set dataset name'):
            lib = xport.Library({'x': xport.Dataset()})
        assert 'x' in lib
        with pytest.raises(ValueError):
            xport.Library({'x': xport.Dataset(name='y')})

    def test_create_from_list(self):
        lib = xport.Library(xport.Dataset())
        assert None in lib
        with pytest.warns(UserWarning, match=r'More than one dataset named'):
            xport.Library([xport.Dataset(), xport.Dataset()])

    def test_create_from_dataframe(self):
        lib = xport.Library(pd.DataFrame())
        assert None in lib


class TestLegacy:
    """
    Verify deprecated API still works.
    """

    # Gotta stay backwards compatible.  The FDA has written docs.

    def test_from_columns(self, library):
        ds = next(iter(library.values()))
        mapping = {k: v for k, v in ds.items()}
        fp = BytesIO()
        with pytest.warns(DeprecationWarning):
            xport.from_columns(mapping, fp)
        fp.seek(0)
        result = next(iter(xport.v56.load(fp).values()))
        assert (result == ds).all(axis=None)

    def test_from_rows(self, library):
        ds = next(iter(library.values()))
        rows = list(ds.itertuples(index=None, name=None))
        fp = BytesIO()
        with pytest.warns(DeprecationWarning):
            xport.from_rows(rows, fp)
        fp.seek(0)
        result = next(iter(xport.v56.load(fp).values()))
        assert (result.values == ds.values).all(axis=None)

    def test_from_dataframe(self, library):
        ds = next(iter(library.values()))
        fp = BytesIO()
        with pytest.warns(DeprecationWarning):
            xport.from_dataframe(ds, fp)
        fp.seek(0)
        result = next(iter(xport.v56.load(fp).values()))
        assert (result == ds).all(axis=None)

    def test_to_rows(self, library, library_bytestring):
        ds = next(iter(library.values()))
        fp = BytesIO(library_bytestring)
        with pytest.warns(DeprecationWarning):
            result = xport.to_rows(fp)
        df = pd.DataFrame(result)
        assert (df.values == ds.values).all(axis=None)

    def test_to_columns(self, library, library_bytestring):
        ds = next(iter(library.values()))
        fp = BytesIO(library_bytestring)
        with pytest.warns(DeprecationWarning):
            result = xport.to_columns(fp)
        df = pd.DataFrame(result)
        assert (df == ds).all(axis=None)

    def test_to_numpy(self, library, library_bytestring):
        ds = next(iter(library.values()))
        fp = BytesIO(library_bytestring)
        with pytest.warns(DeprecationWarning):
            result = xport.to_numpy(fp)
        assert (result == ds.values).all(axis=None)

    def test_to_dataframe(self, library, library_bytestring):
        ds = next(iter(library.values()))
        fp = BytesIO(library_bytestring)
        with pytest.warns(DeprecationWarning):
            result = xport.to_dataframe(fp)
        assert (result == ds).all(axis=None)
