"""
Tests for the core interface.
"""

# Standard Library
from datetime import datetime

# Community Packages
import pytest

# Xport Modules
import xport


class TestInformat:
    """
    Verify parsing and display of input formats.
    """

    def test_init(self):
        """
        Verify data validation can be bypassed by init.
        """
        xport.Informat(name=None, length=None, decimals=None, vtype=None)

    def test_spec_character(self):
        """
        Verify parsing from an informat specification.
        """
        assert xport.Informat.from_spec('$3.') == xport.Informat(
            name='',
            length=3,
            decimals=0,
            vtype=xport.VariableType.CHARACTER,
        )
        assert xport.Informat.from_spec('$CHAR10.') == xport.Informat(
            name='CHAR',
            length=10,
            decimals=0,
            vtype=xport.VariableType.CHARACTER,
        )

    def test_spec_numeric(self):
        """
        Verify parsing from an informat specification.
        """
        assert xport.Informat.from_spec('10.2') == xport.Informat(
            name='',
            length=10,
            decimals=2,
            vtype=xport.VariableType.NUMERIC,
        )
        assert xport.Informat.from_spec('dollar26.') == xport.Informat(
            name='DOLLAR',
            length=26,
            decimals=0,
            vtype=xport.VariableType.NUMERIC,
        )

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
            # TODO: Determine vtype when unpacking an iformat.
            # assert form.vtype == cpy.vtype

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
            assert repr(iform) == f'Informat({s!r})'


class TestFormat:
    """
    Verify parsing and display of variable formats.
    """

    def test_init(self):
        """
        Verify data validation can be bypassed by init.
        """
        xport.Format(name=None, length=None, decimals=None, vtype=None, justify=None)

    def test_spec_character(self):
        """
        Verify parsing from an informat specification.
        """
        justify = xport.FormatAlignment.LEFT
        assert xport.Format.from_spec('$3.', justify) == xport.Format(
            name='',
            length=3,
            decimals=0,
            vtype=xport.VariableType.CHARACTER,
            justify=justify,
        )
        justify = xport.FormatAlignment.RIGHT
        assert xport.Format.from_spec('$CHAR10.', justify) == xport.Format(
            name='CHAR',
            length=10,
            decimals=0,
            vtype=xport.VariableType.CHARACTER,
            justify=justify,
        )

    def test_spec_numeric(self):
        """
        Verify parsing from an informat specification.
        """
        assert xport.Format.from_spec('10.2') == xport.Format(
            name='',
            length=10,
            decimals=2,
            vtype=xport.VariableType.NUMERIC,
        )
        assert xport.Format.from_spec('dollar26.') == xport.Format(
            name='DOLLAR',
            length=26,
            decimals=0,
            vtype=xport.VariableType.NUMERIC,
        )

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
            assert repr(iform) == f'Format({s!r})'


class TestVariableMetadata:
    """
    Verify set/get and validation of variable metadata.
    """

    def test_copy_metadata(self):
        """
        Verify ``Series`` methods that copy will keep SAS metadata.
        """
        v = xport.Variable(name='a', dtype='string')
        v.sas_name = 'Yo!'
        cpy = v.copy()
        assert isinstance(cpy, xport.Variable)
        assert v.sas_name == cpy.sas_name

    def test_sas_name(self):
        v = xport.Variable(name='a', dtype='string')
        assert v.sas_name == v.name
        with pytest.warns(UserWarning):
            v.sas_name = 'a' * 9
        with pytest.raises(ValueError):
            v.sas_name = 'a' * 33

    def test_sas_label(self):
        v = xport.Variable(name='a', dtype='string')
        assert v.sas_label == ''
        v.sas_label = value = 'Example'
        assert v.sas_label == value
        with pytest.warns(UserWarning):
            v.sas_label = 'a' * 41
        with pytest.raises(ValueError):
            v.sas_label = 'a' * 257

    def test_sas_variable_type(self):
        character = ['string', 'object']
        numeric = ['float', 'int', 'bool']
        invalid = ['datetime64[ns]']
        for dtype in character:
            v = xport.Variable(dtype=dtype)
            assert v.sas_variable_type == xport.VariableType.CHARACTER
        for dtype in numeric:
            v = xport.Variable(dtype=dtype)
            assert v.sas_variable_type == xport.VariableType.NUMERIC
        for dtype in invalid:
            with pytest.raises(TypeError):
                xport.Variable(dtype=dtype)

    def test_sas_variable_length(self):
        v = xport.Variable(['a', 'ab'])
        assert v.sas_variable_length == v.str.len().max()
        v.sas_variable_length = value = 10
        assert v.sas_variable_length == value
        with pytest.raises(ValueError):
            v.sas_variable_length = 0

    def test_sas_variable_number(self):
        v = xport.Variable(dtype='object')
        assert v.sas_variable_number is None
        v.sas_variable_number = value = 1
        assert v.sas_variable_number == value

    def test_sas_variable_position(self):
        v = xport.Variable(dtype='object')
        assert v.sas_variable_position is None
        v.sas_variable_position = value = 1
        assert v.sas_variable_position == value

    def test_sas_format(self):
        v = xport.Variable(dtype='object')
        v.sas_format = '$CHAR10.'
        assert v.sas_format.name == 'CHAR'
        assert v.sas_format.length == 10
        assert v.sas_format.decimals == 0
        with pytest.raises(ValueError):
            v.sas_format = ''
        with pytest.raises(ValueError):
            v.sas_format = '$abcdefghi1.'

    def test_sas_iformat(self):
        v = xport.Variable(dtype='object')
        v.sas_iformat = '10.2'
        assert v.sas_iformat.name == ''
        assert v.sas_iformat.length == 10
        assert v.sas_iformat.decimals == 2
        with pytest.raises(ValueError):
            v.sas_iformat = '1.2.3'


class TestDatasetMetadata:
    """
    Verify set/get and validation of dataset metadata.
    """

    def test_sas_name(self):
        df = xport.Dataset()
        df.sas_name = value = 'EXAMPLE1'
        assert df.sas_name == value
        with pytest.raises(ValueError):
            df.sas_name = 'a' * 9
        with pytest.raises(UnicodeEncodeError):
            df.sas_name = '\N{snowman}'
        with pytest.raises((TypeError, AttributeError)):
            df.sas_name = 0

    def test_sas_label(self):
        df = xport.Dataset()
        df.sas_label = value = 'Example label'
        assert df.sas_label == value
        with pytest.raises(ValueError):
            df.sas_label = 'a' * 41

    def test_sas_dataset_type(self):
        df = xport.Dataset()
        df.sas_dataset_type = value = 'DATA'
        assert df.sas_dataset_type == value
        with pytest.raises(ValueError):
            df.sas_dataset_type = 'a' * 9

    def test_sas_dataset_created(self):
        df = xport.Dataset()
        df.sas_dataset_created = value = datetime.now()
        assert df.sas_dataset_created == value
        with pytest.raises(ValueError):
            df.sas_dataset_created = datetime(1800, 1, 1)
        with pytest.raises((TypeError, AttributeError)):
            df.sas_dataset_created = '2000-Jan-01'

    def test_sas_dataset_modified(self):
        df = xport.Dataset()
        df.sas_dataset_modified = value = datetime(1920, 1, 1)
        assert df.sas_dataset_modified == value
        with pytest.raises(ValueError):
            df.sas_dataset_modified = datetime(2100, 1, 1)
        with pytest.raises(TypeError):
            df.sas_dataset_modified = 1

    def test_sas_os(self):
        df = xport.Dataset()
        df.sas_os = value = 'MAC10.15'
        assert df.sas_os == value
        with pytest.raises(ValueError):
            df.sas_os = 'a' * 9

    def test_sas_version(self):
        df = xport.Dataset()
        df.sas_os = value = '9.3'
        assert df.sas_os == value
        with pytest.raises(ValueError):
            df.sas_version = 'a' * 9

    def test_variable_numbers(self):
        """
        Verify enforcement of Variable numbers matching Dataset order.
        """
        v = xport.Variable(name='test_variable_numbers', dtype='float')
        v.sas_variable_number = 10
        with pytest.warns(UserWarning, match=r'SAS variable numbers'):
            xport.Dataset({v.sas_name: v})

    def test_variable_positions(self):
        """
        Verify enforcement of Variable positions matching Dataset order.
        """
        v = xport.Variable(name='test_variable_positions', dtype='float')
        v.sas_variable_position = 10
        with pytest.warns(UserWarning, match=r'SAS variable positions'):
            xport.Dataset({v.sas_name: v})


class TestLibrary:
    """
    Verify get/set of dataset library attributes and members.
    """

    def test_create_empty(self):
        xport.Library()

    def test_create_from_mapping(self):
        lib = xport.Library({'': xport.Dataset()})
        assert '' in lib
        with pytest.raises(ValueError):
            xport.Library({'x': xport.Dataset(sas_name='y')})

    def test_create_from_list(self):
        lib = xport.Library([xport.Dataset()])
        assert '' in lib
        with pytest.warns(UserWarning, match=r'More than one dataset named'):
            xport.Library([xport.Dataset(), xport.Dataset()])
