"""
Tests for the core interface.
"""

# Standard Library
from datetime import datetime

# Community Packages
import pandas as pd
import pytest

# Xport Modules
import xport


class TestVariableMetadata:
    """
    Verify set/get and validation of variable metadata.
    """

    def test_name(self):
        v = pd.Series(name='a', dtype='string').sas
        assert v.name == v.data.name
        with pytest.warns(UserWarning):
            v.name = 'a' * 9
        with pytest.raises(ValueError):
            v.name = 'a' * 33

    def test_label(self):
        v = pd.Series(name='a', dtype='string').sas
        assert v.label == ''
        v.label = value = 'Example'
        assert v.label == value
        with pytest.warns(UserWarning):
            v.label = 'a' * 41
        with pytest.raises(ValueError):
            v.label = 'a' * 257

    def test_type(self):
        assert pd.Series(dtype='string').sas.type == xport.VariableType.CHARACTER
        assert pd.Series(dtype='object').sas.type == xport.VariableType.CHARACTER
        assert pd.Series(dtype='float').sas.type == xport.VariableType.NUMERIC
        assert pd.Series(dtype='int').sas.type == xport.VariableType.NUMERIC
        with pytest.raises(TypeError):
            pd.Series(dtype='bool').sas

    def test_length(self):
        v = pd.Series(['a', 'ab']).sas
        assert v.length == v.data.str.len().max()
        v.length = value = 10
        assert v.length == value
        with pytest.raises(ValueError):
            v.length = 0

    def test_number(self):
        v = pd.Series(dtype='object').sas
        assert v.number is None
        v.number = value = 1
        assert v.number == value

    def test_position(self):
        v = pd.Series(dtype='object').sas
        assert v.position is None
        v.position = value = 1
        assert v.position == value

    def test_format(self):
        v = pd.Series(dtype='object').sas
        v.format = '$CHAR10.'
        assert v.format.name == 'CHAR'
        assert v.format.length == 10
        assert v.format.decimals is None
        with pytest.raises(ValueError):
            v.format = ''
        with pytest.raises(ValueError):
            v.format = '$abcdefghi1.'

    def test_iformat(self):
        v = pd.Series(dtype='object').sas
        v.iformat = '10.2'
        assert v.iformat.name == ''
        assert v.iformat.length == 10
        assert v.iformat.decimals == 2
        with pytest.raises(ValueError):
            v.iformat = '1.2.3'


class TestMemberMetadata:
    """
    Verify set/get and validation of dataset metadata.
    """

    def test_name(self):
        df = pd.DataFrame()
        df.sas.name = value = 'EXAMPLE1'
        assert df.sas.name == value
        with pytest.raises(ValueError):
            df.sas.name = 'a' * 9
        with pytest.raises(UnicodeEncodeError):
            df.sas.name = '\N{snowman}'
        with pytest.raises((TypeError, AttributeError)):
            df.sas.name = 0

    def test_label(self):
        df = pd.DataFrame()
        df.sas.label = value = 'Example label'
        assert df.sas.label == value
        with pytest.raises(ValueError):
            df.sas.label = 'a' * 41

    def test_type(self):
        df = pd.DataFrame()
        df.sas.type = value = 'DATA'
        assert df.sas.type == value
        with pytest.raises(ValueError):
            df.sas.type = 'a' * 9

    def test_created(self):
        df = pd.DataFrame()
        df.sas.created = value = datetime.now()
        assert df.sas.created == value
        with pytest.raises(ValueError):
            df.sas.created = datetime(1800, 1, 1)
        with pytest.raises((TypeError, AttributeError)):
            df.sas.created = '2000-Jan-01'

    def test_modified(self):
        df = pd.DataFrame()
        df.sas.modified = value = datetime(1920, 1, 1)
        assert df.sas.modified == value
        with pytest.raises(ValueError):
            df.sas.modified = datetime(2100, 1, 1)
        with pytest.raises(TypeError):
            df.sas.modified = 1

    def test_os(self):
        df = pd.DataFrame()
        df.sas.os = value = 'MAC10.15'
        assert df.sas.os == value
        with pytest.raises(ValueError):
            df.sas.os = 'a' * 9

    def test_version(self):
        df = pd.DataFrame()
        df.sas.os = value = '9.3'
        assert df.sas.os == value
        with pytest.raises(ValueError):
            df.sas.version = 'a' * 9

    def test_numbers(self):
        df = pd.DataFrame(columns=['a', 'b'], dtype='float')
        for i, k in enumerate(reversed(df)):
            df[k].sas.number = i
        with pytest.raises(ValueError):
            df.sas

    def test_positions(self):
        df = pd.DataFrame({'a': ['b']}, dtype='string')
        for k in df:
            df[k].sas.length = 10
            df[k].sas.position = 0
        with pytest.raises(ValueError):
            df.sas
