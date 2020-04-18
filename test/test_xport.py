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


def test_set_member_metadata():
    """
    Verify the Pandas DataFrame accessor allows storing SAS metadata.
    """
    metadata = {
        'name': 'EXAMPLE',
        'label': 'Example',
        'type': 'DATA',
        'created': datetime(2020, 1, 1),
        'modified': datetime.now(),
        'os': 'MAC10.15',
        'version': '9.3',
    }
    df = pd.DataFrame()
    for name, value in metadata.items():
        setattr(df.sas, name, value)
    for name, value in metadata.items():
        assert getattr(df.sas, name) == value


def test_member_metadata_validation():
    """
    Verify the Pandas DataFrame accessor prevents invalid SAS metadata.
    """
    bad_value = {
        'name': 'a' * 9,
        'label': 'a' * 41,
        'type': 'a' * 9,
        'created': datetime(1800, 1, 1),
        'modified': datetime(2100, 1, 1),
        'os': 'a' * 9,
        'version': 'a' * 9,
    }
    bad_type = {
        'name': 0,
        'label': b'',
        'type': 0,
        'created': '2000-Jan-01',
        'modified': 1234,
        'os': 0,
        'version': (),
    }
    df = pd.DataFrame()
    for name, value in bad_value.items():
        with pytest.raises(ValueError):
            setattr(df.sas, name, value)
    for name, value in bad_type.items():
        with pytest.raises((TypeError, AttributeError)):
            setattr(df.sas, name, value)


def test_set_variable_metadata():
    """
    Verify the Pandas Series accessor allows storing SAS metadata.
    """
    c = pd.Series(['a', 'ab'])
    c.sas.name = 'EXAMPLE'
    c.sas.label = 'Example'
    c.sas.format = '$CHAR10.'
    c.sas.iformat = '$10.'
    assert c.sas.name == 'EXAMPLE'
    assert c.sas.label == 'Example'
    assert c.sas.type == xport.VariableType.CHARACTER
    assert c.sas.length == 2
    assert c.sas.format.name == 'CHAR'
    assert c.sas.format.length == 10
    assert c.sas.format.decimals is None
    assert pd.Series([1]).sas.type == xport.VariableType.NUMERIC


def test_variable_metadata_validation():
    """
    Verify the Pandas Series accessor prevents invalid SAS metadata.
    """
    c = pd.Series(dtype='object')
    with pytest.warns(UserWarning):
        c.sas.name = 'a' * 9
    with pytest.raises(ValueError):
        c.sas.name = 'a' * 33
    with pytest.warns(UserWarning):
        c.sas.label = 'a' * 41
    with pytest.raises(ValueError):
        c.sas.label = 'a' * 257
    with pytest.raises(ValueError):
        c.sas.format = ''
    with pytest.raises(ValueError):
        c.sas.format = '$abcdefghi1.'
    with pytest.raises(ValueError):
        c.sas.iformat = '1.2.3'
