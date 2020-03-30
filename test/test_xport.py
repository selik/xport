"""
Tests for the core interface.
"""

# Xport Modules
import xport


def test_basic_round_trip_to_from_file(tmp_path):
    """
    Verify load/dump are reversible.
    """
    filepath = tmp_path / 'example.xpt'
    data = {
        'a': [1, 2],
        'b': [3, 4],
    }
    with open(filepath, 'wb') as f:
        xport.dump(data, f)
    with open(filepath, 'rb') as f:
        assert data == xport.load(f)


def test_basic_round_trip_to_from_string():
    """
    Verify loads/dumps are reversible.
    """
    data = {
        'a': [1, 2],
        'b': [3, 4],
    }
    s = xport.dumps(data)
    assert data == xport.loads(s)
