"""
Tests for the core interface.
"""

# Community Packages
import pandas as pd

# Xport Modules
import xport


def test_basic_round_trip_to_from_file(tmp_path):
    """
    Verify load/dump are reversible.
    """
    filepath = tmp_path / 'example.xpt'
    df = pd.DataFrame({
        'a': [1, 2],
        'b': [3, 4],
    })
    library = xport.Library(members={'x': xport.Member(observations=df)})
    with open(filepath, 'wb') as f:
        xport.dump(library, f)
    with open(filepath, 'rb') as f:
        assert library == xport.load(f)


def test_basic_round_trip_to_from_string():
    """
    Verify loads/dumps are reversible.
    """
    df = pd.DataFrame({
        'a': [1, 2],
        'b': [3, 4],
    })
    library = xport.Library(members={'x': xport.Member(observations=df)})
    s = xport.dumps(library)
    assert library == xport.loads(s)
