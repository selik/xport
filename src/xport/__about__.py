"""
Project metadata, such as the current version number.
"""
# Standard Library
from collections import namedtuple

__all__ = [
    '__version__',
]


class Version(namedtuple('Version', 'major minor patch')):
    def __str__(self):
        return f'v{".".join(map(str, self))}'


__version__ = Version(3, 0, 0)
