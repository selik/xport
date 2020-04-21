"""
Project metadata, such as the current version number.
"""

# Single-sourcing the package version from the installed package.  This
# has the caveat that the imported version may be different from the
# installed version in some rare circumstances, such as when current
# working directory is the ``src`` folder and the package install did
# not use the ``-e`` or ``--editable`` flag.  Most usage will be from
# the root directory of the project or from outside the project.
# https://packaging.python.org/guides/single-sourcing-package-version/

# Standard Library
import pathlib
from collections import namedtuple

# Community Packages
from pkg_resources import DistributionNotFound, get_distribution

__all__ = [
    '__version__',
]


class Version(namedtuple('Version', 'major minor patch')):
    """
    Version information.
    """

    @classmethod
    def parse(cls, s):
        return Version(*map(int, s.split('.')))

    def __str__(self):
        return '.'.join(map(str, self))


project = pathlib.Path(__file__).parent.name
try:
    __version__ = Version.parse(get_distribution(project).version)
except DistributionNotFound:
    __version__ = None
