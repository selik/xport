"""
Command-line executable, run via ``python -m ...``.
"""
# Standard Library
import pathlib

from .cli import cli

if __name__ == '__main__':
    cli.main(prog_name=pathlib.Path(__file__).parent.name)
