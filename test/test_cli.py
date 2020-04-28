#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Tests for the core CLI.
"""
# Standard Library
import re
import subprocess
from io import StringIO

# Community Packages
import pandas as pd

# Xport Modules
import xport


def test_help():
    """
    Verify we can show usage help.
    """
    cmd = 'python -m xport --help'
    argv = cmd.split()
    proc = subprocess.run(argv, text=True, capture_output=True)
    assert re.match(r'Usage: xport .*', proc.stdout.splitlines()[0])


def test_version():
    """
    Verify we can show the verison.
    """
    cmd = 'python -m xport --version'
    argv = cmd.split()
    proc = subprocess.run(argv, text=True, capture_output=True)
    line, = proc.stdout.splitlines()
    assert re.match(r'xport, version \S+', line)


def test_executable():
    """
    Verify the command line executable installed correctly.
    """
    executable = 'xport --version'.split()
    imported = 'python -m xport --version'.split()
    proc1 = subprocess.run(executable, text=True, capture_output=True)
    proc2 = subprocess.run(imported, text=True, capture_output=True)
    assert proc1.stdout == proc2.stdout


def test_decode(library, library_bytestring):
    """
    Verify the command line executable can decode a library.
    """
    cmd = 'python -m xport -'
    argv = cmd.split()
    proc = subprocess.run(argv, capture_output=True, input=library_bytestring)
    fp = StringIO(proc.stdout.decode())
    df = pd.read_csv(fp)
    ds = xport.Dataset(df)
    assert (ds == next(iter(library.values()))).all(axis=None)


def test_output_file(library, library_bytestring, tmp_path):
    """
    Verify CLI can write output to a file.
    """
    filepath = tmp_path / 'tmp.csv'
    cmd = f'python -m xport - {filepath}'
    argv = cmd.split()
    subprocess.run(argv, capture_output=True, input=library_bytestring)
    with open(filepath) as f:
        df = pd.read_csv(f)
    ds = xport.Dataset(df)
    assert (ds == next(iter(library.values()))).all(axis=None)
