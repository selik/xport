#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Tests for the core CLI.
"""
# Standard Library
import re
import subprocess


def test_help():
    """
    Verify that we can show usage help.
    """
    cmd = 'python -m xport --help'
    argv = cmd.split()
    proc = subprocess.run(argv, text=True, capture_output=True)
    assert re.match(r'Usage: xport .*', proc.stdout.splitlines()[0])


def test_version():
    """
    Verify that we can show the verison.
    """
    cmd = 'python -m xport --version'
    argv = cmd.split()
    proc = subprocess.run(argv, text=True, capture_output=True)
    line, = proc.stdout.splitlines()
    assert re.match(r'xport v\S+', line)


def test_executable():
    """
    Verify that the command line executable installed correctly.
    """
    executable = 'xport --version'.split()
    imported = 'python -m xport --version'.split()
    proc1 = subprocess.run(executable, text=True, capture_output=True)
    proc2 = subprocess.run(imported, text=True, capture_output=True)
    assert proc1.stdout == proc2.stdout
