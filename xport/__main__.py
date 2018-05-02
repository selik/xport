#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
Read SAS XPORT/XPT files
and write CSV format.
'''

import argparse
import sys
import xport
import csv

def parse_args(*args, **kwargs):
    if sys.version_info < (3, 0):
        stdin = sys.stdin
        stdout = sys.stdout
    else:
        stdin = sys.stdin.buffer
        stdout = sys.stdout

    parser = argparse.ArgumentParser(description=__doc__)
    
    parser.add_argument('input',
                        type=argparse.FileType('rb'),
                        nargs='?',
                        default=stdin,
                        help='XPORT/XPT file to read, defaults to stdin')

    parser.add_argument('output',
                        type=argparse.FileType('w'),
                        nargs='?',
                        default=stdout,
                        help='CSV file to write, defaults to stdout')
    
    return parser.parse_args(*args, **kwargs)


if __name__ == '__main__':
    args = parse_args()

    with args.input:
        columns = xport.load(args.input)
    rows = map(tuple, zip(*columns.values()))

    with args.output:
        writer = csv.writer(args.output)
        writer.writerow(columns)
        for row in rows:
            writer.writerow(row)
