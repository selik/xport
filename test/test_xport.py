'''
Tests for ``xport.py``.
'''

import csv
from collections import OrderedDict, namedtuple
import glob
import math
import os
import string
import unittest
from io import BytesIO
import xport



class TestCSVs(unittest.TestCase):

    def convert_types(self, row):
        typed = []
        for s in row:
            try:
                value = int(s)
            except ValueError:
                try:
                    value = float(s)
                except ValueError:
                    try:
                        value = s.decode('utf-8')
                    except UnicodeDecodeError:
                        value = s
                    except AttributeError: # Python 3, no need to convert
                        value = s
            typed.append(value)
        return tuple(typed)


    def test_csvs(self):
        for csvfile in glob.glob('test/data/*.csv'):
            directory, filename = os.path.split(csvfile)
            xptfile = os.path.join(directory, filename[:-4] + '.xpt')

            with open(csvfile) as fcsv, open(xptfile, 'rb') as fxpt:
                csvreader = csv.reader(fcsv)
                xptreader = xport.Reader(fxpt)

                self.assertEqual(tuple(next(csvreader)), xptreader.fields)

                values = (self.convert_types(row) for row in csvreader)
                list(map(self.assertEqual, values, xptreader))



class TestStringsDataset(unittest.TestCase):


    def test_header(self):
        with open('test/data/strings.xpt', 'rb') as f:
            reader = xport.Reader(f)
            x, = reader._variables

            assert reader.fields == ('X',)

            assert x.name == 'X'
            assert x.numeric == False
            assert x.position == 0
            assert x.size == 100


    def test_length(self):
        with open('test/data/strings.xpt', 'rb') as f:
            assert len(list(xport.Reader(f))) == 2


    def test_values(self):
        with open('test/data/strings.xpt', 'rb') as f:
            it = (row.X for row in xport.Reader(f))
            assert next(it) == ''.join(chr(i) for i in range(1, 101))
            assert next(it) == ''.join(chr(i) for i in range(101,128))




class TestKnownValuesDataset(unittest.TestCase):


    def test_header(self):
        with open('test/data/known_values.xpt', 'rb') as f:
            reader = xport.Reader(f)
            x, = reader._variables

            assert reader.fields == ('X',)

            assert x.name == 'X'
            assert x.numeric == True
            assert x.position == 0
            assert x.size == 8


    def test_length(self):
        with open('test/data/known_values.xpt', 'rb') as f:
            assert len(list(xport.Reader(f))) == 2123


    def test_values(self):
        with open('test/data/known_values.xpt', 'rb') as f:
            it = (row.X for row in xport.Reader(f))
            for value in [float(e) for e in range(-1000, 1001)]:
                assert value == next(it)
            for value in [math.pi ** e for e in range(-30, 31)]:
                self.assertAlmostEqual(value, next(it), places=30)
            for value in [-math.pi ** e for e in range(-30, 31)]:
                self.assertAlmostEqual(value, next(it), places=30)



class TestMultipleColumnsDataset(unittest.TestCase):


    def test_header(self):
        with open('test/data/multi.xpt', 'rb') as f:
            reader = xport.Reader(f)
            x, y = reader._variables

            assert reader.fields == ('X', 'Y')

            assert x.name == 'X'
            assert x.numeric == False
            assert x.position == 0
            assert x.size == 10

            assert y.name == 'Y'
            assert y.numeric == True
            assert y.position == 10
            assert y.size == 8


    def test_length(self):
        with open('test/data/multi.xpt', 'rb') as f:
            assert len(list(xport.Reader(f))) == 20


    def test_values(self):
        strings = '''
            This is one time where television really fails to capture
            the true excitement of a large squirrel predicting the weather.
            '''.split()
        with open('test/data/multi.xpt', 'rb') as f:
            for (i, s), (x, y) in zip(enumerate(strings, 1), xport.Reader(f)):
                assert (x, y) == (s, i)


class TestIEEEtoIBM(unittest.TestCase):

    def roundtrip(self, n):
        ibm = xport.ieee_to_ibm(n)
        ieee = xport.ibm_to_ieee(ibm)
        return round(ieee, 9)

    def test_overflow(self):
        with self.assertRaises(xport.Overflow):
            xport.ieee_to_ibm(16 ** 63)

    def test_underflow(self):
        with self.assertRaises(xport.Underflow):
            xport.ieee_to_ibm(16 ** -66)

    def test_nan(self):
        n = float('nan')
        self.assertTrue(math.isnan(self.roundtrip(n)))

    def test_zero(self):
        self.assertEqual(0, self.roundtrip(0))

    def test_small_magnitude_integers(self):
        for i in range(-1000, 1000):
            self.assertEqual(i, self.roundtrip(i))

    def test_small_magnitude_floats(self):
        for i in range(-10, 10):
            i /= 1000
            self.assertEqual(i, self.roundtrip(i))

    def test_large_magnitude_floats(self):
        n = int(1e9)
        for i in range(n, n + 100):
            self.assertEqual(i, self.roundtrip(i))

    def test_large_magnitude_floats_with_fraction(self):
        offset = 1e9
        for i in range(100):
            i /= 1e9
            x = i + offset
            self.assertEqual(x, self.roundtrip(x))

    def test_very_small_magnitude_floats(self):
        for i in range(-10, 10):
            i /= 1e6
            self.assertEqual(i, self.roundtrip(i))



class TestFromToColumns(unittest.TestCase):

    def roundtrip(self, mapping):
        fp = BytesIO()
        xport.from_columns(mapping, fp)
        fp.seek(0)
        duplicate = xport.to_columns(fp)
        for label, column in mapping.items():
            for a, b in zip(column, duplicate[label]):
                self.assertEqual(a, b)

    def test_roundtrip_dict(self):
        columns = {'whole': list(range(10)),
                   'fraction': [i ** -0.5 for i in range(1, 11)],
                   'letters': list(string.ascii_letters[:10]),
                   'words': '''apple banana cantaloupe domino elephant
                               frog guantanamo hooli igloo jarjar
                            '''.split()}
        self.roundtrip(columns)



class TestDumpRows(unittest.TestCase):

    def roundtrip(self, rows):
        fp = BytesIO()
        xport.from_rows(rows, fp)
        fp.seek(0)
        duplicate = xport.to_rows(fp)
        self.assertEqual(rows, duplicate)

    def test_roundtrip_tuples(self):
        rows = [('life', 1),
                ('universe', 3.14),
                ('everything', 42)]
        self.roundtrip(rows)

    def test_rows_as_namedtuple(self):
        Row = namedtuple('Row', 's n')
        rows = [Row(s='life', n=1.0),
                Row(s='universe', n=3.14),
                Row(s='everything', n=42.0)]
        self.roundtrip(rows)

    def test_rows_as_ordered_dict(self):
        rows = [OrderedDict([('s', 'life'), ('n', 1.0)]),
                OrderedDict([('s', 'universe'), ('n', 3.14)]),
                OrderedDict([('s', 'everything'), ('n', 42.0)])]
        fp = BytesIO()
        xport.from_rows(rows, fp)
        fp.seek(0)
        dup = [t._asdict() for t in xport.to_rows(fp)]
        self.assertEqual(rows, dup)



if __name__ == '__main__':
    unittest.main()


