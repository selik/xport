import unittest
import xport
import math

class TestStringsDataset(unittest.TestCase):

    STRINGS_NUM_RECORDS = 2

    def setUp(self):
        self.reader = xport.XportReader(open("xport/tests/strings.xpt", "rb"))
        self.expected_sequence = [
            "".join(chr(e) for e in range(1, 101)),
            "".join(chr(e) for e in range(101,128)),
            ]
        self.data_name = "X"

    def test_header(self):
        assert self.reader.header

    def test_length(self):
        num_records = len([row for row in self.reader])
        assert num_records == self.STRINGS_NUM_RECORDS, "Wrong number of records: %d" % num_records

    def test_values(self):
        observed = ([row[self.data_name] for row in self.reader])
        for obs, exp in zip(observed, self.expected_sequence):
            assert obs == exp, (obs, exp)

    def tearDown(self):
        del self.reader


class TestKnownValuesDataset(unittest.TestCase):

    KNOWN_VALUES_NUM_RECORDS = 2123

    def setUp(self):
        self.reader = xport.XportReader(open("xport/tests/known_values.xpt", "rb"))
        self.expected_sequence = [float(e) for e in range(-1000, 1001)] + \
            [math.pi ** e for e in range(-30, 31)] + \
            [-math.pi ** e for e in range(-30, 31)]
        self.data_name = "X"

    def test_header(self):
        assert self.reader.header

    def test_length(self):
        num_records = len([row for row in self.reader])
        assert num_records == self.KNOWN_VALUES_NUM_RECORDS, "Wrong number of records: %d" % num_records

    def test_values(self):
        observed = ([row[self.data_name] for row in self.reader])
        for obs, exp in zip(observed, self.expected_sequence):
            self.assertAlmostEqual(obs, exp, places=30)

    def tearDown(self):
        del self.reader



if __name__ == '__main__':
    unittest.main()
