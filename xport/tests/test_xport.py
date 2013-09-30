import unittest
import xport
import math

XPORT_NUMERIC_TYPECODE=1
XPORT_CHARACTER_TYPECODE=2

class TestStringsDataset(unittest.TestCase):

    STRINGS_NUM_RECORDS = 2
    STRINGS_RECORD_WIDTH = 100

    def setUp(self):
        self.reader = xport.XportReader(open("xport/tests/strings.xpt", "rb"))
        self.expected_sequence = [
            "".join(chr(e) for e in range(1, 101)),
            "".join(chr(e) for e in range(101,128)),
            ]
        self.data_name = "X"

    def test_header(self):
        members = list(self.reader.members)
        assert len(members) == 1
        assert len(members[0].varlist) == 1
        assert members[0].varlist[0].type == XPORT_CHARACTER_TYPECODE
        assert members[0].varlist[0].length == self.STRINGS_RECORD_WIDTH

    def test_length(self):
        num_records = len([row for row in self.reader])
        assert num_records == self.STRINGS_NUM_RECORDS, "Wrong number of records: %d" % num_records

    def test_values(self):
        observed = [row[self.data_name] for row in self.reader]
        for obs, exp in zip(observed, self.expected_sequence):
            assert obs == exp, (obs, exp)

    def tearDown(self):
        del self.reader


class TestKnownValuesDataset(unittest.TestCase):

    KNOWN_VALUES_NUM_RECORDS = 2123
    KNOWN_VALUES_RECORD_WIDTH = 8

    def setUp(self):
        self.reader = xport.XportReader(open("xport/tests/known_values.xpt", "rb"))
        self.expected_sequence = [float(e) for e in range(-1000, 1001)] + \
            [math.pi ** e for e in range(-30, 31)] + \
            [-math.pi ** e for e in range(-30, 31)]
        self.data_name = "X"

    def test_header(self):
        members = list(self.reader.members)
        assert len(members) == 1
        assert len(members[0].varlist) == 1
        assert members[0].varlist[0].type == XPORT_NUMERIC_TYPECODE
        assert members[0].varlist[0].length == self.KNOWN_VALUES_RECORD_WIDTH

    def test_length(self):
        num_records = len([row for row in self.reader])
        assert num_records == self.KNOWN_VALUES_NUM_RECORDS, "Wrong number of records: %d" % num_records

    def test_values(self):
        observed = [row[self.data_name] for row in self.reader]
        for obs, exp in zip(observed, self.expected_sequence):
            self.assertAlmostEqual(obs, exp, places=30)

    def tearDown(self):
        del self.reader

class TestMultipleColumnsDataset(unittest.TestCase):

    MULTI_NUM_RECORDS = 20
    MULTI_RECORD_WIDTHS = (10, 8)

    def setUp(self):
        self.reader = xport.XportReader(open("xport/tests/multi.xpt", "rb"))
        self.expected_sequence_str = "This is one time where television really fails to capture the true excitement of a large squirrel predicting the weather.".split()
        self.expected_sequence_float = range(1, len(self.expected_sequence_str) + 1)
        self.data_name_str = "X"
        self.data_name_float = "Y"

    def test_header(self):
        members = list(self.reader.members)
        assert len(members) == 1
        assert len(members[0].varlist) == 2
        assert members[0].varlist[0].type == XPORT_CHARACTER_TYPECODE
        assert members[0].varlist[1].type == XPORT_NUMERIC_TYPECODE
        assert members[0].varlist[0].length == self.MULTI_RECORD_WIDTHS[0]
        assert members[0].varlist[1].length == self.MULTI_RECORD_WIDTHS[1]


    def test_length(self):
        num_records = len([row for row in self.reader])
        assert num_records == self.MULTI_NUM_RECORDS, "Wrong number of records: %d" % num_records

    def test_values(self):
        observed = [(row[self.data_name_str], row[self.data_name_float]) for row in self.reader]
        for (obs_s, obs_f), exp_s, exp_f in zip(observed, self.expected_sequence_str, self.expected_sequence_float):
            assert obs_s == exp_s, (obs_s, exp_s)
            assert obs_f == exp_f, (obs_f, exp_f)

    def tearDown(self):
        del self.reader



if __name__ == '__main__':
    unittest.main()
