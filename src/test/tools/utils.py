import unittest
from src.tools import utils


class UtilsTest(unittest.TestCase):

    ###########################
    # Test get_string_chunks  #
    ###########################
    def test_get_string_chunks(self):
        string = 'I am a random sentence.'
        lengths_list = [2, 5, 8, 10]
        for length in lengths_list:
            result = utils.get_string_chunks(string, length)
            for chunk in result:
                self.assertLessEqual(len(chunk), length)
