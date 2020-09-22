from unittest import TestCase
from csep.core.repositories import *


class TestFileSystem(TestCase):
    def setUp(self):
        self.test_dict = {'url': 'test',
                          'name': 'name'}

    def test_to_dict(self):
        test=FileSystem(**self.test_dict)
        self.assertEqual(test.to_dict(), self.test_dict)


    def test_from_dict(self):
        test=FileSystem.from_dict(self.test_dict)
        test_class=FileSystem(**self.test_dict)
        self.assertEqual(test, test_class)
