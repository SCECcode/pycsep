from csep.core.system import *

from unittest import TestCase


class TestSystem(TestCase):
    def setUp(self):
        self.test_dict = {
                'name':'test',
                'url':'test',
                'hostname':'hostname',
                'email':'email',
                'max_cores':'max_cores',
                'mem_per_node':'mem_per_node'
            }

    def test_from_dict(self):
        test = System.from_dict(self.test_dict)
        test_class = System(**self.test_dict)
        self.assertEqual(test, test_class)


    def test_to_dict(self):
            test = System(**self.test_dict)
            self.assertEqual(test.to_dict(), self.test_dict)

class TestSlurmSystem(TestCase):
    def setUp(self):
        self.test_dict = {
                'name':'test',
                'url':'test',
                'hostname':'hostname',
                'email':'email',
                'max_cores':'max_cores',
                'mem_per_node':'mem_per_node',
                'mpj_home': 'test',
                'partition': 'partition'
            }
    def test_to_dict(self):
        test = SlurmSystem(**self.test_dict)
        self.assertEqual(test.to_dict(), self.test_dict)

    def test_from_dict(self):
        test = SlurmSystem.from_dict(self.test_dict)
        test_class = SlurmSystem(**self.test_dict)
        self.assertEqual(test, test_class)