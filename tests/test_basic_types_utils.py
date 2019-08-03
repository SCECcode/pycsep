from unittest import TestCase
from csep.utils import keys_in_dict


class TestBasicTypes(TestCase):
    def test_keys_in_dict_found(self):
        adict = {'_val': 1}
        keys = ['_val', 'val']
        out = keys_in_dict(adict, keys)
        self.assertListEqual(out, ['_val'])

