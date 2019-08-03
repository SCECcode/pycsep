import functools
import operator
import numpy

def keys_in_dict(adict, keys):
    """
    Searches adict and returns vals in keys found in adict.keys()

    Args:
        adict (dict): dictionary
        keys (list): list of keys to search

    Returns:
        out (list): iterable of keys found

    Example:

        adict = {'_val': 1}
        key_in_dict(adict, ['_val', 'val']) -> ['_val']

    """
    return [key for key in keys if key in adict.keys()]

def flat_map_to_ndarray(l):
    out = numpy.array(functools.reduce(operator.iconcat, l, []))
    return out

