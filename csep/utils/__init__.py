import os
import functools
import operator
import subprocess

# Third-party imports
import numpy

def current_git_hash():
    root_dir = os.path.dirname(os.path.abspath(__file__))
    return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root_dir).decode().strip()

def git_remote_url():
    root_dir = os.path.dirname(os.path.abspath(__file__))
    remotes = subprocess.check_output(["git", "remote", "-v"], cwd=root_dir).decode().strip().split()
    return list(set([item for item in remotes if '.git' in item]))

def git_status_string():
    root_dir = os.path.dirname(os.path.abspath(__file__))
    return subprocess.check_output(["git", "status"], cwd=root_dir).decode().strip()

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

def join_struct_arrays(arrays):
    """
    Joins structured numpy arrays. Thanks: https://stackoverflow.com/questions/5355744/numpy-joining-structured-arrays

    Args:
        arrays:

    Returns:

    """
    sizes = numpy.array([a.itemsize for a in arrays])
    offsets = numpy.r_[0, sizes.cumsum()]
    n = len(arrays[0])
    joint = numpy.empty((n, offsets[-1]), dtype=numpy.uint8)
    for a, size, offset in zip(arrays, sizes, offsets):
        joint[:,offset:offset+size] = a.view(numpy.uint8).reshape(n,size)
    dtype = sum((a.dtype.descr for a in arrays), [])
    return joint.ravel().view(dtype)
