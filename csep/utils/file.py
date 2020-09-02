import os
import shutil
import collections
from tempfile import mkdtemp
from contextlib import contextmanager

@contextmanager
def TemporaryDirectory(suffix='', prefix=None, dir=None):
    """
    Creates temporary directory using context managers.

    :param
    :type suffix: str
    :param prefix
    :type prefix: str
    :type dir
    :param dir: str
    """
    dir_name = mkdtemp(suffix=suffix, prefix=prefix, dir=dir)
    try:
        yield dir_name
    finally:
        try:
            shutil.rmtree(dir_name)
        except OSError as e:
            # ENOENT - no such file or directory
            if e.errno != e.errno.ENOENT:
                raise e

def mkdirs(path, mode=0o755):
    """
    Creates the directory specified by path, creating intermediate directories
    as necessary. If directory already exists, this is a no-op.

    :param path: The directory to create
    :type path: str
    :param mode: The mode to give to the directory e.g. 0o755, ignores umask
    :type mode: int
    """
    try:
        o_umask = os.umask(0)
        os.makedirs(path, mode)
    except OSError:
        if not os.path.isdir(path):
            raise
    finally:
        os.umask(o_umask)

def copy_file(src, dest):
    """
    Copies a file on the system.

    Args:
        src (str): absolute path to file object
        dest (str):

    Returns:

    """
    new_path = shutil.copy(src, dest)
    return new_path

def file_exists(path):
    """
    Checks if file exists and returns boolean

    Args:
        path: filename

    Returns:

    """


def get_relative_path(abs_path):
    """
    Given an absolute path return the relative path.

    Args:
        abs_path (str):

    Returns:
        relative path (str)

    Example:

        abs_path = '/home/csep/test_dir/test.file'
        returns =  'test_dir/test.file'

    """
    basename = os.path.basename(abs_path)
    dir_name = os.path.basename(os.path.dirname(abs_path))
    return os.path.join(dir_name, basename)

def iterable(arg):
    return (
        isinstance(arg, collections.Iterable)
        and not isinstance(arg, str)
    )

def get_file_extension(fname):
    """ Returns the extension from a filepath string ignoring the '.' character """
    return os.path.splitext(fname)[-1][1:]