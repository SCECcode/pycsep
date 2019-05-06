import os
import shutil
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


def mkdirs(path, mode):
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