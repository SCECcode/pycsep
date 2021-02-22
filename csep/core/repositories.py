"""
Repository layer. Add in different storage locations here. Used for archiving state of software
data models. Job specific files and other associated items probably should not be stored in repositories, other than
as file names.

This module might be deprecated in favor of the write_json and load_json files. It doesn't seem to add much value
unless we start storing things into a database.

"""
import datetime
import os
import json

# PyCSEP imports
from csep.utils.log import LoggingMixin
from csep.utils.file import copy_file


class Repository(LoggingMixin):
    def __eq__(self, other):
        try:
            return self.to_dict() == other.to_dict()
        except:
            return False

    def load(self, object):
        raise NotImplementedError

    def save(self, object):
        raise NotImplementedError

    def delete(self):
        raise NotImplementedError

    def to_dict(self):
        raise NotImplementedError

    def from_dict(self, adict):
        raise NotImplementedError

class FileSystem(Repository):
    def __init__(self, url="", name='filesystem', **kwargs):
        super().__init__(**kwargs)
        expand_url = os.path.expandvars(os.path.expanduser(url))
        self.url = expand_url
        self.name = name

    def load(self, object):
        """
        Will list all experiments stored in JSON files. Method returns an experiment class
        with identical state to the JSON manifest file.

        Returns:
            csep.core.managers.experiment

        """
        try:
            with open(self.url, 'r') as f:
                adict=json.load(f)
        except IOError:
            raise IOError(f"Unable to access file at {self.url}.")
        return object.from_dict(adict)

    def save(self, data, backup=False):
        """
        Saves file to location in repository. Changes state on the system, should be careful
        about how to approach having multiple monitor classes. Maybe use singleton if there
        isn't a database server backend.

        Args:
            data (dict-like): data to store to file-system. must be JSON serializable


        Returns:
            success (bool)
        """
        success = True
        if backup:
            if os.path.isfile(self.url):
                time_fmt = '%Y-%m-%dT%H:%M:%S:%f'
                time_str = str(datetime.datetime.now().strftime(time_fmt))
                fname = os.path.splitext(self.url)[0] + '_backup_' + time_str + '.json'
                try:
                    copy_file(self.url, fname)
                    self.log.info(f'Found file at {self.url} backing up to {fname}.')
                except Exception as e:
                    self.log.exception(e)
        try:
            with open(self.url, 'w') as f:
                self.log.info(f'Writing file to {self.url}.')
                json.dump(data, f, indent=4, separators=(',', ': '), sort_keys=True, default=str)
        except IOError:
            raise IOError(f"FileSystem Repository Error: saving file to {self.url}")
        return success

    def to_dict(self):
        return {'name': self.name,
                'url': self.url}

    @classmethod
    def from_dict(cls, adict):
        return cls(**adict)


def write_json(object, fname):
    """ Easily write object to json file that implements the to_dict() method

    Args:
        object (class): must implement a method called to_dict()
        fname (str): path of the file to write evaluation results

    Returns:
        NoneType
    """
    repo = FileSystem(url=fname)
    repo.save(object.to_dict())

def load_json(object, fname):
    repo = FileSystem(url=fname)
    return repo.load(object)

