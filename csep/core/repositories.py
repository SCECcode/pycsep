"""
Repository layer. Add in different storage locations here. Used for archiving state of software
data models. Job specific files and other associated items probably should not be stored in repositories, other than
as file names.

"""
import os
import json
from abc import ABC, abstractmethod
from csep.core.factories import ObjectFactory


class Repository(ABC):
    @abstractmethod
    def list(self):
        raise NotImplementedError

    @abstractmethod
    def save(self):
        raise NotImplementedError

    @abstractmethod
    def update(self):
        raise NotImplementedError


class FileSystem(Repository):
    def __init__(self, url="", name=None):
        expand_url = os.path.expandvars(os.path.expanduser(url))
        self.url = expand_url
        self.name = name

    def list(self):
        """
        Will list all experiments stored in JSON files.

        Returns:

        """
        raise NotImplementedError

    def save(self, data):
        """
        Saves file to location in repository.

        Args:
            data (dict-like): data to store to file-system. must be JSON serializable


        Returns:
            success (bool)
        """
        success = True
        try:
            with open(self.url, 'w') as f:
                print(f'Writing archive file to {self.url}.')
                json.dump(data, f, indent=4, separators=(',', ': '))
        except IOError:
            raise
            print(f'Error saving file to {self.url}')
            success = False
        return success

    def to_dict(self):
        return {'name': self.name,
                'url': self.url}

    def update(self):
        raise NotImplementedError


# Register repository builders
repo_builder = ObjectFactory()
repo_builder.register_builder('filesystem', FileSystem)

