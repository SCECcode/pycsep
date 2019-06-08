"""
Repository layer. Add in different storage locations here. Used for archiving state of software
data models. Job specific files and other associated items probably should not be stored in repositories, other than
as file names.

"""
import os
import json
from csep.core.factories import ObjectFactory

class FileSystem:
    def __init__(self, url="", name=None):
        expand_url = os.path.expandvars(os.path.expanduser(url))
        self.url = expand_url
        self.name = name

    def list_experiment(self):
        """
        Will list all experiments stored in JSON files. Method returns an experiment class
        with identical state to the JSON manifest file.

        Returns:
            csep.core.managers.experiment

        """
        try:
            with open(self.url, 'r') as f:
                manifest=json.load(f)
        except (FileNotFoundError, IOError):
            print(f'Error: Unable to load manifest.\nAttempted url {self.url}')

    def save(self, data):
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

