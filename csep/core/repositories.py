"""
Repository layer. Add in different storage locations here. Used for archiving state of software
data models. Job specific files and other associated items probably should not be stored in repositories, other than
as file names.

"""
import os
import shutil
from urllib.parse import urlparse
from abc import ABC, abstractmethod
from csep.core.factories import ObjectFactory


class Repository(ABC):
    @abstractmethod
    def list(self):
        raise NotImplementedError

class FileSystem(Repository):
    def __init__(self, url=""):
        # TODO: Write class to build correct objects based on URLs. For example, file:// vs. sql:// vs mongo://
        expnd_url = os.path.expandvars(os.path.expanduser(url))
        self.url=urlparse(expnd_url).path

    def list(self):
        """
        Will list all experiments stored in JSON files.

        Returns:

        """
        raise NotImplementedError


# Register repository builders
repo_builder = ObjectFactory()
repo_builder.register_builder('filesystem', FileSystem)