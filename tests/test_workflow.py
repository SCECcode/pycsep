from unittest import TestCase
from csep.core.managers import Workflow
from csep.core.repositories import Repository, FileSystem


class TestWorkflow(TestCase):
    def test_create_repo_with_dict(self):
        repo = {'name': 'filesystem',
                'url': 'testing'}
        b = FileSystem(**repo)
        a = Workflow(repository=repo)
        assert isinstance(a.repository, Repository)
        assert a.repository.to_dict() == b.to_dict()

    def test_create_default(self):
        a = Workflow()
        assert a.repository == None
        assert a.base_dir == ''
        assert a.owner == ()
        assert a.description == ''
        assert a.jobs == []


