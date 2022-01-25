Developer Notes
===============

Last updated: 25 January 2022

Creating a new release of pyCSEP
--------------------------------

These are the steps required to create a new release of pyCSEP. This requires a combination of updates to the repository
and Github. You will need to build the wheels for distribution on PyPI and upload them to GitHub to issue a release.
The final step involves uploading the tar-ball of the release to PyPI. CI tools provided by `conda-forge` will automatically
bump the version on `conda-forge`. Note: permissions are required to push new versions to PyPI.

1. Code changes
***************
1. Bump the version number in `_version.py <https://github.com/SCECcode/pycsep/tree/master/csep/_version.py>`_
2. Update `codemeta.json <https://github.com/SCECcode/pycsep/blob/master/codemeta.json>`_
3. Update `CHANGELOG.md <https://github.com/SCECcode/pycsep/blob/master/CHANGELOG.md>`_. Include links to Github pull requests if possible.
4. Update `CREDITS.md <https://github.com/SCECcode/pycsep/blob/master/CREDITS.md>`_ if required.
5. Update the version in `conf.py <https://github.com/SCECcode/pycsep/blob/master/docs/conf.py>`_.
6. Issue a pull request that contains these changes.
7. Merge pull request when all changes are merged into `master` and versions are correct.

2. Creating source distribution
*******************************

Issue these commands from the top-level directory of the project::

    python setup.py check

If that executes with no warnings or failures build the source distribution using the command::

    python setup.py sdist

This creates a folder called `dist` that contains a file called `pycsep-X.Y.Z.tar.gz`. This is the distribution
that will be uploaded to `PyPI`, `conda-forge`, and Github.

Upload to PyPI using `twine`. This requires permissions to push to the PyPI account::

    twine upload dist/pycsep-X.Y.Z.tar.gz

3. Create release on Github
***************************
1. Create a new `release <https://github.com/SCECcode/pycsep/releases>`_ on GitHub. This can be saved as a draft until it's ready.
2. Copy new updates information from `CHANGELOG.md <https://github.com/SCECcode/pycsep/blob/master/CHANGELOG.md>`_.
3. Upload tar-ball created from `setup.py`.
4. Publish release.