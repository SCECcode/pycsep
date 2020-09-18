Installing PyCSEP
=================

This package must be installed from `GitHub <https://github.com/SCECcode/csep2>`_ until an official release is made
available on PyPI and conda-Forge.
We recommend making a virtual environment to ensure there are no conflicts in dependencies.

.. note::
    If you'd like to install an editable version of the package. First, fork the repository and follow these instructions
    using your own copy of the PyCSEP codebase. You will need to run ``pip install -e .`` instead of the
    ``pip install .`` command as listed below to direct python to make an editable installation. Instead of cloning from
    ``https://github.com/SCECcode/csep2`` you would clone from ``https://github.com/<YOUR_GITHUB_USERNAME>/csep2`` in
    step 1.

1. Clone repository from ``https://github.com/SCECcode/csep2``
2. Create environment for installation
    * `Miniconda <https://docs.conda.io/en/latest/miniconda.html>`_ (recommended):

    | ``conda env create -f requirements.yaml``
    | ``conda activate csep-dev``

    .. note::
        If you want to go back to your default environment use the command ``conda deactivate``.

    * `Virtualenv <https://packaging.python.org/guides/installing-using-pip-and-virtual-environments/>`_:

    We highly recommend using Conda, because this tools helps to manage binary dependencies on Python packages. If you
    must use ``virtaulenv`` follow these instructions:

    | ``cd csep2``
    | ``mkdir venv``
    | ``cd venv``
    | ``python3 -m venv csep-dev``
    | ``source csep-dev/bin/activate``
    | ``cd ..``

    .. note::
        If you want to go back to your default environment use the command ``deactivate``.

    Also python 3.7 is required.

3. Navigate to repo ``cd csep2`` (If you are not already there...)
4. Install package ``pip install .``

You can verify the installation works by opening a python interpreter and typing ``import csep``. If you see
no errors the installation worked.

Additionally, you can run the test suite by navigating to the project root directory and running ``./run_tests.sh``.
The test suite requires a properly configured environment to run correctly. For the tests to run properly you need
to install using conda (1) or install with ``pip install .[test]``.

.. note::
    If you need to install this package on a Linux system we recommend to use a Ubuntu Linux v18.04 LTS based system.
    The prerequisites for using this package by venv are the linux packages:

      * ``build-essential``
      * ``python3-dev``
      * ``python3-venv``
      * ``python3-pip``

.. warning::
    There is an issue installing Cartopy on MacOS with Proj >=6.0.0 and will be addressed in 0.18 release of Cartopy.
    If this package is needed please manually install or use Conda instructions above. Additionally, if you choose the
    manual build, you might need to resolve build issues as they arise. This is usually caused by `not having the proper
    python statics installed <https://stackoverflow.com/questions/21530577/fatal-error-python-h-no-such-file-or-directory/>`_)
    to build the binary packages or poorly written setup.py scripts from other packages.