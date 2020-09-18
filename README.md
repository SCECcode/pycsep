# PyCSEP: Collaboratory for the Study of Earthquake Predictability

![Python application](https://github.com/SCECCode/csep2/workflows/Python%20application/badge.svg)
![Build sphinx documentation](https://github.com/SCECCode/csep2/workflows/Build%20sphinx%20documentation/badge.svg)
[![codecov](https://codecov.io/gh/SCECcode/csep2/branch/dev/graph/badge.svg)](https://codecov.io/gh/SCECcode/csep2)

The PyCSEP tools help earthquake forecast model developers evaluate their forecasts with the goal of understanding
earthquake predictability.

PyCSEP should:
1. Help modelers become familiar with formats, procedures, and evaluations used in CSEP Testing Centers.
2. Provide vetted software for model developers to use in their research.
3. Provide quantative and visual tools to assess earthquake forecast quality.
4. Promote open-science ideas by ensuring transparency and availability of scientific code and results.
5. Curate benchmark models and data sets for modelers to conduct retrospective experiments of their forecasts.

## Installing PyCSEP

This package must be installed from [GitHub](https://github.com/SCECcode/csep2) until an offical release is made available on PyPI and Conda-Forge.
We recommend making a virtual environment to ensure there are no conflicts in dependencies.

This installation will be editable and used for development. This way any changes made to the package will be usable
within in the python environment.


1. Clone repository from `git clone https://github.com/SCECcode/pycsep`
2. Create environment for installation
    * [Miniconda](https://docs.conda.io/en/latest/miniconda.html) (recommended):  
    `conda env create -f requirements.yml`  
    `conda activate csep-dev`  
    
    Note: If you want to go back to your default environment use the command `conda deactivate`.

    * [Virtualenv](https://packaging.python.org/guides/installing-using-pip-and-virtual-environments/):  
    We highly recommend using Conda, because this tools helps to manage binary dependencies on Python pacakages. If you
    must use `virtaulenv` follow these instructions:  
    `cd pycsep`  
    `mkdir venv`  
    `cd venv`  
    `python3 -m venv csep-dev`  
    `source csep-dev/bin/activate`  
    `cd ..`
    
    Note: If you want to go back to your default environment use the command `deactivate`.
    
    Note: There is an issue installing Cartopy on MacOS with Proj >=6.0.0 and will be addressed in 0.18 release of Cartopy. 
    If this package is needed please manually install or use Conda instructions above. Additionally, if you choose the 
    manual build, you might need to resolve build issues as they arise. This is usually caused by [not having the proper 
    python statics installed](https://stackoverflow.com/questions/21530577/fatal-error-python-h-no-such-file-or-directory) to build the binary packages or poorly written setup.py scripts from other packages.
    
    Also python 3.7 is required.
    
3. Navigate to repo `cd pycsep` (If you are not already there...)
4. Install package `pip install .`

You can verify the installation works by opening a python interpreter and typing `import csep`. If you see
no errors the installation worked.

Additionally, you can run the test suite by navigating to the project root directory and running `./run_tests.sh`.
The test suite requires a properly configured environment to run correctly. For the tests to run properly you need
to install using conda (1) or install with `pip install .[test]`.

With this editable installation you can freely edit the package and have the changes propagate to the python 
installation.

### Note for Ubuntu Linux users

If you need to install this package on a Linux system we recommend to use a Ubuntu Linux v18.04 LTS based system.
The prerequsites for using this package by venv are the linux packages:
  * `build-essential`
  * `python3-dev`
  * `python3-venv`
  * `python3-pip`
