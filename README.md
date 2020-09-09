# PyCSEP: Collaboratory for the Study of Earthquake Predictability

![](http://hypocenter.usc.edu/research/badges/CSEP2_Logo_CMYK.png)

![Python version](http://hypocenter.usc.edu/research/badges/pycsep-python.svg)
![Python application](https://github.com/SCECCode/csep2/workflows/Python%20application/badge.svg)
[![Build sphinx documentation](https://github.com/SCECCode/csep2/workflows/Build%20sphinx%20documentation/badge.svg)](https://cseptesting.org)
[![codecov](https://codecov.io/gh/SCECcode/csep2/branch/dev/graph/badge.svg)](https://codecov.io/gh/SCECcode/csep2)

The PyCSEP tools help earthquake forecast model developers evaluate their forecasts with the goal of understanding
earthquake predictability.

PyCSEP should:
1. Help modelers become familiar with formats, procedures, and evaluations used in CSEP Testing Centers.
2. Provide vetted software for model developers to use in their research.
3. Provide quantative and visual tools to assess earthquake forecast quality.
4. Promote open-science ideas by ensuring transparency and availability of scientific code and results.
5. Curate benchmark models and data sets for modelers to conduct retrospective experiments of their forecasts.

## Using Conda

The easiest way to install PyCSEP is using `conda`. It can also be installed using `pip` or built from source. 
If you plan on contributing to this package, visit the 
[contribution guidelines](https://github.com/SCECcode/pycsep/blob/master/CONTRIBUTING.md) for 
installation instructions.

    conda install --channel conda-forge pycsep
    
## Using Pip

Before this installation will work, you must **first** install the following system dependencies. The remaining dependencies
should be installed by the installation script. To help manage dependency issues, we recommend using virtual environments 
like `virtualenv`.

Python 3.7 or later (https://python.org)

NumPy 1.10 or later (https://numpy.org)  
&nbsp;&nbsp;&nbsp;&nbsp;Python package for scientific computing and numerical calculations.

GEOS 3.3.3 or later (https://trac.osgeo.org/geos/)  
&nbsp;&nbsp;&nbsp;&nbsp;C++ library for processing geometry.

PROJ 4.9.0 or later (https://proj4.org/)  
&nbsp;&nbsp;&nbsp;&nbsp;Library for cartographic projections. 

Example for Ubuntu:

    sudo apt-get install libproj-dev proj-data proj-bin  
    sudo apt-get install libgeos-dev 
    pip install --upgrade pip
    pip install numpy
    
Example for MacOS:

    brew install proj geos
    pip install --upgrade pip
    pip install numpy
    
### From Source

Use this approach if you want the most up-to-date code. This creates an editable installation that can be synced with 
the latest GitHub commit. 

We recommend using virtual environments when installing python packages from source to avoid any dependency conflicts. We prefer 
`conda` as the package manager over `pip`, because `conda` does a good job of handling binary distributions of packages
across multiple platforms. Also, we recommend using the `miniconda` installer, because it is lightweight and only includes
necessary pacakages like `pip` and `zlib`. 

#### Using Conda
If you don't have `conda` on your machine, download and install [Miniconda](https://docs.conda.io/en/latest/miniconda.html).

    git clone https://github.com/SCECcode/pycsep
    cd pycsep
    conda env create -f requirements.yml
    conda activate csep-dev
    # Installs in editor mode with all dependencies
    pip install -e .
    
Note: If you want to go back to your default environment use the command `conda deactivate`.

#### Using Pip / Virtualenv

We highly recommend using Conda, because this tools helps to manage binary dependencies on Python packages. If you
must use [Virtualenv](https://packaging.python.org/guides/installing-using-pip-and-virtual-environments/)
follow these instructions:  

    git clone https://github.com/SCECcode/pycsep
    cd pycsep
    python -m virtualenv venv
    source venv/bin/activate
    # Installs in editor mode with all dependencies
    pip install -e .[all]
    
 Note: If you want to go back to your default environment use the command `deactivate`.   

## Documentation and Changelog

The documentation can be found at [here](https://cseptesting.org), and the changelog can be found 
[here](https://github.com/SCECcode/pycsep/blob/master/CHANGELOG.txt).

## Releases 

We follow [semver](https://semver.org/) for our versioning strategy. 
