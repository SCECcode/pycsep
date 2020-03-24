# PyCSEP: Collaboratory for the Study of Earthquake Predictability

The PyCSEP tools help earthquake forecast model developers evaluate their forecasts with the goal of understanding
earthquake predictability.

PyCSEP should:
1. Help modelers become familiar with formats, procedures, and evaluations used in CSEP Testing Centers.
2. Provide vetted software for model developers to use in their research.
3. Provide quantative and visual tools to assess earthquake forecast quality.
4. Promote open-science ideas by ensuring transparency and availability of scientific code and results.
5. Curate benchmark models and data sets for modelers to conduct retrospective experiments of their forecasts.

## Installing PyCSEP

This package must be installed from GitHub until an offical release is made available on PyPI and Conda-Forge.
We recommend making a virtual environment to ensure there are no conflicts in dependencies.

This installation will be editable and used for development. This way any changes made to the package will be usable
within in the python environment.


1. Clone repository from `https://github.com/SCECcode/csep2`
2. Create environment for installation
    * [Miniconda](https://docs.conda.io/en/latest/miniconda.html) (recommended):  
    `conda env create -f requirements.yaml`  
    `conda activate csep`  
    
    Note: If you want to go back to your default environment use the command `conda deactivate`.

    * [Virtualenv](https://packaging.python.org/guides/installing-using-pip-and-virtual-environments/):  
    We highly recommend using Conda, because this tools helps to manage binary dependencies on Python pacakages. If you
    must use `virtaulenv` follow these instructions:  
    `cd csep2`  
    `mkdir venv`  
    `cd venv`  
    `python3 -m venv csep`  
    `source csep/bin/activate`  
    `cd ..`  
    `pip3 install numpy` (Because of obspy and scipy)  
    `pip3 install wheel`  
    `pip3 install -r requirements.txt`
    
    Note: If you want to go back to your default environment use the command `deactivate`.
    
    Note: There is an issue installing Cartopy on MacOS with Proj >=6.0.0 and will be addressed in 0.18 release of Cartopy. 
    If this package is needed please manually install or use Conda instructions above. Additionally, if you choose the 
    manual build, you might need to resolve build issues as they arise. This is usually caused by [not having the proper 
    python statics installed](https://stackoverflow.com/questions/21530577/fatal-error-python-h-no-such-file-or-directory) to build the binary packages or poorly written setup.py scripts from other packages.
    
3. Navigate to repo `cd csep2` (If you are not already there...)
4. Install editable version of package `pip install -e .`

You can verify the installation works by opening a python interpreter and typing `import csep`. If you see
no errors the installation worked.

Additionally, you can run the test suite by navigating to the project root directory and running `./run_tests.sh`. The test suite requires a properly configured environment to run correctly.

With this editable installation you can freely edit the package and have the changes propagate to the python 
installation.
