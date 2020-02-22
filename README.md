# PyCSEP: Collaboratory for the Study of Earthquake Predictability

The CSEP tools help earthquake forecast model developers evaluate their forecasts with the goal of understanding
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
    * [Anaconda](https://www.anaconda.com/) (recommended): `conda env create -f requirements.yaml`
    * [Virtualenv](https://packaging.python.org/guides/installing-using-pip-and-virtual-environments/):  
    `mkdir venv`  
    `cd venv`         
    `python3 -m venv csep`    
    `source venv/csep/bin/activate`      
    `pip3 install -r requirements.txt`   
3. Navigate to repo `cd csep2`
4. Install package `pip install -e .`

You can verify the installation works by opening a python interpreter and typing `import csep`. If you see
no errors the installation worked.

With this editable installation you can freely edit the package and have the changes propagate to the python 
installation.





