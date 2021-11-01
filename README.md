# PyCSEP: Collaboratory for the Study of Earthquake Predictability
![](https://g-c662a6.a78b8.36fe.data.globus.org/csep/badges/CSEP2_Logo_CMYK.png)
![Python version](https://g-c662a6.a78b8.36fe.data.globus.org/csep/badges/pycsep-python.svg)
![Python application](https://github.com/SCECCode/csep2/workflows/Python%20application/badge.svg)
[![Build sphinx documentation](https://github.com/SCECCode/csep2/workflows/Build%20sphinx%20documentation/badge.svg)](https://cseptesting.org)
[![codecov](https://codecov.io/gh/SCECcode/pycsep/branch/master/graph/badge.svg?token=HTMKM29MAU)](https://codecov.io/gh/SCECcode/pycsep)

# Description:
The PyCSEP Toolkit helps earthquake forecast model developers evaluate their forecasts with the goal of understanding
earthquake predictability.

PyCSEP should:
1. Help modelers become familiar with formats, procedures, and evaluations used in CSEP Testing Centers.
2. Provide vetted software for model developers to use in their research.
3. Provide quantative and visual tools to assess earthquake forecast quality.
4. Promote open-science ideas by ensuring transparency and availability of scientific code and results.
5. Curate benchmark models and data sets for modelers to conduct retrospective experiments of their forecasts.

# Table of Contents:
1. [Software Documentation](https://docs.cseptesting.org)
2. [Installation](#installation)
3. [Usage](#usage)
4. [Contributing](#contributing)
5. [Change Log](https://github.com/SCECcode/pycsep/blob/master/CHANGELOG.txt)
6. [Credits](#credit)
7. [License](#license)

# Installation:
PyCSEP can be installed in several ways. It can be installed using conda or pip package managers or from the 
source code found in the PyCSEP github repo. Researchers interested in contributing to PyCSEP development should 
install PyCSEP from source code. PyCSEP depends on the following software packages. 
These which may be installed automatically, or manually, depending on the installation method used.
* Python 3.7 or later (https://python.org)
* NumPy 1.10 or later (https://numpy.org)  
* GEOS 3.3.3 or later (https://trac.osgeo.org/geos/)  
* PROJ 4.9.0 or later (https://proj4.org/)  

[Detailed PyCSEP Installation Instructions](https://docs.cseptesting.org/getting_started/installing.html) can be found in the online PyCSEP documentation.

# Usage: 
Once installed, PyCSEP methods can be invoked from python code by importing package csep. PyCSEP provides objects and utilities related to several key concepts:
* Earthquake Catalogs
* Earthquake Forecasts
* Earthquake Forecast Evaluations
* Regions

An example of PyCSEP catalog operations is shown below:
<pre>
import csep
from csep.core import regions
from csep.utils import time_utils, comcat
start_time = csep.utils.time_utils.strptime_to_utc_datetime('2019-01-01 00:00:00.0')
end_time = csep.utils.time_utils.utc_now_datetime()
catalog = csep.query_comcat(start_time, end_time)
print(catalog)
</pre>

Please see [PyCSEP Getting Started](https://docs.cseptesting.org/getting_started/core_concepts) documentation for more examples and tutorials.

# Software Support:
Software support for PyCSEP is provided by that Southern California Earthquake Center (SCEC) Research Computing Group. This group supports several research software distributions including UCVM. Users can report issues and feature requests using the PyCSEP github-based issue tracking link below. Developers will also respond to emails sent to the SCEC software contact listed below.
1. [UCVM Github Issue Tracker:](https://github.com/SCECcode/pycep/issues)
2. Email Contact: software@scec.usc.edu

# Contributing:
We welcome contributions to the PyCSEP Toolkit.  If you would like to contribute to this package, including software, tests, and documentation, 
please visit the [contribution guidelines](https://github.com/SCECcode/pycsep/blob/master/CONTRIBUTING.md) for guidelines on how to contribute to PyCSEP development.
PyCSEP contributors agree to abide by the code of conduct found in our [Code of Conduct](CODE_OF_CONDUCT.md) guidelines.

# Credits:
Development of PyCSEP is a group effort. A list of developers that have contributed to the PyCSEP Toolkit 
are listed in the [Credits](CREDITS.md) file in this repository.

# License:
The PyCSEP software is distributed under the BSD 3-Clause open-source license. Please see the [LICENSE.txt](LICENSE.txt) file for more information.
