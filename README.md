# pyCSEP: Collaboratory for the Study of Earthquake Predictability
![](https://i.postimg.cc/Bb60rVQP/CSEP2-Logo-CMYK.png)
<p align=center>
    <a target="_blank" href="https://python.org" title="Python version"><img src="https://gist.githubusercontent.com/wsavran/efce311162c32460336a4f9892218532/raw/1b9c060efd1c6e52eb53f82d4249107417d6a5ec/pycsep_python_badge.svg">
    <a target="_blank" href="https://pypi.org/project/pycsep"><img src="https://anaconda.org/conda-forge/pycsep/badges/downloads.svg">
    <a target="_blank" href="https://github.com/SCECcode/pycsep/actions"><img src="https://github.com/SCECcode/pycsep/actions/workflows/build-test.yml/badge.svg">
    <a target="_blank" href="https://github.com/SCECcode/pycsep/actions"><img src="https://github.com/SCECcode/pycsep/actions/workflows/build-sphinx.yml/badge.svg">
    <a target="_blank" href="https://codecov.io/gh/SCECcode/pycsep"><img src="https://codecov.io/gh/SCECcode/pycsep/branch/master/graph/badge.svg?token=HTMKM29MAU">
    <a target="_blank" href="https://www.zenodo.org/badge/latestdoi/149362283"><img src="https://www.zenodo.org/badge/149362283.svg" alt="DOI"></a>
        <a target="_blank" a style="border-width:0" href="https://doi.org/10.21105/joss.03658">
  <img src="https://joss.theoj.org/papers/10.21105/joss.03658/status.svg" alt="DOI badge" ></a>
</p>

# Description:
The pyCSEP Toolkit helps earthquake forecast model developers evaluate their forecasts with the goal of understanding
earthquake predictability.

pyCSEP should:
1. Help modelers become familiar with formats, procedures, and evaluations used in CSEP Testing Centers.
2. Provide vetted software for model developers to use in their research.
3. Provide quantitative and visual tools to assess earthquake forecast quality.
4. Promote open-science ideas by ensuring transparency and availability of scientific code and results.
5. Curate benchmark models and data sets for modelers to conduct retrospective experiments of their forecasts.

# Table of Contents:
1. [Software Documentation](https://docs.cseptesting.org)
2. [Installation](#installation)
3. [Usage](#usage)
4. [Contributing](#contributing)
5. [Change Log](https://github.com/SCECcode/pycsep/blob/master/CHANGELOG.md)
6. [Credits](#credits)
7. [License](#license)

# Installation:
pyCSEP can be installed in several ways. It can be installed using conda or pip package managers or from the 
source code found in the pyCSEP github repo. Researchers interested in contributing to pyCSEP development should 
install pyCSEP from source code. pyCSEP depends on the following software packages. 
These which may be installed automatically, or manually, depending on the installation method used.
* Python 3.7 or later (https://python.org)
* NumPy 1.21.3 or later (https://numpy.org)
* SciPy 1.7.1 or later (https://scipy.org)
* pandas 1.3.4 or later (https://pandas.pydata.org)
* cartopy 0.20.0 or later (https://scitools.org.uk/cartopy/docs/latest)
* GEOS 3.7.2 or later (https://trac.osgeo.org/geos/)
* PROJ 8.0.0 or later (https://proj.org/)

Please see the [requirements file](https://github.com/SCECcode/pycsep/blob/master/requirements.yml) for a complete list 
of requirements. These are installed automatically when using the `conda` distribution.

Detailed pyCSEP [installation instructions](https://docs.cseptesting.org/getting_started/installing.html) can be found 
in the online pyCSEP documentation.

# Usage: 
Once installed, pyCSEP methods can be invoked from python code by importing package csep. pyCSEP provides objects and 
utilities related to several key concepts:
* Earthquake Catalogs
* Earthquake Forecasts
* Earthquake Forecast Evaluations
* Regions

An simple example to download and plot an earthquake catalog from the USGS ComCat:
<pre>
import csep
from csep.core import regions
from csep.utils import time_utils
start_time = time_utils.strptime_to_utc_datetime('2019-01-01 00:00:00.0')
end_time = time_utils.utc_now_datetime()
catalog = csep.query_comcat(start_time, end_time)
catalog.plot(show=True)
</pre>

Please see [pyCSEP Getting Started](https://docs.cseptesting.org/getting_started/core_concepts) documentation for more examples and tutorials.

# Software Support:
Software support for pyCSEP is provided by that Southern California Earthquake Center (SCEC) Research Computing Group. 
This group supports several research software distributions including UCVM. Users can report issues and feature requests 
using the pyCSEP github-based issue tracking link below. Developers will also respond to emails sent to the SCEC software contact listed below.
1. [pyCSEP Issues](https://github.com/SCECcode/pycsep/issues)
2. Email Contact: software [at] scec [dot] org

# Contributing:
We welcome contributions to the pyCSEP Toolkit.  If you would like to contribute to this package, including software, tests, and documentation, 
please visit the [contribution guidelines](https://github.com/SCECcode/pycsep/blob/master/CONTRIBUTING.md) for guidelines on how to contribute to pyCSEP development.
pyCSEP contributors agree to abide by the code of conduct found in our [Code of Conduct](CODE_OF_CONDUCT.md) guidelines.

# Credits:
Development of pyCSEP is a group effort. A list of developers that have contributed to the PyCSEP Toolkit 
are listed in the [credits](CREDITS.md) file in this repository.

# License:
The pyCSEP software is distributed under the BSD 3-Clause open-source license. Please see the [license](LICENSE.txt) file for more information.
