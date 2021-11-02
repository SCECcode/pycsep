# Contributing to pyCSEP

This document provides an overview on how to contribute to pyCSEP. It will provide step-by-step instructions and hope to 
answer some questions.


## Getting Started

* Make sure you have an active GitHub account
* Download and install `git`
* Read git documentaion if you aren't familiar with `git`
* Install the **development version** of PyCSEP
* If you haven't worked with git Forks before, make sure to read the documentation linked here:
[some helping info](https://help.github.com/en/github/collaborating-with-issues-and-pull-requests/working-with-forks).

## Developer Installation

This shows you how to install a copy of the repository that you can use to create Pull Requests and sync with the upstream
repository. First, fork the repo on GitHub. It will now live at `https://github.com/<YOUR_GITHUB_USERNAME>/pycsep`. 
We recommend using `conda` to install the development environment.

    git clone https://github.com/<YOUR_GITHUB_USERNAME>/pycsep.git
    cd pycsep
    conda env create -f requirements.yml
    conda activate csep-dev
    pip install -e .[all]
    # add upstream repository
    git remote add upstream https://github.com/SCECCode/pycsep.git
    
Note: use the command `conda deactivate` to go back to your regular environment when you are done working with pyCSEP.

## Submitting a Pull Request

### Some notes for starting a pull request

Pull requests are how we use your changes to the code! Please submit them to us! Here's how:

1. Make a new branch. For features/additions base your new branch at `master`.
2. Make sure to add tests! Only pull requests for documentation, refactoring, or plotting features do not require a test.
3. Also, documentation must accompany new feature requests.
    - Note: We would really appreciate pull requests that help us improve documentation.
3. Make sure the tests pass. Run `./run_tests.sh` in the top-level directory of the repo.
4. Push your changes to your fork and submit a pull request. Make sure to set the branch to `pycsep:master`.
5. Wait for our review. There may be some suggested changes or improvements. Changes can be made after
the pull request has been opening by simply adding more commits to your branch.

Pull requests can be changed after they are opened, so create a pull request as early as possible.
This allows us to provide feedback during development and to answer any questions.

Also, if you find pyCSEP to be useful, but don't want to contribute to the code we highly encourage updates to the documentation!

Please make sure to set the correct branch for your pull request. Also, please do not include large files in your pull request.
If you feel that you need to add large files, such as a benchmark forecast, let us know and we can figure something out.

### Tips to get your pull request accepted quickly

1. Any new feature that contains calculations must contain unit-tests to ensure that the calculations are doing what you
expect. Some exceptions to this are documentation changes and new plotting features. 
2. Documentation should accompany any new feature additions into the package.
    * Plotting functions should provide a sphinx-gallery example, which can be found [here](https://github.com/SCECcode/pycsep/blob/master/examples/tutorials/catalog_filtering.py).
    * More complex features might require additional documentation. We will let you know upon seeing your pull request.
    * The documentation use sphinx which compiles reST. Some notes on that can be found [here](https://www.sphinx-doc.org/en/master/usage/quickstart.html).
3. pyCSEP uses pytest as a test runner. Add new tests to the `tests` folder in an existing file or new file starting matching `test_*.py`
4. New scientific capabilities that are not previously published should be presented to the CSEP science group as part of a 
science review. This will consist of a presentation that provides a scientific justification for the feature.
5. Code should follow the [pep8](https://pep8.org/) style-guide.
6. Functions should use [Google style docstrings](https://www.sphinx-doc.org/en/master/usage/extensions/example_google.html). These 
get compiled by Sphinx to become part of the documentation.
 
## Submitting an Issue

Please open an issue if you want to ask a question about PyCSEP.

* Please search through the past issues to see if your question or the bug has already been addressed
* Please apply the correct tag to your issue so others can search

If you want to submit a bug report, please provide the information below:
* pyCSEP version, Python version, and Platform (Linux, Windows, Mac OSX, etc)
* How did you install pyCSEP (pip, anaconda, from source...)
* Please provide a short, complete, and correct example that demonstrates the issue.
* If this broke in a recent update, please tell us when it used to work.

## Additional Resources
* [Working with Git Forks](https://help.github.com/en/github/collaborating-with-issues-and-pull-requests/working-with-forks)
* [Style Guide](http://google.github.io/styleguide/pyguide.html)
* [Docs or it doesn’t exist](https://lukeplant.me.uk/blog/posts/docs-or-it-doesnt-exist/)
* [Quickstart guide for Sphinx](https://www.sphinx-doc.org/en/master/usage/quickstart.html)
* [Pep8 style guide](https://pep8.org/)
* Performance Tips:
  * [Python](https://wiki.python.org/moin/PythonSpeed/PerformanceTips)
  * [NumPy and ctypes](https://scipy-cookbook.readthedocs.io/)
  * [SciPy](https://www.scipy.org/docs.html)
  * [NumPy Book](http://csc.ucdavis.edu/~chaos/courses/nlp/Software/NumPyBook.pdf)
