# Contributing to PyCSEP

This document provides an overview on how to contribute to PyCSEP. It will provide step-by-step instructions and hope to 
answer some questions.


## Getting Started

* Make sure you have an active GitHub account
* Download and install git
* Read the git documentation
* Install a development version of PyCSEP
* If you haven't worked with Git Forks before, make sure to read the documentation linked here:
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
    # sync with default repository
    git remote add upstream https://github.com/SCECCode/pycsep.git
    
Note: use the commend `conda deactivate` to go back to your regular environment.

## Submitting a Pull Request

Pull requests are great! Please submit them to us! Here's how:

1. Make a new branch. For features/additions base your new branch at `master`.
2. Add a test! Only pull requests for documentation and refactoring do not require a test.
3. Make sure the tests pass. Run `./run_tests.sh` in the top-level directory of the repo.
4. Push your changes to your fork and submit a pull request. Make sure to set the branch to `pycsep:master`.
5. Wait for our review. There may be some suggested changes or improvements. Changes can be made after
the pull request has been opening by simply adding more commits to your branch.

Pull requests can be changed after they are opened, so create a pull request as early as possible.
This allows us to provide feedback during development and to answer any questions.

Please make sure to set the correct branch for your pull request. Also, please do not include large files in your pull request.
If you feel that you need to add large files, let us know and we can figure something out.

## Submitting an Issue

Please open an issue if you want to ask a question about PyCSEP.

* Please search through the past issues to see if your question or the bug has already been addressed
* Please apply the correct tag to your issue so others can search

If you want to submit a bug report, please provide the information below:
* PyCSEP version, Python version, and Platform (Linux, Windows, Mac OSX, etc)
* How did you install PyCSEP (pip, anaconda, from source...)
* Please provide a short, complete, and correct example that demonstrates the issue.
* If this broke in a recent update, please tell us when it used to work.

## Additional Resources
* [Working with Git Forks](https://help.github.com/en/github/collaborating-with-issues-and-pull-requests/working-with-forks)
* [Style Guide](http://google.github.io/styleguide/pyguide.html)
* [Docs or it doesn’t exist](https://lukeplant.me.uk/blog/posts/docs-or-it-doesnt-exist/)
* Performance Tips:
  * [Python](https://wiki.python.org/moin/PythonSpeed/PerformanceTips)
  * [NumPy and ctypes](https://scipy-cookbook.readthedocs.io/)
  * [SciPy](https://www.scipy.org/docs.html)
  * [NumPy Book](http://csc.ucdavis.edu/~chaos/courses/nlp/Software/NumPyBook.pdf)
