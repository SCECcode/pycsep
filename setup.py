import re
from setuptools import setup, find_packages

def get_version():
    VERSIONFILE = "csep/_version.py"
    verstrline = open(VERSIONFILE, "rt").read()
    VSRE = r"^__version__ = ['\"]([^'\"]*)['\"]"
    mo = re.search(VSRE, verstrline, re.M)
    if mo:
        verstr = mo.group(1)
    else:
        raise RuntimeError("Unable to find version string in %s." % (VERSIONFILE,))
    return verstr


with open("README.md",'r') as fh:
    long_description = fh.read()

setup(
    name='pycsep',
    version=get_version(),
    author='William Savran',
    author_email='wsavran@usc.edu',
    packages=find_packages(),
    license='LICENSE',
    description='Python tools from the Collaboratory for the Study of Earthquake Predictability',
    long_description=long_description,
    long_description_content_type='text/markdown',
    install_requires=[
        'numpy',
        'scipy',
        'pandas',
        'matplotlib',
        'cartopy',
        'obspy',
        'pyproj',
        'python-dateutil',
        'mercantile',
        'shapely'
    ],
    extras_require = {
        'test': [
            'pytest',
            'vcrpy',
            'pytest-cov'
        ],
        'dev': [
            'sphinx',
            'sphinx-gallery',
            'sphinx-rtd-theme',
            'pillow'
        ],
        'all': [
            'pytest',
            'vcrpy',
            'pytest-cov',
            'sphinx',
            'sphinx-gallery',
            'sphinx-rtd-theme',
            'pillow'
        ]
    },
    include_package_data=True,
    python_requires=">=3.7",
    zip_safe=False,
    url='https://github.com/SCECCode/pycsep'
)
