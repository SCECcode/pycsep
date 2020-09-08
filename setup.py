import os
from setuptools import setup, find_packages

with open("README.md",'r') as fh:
    long_description = fh.read()

setup(
    name='pycsep',
    version='0.1.0',
    author='William Savran',
    author_email='wsavran@usc.edu',
    packages=find_packages(),
    license='LICENSE',
    description='Python tools from the Collaboratory for the Study of Earthquake Predictability',
    long_description=long_description,
    long_description_content_type='text/markdown',
    install_requires = [
        'numpy',
        'scipy',
        'pandas',
        'matplotlib',
        'cartopy',
        'obspy',
        'pyproj',
        'python-dateutil'
    ],
    extras_require = {
        'test': [
            'pytest',
            'vcrpy',
            'pytest-cov'
        ],
        'dev': [
            'sphinx-gallery',
            'sphinx-rtd-theme',
            'pillow'
        ],
        'all': [
            'pytest',
            'vcrpy',
            'pytest-cov',
            'sphinx-gallery',
            'sphinx-rtd-theme',
            'pillow'
        ]
    },
    include_package_data=True,
    python_requires=">=3.7",
    zip_safe=False,
    url='https://github.com/SCECCode/csep2'
)
