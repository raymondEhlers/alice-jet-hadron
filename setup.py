#/usr/bin/env python

# Setup Jet-H analysis
# Derived from the setup.py in aliBuild and Overwatch
# and based on: https://python-packaging.readthedocs.io/en/latest/index.html

# Always prefer setuptools over distutils
from setuptools import setup, find_packages
# To use a consistent encoding
from codecs import open
import os

def get_version():
    versionModule = {}
    with open(os.path.join("jet_hadron", "version.py")) as f:
        exec(f.read(), versionModule)
    return versionModule["__version__"]

# Get the long description from the README file
here = os.path.abspath(os.path.dirname(__file__))
with open(os.path.join(here, 'README.md'), encoding='utf-8') as f:
    long_description = f.read()

setup(
    name="alice_jet_hadron_correlations",
    version=get_version(),

    description="ALICE jet-hadron correlations analysis",
    long_description=long_description,
    long_description_content_type="text/markdown",

    author="Raymond Ehlers",
    author_email="raymond.ehlers@cern.ch",

    url="https://github.com/raymondEhlers/alice-jet-hadron",
    license="BSD 3-Clause",

    # See https://pypi.python.org/pypi?%3Aaction=list_classifiers
    classifiers=[
        # How mature is this project? Common values are
        #   3 - Alpha
        #   4 - Beta
        #   5 - Production/Stable
        'Development Status :: 4 - Beta',

        # Indicate who your project is intended for
        'Intended Audience :: Science/Research',

        # Specify the Python versions you support here. In particular, ensure
        # that you indicate whether you support Python 2, Python 3 or both.
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7'
    ],

    # What does your project relate to?
    keywords='HEP ALICE',

    packages=find_packages(exclude=(".git", "tests")),

    # Rename scripts to the desired executable names
    # See: https://stackoverflow.com/a/8506532
    entry_points = {
        "console_scripts": [
            "jetHAnalysis = jet_hadron.analysis.correlations:runFromTerminal",
            #"testDev = jet_hadron.analysis.rm_dev:run_from_terminal",
            "jetHResponse = jet_hadron.analysis.response_matrix:runFromTerminal",
            "plotEMCalCorrections = jet_hadron.analysis.EMCal_analysis_tasks:runEMCalCorrectionsHistsFromTerminal",
            "plotEMCalEmbedding = jet_hadron.analysis.EMCal_analysis_tasks:runEMCalEmbeddingHistsFromTerminal",
            "plotRPFRegions = jet_hadron.plot.highlight_RPF:runFromTerminal"
        ],
    },

    # This is usually the minimal set of the required packages.
    # Packages should be installed via pip -r requirements.txt !
    install_requires=[
        "future",
        "ruamel.yaml",
        "IPython",
        "scipy",
        "numpy",
        "matplotlib",
        "seaborn",
        # Not required at the moment, but hopefully for the future.
        #"uproot",
        # Skip rootpy so we can install the package without ROOT being available (so we can run flake8, etc,
        # even if we skip the tests).
        #"rootpy",
        #"root_numpy",  # As of Dec 2018, they have a new tag, but it's not yet on PyPI
        "iminuit>=1.3",
        "probfit",
        "numdifftools",
        "pachyderm",
    ],

    # Include additional files
    include_package_data=True,

    extras_require = {
        "tests": [
            "pytest",
            "pytest-cov",
            "pytest-mock",
            "codecov",
        ],
        "docs": [
            "sphinx",
            # Allow markdown files to be used
            "recommonmark",
            # Allow markdown tables to be used
            "sphinx_markdown_tables",
        ],
        "dev": [
            "pre-commit",
            "flake8",
            # Makes flake8 easier to parse
            "flake8-colors",
            "mypy",
            "yamllint",
        ]
    }
)
