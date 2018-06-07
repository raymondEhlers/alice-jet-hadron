#/usr/bin/env python

# Setup Jet-H analysis
# Derived from the setup.py in aliBuild and Overwatch
# and based on: https://python-packaging.readthedocs.io/en/latest/index.html

# Always prefer setuptools over distutils
from setuptools import setup, find_packages
# To use a consistent encoding
from codecs import open
from os import path

# Get the long description from the README file
here = path.abspath(path.dirname(__file__))
with open(path.join(here, 'README.md'), encoding='utf-8') as f:
    long_description = f.read()

setup(
    name="aliceJetHCorrelations",
    version="0.8",

    description="ALICE jet-hadron correlations analysis",
    long_description=long_description,

    author="Raymond Ehlers",
    author_email="raymond.ehlers@cern.ch",

    url="https://github.com/ALICEYale/alice-yale-dev",

    # See https://pypi.python.org/pypi?%3Aaction=list_classifiers
    classifiers=[
        # How mature is this project? Common values are
        #   3 - Alpha
        #   4 - Beta
        #   5 - Production/Stable
        'Development Status :: 4 - Beta',

        # Indicate who your project is intended for
        'Intended Audience :: Developers',

        # Specify the Python versions you support here. In particular, ensure
        # that you indicate whether you support Python 2, Python 3 or both.
        'Programming Language :: Python :: 2.7',
        'Programming Language :: Python :: 3.6',
    ],

    # What does your project relate to?
    keywords='HEP ALICE',

    packages=find_packages(exclude=(".git", "tests")),

    # Rename scripts to the desired executable names
    # See: https://stackoverflow.com/a/8506532
    entry_points = {
        "console_scripts" : [
            "jetHAnalysis = jetH.analysis.correlations:runFromTerminal",
            "jetHResponse = jetH.analysis.responseMatrix:runFromTerminal",
            "plotEMCalCorrections = jetH.analysis.EMCalAnalysisTasks:runEMCalCorrectionsHistsFromTerminal",
            "plotEMCalEmbedding = jetH.analysis.EMCalAnalysisTasks:runEMCalEmbeddingHistsFromTerminal",
            "plotRPFRegions = jetH.plot.highlightRPF:runFromTerminal"
            ],
        },

    # This is usually the minimal set of the required packages.
    # Packages should be installed via pip -r requirements.txt !
    install_requires=[
        "future",
        "ruamel.yaml",
        "aenum",
        "IPython",
        "scipy",
        "numpy",
        "matplotlib",
        "seaborn",
        "uproot", # Not strictly required at the moment, but hopefully for the future.
        "rootpy",
        "root_numpy",
        "iminuit",
        "probfit",
        "numdifftools"
    ],

    # Include additional files
    include_package_data=True,

    # Test packages
    tests_require = [
        "pytest",
        "pytest-cov",
        "pytest-mock"
    ]
  )
