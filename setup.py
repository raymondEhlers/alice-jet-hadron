#/usr/bin/env python

# Setup Jet-H analysis
# Derived from the setup.py in aliBuild and Overwatch
# and based on: https://python-packaging.readthedocs.io/en/latest/index.html

# Always prefer setuptools over distutils
from setuptools import setup, find_packages
# To use a consistent encoding
from codecs import open
import os
from typing import Any, cast, Dict

def get_version() -> str:
    version_module: Dict[str, Any] = {}
    with open(os.path.join("jet_hadron", "version.py")) as f:
        exec(f.read(), version_module)
    return cast(str, version_module["__version__"])

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
            "jetHCorrelations = jet_hadron.analysis.correlations:run_from_terminal",
            "jetHResponse = jet_hadron.analysis.response_matrix:run_from_terminal",
            "plotEMCalCorrections = jet_hadron.analysis.EMCal_analysis_tasks:run_plot_EMCal_corrections_hists_from_terminal",
            "plotEMCalEmbedding = jet_hadron.analysis.EMCal_analysis_tasks:run_plot_EMCal_embedding_hists_from_terminal",
            "plotRPFRegions = jet_hadron.plot.highlight_RPF:run_from_terminal"
        ],
    },

    # This is usually the minimal set of the required packages.
    # Packages should be installed via pip -r requirements.txt !
    install_requires=[
        "ruamel.yaml",
        "IPython",
        "scipy",
        "numpy",
        "matplotlib",
        "seaborn",
        "numdifftools",
        "pachyderm",
        "reaction_plane_fit",
        "coloredlogs",
        "enlighten",
        "numpythia",
        "pyjet",
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
