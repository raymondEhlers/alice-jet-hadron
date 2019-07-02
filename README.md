# Jet-hadron analysis

This repository contains jet-hadron analysis code, run macros, tools, etc. Most are predominately documented
through the docstrings in the code itself.

# Quick start

First, setup your environment as you would like (virtualenv, alibuild, etc). This package requires python 3.6
and above. First, a few prerequisites are required which unfortunately cannot be resolved solely by pip
because of the packaging details of `probfit` (which is depended on by `reaction_plane_fit`).

```bash
$ pip install numpy cython
```

Then the analysis package can be installed as normal with:

```bash
$ pip install -e .
```

## Running the tests

Tests are implemented using `pytest`. They should be run via

```
$ pytest -l --cov=jetH --cov-report html --cov-branch . --durations=5
```

Coverage can be measured by adding the argument `--cov=module` (for example, `--cov=JetHConfig`). When the
Jet-H analysis code is entirely packaged up, the module can just be `jetHAnalysis`, and it will test
everything.

## Development

You're now ready to develop.

The code is structured as:

- Main analysis code in the `jet_hadron/analysis` package:
    - Correlations analysis in the `correlations` module.
    - ALICE jet-hadron response matrix in the `response_matrix` module.
    - STAR jet-hadron response matrix from simulated data in the `STAR_response_matrix` module.
        - This is based on the ALICE response matrix classes.
    - A number of supporting modules.
    - A comprehensive list of executables can be found in the `entry_points` of `setup.py`.
- Plotting code is predominately in the `jet_hadron/plot` package.
        - `PlotBase` contains shared plotting functions
        - The other modules are the main plotting modules, split up by functionality.
- Event generator steering in `jet_hadron/event_gen`.
    - Includes simulating PYTHIA 6 for STAR at particle and detector level (when provided the appropriate
    input files).
    - PYTHIA 8 is also supported.
- Toy models tests are in `jet_hadron/toy_models`.

Most executables take a uniform set of arguments consisting of a YAML configuration file, along with the
parameters of the analysis (collision system, energy, event activity, and leading hadron bias). The flags can
be accessed through the `--help` flag. A YAML config file that stores most of the configuration options is in
the `config` directory.

For example, the jet-hadron correlations analysis can be run with:

```bash
$ jetHCorrelations -c config/analysisConfigDev.yaml -e 5.02 -s PbPb -a semi_central -b track
```

This would analyze semi-central Pb--Pb collisions at 5.02 TeV with a leading track bias as defined in the
configuration file. Note that the location of input and output files are specified in the YAML configuration.

### Pre-commit checks

It is strongly recommended to install `pre-commit` (from pip or elsewhere) and then run

```bash
$ pre-commit install
```

to utilize the git pre-commit checks. They will be run automatically with each commit to help ensure code
quality. (They can also be run manually with `pre-commit run`).

# Tools

## Highlight Reaction Plane Fit Regions

It is useful to be able to highlight regions of a 2D surface plot to show where the Reaction Plane Fit (RPF)
is actually fitting. Code to create this plot is in `PlotRPFRegions.py`. In short, it modifies the colors the
of the plot in the regions that we want to highlight.

To execute it, the user must specify the root file and the histogram name in that root file. It should be
something like

```bash
$ plotRPFRegions -f output/plotting/PbPb/3360/dev/Track/PbPb_correlations.root -i "jetHDEtaDPhi_jetPt1_trackPt4_corr"
```

All command line options are available with the `-h` option. While the Jet-hadron plot is integrated into the
plotting code, others should look at the code in `PlotRPFRegions`. In particularly, look at the standalone
functions located at the bottom of the file. These specify a variety of additional options that would be too
cumbersome to pass from the command line.

Some options include:

- Changing highlighted colors and ranges in the function `defineHighlightRegions()`
    - Can also set them to use transparency, so the underlying plot will show through. NOTE: This is not a
      perfect effect, but still probably worth taking a look at
- Changing color scheme (by default it is the ROOT6 kBird color scheme) in the function `plotRPFRegions()`
- Modify labels, etc in the function `plotRPFRegions()`

Note that it is best to execute it in the `jetH` folder, as a number of other modules are utilized and would
also need to be copied!

