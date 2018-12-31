# Jet-hadron analysis

This section of the repository contains analysis code, run macros, tools, abstracts, presentations, etc. Most
are predominately documented in the code themselves.

The code is structured as:

- Main analysis code:
    - `jet_hadron/analysis` contains the main analysis executables, including the correlations analysis and
      the response matrix.
        - The directory also contains a number of supporting modules.
- Plotting code:
    - Predominately in the `jet_hadron/plot` package.
        - `PlotBase` contains shared plotting functions
        - The other modules are the main plotting modules, split up by functionality.

# Quick start

This package requires python 3.6 and above. A few prerequisites are required which unfortunately cannot be
resolved solely by pip because of the packaging details of `probfit`.

```bash
$ pip install numpy cython
```

Then proceed with the normal installation:

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

It is strongly recommended to run

```bash
$ pre-commit install
```

to utilize the git pre-commit checks. They can be run with `pre-commit run` (and they will be run
automatically on each commit).

# Highlight Reaction Plane Fit Regions

It is useful to be able to highlight regions of a 2D surface plot to show where the Reaction Plane Fit (RPF)
is actually fitting. Code to create this plot is in `PlotRPFRegions.py`. In short, it modifies the colors the
of the plot in the regions that we want to highlight.

To execute it, the user must specify the root file and the histogram name in that root file. It should be
something like

```
python PlotRPFRegions.py -f output/plotting/PbPb/3360/dev/Track/PbPb_correlations.root -i "jetHDEtaDPhi_jetPt1_trackPt4_corr"
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

## Dependencies

Install the required dependencies by running

```
pip install --user numpy matplotlib seaborn rootpy root_numpy
```

Be certain that ROOT is available (for exampled, loaded with `alibuild`). The packages will be installed to
your user pip directory, so sudo isn't needed.

# Import profiling

To build a description of which modules are imported by which modules, use `profimp`, available
[here](https://github.com/boris-42/profimp) (and via `pip`). Invoke with (for example), `profimp "import
JetHAnalysis" --html > prof.html`. Note that this can be slow for modules which import ROOT.
