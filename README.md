# Electrolyte speciation for MD trajectories

electrolyte-speciation is a Python library and command-line tool for analysing
molecular-dynamics trajectories with [MDAnalysis](https://www.mdanalysis.org/).
It calculates coordination numbers, Free/SSIP/CIP/AGG ion-pairing populations,
local-composition state frequencies, and radial distribution functions (RDFs)
with periodic boundary conditions.

## Tested system

The supplied example configuration was tested with a **LiFSI in DME:TOL
electrolyte**. The associated study is in preparation and will be cited here
when published. This does not validate the package for every electrolyte,
force field, topology, or choice of analysis cutoffs.

## Installation

From a clone of this repository:

```bash
python -m pip install .
```

For test dependencies:

```bash
python -m pip install ".[test]"
```

## Command-line use

Copy and adapt [the LiFSI/DME:TOL example](examples/lifsi_dme_tol.toml). Set
the topology and trajectory paths, MDAnalysis selections, and cutoffs for your
system, then run:

```bash
speciation examples/lifsi_dme_tol.toml
```

Equivalent module invocation:

```bash
python -m speciation examples/lifsi_dme_tol.toml
```

Distances are specified in Ångström. Results are written to the configured
output directory: time series, state frequencies, coordination-number summary,
and RDF CSV files.

The example configuration is included in the repository, but trajectory and
topology files are not. Provide your own matching GROMACS input files before
running an analysis.

## Python API

```python
from speciation import load_config, run_analysis

config = load_config("my_analysis.toml")
run_analysis(config)
```

`AnalysisConfig` and `SpeciesSpec` are also public, so configurations can be
constructed directly in Python for programmatic workflows.

## Development

```bash
pytest
```

Trajectory and output files are intentionally excluded from version control.
Before a public release, add a license that reflects the rights holder's choice
and add the paper citation/DOI when available.
