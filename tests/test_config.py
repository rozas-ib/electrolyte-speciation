from pathlib import Path

import pytest

from speciation import AnalysisConfig, SpeciesSpec, load_config


def test_lifsi_example_loads() -> None:
    config_path = Path(__file__).parents[1] / "examples" / "lifsi_dme_tol.toml"
    config = load_config(config_path)

    assert config.ref_sel_str == "resname Li"
    assert config.pair_cut1 == 3.0
    assert len(config.species) == 7
    assert [item.name for item in config.species if item.pairing_partner] == ["FSI_solv"]


def test_invalid_species_cutoffs_are_rejected() -> None:
    with pytest.raises(ValueError, match="cut2"):
        SpeciesSpec("FSI", "resname FSI", "atoms", cut1=3.0, cut2=3.0)


def test_analysis_requires_species() -> None:
    with pytest.raises(ValueError, match="species"):
        AnalysisConfig(topology_file=Path("top.tpr"), trajectory_file=Path("traj.xtc"), species=())
