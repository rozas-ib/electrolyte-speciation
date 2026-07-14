"""Configuration models and TOML loading for speciation analyses."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal
import tomllib

PointMode = Literal["atoms", "residue_com"]
Rule = Literal["POINT", "ATOMWISE", "RESIDUE_ANY", "RESIDUE_ATLEAST", "RESIDUE_ALL"]


@dataclass(frozen=True)
class SpeciesSpec:
    """Definition of one entity type counted around each reference point."""

    name: str
    target_sel: str
    target_mode: PointMode
    rule: Rule = "POINT"
    atleast_k: int = 1
    cut1: float = 3.0
    cut2: float | None = 8.0
    enabled: bool = True
    include_in_state: bool = True
    pairing_partner: bool = False
    compute_rdf: bool = True

    def __post_init__(self) -> None:
        if self.cut1 <= 0:
            raise ValueError(f"[{self.name}] cut1 must be positive.")
        if self.cut2 is not None and self.cut2 <= self.cut1:
            raise ValueError(f"[{self.name}] cut2 must be greater than cut1.")
        if self.atleast_k < 1:
            raise ValueError(f"[{self.name}] atleast_k must be at least 1.")

    def as_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "target_sel": self.target_sel,
            "target_mode": self.target_mode,
            "rule": self.rule,
            "atleast_k": self.atleast_k,
            "cut1": self.cut1,
            "cut2": self.cut2,
            "enabled": self.enabled,
            "include_in_state": self.include_in_state,
            "pairing_partner": self.pairing_partner,
            "compute_rdf": self.compute_rdf,
        }


@dataclass(frozen=True)
class AnalysisConfig:
    """All inputs and controls required to run one speciation analysis."""

    topology_file: Path
    trajectory_file: Path
    species: tuple[SpeciesSpec, ...]
    output_dir: Path = Path("speciation_outputs")
    start_frame: int | None = None
    stop_frame: int | None = None
    step_frame: int = 1
    apply_unwrap: bool = True
    guess_bonds_if_missing: bool = False
    ref_sel_str: str = "resname Li"
    ref_mode: PointMode = "atoms"
    pair_cut1: float = 3.0
    pair_cut2: float = 8.0
    include_shell2_in_states: bool = False
    count_cap: int | None = 4
    top_states_n: int = 15
    rdf_default_rmax: float = 20.0
    rdf_default_nbins: int = 1000
    compute_total_rdf: bool = True
    write_counts_instead_of_fractions: bool = False

    def __post_init__(self) -> None:
        if not self.species:
            raise ValueError("At least one species definition is required.")
        if self.step_frame < 1:
            raise ValueError("step_frame must be at least 1.")
        if self.pair_cut1 <= 0 or self.pair_cut2 <= self.pair_cut1:
            raise ValueError("pair_cut2 must be greater than positive pair_cut1.")
        if self.rdf_default_rmax <= 0 or self.rdf_default_nbins < 1:
            raise ValueError("RDF range and bin count must be positive.")


def load_config(path: str | Path) -> AnalysisConfig:
    """Load an :class:`AnalysisConfig` from a TOML file."""
    config_path = Path(path)
    with config_path.open("rb") as handle:
        raw = tomllib.load(handle)

    analysis = raw.get("analysis", {})
    reference = raw.get("reference", {})
    pairing = raw.get("pairing", {})
    state = raw.get("state", {})
    rdf = raw.get("rdf", {})
    execution = raw.get("execution", {})
    species = tuple(SpeciesSpec(**item) for item in raw.get("species", []))

    return AnalysisConfig(
        topology_file=Path(analysis["topology_file"]),
        trajectory_file=Path(analysis["trajectory_file"]),
        output_dir=Path(analysis.get("output_dir", "speciation_outputs")),
        species=species,
        start_frame=execution.get("start_frame"),
        stop_frame=execution.get("stop_frame"),
        step_frame=execution.get("step_frame", 1),
        apply_unwrap=execution.get("apply_unwrap", True),
        guess_bonds_if_missing=execution.get("guess_bonds_if_missing", False),
        ref_sel_str=reference.get("selection", "resname Li"),
        ref_mode=reference.get("mode", "atoms"),
        pair_cut1=pairing.get("cut1", 3.0),
        pair_cut2=pairing.get("cut2", 8.0),
        include_shell2_in_states=state.get("include_shell2", False),
        count_cap=state.get("count_cap", 4),
        top_states_n=state.get("top_states_n", 15),
        rdf_default_rmax=rdf.get("rmax", 20.0),
        rdf_default_nbins=rdf.get("nbins", 1000),
        compute_total_rdf=rdf.get("compute_total", True),
        write_counts_instead_of_fractions=analysis.get("write_counts_instead_of_fractions", False),
    )
