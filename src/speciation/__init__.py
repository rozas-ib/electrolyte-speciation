"""Speciation analysis for molecular-dynamics trajectories."""

from .analysis import classify_ion_pairing, run_analysis
from .config import AnalysisConfig, SpeciesSpec, load_config

__all__ = [
    "AnalysisConfig",
    "SpeciesSpec",
    "classify_ion_pairing",
    "load_config",
    "run_analysis",
]

__version__ = "0.1.0"
