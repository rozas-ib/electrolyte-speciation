"""Command-line interface for speciation."""

from __future__ import annotations

import argparse

from . import __version__
from .analysis import run_analysis
from .config import load_config


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze molecular-dynamics speciation.")
    parser.add_argument("config", help="Path to an analysis TOML configuration file.")
    parser.add_argument("--version", action="version", version=f"speciation {__version__}")
    args = parser.parse_args()
    run_analysis(load_config(args.config))
