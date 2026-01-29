"""
Analysis modules for TE-piRNA Arms Race simulations.

This package provides tools for analyzing simulation output:
- phylogeny: TE family tree reconstruction
- transposition_rates: Transposition rate dynamics
- subfamily_dynamics: Subfamily emergence and replacement
- migration_analysis: Multi-population analysis
"""

from .phylogeny import TEPhylogeny
from .transposition_rates import TranspositionAnalyzer
from .subfamily_dynamics import SubfamilyAnalyzer

__all__ = ['TEPhylogeny', 'TranspositionAnalyzer', 'SubfamilyAnalyzer']
