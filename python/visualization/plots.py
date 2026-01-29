#!/usr/bin/env python3
"""
Visualization Utilities for TE-piRNA Simulations

Provides plotting functions for:
- Population dynamics (TE counts, fitness)
- Transposition rate time series
- Phylogenetic structure
- Subfamily dynamics
- Parameter exploration heatmaps
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
import sys

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from analysis.phylogeny import TEPhylogeny
from analysis.transposition_rates import TranspositionAnalyzer
from analysis.subfamily_dynamics import SubfamilyAnalyzer


class SimulationPlotter:
    """Generate plots for simulation analysis."""

    def __init__(self, output_dir: str, style: str = 'seaborn-v0_8-whitegrid'):
        """
        Initialize the plotter.

        Args:
            output_dir: Directory containing simulation output
            style: Matplotlib style to use
        """
        self.output_dir = Path(output_dir)
        self.figures_dir = self.output_dir / "figures"
        self.figures_dir.mkdir(exist_ok=True)

        # Try to set style, fall back to default if not available
        try:
            plt.style.use(style)
        except OSError:
            plt.style.use('default')

        # Load data
        self.summary_file = self.output_dir / "population_summary.tsv"
        self.census_file = self.output_dir / "te_census.tsv"

        self.summary_df = None
        self.census_df = None

        if self.summary_file.exists():
            self.summary_df = pd.read_csv(self.summary_file, sep='\t')
        if self.census_file.exists():
            self.census_df = pd.read_csv(self.census_file, sep='\t')

    def plot_population_dynamics(self, save: bool = True) -> plt.Figure:
        """
        Plot population-level dynamics over time.

        Shows TE counts, fitness, and piRNA insertions.
        """
        if self.summary_df is None:
            raise ValueError("No summary data available")

        fig, axes = plt.subplots(2, 2, figsize=(12, 10))

        # Mean TE count
        ax = axes[0, 0]
        ax.plot(self.summary_df['generation'], self.summary_df['mean_te_count'],
                color='steelblue', linewidth=1.5)
        ax.set_xlabel('Generation')
        ax.set_ylabel('Mean TE Count per Individual')
        ax.set_title('TE Burden Over Time')
        ax.grid(True, alpha=0.3)

        # Transpositions per generation
        ax = axes[0, 1]
        ax.plot(self.summary_df['generation'], self.summary_df['total_transpositions'],
                color='coral', linewidth=1.5)
        ax.set_xlabel('Generation')
        ax.set_ylabel('Transpositions')
        ax.set_title('Transposition Events per Generation')
        ax.grid(True, alpha=0.3)

        # Mean fitness
        ax = axes[1, 0]
        ax.plot(self.summary_df['generation'], self.summary_df['mean_fitness'],
                color='forestgreen', linewidth=1.5)
        ax.set_xlabel('Generation')
        ax.set_ylabel('Mean Fitness')
        ax.set_title('Population Fitness Over Time')
        ax.grid(True, alpha=0.3)

        # Active lineages and piRNA TEs
        ax = axes[1, 1]
        ax.plot(self.summary_df['generation'], self.summary_df['active_lineages'],
                color='purple', linewidth=1.5, label='Active Lineages')
        ax2 = ax.twinx()
        ax2.plot(self.summary_df['generation'], self.summary_df['pirna_te_count'],
                 color='orange', linewidth=1.5, linestyle='--', label='piRNA TEs')
        ax.set_xlabel('Generation')
        ax.set_ylabel('Active Lineages', color='purple')
        ax2.set_ylabel('piRNA TE Count', color='orange')
        ax.set_title('Lineage Diversity and piRNA Silencing')
        ax.legend(loc='upper left')
        ax2.legend(loc='upper right')
        ax.grid(True, alpha=0.3)

        plt.tight_layout()

        if save:
            fig.savefig(self.figures_dir / "population_dynamics.png", dpi=150, bbox_inches='tight')

        return fig

    def plot_transposition_bursts(self, save: bool = True) -> plt.Figure:
        """
        Plot transposition rates with burst detection.
        """
        analyzer = TranspositionAnalyzer(str(self.output_dir))
        rates = analyzer.get_rates_per_generation()
        bursts = analyzer.detect_bursts()

        if rates.empty:
            raise ValueError("No transposition data available")

        fig, ax = plt.subplots(figsize=(12, 5))

        # Plot transposition rate
        ax.plot(rates['generation'], rates['n_transpositions'],
                color='steelblue', linewidth=0.8, alpha=0.7, label='Raw')

        # Plot smoothed rate
        smoothed = analyzer.smooth_rates(window_size=50)
        ax.plot(smoothed['generation'], smoothed['smoothed_transpositions'],
                color='darkblue', linewidth=2, label='Smoothed (50 gen)')

        # Highlight bursts
        for burst in bursts:
            ax.axvspan(burst.start_generation, burst.end_generation,
                      alpha=0.3, color='coral', label='_nolegend_')
            ax.axvline(burst.peak_generation, color='red', linestyle='--',
                      alpha=0.5, label='_nolegend_')

        ax.set_xlabel('Generation')
        ax.set_ylabel('Transpositions per Generation')
        ax.set_title(f'Transposition Dynamics ({len(bursts)} bursts detected)')
        ax.legend()
        ax.grid(True, alpha=0.3)

        plt.tight_layout()

        if save:
            fig.savefig(self.figures_dir / "transposition_bursts.png", dpi=150, bbox_inches='tight')

        return fig

    def plot_lineage_dynamics(self, save: bool = True) -> plt.Figure:
        """
        Plot lineage abundance over time as stacked area chart.
        """
        analyzer = SubfamilyAnalyzer(str(self.output_dir))
        history = analyzer.get_lineage_history()

        if history.empty:
            raise ValueError("No lineage data available")

        fig, ax = plt.subplots(figsize=(12, 6))

        # Create stacked area plot
        colors = plt.cm.tab20(np.linspace(0, 1, min(20, len(history.columns))))

        ax.stackplot(history.index, history.T, labels=history.columns,
                    colors=colors, alpha=0.8)

        ax.set_xlabel('Generation')
        ax.set_ylabel('TE Count')
        ax.set_title('TE Lineage Dynamics Over Time')

        # Only show legend if not too many lineages
        if len(history.columns) <= 10:
            ax.legend(title='Lineage ID', bbox_to_anchor=(1.02, 1), loc='upper left')

        ax.set_xlim(history.index.min(), history.index.max())
        ax.grid(True, alpha=0.3)

        plt.tight_layout()

        if save:
            fig.savefig(self.figures_dir / "lineage_dynamics.png", dpi=150, bbox_inches='tight')

        return fig

    def plot_divergence_distribution(self, generations: Optional[List[int]] = None,
                                     save: bool = True) -> plt.Figure:
        """
        Plot distribution of TE divergence values.
        """
        phylo = TEPhylogeny(str(self.output_dir))
        phylo.load_genealogy()

        if phylo.genealogy_df is None or phylo.genealogy_df.empty:
            raise ValueError("No genealogy data available")

        if generations is None:
            # Select evenly spaced generations
            max_gen = phylo.genealogy_df['generation'].max()
            generations = [int(g) for g in np.linspace(1, max_gen, 5)]

        fig, ax = plt.subplots(figsize=(10, 6))

        colors = plt.cm.viridis(np.linspace(0, 1, len(generations)))

        for gen, color in zip(generations, colors):
            divergences = phylo.get_divergence_distribution(generation=gen)
            if len(divergences) > 0:
                ax.hist(divergences, bins=30, alpha=0.5, color=color,
                       label=f'Gen {gen}', density=True)

        ax.set_xlabel('Divergence (mutations from ancestor)')
        ax.set_ylabel('Density')
        ax.set_title('TE Divergence Distribution Over Time')
        ax.legend()
        ax.grid(True, alpha=0.3)

        plt.tight_layout()

        if save:
            fig.savefig(self.figures_dir / "divergence_distribution.png", dpi=150, bbox_inches='tight')

        return fig

    def plot_phase_timeline(self, save: bool = True) -> plt.Figure:
        """
        Plot timeline of evolutionary phases (monomorphic/dimorphic/polymorphic).
        """
        phylo = TEPhylogeny(str(self.output_dir))
        phases = phylo.identify_phases(window_size=100)

        if phases.empty:
            raise ValueError("No phase data available")

        fig, ax = plt.subplots(figsize=(14, 4))

        phase_colors = {
            'monomorphic': 'lightblue',
            'dimorphic': 'lightgreen',
            'polymorphic': 'coral'
        }

        for _, row in phases.iterrows():
            ax.axhspan(0, 1, xmin=row['start_generation'] / phases['end_generation'].max(),
                      xmax=row['end_generation'] / phases['end_generation'].max(),
                      color=phase_colors.get(row['phase'], 'gray'), alpha=0.7)

        # Add legend
        from matplotlib.patches import Patch
        legend_elements = [Patch(facecolor=color, label=phase.capitalize())
                         for phase, color in phase_colors.items()]
        ax.legend(handles=legend_elements, loc='upper right')

        ax.set_xlim(0, phases['end_generation'].max())
        ax.set_ylim(0, 1)
        ax.set_xlabel('Generation')
        ax.set_yticks([])
        ax.set_title('Evolutionary Phases Over Time')

        plt.tight_layout()

        if save:
            fig.savefig(self.figures_dir / "phase_timeline.png", dpi=150, bbox_inches='tight')

        return fig

    def plot_silencing_correlation(self, save: bool = True) -> plt.Figure:
        """
        Plot relationship between piRNA count and transposition rate.
        """
        analyzer = TranspositionAnalyzer(str(self.output_dir))
        corr_df = analyzer.correlate_with_pirna()

        if corr_df.empty or 'pirna_te_count' not in corr_df.columns:
            raise ValueError("No piRNA correlation data available")

        fig, axes = plt.subplots(1, 2, figsize=(12, 5))

        # Time series comparison
        ax = axes[0]
        ax.plot(corr_df['generation'], corr_df['n_transpositions'],
                color='coral', label='Transpositions', alpha=0.7)
        ax2 = ax.twinx()
        ax2.plot(corr_df['generation'], corr_df['pirna_te_count'],
                 color='steelblue', label='piRNA TEs', alpha=0.7)
        ax.set_xlabel('Generation')
        ax.set_ylabel('Transpositions', color='coral')
        ax2.set_ylabel('piRNA TE Count', color='steelblue')
        ax.set_title('Transposition vs piRNA Silencing Over Time')

        # Scatter plot
        ax = axes[1]
        valid = corr_df.dropna()
        ax.scatter(valid['pirna_te_count'], valid['n_transpositions'],
                  alpha=0.5, c=valid['generation'], cmap='viridis')
        ax.set_xlabel('piRNA TE Count')
        ax.set_ylabel('Transpositions')
        ax.set_title('Transposition-piRNA Relationship')

        # Add correlation coefficient
        if len(valid) > 2:
            corr = np.corrcoef(valid['pirna_te_count'], valid['n_transpositions'])[0, 1]
            ax.text(0.05, 0.95, f'r = {corr:.3f}', transform=ax.transAxes,
                   verticalalignment='top')

        plt.tight_layout()

        if save:
            fig.savefig(self.figures_dir / "silencing_correlation.png", dpi=150, bbox_inches='tight')

        return fig

    def plot_parameter_heatmap(self, results_df: pd.DataFrame,
                              param1: str, param2: str, metric: str,
                              save: bool = True,
                              filename: str = "parameter_heatmap.png") -> plt.Figure:
        """
        Plot heatmap of metric values across two parameters.

        Args:
            results_df: DataFrame with parameter exploration results
            param1: First parameter (x-axis)
            param2: Second parameter (y-axis)
            metric: Metric to plot (column name)
            save: Whether to save figure
            filename: Output filename
        """
        pivot = results_df.pivot_table(
            index=param2, columns=param1, values=metric, aggfunc='mean'
        )

        fig, ax = plt.subplots(figsize=(10, 8))

        im = ax.imshow(pivot.values, cmap='viridis', aspect='auto')

        ax.set_xticks(range(len(pivot.columns)))
        ax.set_xticklabels([f'{v:.3g}' for v in pivot.columns], rotation=45)
        ax.set_yticks(range(len(pivot.index)))
        ax.set_yticklabels([f'{v:.3g}' for v in pivot.index])

        ax.set_xlabel(param1)
        ax.set_ylabel(param2)
        ax.set_title(f'{metric} by {param1} and {param2}')

        plt.colorbar(im, ax=ax, label=metric)

        plt.tight_layout()

        if save:
            fig.savefig(self.figures_dir / filename, dpi=150, bbox_inches='tight')

        return fig

    def generate_all_plots(self) -> List[str]:
        """
        Generate all available plots.

        Returns:
            List of generated figure filenames
        """
        generated = []

        try:
            self.plot_population_dynamics()
            generated.append("population_dynamics.png")
        except Exception as e:
            print(f"Could not generate population_dynamics: {e}")

        try:
            self.plot_transposition_bursts()
            generated.append("transposition_bursts.png")
        except Exception as e:
            print(f"Could not generate transposition_bursts: {e}")

        try:
            self.plot_lineage_dynamics()
            generated.append("lineage_dynamics.png")
        except Exception as e:
            print(f"Could not generate lineage_dynamics: {e}")

        try:
            self.plot_divergence_distribution()
            generated.append("divergence_distribution.png")
        except Exception as e:
            print(f"Could not generate divergence_distribution: {e}")

        try:
            self.plot_phase_timeline()
            generated.append("phase_timeline.png")
        except Exception as e:
            print(f"Could not generate phase_timeline: {e}")

        try:
            self.plot_silencing_correlation()
            generated.append("silencing_correlation.png")
        except Exception as e:
            print(f"Could not generate silencing_correlation: {e}")

        print(f"\nGenerated {len(generated)} figures in {self.figures_dir}")
        return generated


def main():
    """Command-line interface for generating plots."""
    import argparse

    parser = argparse.ArgumentParser(description="Generate simulation plots")
    parser.add_argument('output_dir', help='Simulation output directory')
    parser.add_argument('--all', action='store_true', help='Generate all plots')
    parser.add_argument('--dynamics', action='store_true', help='Population dynamics plot')
    parser.add_argument('--bursts', action='store_true', help='Transposition bursts plot')
    parser.add_argument('--lineages', action='store_true', help='Lineage dynamics plot')
    parser.add_argument('--divergence', action='store_true', help='Divergence distribution plot')
    parser.add_argument('--phases', action='store_true', help='Phase timeline plot')
    parser.add_argument('--silencing', action='store_true', help='Silencing correlation plot')
    parser.add_argument('--show', action='store_true', help='Show plots interactively')

    args = parser.parse_args()

    plotter = SimulationPlotter(args.output_dir)

    if args.all:
        plotter.generate_all_plots()
    else:
        if args.dynamics:
            plotter.plot_population_dynamics()
        if args.bursts:
            plotter.plot_transposition_bursts()
        if args.lineages:
            plotter.plot_lineage_dynamics()
        if args.divergence:
            plotter.plot_divergence_distribution()
        if args.phases:
            plotter.plot_phase_timeline()
        if args.silencing:
            plotter.plot_silencing_correlation()

    if args.show:
        plt.show()


if __name__ == "__main__":
    main()
