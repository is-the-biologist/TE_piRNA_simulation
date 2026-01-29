#!/usr/bin/env python3
"""
Migration Analysis for Multi-Population TE Dynamics

Analyzes TE evolution in multi-population scenarios:
- Track TE divergence between populations
- Analyze invasion dynamics after migration resumes
- Compare TE diversity across populations
- Model reproductive isolation effects
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass


@dataclass
class PopulationComparison:
    """Comparison metrics between two populations."""
    generation: int
    pop1_te_count: float
    pop2_te_count: float
    pop1_lineages: int
    pop2_lineages: int
    shared_lineages: int
    unique_to_pop1: int
    unique_to_pop2: int
    lineage_similarity: float  # Jaccard index


@dataclass
class InvasionEvent:
    """Represents a TE invasion from one population to another."""
    generation: int
    source_population: int
    target_population: int
    lineage_id: int
    initial_count: int
    peak_count: int
    peak_generation: int


class MigrationAnalyzer:
    """
    Analyze TE dynamics in multi-population simulations.

    Note: This analyzer is designed for future two-population simulations.
    Currently provides utility functions and framework for migration analysis.
    """

    def __init__(self, output_dir: str, n_populations: int = 2):
        """
        Initialize the migration analyzer.

        Args:
            output_dir: Directory containing simulation output files
            n_populations: Number of populations in the simulation
        """
        self.output_dir = Path(output_dir)
        self.n_populations = n_populations

        # File paths (assuming multi-population output format)
        self.census_files = [
            self.output_dir / f"te_census_pop{i}.tsv"
            for i in range(n_populations)
        ]
        self.summary_files = [
            self.output_dir / f"population_summary_pop{i}.tsv"
            for i in range(n_populations)
        ]

        self.census_dfs: List[Optional[pd.DataFrame]] = [None] * n_populations
        self.summary_dfs: List[Optional[pd.DataFrame]] = [None] * n_populations

    def load_data(self):
        """Load data from all population files."""
        for i in range(self.n_populations):
            if self.census_files[i].exists():
                self.census_dfs[i] = pd.read_csv(self.census_files[i], sep='\t')
            if self.summary_files[i].exists():
                self.summary_dfs[i] = pd.read_csv(self.summary_files[i], sep='\t')

    def compare_populations(self, generation: int) -> Optional[PopulationComparison]:
        """
        Compare TE composition between populations at a given generation.

        Args:
            generation: Generation to compare

        Returns:
            PopulationComparison object or None if data unavailable
        """
        if self.n_populations != 2:
            raise ValueError("Population comparison currently supports 2 populations")

        if self.census_dfs[0] is None or self.census_dfs[1] is None:
            self.load_data()

        df0 = self.census_dfs[0]
        df1 = self.census_dfs[1]

        if df0 is None or df1 is None:
            return None

        # Filter to generation
        pop0 = df0[df0['generation'] == generation]
        pop1 = df1[df1['generation'] == generation]

        if pop0.empty and pop1.empty:
            return None

        # Get lineage sets
        lineages0 = set(pop0['lineage_id'].unique()) if not pop0.empty else set()
        lineages1 = set(pop1['lineage_id'].unique()) if not pop1.empty else set()

        shared = lineages0 & lineages1
        unique0 = lineages0 - lineages1
        unique1 = lineages1 - lineages0

        # Jaccard similarity
        union = lineages0 | lineages1
        similarity = len(shared) / len(union) if union else 0.0

        return PopulationComparison(
            generation=generation,
            pop1_te_count=len(pop0),
            pop2_te_count=len(pop1),
            pop1_lineages=len(lineages0),
            pop2_lineages=len(lineages1),
            shared_lineages=len(shared),
            unique_to_pop1=len(unique0),
            unique_to_pop2=len(unique1),
            lineage_similarity=similarity
        )

    def track_lineage_divergence(self) -> pd.DataFrame:
        """
        Track how lineage composition diverges between populations over time.

        Returns:
            DataFrame with divergence metrics per generation
        """
        if self.census_dfs[0] is None:
            self.load_data()

        if self.census_dfs[0] is None or self.census_dfs[1] is None:
            return pd.DataFrame()

        # Get all generations
        all_gens = set(self.census_dfs[0]['generation'].unique())
        all_gens.update(self.census_dfs[1]['generation'].unique())

        comparisons = []
        for gen in sorted(all_gens):
            comp = self.compare_populations(gen)
            if comp:
                comparisons.append({
                    'generation': comp.generation,
                    'pop1_te_count': comp.pop1_te_count,
                    'pop2_te_count': comp.pop2_te_count,
                    'pop1_lineages': comp.pop1_lineages,
                    'pop2_lineages': comp.pop2_lineages,
                    'shared_lineages': comp.shared_lineages,
                    'lineage_similarity': comp.lineage_similarity
                })

        return pd.DataFrame(comparisons)

    def detect_invasions(self, min_abundance: int = 10) -> List[InvasionEvent]:
        """
        Detect TE invasion events (lineage appearing in new population).

        Args:
            min_abundance: Minimum abundance to count as invasion

        Returns:
            List of InvasionEvent objects
        """
        if self.census_dfs[0] is None:
            self.load_data()

        if self.census_dfs[0] is None or self.census_dfs[1] is None:
            return []

        invasions = []

        for source_pop, target_pop in [(0, 1), (1, 0)]:
            source_df = self.census_dfs[source_pop]
            target_df = self.census_dfs[target_pop]

            if source_df is None or target_df is None:
                continue

            # Get lineages present in source at start
            first_gen = source_df['generation'].min()
            source_initial = set(
                source_df[source_df['generation'] == first_gen]['lineage_id']
            )

            # Track each lineage in target population
            for lineage in source_initial:
                lineage_data = target_df[target_df['lineage_id'] == lineage]

                if lineage_data.empty:
                    continue

                # Check if this lineage wasn't initially in target
                target_initial = set(
                    target_df[target_df['generation'] == first_gen]['lineage_id']
                )

                if lineage in target_initial:
                    continue

                # This is an invasion - track its dynamics
                first_appearance = lineage_data['generation'].min()
                counts_over_time = lineage_data.groupby('generation').size()

                if counts_over_time.max() >= min_abundance:
                    peak_gen = counts_over_time.idxmax()
                    invasion = InvasionEvent(
                        generation=first_appearance,
                        source_population=source_pop,
                        target_population=target_pop,
                        lineage_id=lineage,
                        initial_count=int(counts_over_time.iloc[0]),
                        peak_count=int(counts_over_time.max()),
                        peak_generation=int(peak_gen)
                    )
                    invasions.append(invasion)

        return invasions

    def calculate_fst_proxy(self, generation: int) -> float:
        """
        Calculate a simple FST-like metric based on lineage frequencies.

        This is a proxy for genetic differentiation based on TE lineage frequencies.

        Args:
            generation: Generation to calculate

        Returns:
            FST-like value (0 = identical, 1 = completely different)
        """
        comp = self.compare_populations(generation)
        if comp is None:
            return np.nan

        # Simple FST proxy based on lineage sharing
        # FST ~ 1 - similarity
        return 1.0 - comp.lineage_similarity

    def simulate_migration_scenario(self,
                                   isolation_generations: int,
                                   migration_rate: float) -> Dict:
        """
        Generate parameters for a migration scenario simulation.

        This is a helper to set up migration experiments.

        Args:
            isolation_generations: Generations of complete isolation
            migration_rate: Migration rate after isolation

        Returns:
            Dictionary of simulation parameters
        """
        return {
            'migration': {
                'enabled': True,
                'isolation_period': isolation_generations,
                'migration_rate': migration_rate,
                'scenario': 'isolation_then_contact'
            },
            'expected_outcomes': {
                'during_isolation': 'Lineage divergence increases',
                'after_contact': 'Potential TE invasion and competition',
                'equilibrium': 'Depends on relative fitness and silencing'
            }
        }

    def summary(self) -> Dict:
        """Generate summary of migration analysis."""
        divergence = self.track_lineage_divergence()
        invasions = self.detect_invasions()

        summary = {
            'n_populations': self.n_populations,
            'n_invasions_detected': len(invasions),
            'data_available': all(df is not None for df in self.census_dfs)
        }

        if not divergence.empty:
            summary['initial_similarity'] = float(divergence.iloc[0]['lineage_similarity'])
            summary['final_similarity'] = float(divergence.iloc[-1]['lineage_similarity'])
            summary['min_similarity'] = float(divergence['lineage_similarity'].min())

        return summary


def main():
    """Command-line interface for migration analysis."""
    import argparse
    import json

    parser = argparse.ArgumentParser(description="Analyze multi-population TE dynamics")
    parser.add_argument('output_dir', help='Simulation output directory')
    parser.add_argument('--divergence', action='store_true', help='Track population divergence')
    parser.add_argument('--invasions', action='store_true', help='Detect invasion events')
    parser.add_argument('--summary', action='store_true', help='Print summary')
    parser.add_argument('--fst', type=int, help='Calculate FST at generation')

    args = parser.parse_args()

    analyzer = MigrationAnalyzer(args.output_dir)

    if args.summary:
        summary = analyzer.summary()
        print(json.dumps(summary, indent=2))

    if args.divergence:
        div = analyzer.track_lineage_divergence()
        if not div.empty:
            print("\nPopulation divergence over time:")
            print(div.to_string())
        else:
            print("\nNo multi-population data available")

    if args.invasions:
        invasions = analyzer.detect_invasions()
        print(f"\nDetected {len(invasions)} invasion events:")
        for inv in invasions:
            print(f"  Gen {inv.generation}: lineage {inv.lineage_id} "
                  f"pop{inv.source_population} -> pop{inv.target_population}")

    if args.fst is not None:
        fst = analyzer.calculate_fst_proxy(args.fst)
        print(f"\nFST proxy at generation {args.fst}: {fst:.4f}")


if __name__ == "__main__":
    main()
