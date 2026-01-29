#!/usr/bin/env python3
"""
TE Subfamily Dynamics Analysis

Analyzes the emergence, expansion, and replacement of TE subfamilies:
- Cluster TEs by sequence similarity (divergence) into subfamilies
- Track subfamily abundance over time
- Calculate subfamily turnover rates
- Identify dominant and emerging subfamilies
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Set
from dataclasses import dataclass
from collections import defaultdict
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import squareform


@dataclass
class Subfamily:
    """Represents a TE subfamily."""
    subfamily_id: int
    founding_te_id: int
    founding_generation: int
    lineage_id: int
    mean_divergence: float
    member_count: int


@dataclass
class SubfamilyTransition:
    """Represents a transition in subfamily dominance."""
    generation: int
    old_dominant: int
    new_dominant: int
    old_abundance: int
    new_abundance: int


class SubfamilyAnalyzer:
    """Analyze TE subfamily dynamics from simulation output."""

    def __init__(self, output_dir: str):
        """
        Initialize the analyzer.

        Args:
            output_dir: Directory containing simulation output files
        """
        self.output_dir = Path(output_dir)
        self.genealogy_file = self.output_dir / "te_genealogy.tsv"
        self.census_file = self.output_dir / "te_census.tsv"
        self.summary_file = self.output_dir / "population_summary.tsv"

        self.genealogy_df: Optional[pd.DataFrame] = None
        self.census_df: Optional[pd.DataFrame] = None
        self.summary_df: Optional[pd.DataFrame] = None

    def load_data(self):
        """Load all relevant data files."""
        if self.genealogy_file.exists():
            self.genealogy_df = pd.read_csv(self.genealogy_file, sep='\t')

        if self.census_file.exists():
            self.census_df = pd.read_csv(self.census_file, sep='\t')

        if self.summary_file.exists():
            self.summary_df = pd.read_csv(self.summary_file, sep='\t')

    def get_lineage_abundances(self) -> pd.DataFrame:
        """
        Get abundance of each lineage over time.

        Returns:
            DataFrame with lineage abundances per generation
        """
        if self.census_df is None:
            self.load_data()

        if self.census_df is None or self.census_df.empty:
            return pd.DataFrame()

        # Count TEs per lineage per generation
        abundances = self.census_df.groupby(['generation', 'lineage_id']).size().reset_index(name='count')

        return abundances

    def get_lineage_history(self) -> pd.DataFrame:
        """
        Get pivoted lineage history (generations as rows, lineages as columns).

        Returns:
            DataFrame with generation index and lineage columns
        """
        abundances = self.get_lineage_abundances()

        if abundances.empty:
            return pd.DataFrame()

        # Pivot
        pivot = abundances.pivot(
            index='generation',
            columns='lineage_id',
            values='count'
        ).fillna(0)

        return pivot

    def identify_subfamilies_by_divergence(self, divergence_threshold: int = 5) -> Dict[int, int]:
        """
        Cluster TEs into subfamilies based on divergence.

        TEs with divergence difference <= threshold are in same subfamily.

        Args:
            divergence_threshold: Maximum divergence difference within subfamily

        Returns:
            Dictionary mapping TE ID to subfamily ID
        """
        if self.genealogy_df is None:
            self.load_data()

        if self.genealogy_df is None or self.genealogy_df.empty:
            return {}

        # Get unique TEs with their divergence
        te_info = self.genealogy_df[['child_id', 'divergence', 'lineage_id']].drop_duplicates()

        # Simple clustering: group by lineage and divergence bin
        te_info['divergence_bin'] = te_info['divergence'] // divergence_threshold

        # Create subfamily ID from lineage + divergence bin
        te_info['subfamily_id'] = (
            te_info['lineage_id'].astype(str) + '_' +
            te_info['divergence_bin'].astype(str)
        )

        # Convert to numeric subfamily IDs
        unique_subfamilies = te_info['subfamily_id'].unique()
        subfamily_map = {sf: i for i, sf in enumerate(unique_subfamilies)}

        te_to_subfamily = {}
        for _, row in te_info.iterrows():
            te_to_subfamily[int(row['child_id'])] = subfamily_map[row['subfamily_id']]

        return te_to_subfamily

    def track_subfamily_abundances(self, divergence_threshold: int = 5) -> pd.DataFrame:
        """
        Track subfamily abundances over time.

        Args:
            divergence_threshold: Threshold for subfamily clustering

        Returns:
            DataFrame with subfamily abundances per generation
        """
        te_to_subfamily = self.identify_subfamilies_by_divergence(divergence_threshold)

        if not te_to_subfamily or self.census_df is None:
            return pd.DataFrame()

        # Add subfamily ID to census
        census = self.census_df.copy()
        census['subfamily_id'] = census['te_id'].map(te_to_subfamily)

        # Count per subfamily per generation
        abundances = census.groupby(['generation', 'subfamily_id']).size().reset_index(name='count')

        return abundances

    def get_dominant_subfamily(self, generation: int) -> Tuple[Optional[int], int]:
        """
        Get the dominant subfamily at a given generation.

        Args:
            generation: Generation number

        Returns:
            Tuple of (subfamily_id, abundance) or (None, 0) if no data
        """
        abundances = self.track_subfamily_abundances()

        if abundances.empty:
            return None, 0

        gen_data = abundances[abundances['generation'] == generation]

        if gen_data.empty:
            return None, 0

        dominant = gen_data.loc[gen_data['count'].idxmax()]
        return int(dominant['subfamily_id']), int(dominant['count'])

    def detect_subfamily_transitions(self, min_dominance_duration: int = 50) -> List[SubfamilyTransition]:
        """
        Detect transitions in subfamily dominance.

        Args:
            min_dominance_duration: Minimum generations of dominance before counting

        Returns:
            List of SubfamilyTransition events
        """
        abundances = self.track_subfamily_abundances()

        if abundances.empty:
            return []

        # Get dominant subfamily per generation
        dominant_per_gen = abundances.loc[
            abundances.groupby('generation')['count'].idxmax()
        ][['generation', 'subfamily_id', 'count']].copy()
        dominant_per_gen.columns = ['generation', 'dominant', 'abundance']
        dominant_per_gen = dominant_per_gen.sort_values('generation')

        transitions = []
        prev_dominant = None
        dominance_start = None

        for _, row in dominant_per_gen.iterrows():
            gen = int(row['generation'])
            dominant = int(row['dominant'])
            abundance = int(row['abundance'])

            if prev_dominant is None:
                prev_dominant = dominant
                dominance_start = gen
            elif dominant != prev_dominant:
                # Check if previous dominance was long enough
                if dominance_start and gen - dominance_start >= min_dominance_duration:
                    # Get previous abundance
                    prev_row = dominant_per_gen[dominant_per_gen['generation'] < gen].iloc[-1]

                    transition = SubfamilyTransition(
                        generation=gen,
                        old_dominant=prev_dominant,
                        new_dominant=dominant,
                        old_abundance=int(prev_row['abundance']),
                        new_abundance=abundance
                    )
                    transitions.append(transition)

                prev_dominant = dominant
                dominance_start = gen

        return transitions

    def calculate_turnover_rate(self, window_size: int = 100) -> pd.DataFrame:
        """
        Calculate subfamily turnover rate over time.

        Turnover is measured as the change in subfamily composition.

        Args:
            window_size: Size of window for turnover calculation

        Returns:
            DataFrame with turnover rates per window
        """
        abundances = self.track_subfamily_abundances()

        if abundances.empty:
            return pd.DataFrame()

        # Get unique generations
        generations = sorted(abundances['generation'].unique())
        max_gen = max(generations)

        turnover_data = []

        for start_gen in range(min(generations), max_gen - window_size + 1, window_size):
            end_gen = start_gen + window_size

            # Get subfamilies at start and end of window
            start_data = abundances[abundances['generation'] == start_gen]
            end_data = abundances[abundances['generation'] == end_gen]

            if start_data.empty or end_data.empty:
                continue

            start_subfamilies = set(start_data['subfamily_id'])
            end_subfamilies = set(end_data['subfamily_id'])

            # Calculate turnover metrics
            new_subfamilies = end_subfamilies - start_subfamilies
            lost_subfamilies = start_subfamilies - end_subfamilies
            persistent = start_subfamilies & end_subfamilies

            total = len(start_subfamilies | end_subfamilies)
            turnover_rate = (len(new_subfamilies) + len(lost_subfamilies)) / max(total, 1)

            turnover_data.append({
                'window_start': start_gen,
                'window_end': end_gen,
                'n_start_subfamilies': len(start_subfamilies),
                'n_end_subfamilies': len(end_subfamilies),
                'n_new': len(new_subfamilies),
                'n_lost': len(lost_subfamilies),
                'n_persistent': len(persistent),
                'turnover_rate': turnover_rate
            })

        return pd.DataFrame(turnover_data)

    def get_subfamily_statistics(self) -> pd.DataFrame:
        """
        Get statistics for each subfamily.

        Returns:
            DataFrame with subfamily statistics
        """
        if self.genealogy_df is None:
            self.load_data()

        te_to_subfamily = self.identify_subfamilies_by_divergence()
        abundances = self.track_subfamily_abundances()

        if not te_to_subfamily or abundances.empty:
            return pd.DataFrame()

        # Get subfamily info from genealogy
        genealogy = self.genealogy_df.copy()
        genealogy['subfamily_id'] = genealogy['child_id'].map(te_to_subfamily)

        stats = []
        for sf_id in abundances['subfamily_id'].unique():
            sf_data = genealogy[genealogy['subfamily_id'] == sf_id]
            sf_abundance = abundances[abundances['subfamily_id'] == sf_id]

            if sf_data.empty:
                continue

            stats.append({
                'subfamily_id': sf_id,
                'founding_generation': int(sf_data['generation'].min()),
                'last_generation': int(sf_abundance['generation'].max()),
                'lineage_id': int(sf_data['lineage_id'].iloc[0]),
                'mean_divergence': float(sf_data['divergence'].mean()),
                'max_divergence': int(sf_data['divergence'].max()),
                'total_members': len(sf_data),
                'peak_abundance': int(sf_abundance['count'].max()),
                'lifespan': int(sf_abundance['generation'].max() - sf_data['generation'].min())
            })

        return pd.DataFrame(stats)

    def summary(self) -> Dict:
        """Generate summary of subfamily dynamics."""
        stats_df = self.get_subfamily_statistics()
        transitions = self.detect_subfamily_transitions()
        turnover = self.calculate_turnover_rate()
        lineage_history = self.get_lineage_history()

        summary = {
            'n_subfamilies': len(stats_df) if not stats_df.empty else 0,
            'n_lineages': lineage_history.shape[1] if not lineage_history.empty else 0,
            'n_transitions': len(transitions),
        }

        if not stats_df.empty:
            summary['mean_subfamily_lifespan'] = float(stats_df['lifespan'].mean())
            summary['max_peak_abundance'] = int(stats_df['peak_abundance'].max())
            summary['mean_divergence_spread'] = float(stats_df['max_divergence'].mean())

        if not turnover.empty:
            summary['mean_turnover_rate'] = float(turnover['turnover_rate'].mean())

        return summary


def main():
    """Command-line interface for subfamily analysis."""
    import argparse
    import json

    parser = argparse.ArgumentParser(description="Analyze TE subfamily dynamics")
    parser.add_argument('output_dir', help='Simulation output directory')
    parser.add_argument('--transitions', action='store_true', help='Detect dominance transitions')
    parser.add_argument('--turnover', action='store_true', help='Calculate turnover rates')
    parser.add_argument('--stats', action='store_true', help='Subfamily statistics')
    parser.add_argument('--summary', action='store_true', help='Print summary')

    args = parser.parse_args()

    analyzer = SubfamilyAnalyzer(args.output_dir)

    if args.summary:
        summary = analyzer.summary()
        print(json.dumps(summary, indent=2))

    if args.transitions:
        transitions = analyzer.detect_subfamily_transitions()
        print(f"\nDetected {len(transitions)} subfamily transitions:")
        for t in transitions:
            print(f"  Gen {t.generation}: subfamily {t.old_dominant} -> {t.new_dominant}")

    if args.turnover:
        turnover = analyzer.calculate_turnover_rate()
        print("\nTurnover rates:")
        print(turnover.to_string())

    if args.stats:
        stats = analyzer.get_subfamily_statistics()
        print("\nSubfamily statistics:")
        print(stats.to_string())


if __name__ == "__main__":
    main()
