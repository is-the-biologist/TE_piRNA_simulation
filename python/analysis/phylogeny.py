#!/usr/bin/env python3
"""
TE Phylogeny Reconstruction and Analysis

Reconstructs and analyzes TE family trees from genealogical data.
Identifies monomorphic, dimorphic, and polymorphic phases.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Set
from dataclasses import dataclass
from collections import defaultdict


@dataclass
class TreeNode:
    """Node in the TE phylogenetic tree."""
    te_id: int
    parent_id: int
    generation: int
    divergence: int
    lineage_id: int
    children: List['TreeNode']
    depth: int = 0


@dataclass
class TreeStats:
    """Statistics for a TE phylogenetic tree."""
    n_nodes: int
    n_leaves: int
    max_depth: int
    mean_depth: float
    n_lineages: int
    total_divergence: int
    mean_divergence: float
    branching_factor: float


class TEPhylogeny:
    """Reconstruct and analyze TE phylogenies from simulation output."""

    def __init__(self, output_dir: str):
        """
        Initialize phylogeny analyzer.

        Args:
            output_dir: Directory containing simulation output files
        """
        self.output_dir = Path(output_dir)
        self.genealogy_file = self.output_dir / "te_genealogy.tsv"
        self.genealogy_df: Optional[pd.DataFrame] = None
        self.trees: Dict[int, TreeNode] = {}  # Root TE ID -> TreeNode
        self.nodes: Dict[int, TreeNode] = {}  # All nodes by TE ID

    def load_genealogy(self) -> pd.DataFrame:
        """Load genealogy data from TSV file."""
        if not self.genealogy_file.exists():
            raise FileNotFoundError(f"Genealogy file not found: {self.genealogy_file}")

        self.genealogy_df = pd.read_csv(self.genealogy_file, sep='\t')
        return self.genealogy_df

    def build_trees(self) -> Dict[int, TreeNode]:
        """
        Build phylogenetic trees from genealogy data.

        Returns:
            Dictionary mapping root TE IDs to their tree nodes
        """
        if self.genealogy_df is None:
            self.load_genealogy()

        # Create all nodes
        self.nodes = {}
        for _, row in self.genealogy_df.iterrows():
            node = TreeNode(
                te_id=int(row['child_id']),
                parent_id=int(row['parent_id']),
                generation=int(row['generation']),
                divergence=int(row['divergence']),
                lineage_id=int(row['lineage_id']),
                children=[]
            )
            self.nodes[node.te_id] = node

        # Build parent-child relationships
        roots = []
        for node in self.nodes.values():
            if node.parent_id == -1:
                roots.append(node.te_id)
            elif node.parent_id in self.nodes:
                parent = self.nodes[node.parent_id]
                parent.children.append(node)

        # Calculate depths
        def set_depths(node: TreeNode, depth: int = 0):
            node.depth = depth
            for child in node.children:
                set_depths(child, depth + 1)

        for root_id in roots:
            self.trees[root_id] = self.nodes[root_id]
            set_depths(self.nodes[root_id])

        return self.trees

    def get_tree_stats(self, root_id: Optional[int] = None) -> TreeStats:
        """
        Calculate statistics for a tree or all trees.

        Args:
            root_id: Specific tree root, or None for all trees combined

        Returns:
            TreeStats object with tree statistics
        """
        if not self.trees:
            self.build_trees()

        if root_id is not None:
            nodes = self._get_all_descendants(self.trees[root_id])
        else:
            nodes = list(self.nodes.values())

        if not nodes:
            return TreeStats(0, 0, 0, 0.0, 0, 0, 0.0, 0.0)

        n_nodes = len(nodes)
        leaves = [n for n in nodes if not n.children]
        n_leaves = len(leaves)
        max_depth = max(n.depth for n in nodes)
        mean_depth = np.mean([n.depth for n in nodes])
        lineages = set(n.lineage_id for n in nodes)
        n_lineages = len(lineages)
        total_divergence = sum(n.divergence for n in nodes)
        mean_divergence = np.mean([n.divergence for n in nodes])

        # Branching factor: mean children per internal node
        internal_nodes = [n for n in nodes if n.children]
        branching_factor = np.mean([len(n.children) for n in internal_nodes]) if internal_nodes else 0.0

        return TreeStats(
            n_nodes=n_nodes,
            n_leaves=n_leaves,
            max_depth=max_depth,
            mean_depth=mean_depth,
            n_lineages=n_lineages,
            total_divergence=total_divergence,
            mean_divergence=mean_divergence,
            branching_factor=branching_factor
        )

    def _get_all_descendants(self, node: TreeNode) -> List[TreeNode]:
        """Get all descendants of a node including itself."""
        result = [node]
        for child in node.children:
            result.extend(self._get_all_descendants(child))
        return result

    def identify_phases(self, window_size: int = 100) -> pd.DataFrame:
        """
        Identify monomorphic, dimorphic, and polymorphic phases.

        A phase is defined by the number of active lineages in each generation window.
        - Monomorphic: 1 dominant lineage
        - Dimorphic: 2 active lineages
        - Polymorphic: 3+ active lineages

        Args:
            window_size: Generation window for phase detection

        Returns:
            DataFrame with phase information per generation window
        """
        if self.genealogy_df is None:
            self.load_genealogy()

        df = self.genealogy_df.copy()
        max_gen = df['generation'].max()

        phases = []
        for start_gen in range(1, max_gen + 1, window_size):
            end_gen = min(start_gen + window_size - 1, max_gen)

            # Get TEs active in this window
            window_tes = df[(df['generation'] >= start_gen) & (df['generation'] <= end_gen)]
            active_lineages = window_tes['lineage_id'].nunique()
            n_transpositions = len(window_tes)

            if active_lineages == 1:
                phase = 'monomorphic'
            elif active_lineages == 2:
                phase = 'dimorphic'
            else:
                phase = 'polymorphic'

            phases.append({
                'start_generation': start_gen,
                'end_generation': end_gen,
                'active_lineages': active_lineages,
                'n_transpositions': n_transpositions,
                'phase': phase
            })

        return pd.DataFrame(phases)

    def get_lineage_history(self) -> pd.DataFrame:
        """
        Track the history of each lineage over time.

        Returns:
            DataFrame with lineage abundance per generation
        """
        if self.genealogy_df is None:
            self.load_genealogy()

        df = self.genealogy_df.copy()

        # Count TEs per lineage per generation
        lineage_counts = df.groupby(['generation', 'lineage_id']).size().reset_index(name='count')

        # Pivot to get lineages as columns
        pivot = lineage_counts.pivot(
            index='generation',
            columns='lineage_id',
            values='count'
        ).fillna(0)

        return pivot

    def get_divergence_distribution(self, generation: Optional[int] = None) -> np.ndarray:
        """
        Get distribution of divergence values.

        Args:
            generation: Specific generation, or None for all

        Returns:
            Array of divergence values
        """
        if self.genealogy_df is None:
            self.load_genealogy()

        df = self.genealogy_df
        if generation is not None:
            df = df[df['generation'] == generation]

        return df['divergence'].values

    def export_newick(self, root_id: int) -> str:
        """
        Export tree in Newick format for visualization.

        Args:
            root_id: Root TE ID

        Returns:
            Newick format string
        """
        if root_id not in self.trees:
            self.build_trees()

        def to_newick(node: TreeNode) -> str:
            if not node.children:
                return f"TE{node.te_id}:{node.divergence}"
            children_str = ",".join(to_newick(c) for c in node.children)
            return f"({children_str})TE{node.te_id}:{node.divergence}"

        return to_newick(self.trees[root_id]) + ";"

    def summary(self) -> Dict:
        """Generate summary of all phylogenetic analyses."""
        if not self.trees:
            self.build_trees()

        stats = self.get_tree_stats()
        phases = self.identify_phases()

        return {
            'tree_stats': {
                'n_nodes': stats.n_nodes,
                'n_leaves': stats.n_leaves,
                'max_depth': stats.max_depth,
                'mean_depth': round(stats.mean_depth, 2),
                'n_lineages': stats.n_lineages,
                'mean_divergence': round(stats.mean_divergence, 2),
                'branching_factor': round(stats.branching_factor, 2)
            },
            'phase_summary': phases['phase'].value_counts().to_dict(),
            'n_trees': len(self.trees)
        }


def main():
    """Command-line interface for phylogeny analysis."""
    import argparse
    import json

    parser = argparse.ArgumentParser(description="Analyze TE phylogenies")
    parser.add_argument('output_dir', help='Simulation output directory')
    parser.add_argument('--phases', action='store_true', help='Identify evolutionary phases')
    parser.add_argument('--newick', type=int, help='Export tree as Newick (specify root ID)')
    parser.add_argument('--summary', action='store_true', help='Print summary statistics')

    args = parser.parse_args()

    phylo = TEPhylogeny(args.output_dir)

    if args.summary:
        phylo.load_genealogy()
        summary = phylo.summary()
        print(json.dumps(summary, indent=2))

    if args.phases:
        phases = phylo.identify_phases()
        print("\nEvolutionary Phases:")
        print(phases.to_string())

    if args.newick is not None:
        newick = phylo.export_newick(args.newick)
        print(f"\nNewick format for tree rooted at TE{args.newick}:")
        print(newick)


if __name__ == "__main__":
    main()
