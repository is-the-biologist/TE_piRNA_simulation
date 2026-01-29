#!/usr/bin/env python3
"""
Parameter Exploration Framework for TE-piRNA Arms Race Simulations

This module provides tools for systematic parameter space exploration:
- Grid search over parameter combinations
- Latin Hypercube Sampling for efficient exploration
- Sensitivity analysis
- Results aggregation and summary statistics
"""

import numpy as np
import pandas as pd
import yaml
import itertools
import os
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime
import json

from launcher import SimulationLauncher


@dataclass
class ExplorationResult:
    """Container for exploration results."""
    parameter_sets: List[Dict[str, Any]]
    results: List[Dict[str, Any]]
    summary: pd.DataFrame
    exploration_type: str
    timestamp: str


class ParameterExplorer:
    """Framework for exploring simulation parameter space."""

    def __init__(self, config_path: str = "python/config/parameters.yaml"):
        """
        Initialize the parameter explorer.

        Args:
            config_path: Path to the YAML configuration file
        """
        self.base_dir = Path(__file__).parent.parent
        self.config_path = self.base_dir / config_path
        self.config = self._load_config()
        self.launcher = SimulationLauncher(config_path=config_path)

    def _load_config(self) -> Dict[str, Any]:
        """Load the YAML configuration file."""
        with open(self.config_path, 'r') as f:
            return yaml.safe_load(f)

    def get_parameter_info(self, param_name: str) -> Dict[str, Any]:
        """Get information about a specific parameter."""
        for category in ['population', 'genome', 'te', 'pirna', 'simulation', 'migration']:
            if category in self.config and param_name in self.config[category]:
                info = self.config[category][param_name].copy()
                info['category'] = category
                return info
        raise KeyError(f"Parameter '{param_name}' not found in configuration")

    def get_exploration_values(self, param_name: str) -> List[Any]:
        """Get the exploration values for a parameter."""
        info = self.get_parameter_info(param_name)
        return info.get('exploration', [info['default']])

    def grid_search(self, parameters: List[str],
                   output_dir: str = "output/grid_search",
                   seeds: Optional[List[int]] = None,
                   replicates: int = 1,
                   max_workers: int = 4,
                   verbose: bool = True) -> ExplorationResult:
        """
        Perform grid search over specified parameters.

        Args:
            parameters: List of parameter names to explore
            output_dir: Output directory for results
            seeds: Random seeds for replicates
            replicates: Number of replicates per parameter combination
            max_workers: Maximum parallel processes
            verbose: Print progress information

        Returns:
            ExplorationResult with all results
        """
        # Get exploration values for each parameter
        param_values = {p: self.get_exploration_values(p) for p in parameters}

        if verbose:
            print("Grid Search Configuration:")
            for p, vals in param_values.items():
                print(f"  {p}: {vals}")

        # Generate all combinations
        keys = list(param_values.keys())
        value_lists = [param_values[k] for k in keys]
        combinations = list(itertools.product(*value_lists))

        if verbose:
            total_runs = len(combinations) * replicates
            print(f"\nTotal parameter combinations: {len(combinations)}")
            print(f"Replicates per combination: {replicates}")
            print(f"Total simulations: {total_runs}")

        # Generate parameter sets with replicates
        parameter_sets = []
        run_seeds = []

        if seeds is None:
            seeds = list(range(replicates))

        for combo in combinations:
            params = dict(zip(keys, combo))
            for rep, seed in enumerate(seeds[:replicates]):
                parameter_sets.append(params.copy())
                run_seeds.append(seed + hash(str(combo)) % 10000)  # Unique seed per combo

        # Create output directory
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        full_output_dir = f"{output_dir}_{timestamp}"

        # Run simulations
        results = self.launcher.run_batch(
            parameter_sets=parameter_sets,
            output_dir=full_output_dir,
            seeds=run_seeds,
            max_workers=max_workers,
            verbose=verbose
        )

        # Create summary DataFrame
        summary = self._create_summary(results, parameters)

        # Save exploration configuration
        config_file = self.base_dir / full_output_dir / "exploration_config.yaml"
        os.makedirs(os.path.dirname(config_file), exist_ok=True)
        with open(config_file, 'w') as f:
            yaml.dump({
                'type': 'grid_search',
                'parameters': parameters,
                'values': param_values,
                'replicates': replicates,
                'total_runs': len(parameter_sets),
                'timestamp': timestamp
            }, f)

        return ExplorationResult(
            parameter_sets=parameter_sets,
            results=results,
            summary=summary,
            exploration_type='grid_search',
            timestamp=timestamp
        )

    def latin_hypercube(self, parameters: List[str],
                       n_samples: int = 100,
                       output_dir: str = "output/lhs",
                       seed: int = 42,
                       max_workers: int = 4,
                       verbose: bool = True) -> ExplorationResult:
        """
        Perform Latin Hypercube Sampling over parameter space.

        Args:
            parameters: List of parameter names to explore
            n_samples: Number of samples to generate
            output_dir: Output directory for results
            seed: Random seed for reproducibility
            max_workers: Maximum parallel processes
            verbose: Print progress information

        Returns:
            ExplorationResult with all results
        """
        np.random.seed(seed)

        # Get parameter ranges
        param_ranges = {}
        param_types = {}
        for p in parameters:
            info = self.get_parameter_info(p)
            param_ranges[p] = info.get('range', [info['default'], info['default']])
            param_types[p] = info.get('type', 'float')

        if verbose:
            print("Latin Hypercube Sampling Configuration:")
            for p, (lo, hi) in param_ranges.items():
                print(f"  {p}: [{lo}, {hi}] ({param_types[p]})")
            print(f"\nNumber of samples: {n_samples}")

        # Generate LHS samples
        n_params = len(parameters)
        samples = np.zeros((n_samples, n_params))

        for i, p in enumerate(parameters):
            lo, hi = param_ranges[p]

            # Create stratified random samples
            cut_points = np.linspace(0, 1, n_samples + 1)
            u = np.random.uniform(cut_points[:-1], cut_points[1:])
            np.random.shuffle(u)

            # Scale to parameter range
            if param_types[p] == 'integer':
                samples[:, i] = np.round(lo + u * (hi - lo)).astype(int)
            else:
                samples[:, i] = lo + u * (hi - lo)

        # Convert to parameter dictionaries
        parameter_sets = []
        for row in samples:
            params = {}
            for i, p in enumerate(parameters):
                if param_types[p] == 'integer':
                    params[p] = int(row[i])
                else:
                    params[p] = float(row[i])
            parameter_sets.append(params)

        # Create output directory
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        full_output_dir = f"{output_dir}_{timestamp}"

        # Generate unique seeds for each run
        run_seeds = [seed + i for i in range(n_samples)]

        # Run simulations
        results = self.launcher.run_batch(
            parameter_sets=parameter_sets,
            output_dir=full_output_dir,
            seeds=run_seeds,
            max_workers=max_workers,
            verbose=verbose
        )

        # Create summary DataFrame
        summary = self._create_summary(results, parameters)

        # Save exploration configuration and samples
        config_file = self.base_dir / full_output_dir / "exploration_config.yaml"
        os.makedirs(os.path.dirname(config_file), exist_ok=True)
        with open(config_file, 'w') as f:
            yaml.dump({
                'type': 'latin_hypercube',
                'parameters': parameters,
                'ranges': {p: list(param_ranges[p]) for p in parameters},
                'n_samples': n_samples,
                'seed': seed,
                'timestamp': timestamp
            }, f)

        # Save samples as CSV
        samples_df = pd.DataFrame(samples, columns=parameters)
        samples_df.to_csv(self.base_dir / full_output_dir / "lhs_samples.csv", index=False)

        return ExplorationResult(
            parameter_sets=parameter_sets,
            results=results,
            summary=summary,
            exploration_type='latin_hypercube',
            timestamp=timestamp
        )

    def two_parameter_sweep(self, param1: str, param2: str,
                           output_dir: str = "output/2d_sweep",
                           replicates: int = 3,
                           max_workers: int = 4,
                           verbose: bool = True) -> ExplorationResult:
        """
        Focused 2D parameter sweep for visualization.

        Args:
            param1: First parameter name
            param2: Second parameter name
            output_dir: Output directory
            replicates: Number of replicates
            max_workers: Maximum parallel processes
            verbose: Print progress

        Returns:
            ExplorationResult with results
        """
        return self.grid_search(
            parameters=[param1, param2],
            output_dir=output_dir,
            replicates=replicates,
            max_workers=max_workers,
            verbose=verbose
        )

    def _create_summary(self, results: List[Dict[str, Any]],
                       parameters: List[str]) -> pd.DataFrame:
        """Create summary DataFrame from results."""
        rows = []

        for r in results:
            row = {
                'run_id': r['run_id'],
                'success': r['success'],
                'duration': r.get('duration', np.nan)
            }

            # Add parameter values
            params = r.get('parameters', {})
            for p in parameters:
                row[p] = params.get(p, np.nan)

            # Parse output files for summary statistics if successful
            if r['success']:
                output_dir = Path(r['output_dir'])
                summary_file = output_dir / "population_summary.tsv"

                if summary_file.exists():
                    try:
                        df = pd.read_csv(summary_file, sep='\t')
                        if not df.empty:
                            final = df.iloc[-1]
                            row['final_mean_te_count'] = final.get('mean_te_count', np.nan)
                            row['final_active_lineages'] = final.get('active_lineages', np.nan)
                            row['final_pirna_te_count'] = final.get('pirna_te_count', np.nan)
                            row['total_transpositions'] = df['total_transpositions'].sum()
                    except Exception:
                        pass

            rows.append(row)

        return pd.DataFrame(rows)

    def sensitivity_analysis(self, parameters: List[str],
                            baseline: Optional[Dict[str, Any]] = None,
                            perturbation: float = 0.1,
                            output_dir: str = "output/sensitivity",
                            replicates: int = 5,
                            max_workers: int = 4,
                            verbose: bool = True) -> pd.DataFrame:
        """
        Perform local sensitivity analysis around baseline parameters.

        Args:
            parameters: Parameters to analyze
            baseline: Baseline parameter values (uses defaults if None)
            perturbation: Fractional perturbation (e.g., 0.1 = +/- 10%)
            output_dir: Output directory
            replicates: Number of replicates
            max_workers: Maximum parallel processes
            verbose: Print progress

        Returns:
            DataFrame with sensitivity metrics
        """
        if baseline is None:
            baseline = self.launcher.get_default_parameters()

        if verbose:
            print("Sensitivity Analysis Configuration:")
            print(f"  Perturbation: +/- {perturbation*100}%")
            print(f"  Parameters: {parameters}")

        # Generate perturbed parameter sets
        parameter_sets = []
        perturbation_info = []

        # Add baseline
        for _ in range(replicates):
            parameter_sets.append(baseline.copy())
            perturbation_info.append(('baseline', 'baseline', 0))

        # Add perturbations for each parameter
        for p in parameters:
            base_val = baseline[p]
            info = self.get_parameter_info(p)
            param_range = info.get('range', [base_val * 0.1, base_val * 10])

            # Low perturbation
            low_val = base_val * (1 - perturbation)
            low_val = max(param_range[0], low_val)
            if info.get('type') == 'integer':
                low_val = int(low_val)

            # High perturbation
            high_val = base_val * (1 + perturbation)
            high_val = min(param_range[1], high_val)
            if info.get('type') == 'integer':
                high_val = int(high_val)

            for _ in range(replicates):
                # Low
                params_low = baseline.copy()
                params_low[p] = low_val
                parameter_sets.append(params_low)
                perturbation_info.append((p, 'low', low_val))

                # High
                params_high = baseline.copy()
                params_high[p] = high_val
                parameter_sets.append(params_high)
                perturbation_info.append((p, 'high', high_val))

        # Run simulations
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        full_output_dir = f"{output_dir}_{timestamp}"

        results = self.launcher.run_batch(
            parameter_sets=parameter_sets,
            output_dir=full_output_dir,
            max_workers=max_workers,
            verbose=verbose
        )

        # Analyze sensitivity
        sensitivity_data = []
        for (param, direction, value), result in zip(perturbation_info, results):
            row = {
                'parameter': param,
                'direction': direction,
                'value': value,
                'success': result['success']
            }

            if result['success']:
                output_dir_path = Path(result['output_dir'])
                summary_file = output_dir_path / "population_summary.tsv"
                if summary_file.exists():
                    try:
                        df = pd.read_csv(summary_file, sep='\t')
                        if not df.empty:
                            final = df.iloc[-1]
                            row['final_te_count'] = final.get('mean_te_count', np.nan)
                            row['final_lineages'] = final.get('active_lineages', np.nan)
                    except Exception:
                        pass

            sensitivity_data.append(row)

        return pd.DataFrame(sensitivity_data)


def main():
    """Command-line interface for parameter exploration."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Parameter exploration for TE-piRNA simulations"
    )

    subparsers = parser.add_subparsers(dest='command', help='Exploration type')

    # Grid search
    grid_parser = subparsers.add_parser('grid', help='Grid search')
    grid_parser.add_argument(
        '-p', '--params',
        nargs='+',
        required=True,
        help='Parameters to explore'
    )
    grid_parser.add_argument(
        '-r', '--replicates',
        type=int,
        default=3,
        help='Replicates per combination'
    )
    grid_parser.add_argument(
        '-w', '--workers',
        type=int,
        default=4,
        help='Parallel workers'
    )

    # LHS
    lhs_parser = subparsers.add_parser('lhs', help='Latin Hypercube Sampling')
    lhs_parser.add_argument(
        '-p', '--params',
        nargs='+',
        required=True,
        help='Parameters to explore'
    )
    lhs_parser.add_argument(
        '-n', '--samples',
        type=int,
        default=100,
        help='Number of samples'
    )
    lhs_parser.add_argument(
        '-s', '--seed',
        type=int,
        default=42,
        help='Random seed'
    )
    lhs_parser.add_argument(
        '-w', '--workers',
        type=int,
        default=4,
        help='Parallel workers'
    )

    args = parser.parse_args()

    explorer = ParameterExplorer()

    if args.command == 'grid':
        result = explorer.grid_search(
            parameters=args.params,
            replicates=args.replicates,
            max_workers=args.workers
        )
        print(f"\nSummary saved to exploration results")
        print(result.summary.describe())

    elif args.command == 'lhs':
        result = explorer.latin_hypercube(
            parameters=args.params,
            n_samples=args.samples,
            seed=args.seed,
            max_workers=args.workers
        )
        print(f"\nSummary saved to exploration results")
        print(result.summary.describe())

    else:
        parser.print_help()


if __name__ == "__main__":
    main()
