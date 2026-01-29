#!/usr/bin/env python3
"""
Simulation Launcher for TE-piRNA Arms Race Model

This module provides functionality to:
- Execute SLiM simulations with custom parameters
- Manage output directories and files
- Support parallel execution of multiple simulations
- Handle parameter passing from YAML config to SLiM
"""

import subprocess
import os
import sys
import yaml
import argparse
import shutil
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, Optional, List
from concurrent.futures import ProcessPoolExecutor, as_completed
import hashlib
import json


class SimulationLauncher:
    """Manages execution of SLiM simulations."""

    def __init__(self, config_path: str = "python/config/parameters.yaml",
                 slim_script: str = "slim/te_pirna_simulation.slim"):
        """
        Initialize the launcher.

        Args:
            config_path: Path to the YAML configuration file
            slim_script: Path to the SLiM simulation script
        """
        self.base_dir = Path(__file__).parent.parent
        self.config_path = self.base_dir / config_path
        self.slim_script = self.base_dir / slim_script

        # Load configuration
        self.config = self._load_config()

        # Find SLiM executable
        self.slim_executable = self._find_slim()

    def _load_config(self) -> Dict[str, Any]:
        """Load the YAML configuration file."""
        if not self.config_path.exists():
            raise FileNotFoundError(f"Configuration file not found: {self.config_path}")

        with open(self.config_path, 'r') as f:
            return yaml.safe_load(f)

    def _find_slim(self) -> str:
        """Find the SLiM executable."""
        # Common locations for SLiM
        possible_paths = [
            "slim",  # If in PATH
            "/usr/local/bin/slim",
            "/opt/homebrew/bin/slim",
            os.path.expanduser("~/bin/slim"),
            "/Applications/SLiM/slim",
        ]

        for path in possible_paths:
            if shutil.which(path):
                return path

        # Try to find it
        result = shutil.which("slim")
        if result:
            return result

        raise FileNotFoundError(
            "SLiM executable not found. Please install SLiM or add it to your PATH.\n"
            "Download from: https://messerlab.org/slim/"
        )

    def get_default_parameters(self) -> Dict[str, Any]:
        """Extract default parameter values from configuration."""
        params = {}

        # Flatten the nested config structure
        for category in ['population', 'genome', 'te', 'pirna', 'simulation']:
            if category in self.config:
                for param_name, param_info in self.config[category].items():
                    if isinstance(param_info, dict) and 'default' in param_info:
                        params[param_name] = param_info['default']

        return params

    def _build_slim_command(self, parameters: Dict[str, Any],
                           output_dir: str, seed: Optional[int] = None) -> List[str]:
        """
        Build the SLiM command with parameters.

        Args:
            parameters: Dictionary of parameter names and values
            output_dir: Directory for output files
            seed: Random seed for reproducibility

        Returns:
            List of command components
        """
        cmd = [self.slim_executable]

        # Add random seed if specified
        if seed is not None:
            cmd.extend(["-s", str(seed)])

        # Add each parameter as a define constant
        for param, value in parameters.items():
            if isinstance(value, str):
                cmd.extend(["-d", f'{param}="{value}"'])
            elif isinstance(value, bool):
                cmd.extend(["-d", f'{param}={"T" if value else "F"}'])
            elif isinstance(value, float):
                cmd.extend(["-d", f'{param}={value}'])
            else:
                cmd.extend(["-d", f'{param}={value}'])

        # Ensure output directory ends with /
        if not output_dir.endswith('/'):
            output_dir += '/'
        cmd.extend(["-d", f'OUTPUT_DIR="{output_dir}"'])

        # Add the script path
        cmd.append(str(self.slim_script))

        return cmd

    def _create_output_dir(self, base_output: str, run_id: str) -> Path:
        """Create a unique output directory for this simulation run."""
        output_path = self.base_dir / base_output / run_id
        output_path.mkdir(parents=True, exist_ok=True)
        return output_path

    def _generate_run_id(self, parameters: Dict[str, Any], seed: Optional[int] = None) -> str:
        """Generate a unique run ID based on parameters."""
        # Create a hash of the parameters for uniqueness
        param_str = json.dumps(parameters, sort_keys=True)
        if seed is not None:
            param_str += f"_seed{seed}"
        param_hash = hashlib.md5(param_str.encode()).hexdigest()[:8]

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        return f"run_{timestamp}_{param_hash}"

    def run_simulation(self, parameters: Optional[Dict[str, Any]] = None,
                      output_dir: str = "output",
                      seed: Optional[int] = None,
                      run_id: Optional[str] = None,
                      verbose: bool = True) -> Dict[str, Any]:
        """
        Run a single SLiM simulation.

        Args:
            parameters: Custom parameters (uses defaults if None)
            output_dir: Base output directory
            seed: Random seed for reproducibility
            run_id: Custom run identifier (auto-generated if None)
            verbose: Print progress information

        Returns:
            Dictionary with run information and results
        """
        # Get parameters (defaults + overrides)
        params = self.get_default_parameters()
        if parameters:
            params.update(parameters)

        # Create run ID and output directory
        if run_id is None:
            run_id = self._generate_run_id(params, seed)

        output_path = self._create_output_dir(output_dir, run_id)

        # Save parameters used for this run
        params_file = output_path / "parameters_used.yaml"
        with open(params_file, 'w') as f:
            yaml.dump({
                'run_id': run_id,
                'seed': seed,
                'parameters': params,
                'timestamp': datetime.now().isoformat()
            }, f)

        # Build command
        cmd = self._build_slim_command(params, str(output_path) + "/", seed)

        if verbose:
            print(f"Running simulation: {run_id}")
            print(f"Output directory: {output_path}")
            print(f"Command: {' '.join(cmd)}")

        # Execute simulation
        start_time = datetime.now()
        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                cwd=str(self.base_dir)
            )

            end_time = datetime.now()
            duration = (end_time - start_time).total_seconds()

            # Save stdout/stderr
            with open(output_path / "slim_stdout.txt", 'w') as f:
                f.write(result.stdout)
            with open(output_path / "slim_stderr.txt", 'w') as f:
                f.write(result.stderr)

            success = result.returncode == 0

            if verbose:
                if success:
                    print(f"Simulation completed in {duration:.1f} seconds")
                else:
                    print(f"Simulation FAILED (exit code {result.returncode})")
                    print(f"Error: {result.stderr[:500]}")

            return {
                'run_id': run_id,
                'success': success,
                'return_code': result.returncode,
                'duration': duration,
                'output_dir': str(output_path),
                'parameters': params,
                'seed': seed,
                'stdout': result.stdout,
                'stderr': result.stderr
            }

        except Exception as e:
            end_time = datetime.now()
            duration = (end_time - start_time).total_seconds()

            if verbose:
                print(f"Simulation error: {str(e)}")

            return {
                'run_id': run_id,
                'success': False,
                'error': str(e),
                'duration': duration,
                'output_dir': str(output_path),
                'parameters': params,
                'seed': seed
            }

    def run_batch(self, parameter_sets: List[Dict[str, Any]],
                 output_dir: str = "output",
                 seeds: Optional[List[int]] = None,
                 max_workers: int = 4,
                 verbose: bool = True) -> List[Dict[str, Any]]:
        """
        Run multiple simulations in parallel.

        Args:
            parameter_sets: List of parameter dictionaries
            output_dir: Base output directory
            seeds: List of random seeds (one per parameter set)
            max_workers: Maximum parallel processes
            verbose: Print progress information

        Returns:
            List of result dictionaries
        """
        if seeds is None:
            seeds = [None] * len(parameter_sets)

        if len(seeds) != len(parameter_sets):
            raise ValueError("Number of seeds must match number of parameter sets")

        results = []

        if verbose:
            print(f"Running {len(parameter_sets)} simulations with {max_workers} workers...")

        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            futures = {}
            for i, (params, seed) in enumerate(zip(parameter_sets, seeds)):
                run_id = f"batch_{i:04d}"
                future = executor.submit(
                    self.run_simulation,
                    parameters=params,
                    output_dir=output_dir,
                    seed=seed,
                    run_id=run_id,
                    verbose=False
                )
                futures[future] = i

            for future in as_completed(futures):
                idx = futures[future]
                result = future.result()
                results.append(result)

                if verbose:
                    status = "OK" if result['success'] else "FAILED"
                    print(f"  [{idx+1}/{len(parameter_sets)}] {result['run_id']}: {status}")

        return results


def main():
    """Command-line interface for the simulation launcher."""
    parser = argparse.ArgumentParser(
        description="Launch TE-piRNA Arms Race SLiM simulations"
    )

    parser.add_argument(
        '-c', '--config',
        default='python/config/parameters.yaml',
        help='Path to configuration file'
    )
    parser.add_argument(
        '-o', '--output',
        default='output',
        help='Output directory'
    )
    parser.add_argument(
        '-s', '--seed',
        type=int,
        default=None,
        help='Random seed for reproducibility'
    )
    parser.add_argument(
        '-p', '--param',
        action='append',
        nargs=2,
        metavar=('NAME', 'VALUE'),
        help='Override parameter (can be used multiple times)'
    )
    parser.add_argument(
        '-q', '--quiet',
        action='store_true',
        help='Suppress verbose output'
    )

    args = parser.parse_args()

    # Initialize launcher
    launcher = SimulationLauncher(config_path=args.config)

    # Parse parameter overrides
    param_overrides = {}
    if args.param:
        for name, value in args.param:
            # Try to parse as number
            try:
                if '.' in value:
                    param_overrides[name] = float(value)
                else:
                    param_overrides[name] = int(value)
            except ValueError:
                param_overrides[name] = value

    # Run simulation
    result = launcher.run_simulation(
        parameters=param_overrides if param_overrides else None,
        output_dir=args.output,
        seed=args.seed,
        verbose=not args.quiet
    )

    # Exit with appropriate code
    sys.exit(0 if result['success'] else 1)


if __name__ == "__main__":
    main()
