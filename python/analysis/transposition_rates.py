#!/usr/bin/env python3
"""
Transposition Rate Analysis

Analyzes transposition dynamics over simulation time:
- Track transposition rates per generation
- Detect bursts of transposition activity
- Correlate bursts with piRNA locus insertions
- Calculate cycle frequencies
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from scipy import signal
from scipy.stats import zscore


@dataclass
class TranspositionBurst:
    """Represents a detected burst of transposition activity."""
    start_generation: int
    peak_generation: int
    end_generation: int
    peak_rate: float
    mean_rate: float
    duration: int
    total_transpositions: int


@dataclass
class BurstCycle:
    """Represents a cycle of burst-silence-burst."""
    burst1: TranspositionBurst
    burst2: TranspositionBurst
    silence_duration: int
    silence_mean_rate: float


class TranspositionAnalyzer:
    """Analyze transposition rate dynamics from simulation output."""

    def __init__(self, output_dir: str):
        """
        Initialize the analyzer.

        Args:
            output_dir: Directory containing simulation output files
        """
        self.output_dir = Path(output_dir)
        self.transposition_file = self.output_dir / "transposition_events.tsv"
        self.summary_file = self.output_dir / "population_summary.tsv"
        self.census_file = self.output_dir / "te_census.tsv"

        self.transposition_df: Optional[pd.DataFrame] = None
        self.summary_df: Optional[pd.DataFrame] = None
        self.census_df: Optional[pd.DataFrame] = None

    def load_data(self):
        """Load all relevant data files."""
        if self.transposition_file.exists():
            self.transposition_df = pd.read_csv(self.transposition_file, sep='\t')

        if self.summary_file.exists():
            self.summary_df = pd.read_csv(self.summary_file, sep='\t')

        if self.census_file.exists():
            self.census_df = pd.read_csv(self.census_file, sep='\t')

    def get_rates_per_generation(self) -> pd.DataFrame:
        """
        Calculate transposition rates per generation.

        Returns:
            DataFrame with generation and rate columns
        """
        if self.transposition_df is None:
            self.load_data()

        if self.transposition_df is None or self.transposition_df.empty:
            return pd.DataFrame(columns=['generation', 'n_transpositions', 'mean_effective_rate',
                                        'silenced_fraction'])

        df = self.transposition_df.copy()

        # Group by generation
        rates = df.groupby('generation').agg({
            'te_id': 'count',
            'effective_rate': 'mean',
            'was_silenced': lambda x: (x == 'TRUE').mean() if len(x) > 0 else 0
        }).reset_index()

        rates.columns = ['generation', 'n_transpositions', 'mean_effective_rate', 'silenced_fraction']

        # Fill missing generations with zeros
        all_gens = pd.DataFrame({'generation': range(1, rates['generation'].max() + 1)})
        rates = all_gens.merge(rates, on='generation', how='left').fillna(0)

        return rates

    def smooth_rates(self, window_size: int = 50) -> pd.DataFrame:
        """
        Apply smoothing to transposition rates.

        Args:
            window_size: Size of rolling window

        Returns:
            DataFrame with smoothed rates
        """
        rates = self.get_rates_per_generation()

        if rates.empty:
            return rates

        rates['smoothed_transpositions'] = rates['n_transpositions'].rolling(
            window=window_size, center=True, min_periods=1
        ).mean()

        rates['smoothed_rate'] = rates['mean_effective_rate'].rolling(
            window=window_size, center=True, min_periods=1
        ).mean()

        return rates

    def detect_bursts(self, threshold_std: float = 2.0,
                     min_duration: int = 10) -> List[TranspositionBurst]:
        """
        Detect bursts of transposition activity.

        A burst is defined as a period where transposition rate exceeds
        threshold_std standard deviations above the mean.

        Args:
            threshold_std: Z-score threshold for burst detection
            min_duration: Minimum generations for a burst

        Returns:
            List of detected TranspositionBurst objects
        """
        rates = self.get_rates_per_generation()

        if rates.empty or len(rates) < min_duration:
            return []

        # Calculate z-scores
        trans = rates['n_transpositions'].values
        if trans.std() == 0:
            return []

        z_scores = zscore(trans)

        # Find burst regions
        is_burst = z_scores > threshold_std
        bursts = []

        i = 0
        while i < len(is_burst):
            if is_burst[i]:
                # Start of potential burst
                start = i
                while i < len(is_burst) and is_burst[i]:
                    i += 1
                end = i - 1

                duration = end - start + 1
                if duration >= min_duration:
                    burst_data = rates.iloc[start:end+1]
                    peak_idx = burst_data['n_transpositions'].idxmax()

                    burst = TranspositionBurst(
                        start_generation=int(rates.loc[start, 'generation']),
                        peak_generation=int(rates.loc[peak_idx, 'generation']),
                        end_generation=int(rates.loc[end, 'generation']),
                        peak_rate=float(burst_data['n_transpositions'].max()),
                        mean_rate=float(burst_data['n_transpositions'].mean()),
                        duration=duration,
                        total_transpositions=int(burst_data['n_transpositions'].sum())
                    )
                    bursts.append(burst)
            else:
                i += 1

        return bursts

    def detect_cycles(self, bursts: Optional[List[TranspositionBurst]] = None) -> List[BurstCycle]:
        """
        Detect burst-silence-burst cycles.

        Args:
            bursts: List of bursts (detected if None)

        Returns:
            List of BurstCycle objects
        """
        if bursts is None:
            bursts = self.detect_bursts()

        if len(bursts) < 2:
            return []

        rates = self.get_rates_per_generation()
        cycles = []

        for i in range(len(bursts) - 1):
            burst1 = bursts[i]
            burst2 = bursts[i + 1]

            # Get silence period between bursts
            silence_start = burst1.end_generation + 1
            silence_end = burst2.start_generation - 1
            silence_duration = silence_end - silence_start + 1

            if silence_duration > 0:
                silence_data = rates[
                    (rates['generation'] >= silence_start) &
                    (rates['generation'] <= silence_end)
                ]
                silence_mean_rate = silence_data['n_transpositions'].mean() if not silence_data.empty else 0

                cycle = BurstCycle(
                    burst1=burst1,
                    burst2=burst2,
                    silence_duration=silence_duration,
                    silence_mean_rate=float(silence_mean_rate)
                )
                cycles.append(cycle)

        return cycles

    def correlate_with_pirna(self) -> pd.DataFrame:
        """
        Correlate transposition rates with piRNA locus insertions.

        Returns:
            DataFrame with generation, transposition rate, and piRNA count
        """
        if self.summary_df is None:
            self.load_data()

        if self.summary_df is None:
            return pd.DataFrame()

        rates = self.get_rates_per_generation()

        # Merge with summary data
        merged = rates.merge(
            self.summary_df[['generation', 'pirna_te_count', 'mean_te_count']],
            on='generation',
            how='left'
        )

        return merged

    def calculate_burst_statistics(self) -> Dict:
        """
        Calculate summary statistics for burst dynamics.

        Returns:
            Dictionary with burst statistics
        """
        bursts = self.detect_bursts()
        cycles = self.detect_cycles(bursts)
        rates = self.get_rates_per_generation()

        stats = {
            'n_bursts': len(bursts),
            'n_cycles': len(cycles),
            'total_generations': len(rates),
            'overall_mean_rate': float(rates['n_transpositions'].mean()) if not rates.empty else 0,
            'overall_max_rate': float(rates['n_transpositions'].max()) if not rates.empty else 0
        }

        if bursts:
            stats['mean_burst_duration'] = np.mean([b.duration for b in bursts])
            stats['mean_burst_peak'] = np.mean([b.peak_rate for b in bursts])
            stats['total_burst_transpositions'] = sum(b.total_transpositions for b in bursts)

        if cycles:
            stats['mean_cycle_length'] = np.mean([
                c.burst2.peak_generation - c.burst1.peak_generation for c in cycles
            ])
            stats['mean_silence_duration'] = np.mean([c.silence_duration for c in cycles])

        return stats

    def get_time_series_features(self) -> Dict:
        """
        Extract time series features from transposition rates.

        Returns:
            Dictionary with time series features
        """
        rates = self.get_rates_per_generation()

        if rates.empty:
            return {}

        trans = rates['n_transpositions'].values

        features = {
            'mean': float(np.mean(trans)),
            'std': float(np.std(trans)),
            'max': float(np.max(trans)),
            'min': float(np.min(trans)),
            'median': float(np.median(trans)),
            'skewness': float(pd.Series(trans).skew()),
            'kurtosis': float(pd.Series(trans).kurtosis())
        }

        # Trend analysis
        if len(trans) > 10:
            x = np.arange(len(trans))
            slope, _ = np.polyfit(x, trans, 1)
            features['trend_slope'] = float(slope)

        # Autocorrelation at lag 1
        if len(trans) > 1:
            autocorr = np.corrcoef(trans[:-1], trans[1:])[0, 1]
            features['autocorrelation_lag1'] = float(autocorr) if not np.isnan(autocorr) else 0

        return features

    def summary(self) -> Dict:
        """Generate comprehensive summary of transposition dynamics."""
        stats = self.calculate_burst_statistics()
        features = self.get_time_series_features()

        # Combine
        summary = {
            'burst_statistics': stats,
            'time_series_features': features
        }

        # Add correlation with piRNA if available
        corr_df = self.correlate_with_pirna()
        if not corr_df.empty and 'pirna_te_count' in corr_df.columns:
            # Calculate correlation between transposition rate and piRNA count
            valid_data = corr_df.dropna()
            if len(valid_data) > 2:
                corr = np.corrcoef(
                    valid_data['n_transpositions'],
                    valid_data['pirna_te_count']
                )[0, 1]
                summary['pirna_correlation'] = float(corr) if not np.isnan(corr) else 0

        return summary


def main():
    """Command-line interface for transposition analysis."""
    import argparse
    import json

    parser = argparse.ArgumentParser(description="Analyze transposition dynamics")
    parser.add_argument('output_dir', help='Simulation output directory')
    parser.add_argument('--bursts', action='store_true', help='Detect and report bursts')
    parser.add_argument('--cycles', action='store_true', help='Detect burst cycles')
    parser.add_argument('--summary', action='store_true', help='Print summary statistics')
    parser.add_argument('--rates', action='store_true', help='Output rates per generation')

    args = parser.parse_args()

    analyzer = TranspositionAnalyzer(args.output_dir)

    if args.summary:
        summary = analyzer.summary()
        print(json.dumps(summary, indent=2))

    if args.bursts:
        bursts = analyzer.detect_bursts()
        print(f"\nDetected {len(bursts)} bursts:")
        for i, b in enumerate(bursts):
            print(f"  Burst {i+1}: gen {b.start_generation}-{b.end_generation}, "
                  f"peak={b.peak_rate:.1f} at gen {b.peak_generation}")

    if args.cycles:
        cycles = analyzer.detect_cycles()
        print(f"\nDetected {len(cycles)} burst cycles:")
        for i, c in enumerate(cycles):
            print(f"  Cycle {i+1}: bursts at gen {c.burst1.peak_generation} -> {c.burst2.peak_generation}, "
                  f"silence={c.silence_duration} gen")

    if args.rates:
        rates = analyzer.get_rates_per_generation()
        print("\nTransposition rates per generation:")
        print(rates.to_string())


if __name__ == "__main__":
    main()
