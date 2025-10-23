#!/usr/bin/env python3
"""
Benchmark script comparing 4 main PoS algorithms
Run each algorithm 10 times with the same parameters and calculate average results
"""

import sys
import os
import random
import numpy as np
import json
import time
from datetime import datetime
from typing import Dict, List, Tuple
from scipy import stats
from scipy.stats import ttest_ind, ttest_rel, mannwhitneyu, wilcoxon

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "src"))

from parameters import Parameters, PoS, Distribution, NewEntry
from simulator import simulate
from utils import generate_peers, gini
import simulator as sim


def calculate_cohens_d(group1: List[float], group2: List[float]) -> float:
    """
    Calculate Cohen's d effect size between two groups (for independent samples)

    Args:
        group1: First group of values
        group2: Second group of values

    Returns:
        Cohen's d effect size
    """
    n1, n2 = len(group1), len(group2)
    s1, s2 = np.std(group1, ddof=1), np.std(group2, ddof=1)

    # Pooled standard deviation
    pooled_std = np.sqrt(((n1 - 1) * s1**2 + (n2 - 1) * s2**2) / (n1 + n2 - 2))

    # Cohen's d
    d = (np.mean(group1) - np.mean(group2)) / pooled_std

    return d


def calculate_cohens_dz_paired(group1: List[float], group2: List[float]) -> float:
    """
    Calculate Cohen's dz effect size for paired samples

    Args:
        group1: First group of values (paired with group2)
        group2: Second group of values (paired with group1)

    Returns:
        Cohen's dz effect size for paired samples
    """
    # Calculate differences
    differences = np.array(group1) - np.array(group2)

    # Cohen's dz = mean_difference / std_difference
    dz = np.mean(differences) / np.std(differences, ddof=1)

    return dz


def calculate_confidence_interval(
    values: List[float], confidence: float = 0.95
) -> Tuple[float, float]:
    """
    Calculate confidence interval for a sample

    Args:
        values: Sample values
        confidence: Confidence level (default 0.95 for 95% CI)

    Returns:
        Tuple of (lower_bound, upper_bound)
    """
    n = len(values)
    mean = np.mean(values)
    std = np.std(values, ddof=1)

    # Calculate t-value for given confidence level
    alpha = 1 - confidence
    t_value = stats.t.ppf(1 - alpha / 2, n - 1)

    # Calculate margin of error
    margin_error = t_value * std / np.sqrt(n)

    return (mean - margin_error, mean + margin_error)


def calculate_statistical_significance(
    algorithm_results: List[Dict], baseline_results: List[Dict]
) -> Dict:
    """
    Calculate P-values and statistical significance for algorithm comparison

    Args:
        algorithm_results: Results from the algorithm being tested
        baseline_results: Results from baseline algorithm (WEIGHTED)

    Returns:
        Dictionary containing statistical test results
    """
    # Extract values
    alg_gini = [r["final_gini"] for r in algorithm_results]
    alg_nakamoto = [r["final_nakamoto"] for r in algorithm_results]
    baseline_gini = [r["final_gini"] for r in baseline_results]
    baseline_nakamoto = [r["final_nakamoto"] for r in baseline_results]

    # Statistical tests for Gini coefficient (lower is better) - PAIRED TEST
    try:
        gini_t_stat, gini_p_value = ttest_rel(
            alg_gini, baseline_gini, alternative="less"
        )
        gini_wilcoxon = wilcoxon(alg_gini, baseline_gini, alternative="less")
    except Exception as e:
        gini_t_stat, gini_p_value = np.nan, np.nan
        gini_wilcoxon = (np.nan, np.nan)

    # Statistical tests for Nakamoto coefficient (higher is better) - PAIRED TEST
    try:
        nakamoto_t_stat, nakamoto_p_value = ttest_rel(
            alg_nakamoto, baseline_nakamoto, alternative="greater"
        )
        nakamoto_wilcoxon = wilcoxon(
            alg_nakamoto, baseline_nakamoto, alternative="greater"
        )
    except Exception as e:
        nakamoto_t_stat, nakamoto_p_value = np.nan, np.nan
        nakamoto_wilcoxon = (np.nan, np.nan)

    # Effect sizes (using paired Cohen's dz for paired samples)
    gini_effect_size = calculate_cohens_dz_paired(alg_gini, baseline_gini)
    nakamoto_effect_size = calculate_cohens_dz_paired(alg_nakamoto, baseline_nakamoto)

    return {
        # Gini coefficient tests
        "gini_t_statistic": (
            round(gini_t_stat, 4) if not np.isnan(gini_t_stat) else None
        ),
        "gini_p_value": round(gini_p_value, 50) if not np.isnan(gini_p_value) else None,
        "gini_wilcoxon_p": (
            round(gini_wilcoxon[1], 6) if not np.isnan(gini_wilcoxon[1]) else None
        ),
        "gini_significant": bool(
            gini_p_value < 0.001 if not np.isnan(gini_p_value) else False
        ),
        "gini_effect_size": round(gini_effect_size, 4),
        # Nakamoto coefficient tests
        "nakamoto_t_statistic": (
            round(nakamoto_t_stat, 4) if not np.isnan(nakamoto_t_stat) else None
        ),
        "nakamoto_p_value": (
            round(nakamoto_p_value, 50) if not np.isnan(nakamoto_p_value) else None
        ),
        "nakamoto_wilcoxon_p": (
            round(nakamoto_wilcoxon[1], 6)
            if not np.isnan(nakamoto_wilcoxon[1])
            else None
        ),
        "nakamoto_significant": bool(
            nakamoto_p_value < 0.001 if not np.isnan(nakamoto_p_value) else False
        ),
        "nakamoto_effect_size": round(nakamoto_effect_size, 4),
        # Interpretation
        "gini_interpretation": interpret_effect_size(gini_effect_size),
        "nakamoto_interpretation": interpret_effect_size(nakamoto_effect_size),
    }


def interpret_effect_size(d: float) -> str:
    """
    Interpret Cohen's d effect size

    Args:
        d: Cohen's d value

    Returns:
        Interpretation string
    """
    abs_d = abs(d)
    if abs_d < 0.2:
        return "negligible"
    elif abs_d < 0.5:
        return "small"
    elif abs_d < 0.8:
        return "medium"
    else:
        return "large"


def run_single_experiment(
    pos_algorithm: PoS,
    run_id: int,
    params: Parameters,
    stakes: List[float],
    corrupted: List[int],
) -> Dict:
    """Run a single experiment with the specified PoS algorithm"""
    print(f"    Run {run_id + 1}/10: {pos_algorithm.name}")
    # Create parameter copy with specific algorithm
    test_params = Parameters(
        n_epochs=params.n_epochs,
        proof_of_stake=pos_algorithm,
        initial_stake_volume=params.initial_stake_volume,
        initial_distribution=params.initial_distribution,
        n_peers=params.n_peers,
        n_corrupted=params.n_corrupted,
        p_fail=params.p_fail,
        p_join=params.p_join,
        p_leave=params.p_leave,
        join_amount=params.join_amount,
        penalty_percentage=params.penalty_percentage,
        reward=params.reward,
    )

    # Create copies of stakes and corrupted to avoid modifying original data
    test_stakes = stakes.copy()
    test_corrupted = corrupted.copy()

    # Run simulation
    start_time = time.time()
    gini_history, peers_history, nakamoto_history, _ = simulate(
        test_stakes, test_corrupted, test_params
    )
    end_time = time.time()

    return {
        "algorithm": pos_algorithm.name,
        "run_id": run_id,
        "starting_gini": gini(stakes),
        "final_gini": gini_history[-1],
        "final_nakamoto": nakamoto_history[-1],
        "final_peers": peers_history[-1],
        "execution_time": end_time - start_time,
        "gini_history": gini_history,
        "nakamoto_history": nakamoto_history,
        "peers_history": peers_history,
    }


def calculate_statistics(
    results: List[Dict], baseline_results: List[Dict] = None
) -> Dict:
    """Calculate statistics from multiple runs, including P-values if baseline is provided"""

    final_gini_values = [r["final_gini"] for r in results]
    final_nakamoto_values = [r["final_nakamoto"] for r in results]
    final_peers_values = [r["final_peers"] for r in results]
    execution_times = [r["execution_time"] for r in results]

    base_stats = {
        "algorithm": results[0]["algorithm"],
        "num_runs": len(results),
        "starting_gini": results[0]["starting_gini"],
        # Final Gini statistics (using sample std: ddof=1)
        "final_gini_mean": round(np.mean(final_gini_values), 3),
        "final_gini_std": round(np.std(final_gini_values, ddof=1), 3),
        "final_gini_min": round(np.min(final_gini_values), 3),
        "final_gini_max": round(np.max(final_gini_values), 3),
        "final_gini_ci_95": list(calculate_confidence_interval(final_gini_values)),
        # Final Nakamoto statistics (using sample std: ddof=1)
        "final_nakamoto_mean": round(np.mean(final_nakamoto_values), 3),
        "final_nakamoto_std": round(np.std(final_nakamoto_values, ddof=1), 3),
        "final_nakamoto_min": int(np.min(final_nakamoto_values)),
        "final_nakamoto_max": int(np.max(final_nakamoto_values)),
        "final_nakamoto_ci_95": list(
            calculate_confidence_interval(final_nakamoto_values)
        ),
        # Final Peers statistics (using sample std: ddof=1)
        "final_peers_mean": round(np.mean(final_peers_values), 3),
        "final_peers_std": round(np.std(final_peers_values, ddof=1), 3),
        "final_peers_min": int(np.min(final_peers_values)),
        "final_peers_max": int(np.max(final_peers_values)),
        "final_peers_ci_95": list(calculate_confidence_interval(final_peers_values)),
        # Execution time statistics (using sample std: ddof=1)
        "execution_time_mean": round(np.mean(execution_times), 3),
        "execution_time_std": round(np.std(execution_times, ddof=1), 3),
        "execution_time_total": round(np.sum(execution_times), 3),
        "execution_time_ci_95": list(calculate_confidence_interval(execution_times)),
    }

    # Add statistical significance if baseline is provided
    if baseline_results is not None:
        statistical_results = calculate_statistical_significance(
            results, baseline_results
        )
        base_stats.update(statistical_results)

    return base_stats


def run_benchmark():
    """Run the main benchmark"""

    # Set seed for reproducible results
    random.seed(42)
    np.random.seed(42)

    # Scheduled joins: [(epoch, stake_amount), ...]
    scheduled_joins = [(5000, 10000), (15000, 50000)]
    # scheduled_joins = []

    # name = "scenario_1"
    # # Common parameters for all algorithms
    # params = Parameters(
    #     n_epochs=20000,  # Reduce epochs for faster execution
    #     initial_stake_volume=10000.0,
    #     initial_distribution=Distribution.UNIFORM,
    #     n_peers=1000,  # Reduce peers for faster execution
    #     n_corrupted=20,
    #     p_fail=0.1,
    #     p_join=0.0005,
    #     p_leave=0.0005,
    #     join_amount=NewEntry.NEW_AVERAGE,
    #     penalty_percentage=0.1,
    #     reward=20.0,
    #     scheduled_joins=scheduled_joins,
    # )

    # name = "scenario_2"
    # # Common parameters for all algorithms
    # params = Parameters(
    #     n_epochs=50000,  # Reduce epochs for faster execution
    #     initial_stake_volume=50000.0,
    #     initial_distribution=Distribution.RANDOM,
    #     n_peers=10000,  # Reduce peers for faster execution
    #     n_corrupted=500,
    #     p_fail=0.5,
    #     p_join=0.005,
    #     p_leave=0.005,
    #     join_amount=NewEntry.NEW_RANDOM,
    #     penalty_percentage=0.5,
    #     reward=50.0,
    #     scheduled_joins=scheduled_joins,
    # )

    name = "scenario_3"
    # Common parameters for all algorithms
    params = Parameters(
        n_epochs=20000,  # Reduce epochs for faster execution
        initial_stake_volume=10000.0,
        initial_distribution=Distribution.GINI,
        initial_gini=0.5,
        n_peers=1000,  # Reduce peers for faster execution
        n_corrupted=50,
        p_fail=0.3,
        p_join=0.001,
        p_leave=0.001,
        join_amount=NewEntry.NEW_MAX,
        penalty_percentage=0.3,
        reward=20.0,
        scheduled_joins=scheduled_joins,
    )

    # Generate stakes and corrupted peers (use same data for all algorithms)
    stakes_original = generate_peers(
        params.n_peers,
        params.initial_stake_volume,
        params.initial_distribution,
        0.5,  # initial_gini
    )
    corrupted_original = random.sample(range(params.n_peers), params.n_corrupted)

    algorithms = [
        PoS.WEIGHTED,
        PoS.SRSW_WEIGHTED,
        PoS.LOG_WEIGHTED,
        PoS.DESW,
    ]

    algorithm_names = {
        PoS.WEIGHTED: "WEIGHTED (Baseline)",
        PoS.SRSW_WEIGHTED: "SRSW_WEIGHTED",
        PoS.LOG_WEIGHTED: "LOG_WEIGHTED (LSW)",
        PoS.DESW: "DESW",
    }

    summary_stats = {}
    all_algorithm_results = {}  # Store all results for statistical comparison

    print(f"Benchmark parameters:")
    print(f"  - Epochs: {params.n_epochs}")
    print(f"  - Peers: {params.n_peers}")
    print(f"  - Corrupted: {params.n_corrupted}")
    print(f"  - Initial Volume: {params.initial_stake_volume}")
    print(f"  - Distribution: {params.initial_distribution.name}")
    if params.scheduled_joins:
        print(f"  - Scheduled Joins: {len(params.scheduled_joins)} events")
        for epoch, stake in params.scheduled_joins:
            print(f"    * Epoch {epoch}: +{stake} stake")
    print()

    print(f"Initial Gini coefficient: {gini(stakes_original):.3f}")
    print(f"Number of peers: {len(stakes_original)}")
    print(f"Number of corrupted peers: {len(corrupted_original)}")
    print()

    total_start_time = time.time()

    # Run benchmark for each algorithm
    for algorithm in algorithms:

        print(f"Running {algorithm_names[algorithm]}...")

        algorithm_results = []

        # Run 10 times for each algorithm
        for run_id in range(10):

            random.seed(42 + run_id)
            np.random.seed(42 + run_id)

            result = run_single_experiment(
                algorithm, run_id, params, stakes_original, corrupted_original
            )
            algorithm_results.append(result)

        # Store results for statistical comparison
        all_algorithm_results[algorithm.name] = algorithm_results

        # Calculate summary statistics (without P-values for now)
        stats = calculate_statistics(algorithm_results)
        summary_stats[algorithm.name] = stats

        print(f"  Completed {algorithm_names[algorithm]}")
        print(
            f"  Final Gini: {stats['final_gini_mean']:.3f} ± {stats['final_gini_std']:.3f}"
        )
        print(
            f"  Final Nakamoto: {stats['final_nakamoto_mean']:.3f} ± {stats['final_nakamoto_std']:.3f}"
        )
        print(f"  Execution time: {stats['execution_time_total']:.3f}s")
        print()

    # Calculate P-values for all algorithms compared to baseline (WEIGHTED)
    baseline_results = all_algorithm_results["WEIGHTED"]

    for algorithm in algorithms:
        if algorithm.name == "WEIGHTED":
            continue

        algorithm_results = all_algorithm_results[algorithm.name]

        # Calculate statistics with P-values
        stats_with_pvalues = calculate_statistics(algorithm_results, baseline_results)

        # Update summary stats with P-values
        summary_stats[algorithm.name].update(
            {
                "gini_p_value": stats_with_pvalues["gini_p_value"],
                "gini_significant": stats_with_pvalues["gini_significant"],
                "gini_effect_size": stats_with_pvalues["gini_effect_size"],
                "gini_interpretation": stats_with_pvalues["gini_interpretation"],
                "nakamoto_p_value": stats_with_pvalues["nakamoto_p_value"],
                "nakamoto_significant": stats_with_pvalues["nakamoto_significant"],
                "nakamoto_effect_size": stats_with_pvalues["nakamoto_effect_size"],
                "nakamoto_interpretation": stats_with_pvalues[
                    "nakamoto_interpretation"
                ],
            }
        )

    total_end_time = time.time()
    total_execution_time = total_end_time - total_start_time

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Create results directory if it doesn't exist
    results_dir = os.path.join(os.path.dirname(__file__), "results")
    os.makedirs(results_dir, exist_ok=True)

    # Save only summary results (no detailed data to save time)
    summary_filename = os.path.join(results_dir, f"benchmark_summary_{name}.json")
    with open(summary_filename, "w", encoding="utf-8") as f:
        summary_results = {
            "metadata": {
                "timestamp": timestamp,
                "total_execution_time": total_execution_time,
                "parameters": {
                    "n_epochs": params.n_epochs,
                    "n_peers": params.n_peers,
                    "n_corrupted": params.n_corrupted,
                    "initial_stake_volume": params.initial_stake_volume,
                    "initial_distribution": params.initial_distribution.name,
                    "p_fail": params.p_fail,
                    "p_join": params.p_join,
                    "p_leave": params.p_leave,
                    "penalty_percentage": params.penalty_percentage,
                    "reward": params.reward,
                    "scheduled_joins": params.scheduled_joins,
                },
                "initial_gini": gini(stakes_original),
                "algorithms_tested": [alg.name for alg in algorithms],
            },
            "summary_statistics": summary_stats,
        }
        json.dump(summary_results, f, indent=2, ensure_ascii=False)

    return summary_stats


if __name__ == "__main__":
    try:
        summary_stats = run_benchmark()

    except Exception as e:
        import traceback

        traceback.print_exc()
