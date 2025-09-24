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

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from parameters import Parameters, PoS, Distribution, NewEntry
from simulator import simulate
from utils import generate_peers, gini


def run_single_experiment(pos_algorithm: PoS, run_id: int, params: Parameters, stakes: List[float], corrupted: List[int]) -> Dict:
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
        reward=params.reward
    )
    
    # Create copies of stakes and corrupted to avoid modifying original data
    test_stakes = stakes.copy()
    test_corrupted = corrupted.copy()
    
    # Run simulation
    start_time = time.time()
    gini_history, peers_history, nakamoto_history, _ = simulate(test_stakes, test_corrupted, test_params)
    end_time = time.time()
    
    return {
        'algorithm': pos_algorithm.name,
        'run_id': run_id,
        'starting_gini': gini(stakes),
        'final_gini': gini_history[-1],
        'final_nakamoto': nakamoto_history[-1],
        'final_peers': peers_history[-1],
        'execution_time': end_time - start_time,
        'gini_history': gini_history,
        'nakamoto_history': nakamoto_history,
        'peers_history': peers_history
    }


def calculate_statistics(results: List[Dict]) -> Dict:
    """Calculate statistics from multiple runs"""
    
    final_gini_values = [r['final_gini'] for r in results]
    final_nakamoto_values = [r['final_nakamoto'] for r in results]
    final_peers_values = [r['final_peers'] for r in results]
    execution_times = [r['execution_time'] for r in results]
    
    return {
        'algorithm': results[0]['algorithm'],
        'num_runs': len(results),
        'starting_gini': results[0]['starting_gini'],
        
        # Final Gini statistics
        'final_gini_mean': round(np.mean(final_gini_values), 3),
        'final_gini_std': round(np.std(final_gini_values), 3),
        'final_gini_min': round(np.min(final_gini_values), 3),
        'final_gini_max': round(np.max(final_gini_values), 3),
        
        # Final Nakamoto statistics
        'final_nakamoto_mean': round(np.mean(final_nakamoto_values), 3),
        'final_nakamoto_std': round(np.std(final_nakamoto_values), 3),
        'final_nakamoto_min': int(np.min(final_nakamoto_values)),
        'final_nakamoto_max': int(np.max(final_nakamoto_values)),
        
        # Final Peers statistics
        'final_peers_mean': round(np.mean(final_peers_values), 3),
        'final_peers_std': round(np.std(final_peers_values), 3),
        'final_peers_min': int(np.min(final_peers_values)),
        'final_peers_max': int(np.max(final_peers_values)),
        
        # Execution time statistics
        'execution_time_mean': round(np.mean(execution_times), 3),
        'execution_time_std': round(np.std(execution_times), 3),
        'execution_time_total': round(np.sum(execution_times), 3)
    }


def run_benchmark():
    """Run the main benchmark"""
    
    print("PoS Algorithms Benchmark - 10 Runs Comparison")
    print("=" * 60)
    
    # Set seed for reproducible results
    random.seed(42)
    np.random.seed(42)

        # Scheduled joins: [(epoch, stake_amount), ...]
    scheduled_joins = []
    
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
    #     scheduled_joins=scheduled_joins
    # )

    name = "scenario_2"
    # Common parameters for all algorithms
    params = Parameters(
        n_epochs=50000,  # Reduce epochs for faster execution
        initial_stake_volume=50000.0,
        initial_distribution=Distribution.RANDOM,
        n_peers=10000,  # Reduce peers for faster execution
        n_corrupted=500,
        p_fail=0.5,
        p_join=0.005,
        p_leave=0.005,
        join_amount=NewEntry.NEW_RANDOM,
        penalty_percentage=0.5,
        reward=50.0,
        scheduled_joins=scheduled_joins
    )

    # name = "scenario_3"
    # # Common parameters for all algorithms
    # params = Parameters(
    #     n_epochs=20000,  # Reduce epochs for faster execution
    #     initial_stake_volume=10000.0,
    #     initial_distribution=Distribution.GINI,
    #     n_peers=1000,  # Reduce peers for faster execution
    #     n_corrupted=50,
    #     p_fail=0.3,
    #     p_join=0.001,
    #     p_leave=0.001,
    #     join_amount=NewEntry.NEW_MAX,
    #     penalty_percentage=0.3,
    #     reward=20.0,
    #     scheduled_joins=scheduled_joins
    # )
    
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
    
    # Generate stakes and corrupted peers (use same data for all algorithms)
    stakes_original = generate_peers(
        params.n_peers, 
        params.initial_stake_volume,
        params.initial_distribution, 
        0.5  # initial_gini
    )
    corrupted_original = random.sample(range(params.n_peers), params.n_corrupted)
    
    print(f"Initial Gini coefficient: {gini(stakes_original):.3f}")
    print(f"Number of peers: {len(stakes_original)}")
    print(f"Number of corrupted peers: {len(corrupted_original)}")
    print()
    
    # 4 main algorithms for comparison
    algorithms = [
        PoS.WEIGHTED,        # Baseline
        PoS.SRSW_WEIGHTED,   # SRSW
        PoS.LOG_WEIGHTED,    # LSW (Log-weighted)
        PoS.DESW             # DESW
    ]
    
    algorithm_names = {
        PoS.WEIGHTED: "WEIGHTED (Baseline)",
        PoS.SRSW_WEIGHTED: "SRSW_WEIGHTED", 
        PoS.LOG_WEIGHTED: "LOG_WEIGHTED (LSW)",
        PoS.DESW: "DESW"
    }
    
    summary_stats = {}
    
    total_start_time = time.time()
    
    # Run benchmark for each algorithm
    for algorithm in algorithms:
        print(f"Running {algorithm_names[algorithm]}...")
        
        algorithm_results = []
        
        # Run 10 times for each algorithm
        for run_id in range(10):
            result = run_single_experiment(
                algorithm, 
                run_id, 
                params, 
                stakes_original, 
                corrupted_original
            )
            algorithm_results.append(result)
        
        # Calculate summary statistics
        stats = calculate_statistics(algorithm_results)
        summary_stats[algorithm.name] = stats
        
        print(f"  Completed {algorithm_names[algorithm]}")
        print(f"  Final Gini: {stats['final_gini_mean']:.3f} ± {stats['final_gini_std']:.3f}")
        print(f"  Final Nakamoto: {stats['final_nakamoto_mean']:.3f} ± {stats['final_nakamoto_std']:.3f}")
        print(f"  Execution time: {stats['execution_time_total']:.3f}s")
        print()
    
    total_end_time = time.time()
    total_execution_time = total_end_time - total_start_time
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Create results directory if it doesn't exist
    results_dir = os.path.join(os.path.dirname(__file__), 'results')
    os.makedirs(results_dir, exist_ok=True)
    
    # Save only summary results (no detailed data to save time)
    summary_filename = os.path.join(results_dir, f'benchmark_summary_{name}.json')
    with open(summary_filename, 'w', encoding='utf-8') as f:
        summary_results = {
            'metadata': {
                'timestamp': timestamp,
                'total_execution_time': total_execution_time,
                'parameters': {
                    'n_epochs': params.n_epochs,
                    'n_peers': params.n_peers,
                    'n_corrupted': params.n_corrupted,
                    'initial_stake_volume': params.initial_stake_volume,
                    'initial_distribution': params.initial_distribution.name,
                    'p_fail': params.p_fail,
                    'p_join': params.p_join,
                    'p_leave': params.p_leave,
                    'penalty_percentage': params.penalty_percentage,
                    'reward': params.reward,
                    'scheduled_joins': params.scheduled_joins
                },
                'initial_gini': gini(stakes_original),
                'algorithms_tested': [alg.name for alg in algorithms]
            },
            'summary_statistics': summary_stats
        }
        json.dump(summary_results, f, indent=2, ensure_ascii=False)
    
    # Print final comparison table
    print("\n" + "=" * 70)
    print("FINAL COMPARISON TABLE (Average of 10 runs)")
    print("=" * 70)
    print(f"{'Algorithm':<20} {'Final Gini':<15} {'Final Nakamoto':<18} {'Exec Time(s)':<12}")
    print("-" * 70)
    
    for algorithm in algorithms:
        stats = summary_stats[algorithm.name]
        print(f"{algorithm_names[algorithm]:<20} "
              f"{stats['final_gini_mean']:.3f}±{stats['final_gini_std']:.3f}   "
              f"{stats['final_nakamoto_mean']:.3f}±{stats['final_nakamoto_std']:.3f}        "
              f"{stats['execution_time_total']:.3f}")
    
    print("\n" + "=" * 70)
    print(f"Total execution time: {total_execution_time:.3f}s")
    print(f"Results saved in: {summary_filename}")
    print("=" * 70)
    
    return summary_stats


if __name__ == "__main__":
    try:
        print("Starting benchmark...")
        summary_stats = run_benchmark()
        print("\nBenchmark completed successfully!")
        
    except Exception as e:
        print(f"Error running benchmark: {e}")
        import traceback
        traceback.print_exc()
