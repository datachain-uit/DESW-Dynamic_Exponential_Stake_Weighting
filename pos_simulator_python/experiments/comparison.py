#!/usr/bin/env python3
"""
Simple example illustrating the Python implementation of the PoS Simulator
Compare 6 PoS algorithms with 2 metrics: Gini and Nakamoto Coefficient
"""

import sys
import os
import random
import numpy as np
import matplotlib.pyplot as plt
import json

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from parameters import Parameters, PoS, Distribution, NewEntry
from simulator import simulate
from utils import generate_peers, gini, HHI_coefficient
from experiment_utils import save_results_to_json, save_plot, create_and_save_plot


# SCHEDULED JOIN SETUP - Edit directly here
# Example: [(5000, 50000), (10000, 30000)] -> At epoch 5000 a validator joins with 50k stake, at epoch 10000 joins with 30k stake
SCHEDULED_JOINS = [
    # (5000, 10000),  
    # (15000, 50000),  
]

def get_scheduled_joins():
    """Return the configured join schedule"""
    if SCHEDULED_JOINS:
        for epoch, stake in SCHEDULED_JOINS:
            print(f"   • Epoch {epoch}: Validator join with stake {stake:,.0f}")
        return SCHEDULED_JOINS
    else:
        print("\nNo join schedule configured.")
        return None


def run_single_experiment(pos_algorithm, experiment_name, starting_gini=0.3, scheduled_joins=None):
    """Run a single experiment with the specified PoS algorithm"""
    print(f"Running {experiment_name}")
    
    # Common parameters
    params = Parameters(
        n_epochs=25000,
        proof_of_stake=pos_algorithm,
        initial_stake_volume=5000.0,
        initial_distribution=Distribution.RANDOM,
        n_peers=10000,
        n_corrupted=50,
        p_fail=0.5,
        p_join=0.001,
        p_leave=0.001,
        join_amount=NewEntry.NEW_RANDOM,
        penalty_percentage=0.5,
        reward=20.0,
        scheduled_joins=scheduled_joins
    )
    
    # Generate initial stakes
    stakes = generate_peers(
        params.n_peers, 
        params.initial_stake_volume, 
        params.initial_distribution, 
        starting_gini
    )
    
    # Create corrupted peers
    corrupted = random.sample(range(params.n_peers), params.n_corrupted)
    
    print(f"  Initial Gini: {gini(stakes):.3f}")
    print(f"  Peers: {len(stakes)}, Corrupted: {len(corrupted)}")
    
    # Run simulation
    gini_history, peers_history, nakamoto_history, hhi_history = simulate(stakes, corrupted, params)
    
    print(f"  Final Gini: {gini_history[-1]:.3f}")
    print(f"  Final Nakamoto: {nakamoto_history[-1]}")
    print(f"  Final Peers: {peers_history[-1]}")
    print(f"  Final HHI: {hhi_history[-1]:.3f}")
    
    # Create filename for plots
    filename = experiment_name.lower().replace(' ', '_').replace(':', '')
    
    # Plot 1: Gini Coefficient
    plt.figure(figsize=(12, 8))
    plt.plot(gini_history, linewidth=2, color='blue', alpha=0.8)
    plt.title(f'{experiment_name} - Gini Coefficient', fontsize=16, fontweight='bold')
    plt.xlabel('Epoch')
    plt.ylabel('Gini Coefficient')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    save_plot('', f'{filename}_gini.png', ' Gini')
    plt.show()
    
    # Plot 2: Nakamoto Coefficient
    plt.figure(figsize=(12, 8))
    plt.plot(nakamoto_history, linewidth=2, color='red', alpha=0.8)
    plt.title(f'{experiment_name} - Nakamoto Coefficient', fontsize=16, fontweight='bold')
    plt.xlabel('Epoch')
    plt.ylabel('Nakamoto Coefficient')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    save_plot('', f'{filename}_nakamoto.png', ' Nakamoto')
    plt.show()
    
    # Plot 3: Peers Count
    plt.figure(figsize=(12, 8))
    plt.plot(peers_history, linewidth=2, color='green', alpha=0.8)
    plt.title(f'{experiment_name} - Peers Count', fontsize=16, fontweight='bold')
    plt.xlabel('Epoch')
    plt.ylabel('Number of Peers')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    save_plot('', f'{filename}_peers.png', ' Peers Count')
    # plt.show()
    
    # Plot 4: HHI Coefficient
    plt.figure(figsize=(12, 8))
    plt.plot(hhi_history, linewidth=2, color='orange', alpha=0.8)
    plt.title(f'{experiment_name} - HHI Coefficient', fontsize=16, fontweight='bold')
    plt.xlabel('Epoch')
    plt.ylabel('HHI Coefficient')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    save_plot('', f'{filename}_hhi.png', ' HHI Coefficient')
    plt.show()
    
    # Save data
    result = {
        'gini_history': gini_history,
        'nakamoto_history': nakamoto_history,
        'hhi_history': hhi_history,
        'peers_history': peers_history,
        'starting_gini': starting_gini,
        'final_gini': gini_history[-1],
        'final_nakamoto': nakamoto_history[-1],
        'final_peers': peers_history[-1],
        'final_hhi': hhi_history[-1]
    }
    # Convert to format compatible with experiment_utils
    results_for_save = {
        'experiment_result': {
            'starting_gini': result['starting_gini'],
            'final_gini': result['final_gini'],
            'final_nakamoto': result['final_nakamoto'],
            'final_peers': result['final_peers'],
            'final_hhi': result['final_hhi']
        }
    }
    save_results_to_json(results_for_save, f'{filename}_data.json', '')
    
    return result


def run_experiment_1():
    """Experiment 1: WEIGHTED PoS"""
    scheduled_joins = get_scheduled_joins()
    return run_single_experiment(PoS.WEIGHTED, "Experiment 1: WEIGHTED PoS", scheduled_joins=scheduled_joins)


def run_experiment_2():
    """Experiment 2: OPPOSITE_WEIGHTED PoS"""
    scheduled_joins = get_scheduled_joins()
    return run_single_experiment(PoS.OPPOSITE_WEIGHTED, "Experiment 2: OPPOSITE_WEIGHTED PoS", scheduled_joins=scheduled_joins)


def run_experiment_3():
    """Experiment 3: GINI_STABILIZED PoS"""
    scheduled_joins = get_scheduled_joins()
    return run_single_experiment(PoS.GINI_STABILIZED, "Experiment 3: GINI_STABILIZED PoS", scheduled_joins=scheduled_joins)


def run_experiment_4():
    """Experiment 4: LOG_WEIGHTED PoS"""
    scheduled_joins = get_scheduled_joins()
    return run_single_experiment(PoS.LOG_WEIGHTED, "Experiment 4: LOG_WEIGHTED PoS", scheduled_joins=scheduled_joins)


def run_experiment_5():
    """Experiment 5: DESW PoS"""
    scheduled_joins = get_scheduled_joins()
    return run_single_experiment(PoS.DESW, "Experiment 5: DESW PoS", scheduled_joins=scheduled_joins)

def run_experiment_6():
    """Experiment 6: SRSW_WEIGHTED PoS"""
    scheduled_joins = get_scheduled_joins()
    return run_single_experiment(PoS.SRSW_WEIGHTED, "Experiment 6: SRSW_WEIGHTED PoS", scheduled_joins=scheduled_joins)

def run_comparison_experiment():
    """Experiment 7: Compare all 6 PoS algorithms"""
    print("Compare all 6 PoS algorithms")
    print("=" * 50)
    
    # Ask for scheduled joins for this experiment
    scheduled_joins = get_scheduled_joins()
    
    # Common parameters for all algorithms
    base_params = {
        'n_epochs': 20000,
        'initial_stake_volume': 5000.0,
        'initial_distribution': Distribution.RANDOM,
        'n_peers': 10000,
        'n_corrupted': 50,
        'initial_gini': 0.3,
        'p_fail': 0.1,
        'p_join': 0.0001,
        'p_leave': 0.0001,
        'join_amount': NewEntry.NEW_RANDOM,
        'penalty_percentage': 0.5,
        'reward': 20.0
    }

      
    # Generate stakes and corrupted peers (use the same data for all)
    stakes_original = generate_peers(
        base_params['n_peers'], 
        base_params['initial_stake_volume'],
        base_params['initial_distribution'], 
        base_params['initial_gini']
    )
    corrupted = random.sample(range(base_params['n_peers']), base_params['n_corrupted'])
    
    print(f"Initial Gini coefficient: {gini(stakes_original):.3f}")
    print(f"Number of peers: {len(stakes_original)}")
    print(f"Number of corrupted peers: {len(corrupted)}")
    print()
    
    # Dictionary to store results for each algorithm
    algorithms = {
        'WEIGHTED': PoS.WEIGHTED,
        'OPPOSITE_WEIGHTED': PoS.OPPOSITE_WEIGHTED,
        'GINI_STABILIZED': PoS.GINI_STABILIZED,
        'LOG_WEIGHTED': PoS.LOG_WEIGHTED,
        'DESW': PoS.DESW,
        'SRSW_WEIGHTED': PoS.SRSW_WEIGHTED
    }
    
    results = {}
    colors = {'WEIGHTED': 'blue',
                'OPPOSITE_WEIGHTED': 'red', 
              'GINI_STABILIZED': 'green',
               'LOG_WEIGHTED': 'purple',  
              'DESW': 'brown', 
              'SRSW_WEIGHTED': 'orange'}
    
    # Run simulation for each algorithm
    for name, pos_type in algorithms.items():
        print(f"Running {name} simulation...")
        
        params = Parameters(
            proof_of_stake=pos_type,
            scheduled_joins=scheduled_joins,
            **base_params
        )
        stakes = stakes_original.copy()
        
        gini_history, peers_history, nakamoto_history, hhi_history = simulate(
            stakes, corrupted.copy(), params
        )
        
        results[name] = {
            'gini_history': gini_history,
            'nakamoto_history': nakamoto_history,
            'hhi_history': hhi_history,
            'peers_history': peers_history,
            'final_gini': gini_history[-1],
            'final_nakamoto': nakamoto_history[-1],
            'final_peers': peers_history[-1],
            'final_hhi': hhi_history[-1]
        }
        
        print(f"  Final Gini: {gini_history[-1]:.3f}")
        print(f"  Final Nakamoto: {nakamoto_history[-1]}")
        print(f"  Final Peers: {peers_history[-1]}")
        print(f"  Final HHI: {hhi_history[-1]:.3f}")
    
    # Plot 1: Gini Coefficient Comparison
    plt.figure(figsize=(12, 8))
    for name, result in results.items():
        plt.plot(result['gini_history'], label=name, linewidth=2, 
                color=colors[name], alpha=0.8)
    
    plt.title('Gini Coefficient Evolution - All PoS Algorithms', 
              fontsize=16, fontweight='bold')
    plt.xlabel('Epoch')
    plt.ylabel('Gini Coefficient')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    save_plot('', 'gini_comparison.png', ' Gini')
    plt.show()
    
    # Plot 2: Nakamoto Coefficient Comparison
    plt.figure(figsize=(12, 8))
    for name, result in results.items():
        plt.plot(result['nakamoto_history'], label=name, linewidth=2, 
                color=colors[name], alpha=0.8)
    
    plt.title('Nakamoto Coefficient Evolution - All PoS Algorithms', 
              fontsize=16, fontweight='bold')
    plt.xlabel('Epoch')
    plt.ylabel('Nakamoto Coefficient')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    save_plot('', 'nakamoto_comparison.png', ' Nakamoto')
    plt.show()
      
    # Detailed stats
    print("\nFINAL COMPARISON RESULTS:")
    print("-" * 70)
    print(f"{'Algorithm':<20} {'Final Gini':<12} {'Final Nakamoto':<15} {'Final Peers':<12} {'Final HHI':<12}")
    print("-" * 70)
    
    for name, result in results.items():
        print(f"{name:<20} {result['final_gini']:<12.3f} {result['final_nakamoto']:<15} {result['final_peers']:<12} {result['final_hhi']:<12.3f}")

    # Save data
    save_results_to_json(results, 'all_pos_comparison_data.json', '')
    
    print("\nComparison completed!")
    return results


def main():
    """Run PoS Simulator experiment"""
    print("PoS Simulator - Compare 6 Proof-of-Stake Algorithms")
    print("=" * 60)
    
    # Set random seed for reproducibility
    random.seed(42)
    np.random.seed(42)
    
    print("Results will be saved in the results/ folder (created automatically)")
    
    try:
        while True:
            print("\nChoose an experiment:")
            print("1. Experiment 1: WEIGHTED PoS")
            print("2. Experiment 2: OPPOSITE_WEIGHTED PoS")
            print("3. Experiment 3: GINI_STABILIZED PoS") 
            print("4. Experiment 4: LOG_WEIGHTED PoS")
            print("5. Experiment 5: DESW PoS")
            print("6. Experiment 6: SRSW_WEIGHTED PoS")
            print("7. Compare all 6 algorithms")
            print("8. Exit")
            
            choice = input("\nEnter choice (1-9): ").strip()
            
            if choice == "1":
                print("\n" + "=" * 60)
                run_experiment_1()
            elif choice == "2":
                print("\n" + "=" * 60)
                run_experiment_2()
            elif choice == "3":
                print("\n" + "=" * 60)
                run_experiment_3()
            elif choice == "4":
                print("\n" + "=" * 60)
                run_experiment_4()
            elif choice == "5":
                print("\n" + "=" * 60)
                run_experiment_5()
            elif choice == "6":
                print("\n" + "=" * 60)
                run_experiment_6()
            elif choice == "7":
                print("\n" + "=" * 60)
                run_comparison_experiment()
            elif choice == "8":
                print("Goodbye!")
                break
            else:
                print("Invalid choice. Please try again.")
        
        print("\n" + "=" * 60)
        print("All experiments completed successfully!")
        
    except Exception as e:
        print(f"Error during execution: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()