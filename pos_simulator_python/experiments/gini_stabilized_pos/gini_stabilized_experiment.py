#!/usr/bin/env python3
"""
GINI_STABILIZED PoS Experiment
Experiment with the GINI_STABILIZED Proof-of-Stake algorithm
"""

import sys
import os
import random
import numpy as np
import matplotlib.pyplot as plt
import json

# Add src and experiments to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', 'src'))
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from parameters import Parameters, PoS, Distribution, NewEntry
from simulator import simulate
from utils import generate_peers, gini, HHI_coefficient
from experiment_utils import save_results_to_json, create_and_save_plot, print_experiment_results, get_experiment_config, get_scheduled_joins


# SCHEDULED JOIN SETUP - Edit directly here
# Example: [(5000, 50000), (10000, 30000)] -> At epoch 5000 a validator joins with 50k stake, at epoch 10000 joins with 30k stake
SCHEDULED_JOINS = [
    # (5000, 10000),  
    # (15000, 50000),  
]

foldername = 'gini_stabilized_pos'


def run_gini_stabilized_experiment(starting_gini=0.3, n_epochs=50000):
    """Run GINI_STABILIZED PoS experiment"""
    print("GINI_STABILIZED PoS EXPERIMENT")
    print("=" * 50)
    
    # Get join schedule
    scheduled_joins = get_scheduled_joins(SCHEDULED_JOINS)
    
    # Set parameters
    params = Parameters(
        n_epochs=n_epochs,
        proof_of_stake=PoS.GINI_STABILIZED,
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
    
    print(f"Initial Gini: {gini(stakes):.3f}")
    print(f"Peers: {len(stakes)}, Corrupted: {len(corrupted)}")
    print(f"Epochs: {n_epochs}")
    
    # Run simulation
    print("\nStarting simulation...")
    gini_history, peers_history, nakamoto_history, hhi_history = simulate(stakes, corrupted, params)
    
    # Print results
    print_experiment_results("GINI_STABILIZED PoS", gini_history, nakamoto_history, peers_history, hhi_history)
    
    # Plot and save charts
    create_and_save_plot(gini_history, 'GINI_STABILIZED PoS - Gini Coefficient Evolution', 
                        'Epoch', 'Gini Coefficient', 'gini_stabilized_gini.png', 'blue',foldername)
    
    create_and_save_plot(nakamoto_history, 'GINI_STABILIZED PoS - Nakamoto Coefficient Evolution', 
                        'Epoch', 'Nakamoto Coefficient', 'gini_stabilized_nakamoto.png', 'red',foldername)
    
    create_and_save_plot(peers_history, 'GINI_STABILIZED PoS - Peers Count Evolution', 
                        'Epoch', 'Number of Peers', 'gini_stabilized_peers.png', 'green',foldername)
    
    create_and_save_plot(hhi_history, 'GINI_STABILIZED PoS - HHI Coefficient Evolution', 
                        'Epoch', 'HHI Coefficient', 'gini_stabilized_hhi.png', 'orange',foldername)
    
    # Save data
    result = {
        'starting_gini': starting_gini,
        'final_gini': gini_history[-1],
        'final_nakamoto': nakamoto_history[-1],
        'final_peers': peers_history[-1],
        'final_hhi': hhi_history[-1],
        'gini_history': gini_history,
        'nakamoto_history': nakamoto_history,
        'peers_history': peers_history,
        'hhi_history': hhi_history
    }
    
    save_results_to_json({0: result}, 'gini_stabilized_results.json',foldername)
    
    return result


def main():
    """Run GINI_STABILIZED PoS experiment"""
    print("GINI_STABILIZED PoS Simulator")
    print("=" * 60)
    
    # Set random seed for reproducibility
    random.seed(42)
    np.random.seed(42)
    
    try:
        # Get configuration from user
        starting_gini, n_epochs = get_experiment_config()
        
        print(f"\nStarting experiment with:")
        print(f"- Starting Gini: {starting_gini}")
        print(f"- Epochs: {n_epochs}")
        print()
        
        # Run experiment
        result = run_gini_stabilized_experiment(starting_gini, n_epochs)
        
        print("\n" + "=" * 60)
        print("GINI_STABILIZED PoS experiment completed successfully!")
        print(f"Results are saved in the 'results/' folder")
        
    except Exception as e:
        print(f"Error during execution: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
