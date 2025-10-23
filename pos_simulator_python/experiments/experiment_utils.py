#!/usr/bin/env python3
"""
Experiment Utilities
Common utility functions for all experiments
"""

import os
import json
import matplotlib.pyplot as plt


def get_results_dir(foldername):
    """Get the results directory path corresponding to the running file"""
    # Get the path of the file calling this function
    import inspect

    caller_frame = inspect.currentframe().f_back
    caller_file = caller_frame.f_globals["__file__"]

    # Create results path in the same directory as the caller file
    caller_dir = os.path.dirname(os.path.abspath(caller_file))
    results_dir = os.path.join(caller_dir, foldername, "results")

    # Create directory if it doesn't exist
    os.makedirs(results_dir, exist_ok=True)

    return results_dir


def save_results_to_json(results, filename, foldername):
    """
    Save experiment results as JSON to the corresponding results directory

    Args:
        results (dict): Dictionary containing experiment results
        filename (str): JSON filename to save
    """
    results_dir = get_results_dir(foldername)

    serializable_results = {}
    for key, value in results.items():
        result_data = {}

        if "starting_gini" in value:
            result_data["starting_gini"] = value["starting_gini"]
        if "final_gini" in value:
            result_data["final_gini"] = value["final_gini"]
        if "final_nakamoto" in value:
            result_data["final_nakamoto"] = value["final_nakamoto"]
        if "final_peers" in value:
            result_data["final_peers"] = value["final_peers"]
        if "final_hhi" in value:
            result_data["final_hhi"] = value["final_hhi"]

        serializable_results[key] = result_data

    json_path = os.path.join(results_dir, filename)
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(serializable_results, f, indent=2, ensure_ascii=False)

    print(f"Data saved: {json_path}")


def save_plot(foldername, filename, title_suffix=""):
    """
    Save matplotlib plot to the corresponding results directory

    Args:
        filename (str): PNG filename to save (no path needed)
        title_suffix (str): Title extension (optional)
    """
    results_dir = get_results_dir(foldername)
    plot_path = os.path.join(results_dir, filename)

    plt.savefig(plot_path, dpi=300, bbox_inches="tight")
    print(f"Plot{title_suffix} saved: {plot_path}")


def create_and_save_plot(
    history_data, title, xlabel, ylabel, filename, color="blue", foldername=""
):
    """
    Create and save complete plot

    Args:
        history_data (list): Historical data to plot
        title (str): Plot title
        xlabel (str): X-axis label
        ylabel (str): Y-axis label
        filename (str): Filename to save
        color (str): Line color
    """
    plt.figure(figsize=(12, 8))
    plt.plot(history_data, linewidth=2, color=color, alpha=0.8)
    plt.title(title, fontsize=16, fontweight="bold")
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    save_plot(
        foldername,
        filename,
        f" {ylabel}",
    )
    # plt.show()


def print_experiment_results(
    algorithm_name, gini_history, nakamoto_history, peers_history, hhi_history
):
    """
    Print final experiment results

    Args:
        algorithm_name (str): Algorithm name
        gini_history (list): Gini coefficient history
        nakamoto_history (list): Nakamoto coefficient history
        peers_history (list): Peers count history
        hhi_history (list): HHI coefficient history
    """
    print(f"\nFinal results for {algorithm_name}:")
    print(f"  Final Gini: {gini_history[-1]:.3f}")
    print(f"  Final Nakamoto: {nakamoto_history[-1]}")
    print(f"  Final Peers: {peers_history[-1]}")
    print(f"  Final HHI: {hhi_history[-1]:.3f}")


def get_experiment_config():
    """
    Ask user for experiment configuration

    Returns:
        tuple: (starting_gini, n_epochs)
    """
    print("\nExperiment configuration:")

    # Ask for starting Gini
    try:
        starting_gini = float(
            input("Starting Gini coefficient (0-1, default 0.3): ") or "0.3"
        )
        if not (0 <= starting_gini <= 1):
            raise ValueError("Gini coefficient must be between 0-1")
    except ValueError as e:
        print(f"Error: {e}. Using default value 0.3")
        starting_gini = 0.3

    # Ask for number of epochs
    try:
        n_epochs = int(input("Number of epochs (default 50000): ") or "50000")
        if n_epochs <= 0:
            raise ValueError("Number of epochs must be greater than 0")
    except ValueError as e:
        print(f"Error: {e}. Using default value 50000")
        n_epochs = 50000

    return starting_gini, n_epochs


def get_scheduled_joins(scheduled_joins_list):
    """
    Display and return join schedule

    Args:
        scheduled_joins_list (list): List of tuples (epoch, stake)

    Returns:
        list or None: Join schedule or None if none exists
    """
    if scheduled_joins_list:
        print("Join schedule:")
        for epoch, stake in scheduled_joins_list:
            print(f"   • Epoch {epoch}: Validator joins with stake {stake:,.0f}")
        return scheduled_joins_list
    else:
        print("No join schedule configured.")
        return None
