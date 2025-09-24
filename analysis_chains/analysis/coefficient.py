import pandas as pd
import numpy as np
import os
from datetime import datetime
import glob

def calculate_gini_coefficient(df, col='tokens'):
    """
    Compute the Gini coefficient using the standard Lorenz curve formula
    
    Args:
        df (pd.DataFrame): DataFrame containing data
        col (str): Column name containing stake values
    
    Returns:
        float: Gini index (0 = perfect equality, 1 = complete inequality)
    """
    # Ensure numpy array for performance
    weights = df[col].to_numpy()

    # Sort values
    weights_sorted = np.sort(weights)

    # Cumulative sum of tokens
    cum_weights = np.cumsum(weights_sorted, dtype=float)
    total_weight = cum_weights[-1]

    # Lorenz curve is cumulative tokens divided by total tokens
    lorenz_curve = cum_weights / total_weight

    # Area under the Lorenz curve
    B = np.trapz(lorenz_curve, dx=1/len(weights))

    # Gini coefficient via G = 1 - 2B
    gini_coefficient = 1 - 2 * B
    
    return gini_coefficient

def calculate_nakamoto_coefficient(df, col='tokens'):
    """
    Compute the Nakamoto coefficient - minimum number of validators to control > 50% stake
    
    Args:
        df (pd.DataFrame): DataFrame containing data
        col (str): Column name containing stake values
    
    Returns:
        int: Number of validators required to exceed 50% of total stake
    """
    tokens = df[col].to_numpy()
    tokens_sorted = np.sort(tokens)[::-1]  # sort descending
    
    total_stake = np.sum(tokens_sorted)
    threshold = total_stake / 2
    
    cumulative_stake = 0
    for i, stake in enumerate(tokens_sorted):
        cumulative_stake += stake
        if cumulative_stake > threshold:
            return i + 1
    
    return len(tokens_sorted)

def calculate_hhi_coefficient(df, col='tokens', normalize=False):
    """
    Compute HHI coefficient (Herfindahl-Hirschman Index) - measures market concentration
    
    Args:
        df (pd.DataFrame): DataFrame containing data
        col (str): Column name containing stake values
        normalize (bool): Normalize HHI by number of validators or not
    
    Returns:
        float: HHI index (0 = fully diffuse, 1 = fully concentrated)
    """
    # Handle NaNs and values <= 0
    weights = df[col].fillna(0).to_numpy()
    weights = weights[weights > 0]

    total_weight = weights.sum()
    if total_weight == 0:
        return 0.0

    market_shares = weights / total_weight
    hhi_index = np.sum(market_shares ** 2)

    if normalize and len(weights) > 1:
        n = len(weights)
        hhi_index = (hhi_index - 1/n) / (1 - 1/n)
    
    return hhi_index
