"""
Main simulation module for PoS Simulator
Equivalent to Simulator.jl from the original Julia version
"""

from typing import List, Tuple
import copy

# Initialize pmin and pmax for DESW
pmin = 0
pmax = 1

try:
    from tqdm import tqdm
except ImportError:
    # Fallback if tqdm is not available
    def tqdm(iterable, **kwargs):
        return iterable

try:
    from .parameters import Parameters, PoS, SType
    from .utils import (
        gini, consensus, d, lerp, try_to_join, try_to_leave,
        nakamoto_coefficient, decentralization_score, HHI_coefficient
    )
except ImportError:
    # Khi chạy như script, sử dụng absolute imports
    from parameters import Parameters, PoS, SType
    from utils import (
        gini, consensus, d, lerp, try_to_join, try_to_leave,
        nakamoto_coefficient, decentralization_score, HHI_coefficient
    )



def simulate(stakes: List[float], corrupted: List[int], params: Parameters) -> Tuple[List[float], List[int], List[int], List[float]]:
    """
    Run PoS simulation
    
    Args:
        stakes: Initial stake for each peer
        corrupted: List of indices of corrupted peers
        params: Simulation parameters
        
    Returns:
        Tuple of (gini_history, n_peers_history, nakamoto_history, hhi_history)
    """
    # Create copies to avoid modifying original data
    stakes = copy.deepcopy(stakes)
    corrupted = copy.deepcopy(corrupted)
    
    gini_history = []
    n_peers_history = []
    nakamoto_history = []
    hhi_history = []
    
    percentage_corrupted = len(corrupted) / len(stakes) if stakes else 0
    
    # Initialize t for GiniStabilized
    t = d(gini(stakes), params.θ)
    
    # Convert scheduled_joins to dictionary for fast lookup
    scheduled_joins_dict = {}
    if params.scheduled_joins:
        for epoch, stake_amount in params.scheduled_joins:
            if epoch not in scheduled_joins_dict:
                scheduled_joins_dict[epoch] = []
            scheduled_joins_dict[epoch].append(stake_amount)
    
    for i in range(params.n_epochs):
        # Handle scheduled joins for current epoch
        if i in scheduled_joins_dict:
            for stake_amount in scheduled_joins_dict[i]:
                stakes.append(stake_amount)
                print(f"  Epoch {i}: Scheduled join with stake {stake_amount:.2f}")
        
        # Try adding/removing peers (random joins)
        try_to_join(stakes, corrupted, params.p_join, params.join_amount, percentage_corrupted)
        try_to_leave(stakes, params.p_leave)
        
        # Calculate current Gini coefficient
        g = gini(stakes)
        gini_history.append(g)
        
        # Calculate current Nakamoto Coefficient
        nc = nakamoto_coefficient(stakes)
        nakamoto_history.append(nc)

        # Calculate current HHI Coefficient
        hhi = HHI_coefficient(stakes)
        hhi_history.append(hhi)
        
        # Select validator based on consensus mechanism
        if params.proof_of_stake == PoS.GINI_STABILIZED:
            # Calculate s based on s_type
            if params.s_type == SType.CONSTANT:
                s = params.k
            elif params.s_type == SType.LINEAR:
                s = abs(g - params.θ) * params.k
            elif params.s_type == SType.QUADRATIC:
                s = (abs(g - params.θ)) ** 2 * params.k
            else:  # SQRT or other
                s = (abs(g - params.θ)) ** 0.5 * params.k
            
            validator = consensus(params.proof_of_stake, stakes, t)
            t = lerp(t, d(g, params.θ), s)
        else:
            if params.proof_of_stake == PoS.DESW:
                validator = consensus(params.proof_of_stake, stakes, pmin, pmax)
            else:
                validator = consensus(params.proof_of_stake, stakes)
        
        # Apply rewards/penalties
        if validator in corrupted and __import__('random').random() > 1 - params.p_fail:
            # Corrupted validator fails - apply penalty
            stakes[validator] *= 1 - params.penalty_percentage
        else:
            # Successful validation - apply reward
            stakes[validator] += params.reward
        
        # Record number of peers
        n_peers_history.append(len(stakes))
    
    return gini_history, n_peers_history, nakamoto_history, hhi_history


def simulate_verbose(stakes: List[float], corrupted: List[int], params: Parameters) -> Tuple[List[float], List[int], List[int], List[float]]:
    """
    Run PoS simulation with progress bar
    
    Args:
        stakes: Initial stake for each peer
        corrupted: List of indices of corrupted peers
        params: Simulation parameters
        
    Returns:
        Tuple of (gini_history, n_peers_history, nakamoto_history, hhi_history)
    """
    # Create copies to avoid modifying original data
    stakes = copy.deepcopy(stakes)
    corrupted = copy.deepcopy(corrupted)
    
    gini_history = []
    n_peers_history = []
    nakamoto_history = []
    hhi_history = []
    
    percentage_corrupted = len(corrupted) / len(stakes) if stakes else 0
    
    # Initialize t for GiniStabilized
    t = d(gini(stakes), params.θ)

    # Initialize pmin and pmax for DESW

    # Convert scheduled_joins to dictionary for fast lookup
    scheduled_joins_dict = {}
    if params.scheduled_joins:
        for epoch, stake_amount in params.scheduled_joins:
            if epoch not in scheduled_joins_dict:
                scheduled_joins_dict[epoch] = []
            scheduled_joins_dict[epoch].append(stake_amount)
    
    # Use tqdm for progress bar (equivalent to @showprogress in Julia)
    for i in tqdm(range(params.n_epochs), desc="Simulating epochs"):
        # Handle scheduled joins for current epoch
        if i in scheduled_joins_dict:
            for stake_amount in scheduled_joins_dict[i]:
                stakes.append(stake_amount)
                print(f"  Epoch {i}: Scheduled join with stake {stake_amount:.2f}")
        
        # Try adding/removing peers (random joins)
        try_to_join(stakes, corrupted, params.p_join, params.join_amount, percentage_corrupted)
        try_to_leave(stakes, params.p_leave)
        
        # Calculate current Gini coefficient
        g = gini(stakes)
        gini_history.append(g)
        
        # Calculate current Nakamoto Coefficient
        nc = nakamoto_coefficient(stakes)
        nakamoto_history.append(nc)

        # Calculate current HHI Coefficient
        hhi = HHI_coefficient(stakes)
        hhi_history.append(hhi)
        
        # Select validator based on consensus mechanism
        if params.proof_of_stake == PoS.GINI_STABILIZED:
            # Calculate s based on s_type
            if params.s_type == SType.CONSTANT:
                s = params.k
            elif params.s_type == SType.LINEAR:
                s = abs(g - params.θ) * params.k
            elif params.s_type == SType.QUADRATIC:
                s = (abs(g - params.θ)) ** 2 * params.k
            else:  # SQRT or other
                s = (abs(g - params.θ)) ** 0.5 * params.k
            
            validator = consensus(params.proof_of_stake, stakes, t)
            t = lerp(t, d(g, params.θ), s)
        else:
            if params.proof_of_stake == PoS.DESW:
                validator = consensus(params.proof_of_stake, stakes, pmin, pmax)
            else:
                validator = consensus(params.proof_of_stake, stakes)
        
        # Apply rewards/penalties
        if validator in corrupted and __import__('random').random() > 1 - params.p_fail:
            # Corrupted validator fails - apply penalty
            stakes[validator] *= 1 - params.penalty_percentage
        else:
            # Successful validation - apply reward
            stakes[validator] += params.reward
        
        # Record number of peers
        n_peers_history.append(len(stakes))
    
    return gini_history, n_peers_history, nakamoto_history, hhi_history


# Convenience function to run a single experiment
def run_experiment(n_peers: int, initial_volume: float, initial_gini: float, 
                  params: Parameters, verbose: bool = False) -> Tuple[List[float], List[int], List[int], List[float]]:
    """
    Run a single experiment with given parameters
    
    Args:
        n_peers: Initial number of peers
        initial_volume: Initial stake volume
        initial_gini: Initial Gini coefficient
        params: Simulation parameters
        verbose: Whether to show progress bar
        
    Returns:
        Tuple of (gini_history, n_peers_history, nakamoto_history, hhi_history)
    """
    from .utils import generate_peers
    import random
    
    # Generate initial stakes
    stakes = generate_peers(n_peers, initial_volume, params.initial_distribution, initial_gini)
    
    # Create corrupted peers
    corrupted = random.sample(range(n_peers), params.n_corrupted)
    
    # Run simulation
    if verbose:
        return simulate_verbose(stakes, corrupted, params)
    else:
        return simulate(stakes, corrupted, params)