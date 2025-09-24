"""
Utility functions for PoS simulation
"""

import numpy as np
import random
from typing import List, Tuple, Optional
try:
    from .parameters import Distribution, PoS, NewEntry, Parameters
except ImportError:
    from parameters import Distribution, PoS, NewEntry, Parameters


def gini(data: List[float]) -> float:
    """
    Compute the Gini coefficient of a given dataset.

    The Gini coefficient is a statistical measure of dispersion representing
    inequality within a distribution. Commonly used to measure income inequality.

    Args:
        data: List of numeric data points.

    Returns:
        The Gini coefficient of the input dataset in [0, 1], where 0 means
        perfect equality and 1 means maximal inequality.

    Examples:
        >>> data = [100.0, 200.0, 300.0, 400.0, 500.0]
        >>> gini(data)  # Output: 0.2
    """
    if not data or len(data) == 0:
        return 0.0
    
    # Number of data points
    n = len(data)
    
    # Sum of data points
    total = sum(data)
    
    if total == 0:
        return 0.0
    
    # Sort data points in ascending order
    sorted_data = sorted(data)
    
    # Cumulative share of sorted data
    cumulative_percentage = np.cumsum(sorted_data) / total
    
    # Lorenz curve
    lorenz_curve = cumulative_percentage - 0.5 * (np.array(sorted_data) / total)
    
    # Gini coefficient
    G = 1 - 2 * np.sum(lorenz_curve) / n
    
    return G


def nakamoto_coefficient(data: List[float], threshold: float = 0.51) -> int:
    """
    Compute the Nakamoto Coefficient of a given dataset.

    The Nakamoto Coefficient is the smallest number of entities required to
    control a given percentage (default 51%) of the total resource. It is a key
    metric of decentralization in blockchains.

    Args:
        data: List of numeric data (e.g., validators' stakes).
        threshold: Target fraction to control (default 0.51 = 51%).

    Returns:
        The Nakamoto Coefficient: the minimal number of entities required to
        control threshold% of the total resource.

    Examples:
        >>> data = [100.0, 200.0, 300.0, 400.0, 500.0]  # Total: 1500
        >>> nakamoto_coefficient(data)  # 2 largest validators (500+400=900 > 765)
        >>> nakamoto_coefficient(data, 0.33)  # 1 validator (500 > 495)
    """
    if not data or len(data) == 0:
        return 0
    
    # Total resource
    total = sum(data)
    
    if total == 0:
        return 0
    
    # Sort descending (largest to smallest)
    sorted_data = sorted(data, reverse=True)
    
    # Target amount to control
    target_amount = total * threshold
    
    # Cumulative sum from the largest
    cumulative_sum = 0
    for i, value in enumerate(sorted_data):
        cumulative_sum += value
        if cumulative_sum >= target_amount:
            return i + 1  # Return number of entities needed
    
    # Fallback in case of insufficient resource (shouldn't happen)
    return len(data)


def nakamoto_coefficient_analysis(data: List[float]) -> dict:
    """
    Detailed analysis of the Nakamoto Coefficient across multiple thresholds.

    Args:
        data: List of numeric data points.

    Returns:
        Dictionary mapping percentage thresholds to their Nakamoto Coefficient.

    Examples:
        >>> data = [100.0, 200.0, 300.0, 400.0, 500.0]
        >>> result = nakamoto_coefficient_analysis(data)
        >>> print(result)
        {'25%': 1, '33%': 1, '50%': 2, '51%': 2, '66%': 3, '75%': 4}
    """
    if not data or len(data) == 0:
        return {}
    
    thresholds = [0.25, 0.33, 0.50, 0.51, 0.66, 0.75]
    results = {}
    
    for threshold in thresholds:
        nc = nakamoto_coefficient(data, threshold)
        results[f"{int(threshold * 100)}%"] = nc
    
    return results


def decentralization_score(data: List[float]) -> float:
    """
    Compute a decentralization score based on the Nakamoto Coefficient.

    The score normalizes the Nakamoto Coefficient by the total number of
    entities. Higher score = more decentralized.

    Args:
        data: List of numeric data points.

    Returns:
        Decentralization score in [0, 1] (1 = fully decentralized).

    Examples:
        >>> data = [100.0, 100.0, 100.0, 100.0, 100.0]  # Even distribution
        >>> decentralization_score(data)  # ~1.0
        >>> data = [1000.0, 10.0, 10.0, 10.0, 10.0]  # Concentrated
        >>> decentralization_score(data)  # ~0.0
    """
    if not data or len(data) == 0:
        return 0.0
    
    n_entities = len(data)
    nc_51 = nakamoto_coefficient(data, 0.51)
    
    # Decentralization score = (n_entities - nc_51) / (n_entities - 1)
    # When nc_51 = 1 (highly centralized) → score = 0
    # When nc_51 = n_entities (fully decentralized) → score = 1
    # When nc_51 = n_entities/2 → score = 0.5
    if n_entities == 1:
        return 0.0
    
    score = (n_entities - nc_51) / (n_entities - 1)
    
    return score


def HHI_coefficient(data: List[float]) -> float:
    """
    Compute the Herfindahl-Hirschman Index (HHI) of a dataset.

    HHI is a decentralization measure used to quantify market concentration.

    Args:
        data: List of numeric data points.

    Returns:
        The HHI of the input dataset.

    Examples:
        >>> data = [100.0, 200.0, 300.0, 400.0, 500.0]
        >>> HHI_coefficient(data)  # Output: 0.2
    """
    if not data or len(data) == 0:
        return 0.0
    
    # Total resource
    total = sum(data)
    
    if total == 0:
        return 0.0
    
    # Percentage of each data point
    percentages = [value / total for value in data]

    # HHI coefficient
    HHI = sum(percentage ** 2 for percentage in percentages)
    
    return HHI


def lerp_vector(a: List[float], b: List[float], l: float) -> List[float]:
    """
    Perform linear interpolation between two vectors a and b with interpolation factor l.
    Linear interpolation (lerp) computes a point between vectors a and b based on scalar l in [0,1].
    When l=0, result equals a; when l=1, result equals b.

    Parameters:
        a: Starting vector.
        b: Ending vector.
        l: Interpolation factor in range [0, 1].

    Returns:
        A new list representing the linear interpolation result between a and b with factor l.
            
    Examples:
        >>> a = [1.0, 2.0, 3.0]
        >>> b = [4.0, 5.0, 6.0]
        >>> l = 0.5
        >>> lerp_vector(a, b, l)  # Output: [2.5, 3.5, 4.5]
    """
    if len(a) != len(b):
        raise ValueError("Vectors a and b must have the same length")
    
    interpolated_vector = []
    for i in range(len(a)):
        interpolated_vector.append((1 - l) * a[i] + l * b[i])
    
    return interpolated_vector


def lerp(a: float, b: float, l: float) -> float:
    """
    Perform linear interpolation between two values a and b with interpolation factor l.

    Linear interpolation (lerp) computes a point between a and b based on scalar l in [0,1].
    When l=0, result equals a; when l=1, result equals b.

    Parameters:
        a: Starting value.
        b: Ending value.
        l: Interpolation factor in range [0, 1].

    Returns:
        Result of linear interpolation between a and b with factor l.
        
    Examples:
        >>> a = 1.0
        >>> b = 4.0
        >>> l = 0.5
        >>> lerp(a, b, l)  # Output: 2.5
    """
    return (1 - l) * a + l * b


def weighted_consensus(peers: List[float]) -> int:
    """
    Determine weighted consensus among a group of peer nodes based on their probabilities.
    This function computes weighted consensus among peer nodes, where each node is represented
    by a probability value. Nodes with higher probabilities have greater influence on consensus.

    Parameters:
        peers: A list containing the staked token amounts of each node.

    Returns:
        Index of the node selected as consensus result, based on weighted probability.

    Examples:
    >>> peers = [0.2, 0.3, 0.5]
    >>> weighted_consensus(peers)  # Result: index of selected node
    """
    if not peers or len(peers) == 0:
        raise ValueError("Peers list cannot be empty")
    
    total = sum(peers)
    if total == 0:
        return random.randint(0, len(peers) - 1)
    
    cumulative_probabilities = np.cumsum(np.array(peers) / total)
    random_number = random.random()
    
    # Find first index with cumulative probability >= random number
    for i, cum_prob in enumerate(cumulative_probabilities):
        if cum_prob >= random_number:
            return i
    
    # Fallback (shouldn't happen with proper cumulative probabilities)
    return len(peers) - 1


def opposite_weighted_consensus(peers: List[float]) -> int:
    """
    Determine opposite weighted consensus among peer group based on 
    their probabilities.
    
    This function computes opposite weighted consensus among peer group, 
    where each peer is represented by a probability value. 
    Peers with lower probabilities (opposite influence) have 
    greater influence on consensus result.
    
    Args:
        peers: List containing staked token amounts of each peer.
        
    Returns:
        Index of peer selected as opposite consensus, based on 
        opposite weighted probability.
        
    Examples:
        >>> peers = [0.2, 0.3, 0.5]
        >>> opposite_weighted_consensus(peers)  # Output: index of the selected peer
    """
    if not peers or len(peers) == 0:
        raise ValueError("Peers list cannot be empty")
    
    max_peer = max(peers)
    opposite_peers = [abs(max_peer - peer) for peer in peers]
    
    total = sum(opposite_peers)
    if total == 0:
        return random.randint(0, len(peers) - 1)
    
    cumulative_probabilities = np.cumsum(np.array(opposite_peers) / total)
    random_number = random.random()
    
    # Find first index with cumulative probability >= random number
    for i, cum_prob in enumerate(cumulative_probabilities):
        if cum_prob >= random_number:
            return i
    
    # Fallback (shouldn't happen with proper cumulative probabilities)
    return len(peers) - 1


def gini_stabilized_consensus(peers: List[float], t: float) -> int:
    """
    Determine Gini stabilized consensus among peer group based on 
    their probabilities, using linear interpolation between two consensus methods.
    
    It combines two consensus methods: weighted consensus and 
    opposite weighted consensus, using linear interpolation based on parameter t.
    
    Args:
        peers: List containing staked token amounts of each peer.
        t: Interpolation parameter from 0 to 1, determining weight 
           of weighted consensus (when t=0) and opposite weighted consensus (when t=1).
           
    Returns:
        Index of peer selected as dynamic consensus.
        
    Examples:
        >>> peers = [0.2, 0.3, 0.5]
        >>> t = 0.5
        >>> gini_stabilized_consensus(peers, t)  # Output: index of the selected peer
    """
    if t == -1:
        raise ValueError("Cannot launch GiniStabilized with t = -1")
    
    if not peers or len(peers) == 0:
        raise ValueError("Peers list cannot be empty")
    
    total = sum(peers)
    if total == 0:
        return random.randint(0, len(peers) - 1)
    
    # Calculate weighted probabilities
    weighted = np.cumsum(np.array(peers) / total)
    
    # Calculate opposite weighted probabilities
    max_peer = max(peers)
    processed_peers = [abs(max_peer - peer) for peer in peers]
    total_processed = sum(processed_peers)
    
    if total_processed == 0:
        opposite_weighted = np.cumsum(np.ones(len(peers)) / len(peers))
    else:
        opposite_weighted = np.cumsum(np.array(processed_peers) / total_processed)
    
    # Linear interpolation between the two methods
    cumulative_probabilities = lerp_vector(opposite_weighted.tolist(), weighted.tolist(), t)
    
    random_number = random.random()
    
    # Find first index with cumulative probability >= random number
    for i, cum_prob in enumerate(cumulative_probabilities):
        if cum_prob >= random_number:
            return i
    
    # Fallback (shouldn't happen with proper cumulative probabilities)
    return len(peers) - 1

def desw_consensus(peers: List[float], pmin: float = 0, pmax:float = 1) -> int:
    """
    Determine DESW (Dynamic Exponential Stake Weighting) consensus among peer group.
    Weight combines Power-Law (stake^p, where p = 1 - Gini) to create dynamic balance.
    
    Args:
        peers: List containing staked token amounts of each peer.
    
    Returns:
        Index of peer selected as DESW consensus.
    
    Examples:
        >>> peers = [0.2, 0.3, 0.5]
        >>> desw_consensus(peers)  # Output: index of the selected peer
    """
    if not peers or len(peers) == 0:
        raise ValueError("Peers list cannot be empty")
    
    
    total = sum(peers)
    if total == 0:
        return random.randint(0, len(peers) - 1)
    
    # Calculate Gini coefficient using existing function
    gini_stake = gini(peers)
    
    # Calculate dynamic p: p = 1 - Gini, bounded in [0.2, 0.8]
    p_dynamic = max(pmin, min(pmax, 1 - gini_stake))
    
    # Calculate Power-Law weights
    power_weights = np.array(peers) ** p_dynamic
    
    
    # Normalize to probabilities
    total_weight = sum(power_weights)
    probabilities = power_weights / total_weight
    cumulative_probabilities = np.cumsum(probabilities)
    
    # Select validator randomly
    random_number = random.random()
    for i, cum_prob in enumerate(cumulative_probabilities):
        if cum_prob >= random_number:
            return i
    
    # Fallback
    return len(peers) - 1

def srsw_weighted_consensus(peers: List[float]) -> int:
    """
    Determine SRSW weighted consensus among a group of peer nodes.
    
    This function applies square root to stake before calculating probabilities,
    helping reduce influence of validators with very high stake and decrease
    the system's Gini coefficient.
    
    Args:
        peers: List containing staked token amounts of each node.
        
    Returns:
        Index of node selected as consensus result, 
        based on square root weighted probability.
        
    Examples:
        >>> peers = [1, 10, 100]
        >>> srsw_weighted_consensus(peers)  # Reduce influence of peer with stake = 100
    """
    if not peers or len(peers) == 0:
        raise ValueError("Peers list cannot be empty")
    
    # Calculate square root of stakes
    srsw_stakes = [np.sqrt(stake) for stake in peers]
    
    total = sum(srsw_stakes)
    if total == 0:
        return random.randint(0, len(peers) - 1)
    
    cumulative_probabilities = np.cumsum(np.array(srsw_stakes) / total)
    random_number = random.random()
    
    # Find first index with cumulative probability >= random number
    for i, cum_prob in enumerate(cumulative_probabilities):
        if cum_prob >= random_number:
            return i
    
    # Fallback (shouldn't happen with proper cumulative probabilities)
    return len(peers) - 1


def log_weighted_consensus(peers: List[float]) -> int:
    """
    Determine logarithmic weighted consensus among a group of peer nodes.
    
    This function applies natural logarithm to stake before calculating probabilities,
    helping reduce influence of validators with very high stake and decrease
    the system's Gini coefficient.
    
    Args:
        peers: List containing staked token amounts of each node.
        
    Returns:
        Index of node selected as consensus result, 
        based on logarithmic weighted probability.
        
    Examples:
        >>> peers = [1, 10, 100]
        >>> log_weighted_consensus(peers)  # Reduce influence of peer with stake = 100
    """
    if not peers or len(peers) == 0:
        raise ValueError("Peers list cannot be empty")
    
    # Avoid log(0) by adding small value
    epsilon = 1e-8
    log_stakes = [np.sqrt(max(stake, epsilon)) for stake in peers]
    
    total = sum(log_stakes)
    if total == 0:
        return random.randint(0, len(peers) - 1)
    
    cumulative_probabilities = np.cumsum(np.array(log_stakes) / total)
    random_number = random.random()
    
    # Find first index with cumulative probability >= random number
    for i, cum_prob in enumerate(cumulative_probabilities):
        if cum_prob >= random_number:
            return i
    
    # Fallback (shouldn't happen with proper cumulative probabilities)
    return len(peers) - 1

def random_consensus(peers: List[float]) -> int:
    """
    Determine random consensus among peer group.
    
    This function selects a random agent from the group based on uniform distribution.
    
    Args:
        peers: List containing staked token amounts of each peer.
        
    Returns:
        Index of randomly selected agent for consensus.
        
    Examples:
        >>> peers = [0.2, 0.3, 0.5]
        >>> random_consensus(peers)  # Output: index of the randomly selected agent
    """
    if not peers or len(peers) == 0:
        raise ValueError("Peers list cannot be empty")
    
    return random.randint(0, len(peers) - 1)


def constant_reward(total_reward: float, n_epochs: int) -> float:
    """Calculate fixed reward per epoch"""
    return total_reward / n_epochs


def dynamic_reward(total_reward: float, n_epochs: int, current_epoch: int) -> float:
    """Calculate dynamic reward based on current epoch"""
    return (total_reward / n_epochs) + ((current_epoch / n_epochs) * total_reward)


def generate_peers(n_peers: int, initial_volume: float, distribution_type: Distribution, initial_gini: float = -1.0) -> List[float]:
    """
    Generate initial peer stakes based on distribution type
    
    Args:
        n_peers: Number of peers to create
        initial_volume: Total initial stake volume
        distribution_type: Distribution type (Uniform, Gini, Random)
        initial_gini: Initial Gini coefficient (for Gini distribution)
        
    Returns:
        List of initial stakes for each peer
    """
    if distribution_type == Distribution.UNIFORM:
        return generate_vector_uniform(n_peers, initial_volume)
    elif distribution_type == Distribution.GINI:
        if initial_gini == -1.0:
            print("In order to generate peers with a Gini distribution, call 'generate_peers' with the 'initial_gini' " + 
                  "parameter positive and less or equal to 1. Automatically setting 'initial_gini' equal to 0.3")
            initial_gini = 0.3
        return generate_vector_with_gini(n_peers, initial_volume, initial_gini)
    elif distribution_type == Distribution.RANDOM:
        return generate_vector_random(n_peers, initial_volume)
    else:
        raise ValueError(f"Unknown distribution type: {distribution_type}")


def consensus(pos: PoS, stakes: List[float], t: float = -1.0, pmin: float = 0, pmax:float = 1) -> int:
    """
    Execute consensus algorithm based on PoS type
    
    Args:
        pos: Proof of Stake type
        stakes: List of peer stakes
        t: Parameter for GiniStabilized consensus
        
    Returns:
        Index of selected validator
    """
    if pos == PoS.WEIGHTED:
        return weighted_consensus(stakes)
    elif pos == PoS.OPPOSITE_WEIGHTED:
        return opposite_weighted_consensus(stakes)
    elif pos == PoS.GINI_STABILIZED:
        return gini_stabilized_consensus(stakes, t)
    elif pos == PoS.LOG_WEIGHTED:
        return log_weighted_consensus(stakes)
    elif pos == PoS.DESW:
        return desw_consensus(stakes, pmin, pmax)
    elif pos == PoS.SRSW_WEIGHTED:
        return srsw_weighted_consensus(stakes)
    elif pos == PoS.RANDOM:
        return random_consensus(stakes)
    else:
        raise ValueError(f"Unknown PoS type: {pos}")


def generate_vector_with_gini(n_peers: int, initial_volume: float, gini_coeff: float) -> List[float]:
    """
    Generate a vector with specific Gini coefficient
    
    Args:
        n_peers: Number of peers
        initial_volume: Total volume to distribute
        gini_coeff: Target Gini coefficient
        
    Returns:
        List of stakes with specified Gini coefficient
    """
    def lorenz_curve(x1: float, y1: float, x2: float, y2: float):
        """Create Lorenz curve function"""
        m = (y2 - y1) / (x2 - x1) if x2 != x1 else 0
        return lambda x: m * x
    
    max_r = (n_peers - 1) / 2
    r = gini_coeff * max_r
    prop = ((n_peers - 1) / n_peers) * ((max_r - r) / max_r)
    lc = lorenz_curve(0, 0, (n_peers - 1) / n_peers, prop)
    
    # Create cumulative distribution
    q = [lc(i / n_peers) for i in range(1, n_peers)]
    q.append(1.0)
    
    # Convert to actual stakes
    cumulative_sum = [i * initial_volume for i in q]
    stakes = [cumulative_sum[0]]
    
    for i in range(1, n_peers):
        stakes.append(cumulative_sum[i] - cumulative_sum[i - 1])
    
    return stakes


def generate_vector_uniform(n: int, volume: float) -> List[float]:
    """
    Generate uniform distribution of stakes
    
    Args:
        n: Number of peers
        volume: Total volume to distribute
        
    Returns:
        List of equal stakes
    """
    return [volume / n for _ in range(n)]


def generate_vector_random(n: int, volume: float) -> List[float]:
    """
    Generate random distribution of stakes
    
    Args:
        n: Number of peers
        volume: Total volume to distribute
        
    Returns:
        List of randomly distributed stakes
    """
    if n <= 0:
        return []
    
    if n == 1:
        return [volume]
    
    # Create n-1 random cut points in range [0, volume]
    cut_points = sorted([random.uniform(0, volume) for _ in range(n - 1)])
    
    # Calculate stake for each peer based on cut points
    stakes = []
    
    # First peer: from 0 to first cut_point
    stakes.append(cut_points[0])
    
    # Middle peers: from previous cut_point to next cut_point
    for i in range(1, n - 1):
        stakes.append(cut_points[i] - cut_points[i - 1])
    
    # Last peer: from last cut_point to volume
    stakes.append(volume - cut_points[-1])
    
    # Ensure no negative stakes (in very rare cases)
    stakes = [max(0.0, stake) for stake in stakes]
    
    # Normalize to ensure total equals volume exactly
    total = sum(stakes)
    if total > 0:
        stakes = [stake * volume / total for stake in stakes]
    else:
        # Emergency case: return uniform distribution
        return generate_vector_uniform(n, volume)
    
    return stakes


def compute_smooth_parameter(current_gini: float, target_gini: float, r: float) -> float:
    """
    Calculate smoothing parameter for Gini stabilization
    
    Args:
        current_gini: Current Gini coefficient
        target_gini: Target Gini coefficient
        r: Smoothing coefficient
        
    Returns:
        Smoothing parameter value
    """
    diff = abs(current_gini - target_gini)
    diff = diff * (1 / r)
    
    res = (diff + 0j) ** (1 / 7.0)
    res = res.real * (1 if current_gini >= target_gini else -1)
    
    res = (res / 2) + 0.5
    
    if res > 1.0:
        res = 1.0
    if res < 0.0:
        res = 0.0
    
    return 1 - res


def compute_smooth_parameter2(current_gini: float, target_gini: float, r: float) -> float:
    """
    Alternative smoothing parameter calculation
    
    Args:
        current_gini: Current Gini coefficient
        target_gini: Target Gini coefficient
        r: Smoothing coefficient
        
    Returns:
        Smoothing parameter value
    """
    denom = target_gini + r
    if denom == 0:
        return 0.5
    
    res = 0.5 - ((current_gini / denom) - (target_gini / denom)) * (1 / (1 - target_gini / denom))
    
    if res > 1.0:
        res = 1.0
    if res < 0.0:
        res = 0.0
    
    return res


def compute_smooth_parameter3(current_gini: float, target_gini: float) -> float:
    """
    Simple smoothing parameter computation
    
    Args:
        current_gini: Current Gini coefficient
        target_gini: Target Gini coefficient
        
    Returns:
        Smoothing parameter value
    """
    if current_gini > target_gini:
        return 0.0
    elif current_gini < target_gini:
        return 1.5
    else:
        return 0.75  # Default when equal


def d(g: float, θ: float) -> float:
    """
    Compute function d for GiniStabilized consensus
    
    Args:
        g: Current Gini coefficient
        θ: Target Gini coefficient (theta)
        
    Returns:
        The value of d
    """
    if g > θ:
        return 0.5
    else:
        return 1.5


def try_to_join(stakes: List[float], corrupted: List[int], p: float, 
                join_amount: NewEntry, percentage_corrupted: float) -> None:
    """
    Try to add new peer to the network
    
    Args:
        stakes: Current stake list (modified in place)
        corrupted: List of corrupted peer indices (modified in place)
        p: Join probability
        join_amount: Amount type for new peer
        percentage_corrupted: Percentage of corrupted peers
    """
    if random.random() <= p:
        # Add new peer
        if join_amount == NewEntry.NEW_AVERAGE:
            new_stake = sum(stakes) / len(stakes) if stakes else 0
        elif join_amount == NewEntry.NEW_MAX:
            new_stake = max(stakes) if stakes else 0
        elif join_amount == NewEntry.NEW_MIN:
            new_stake = min(stakes) if stakes else 0
        elif join_amount == NewEntry.NEW_RANDOM:
            new_stake = stakes[random.randint(0, len(stakes) - 1)] if stakes else 0
        else:
            new_stake = 0
        
        stakes.append(new_stake)
        
        # Check if new peer is corrupted
        if random.random() <= percentage_corrupted:
            corrupted.append(len(stakes) - 1)
        
        # Recursive call to try adding another peer
        try_to_join(stakes, corrupted, p, join_amount, percentage_corrupted)


def try_to_leave(stakes: List[float], p: float) -> None:
    """
    Try to remove peer from network
    
    Args:
        stakes: Current stake list (modified in place)
        p: Leave probability
    """
    if stakes and random.random() <= p:
        # Remove random peer
        index_to_remove = random.randint(0, len(stakes) - 1)
        stakes.pop(index_to_remove)