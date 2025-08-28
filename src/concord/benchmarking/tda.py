
# Code to compute persistent homology of data
import numpy as np
from typing import Optional

from .. import logger

def compute_persistent_homology(
        adata, 
        key='X_pca', 
        homology_dimensions=[0,1,2],
        *,
        max_points: Optional[int] = None,
        random_state: Optional[int] = None,
):
    """
    Computes persistent homology using Vietoris-Rips complex.

    Args:
        adata : anndata.AnnData
            The AnnData object containing the data.
        key : str, optional
            The key in `adata.obsm` specifying the embedding to use. Default is 'X_pca'.
        homology_dimensions : list, optional
            List of homology dimensions to compute. Default is [0, 1, 2].

    Returns:
        np.ndarray
            Persistence diagrams representing homology classes across filtration values.
    """ 
    from gtda.homology import VietorisRipsPersistence
    X = adata.obsm[key]
    if max_points is not None and X.shape[0] > max_points:
        rng = np.random.default_rng(random_state)
        idx = rng.choice(X.shape[0], size=max_points, replace=False)
        X = X[idx]
    
    logger.info(f"Computing persistent homology for {X.shape[0]} points in {X.shape[1]} dimensions...")
    VR = VietorisRipsPersistence(homology_dimensions=homology_dimensions)  # Parameter explained in the text
    diagrams = VR.fit_transform(X[None, :, :])
    return diagrams



def compute_betti_median_or_mode(betti_values, statistic="median"):
    """
    Computes the median or mode of Betti numbers.

    Args:
        betti_values : np.ndarray
            Array of Betti numbers across filtration values.
        statistic : str, optional
            Statistic to compute ('median' or 'mode'). Default is 'median'.

    Returns:
        float
            The computed median or mode of the Betti numbers.

    Raises:
        ValueError
            If the provided statistic is not 'median' or 'mode'.
    """
    from scipy.stats import mode
    if statistic == "median":
        return np.median(betti_values)
    elif statistic == "mode":
        return mode(betti_values)[0]
    else:
        raise ValueError("Statistic must be 'median' or 'mode'.")


def compute_betti_entropy(betti_values):
    """
    Computes the entropy of the Betti curve.

    Args:
        betti_values : np.ndarray
            Array of Betti numbers across filtration values.

    Returns:
        float
            The entropy of the Betti curve.
    """
    from scipy.stats import entropy
    total = np.sum(betti_values)
    if total == 0:
        return 0.0  # Entropy is zero if the Betti curve sums to zero
    # Normalize values to get a probability distribution
    prob_dist = betti_values / total
    return entropy(prob_dist)


# ──────────────────────────────────────────────────────────────
def betti_stability(betti_values: np.ndarray) -> float:
    """
    Return a stability score in [0,1] from a Betti curve vector.

    Stability := 1 / (1 + variance), where variance is np.var(betti_values).
    Caps large variances; 1 for var=0 (constants), approaches 0 for high var.
    """
    var = np.var(betti_values)
    return 1.0 / (1.0 + var)


def interpolate_betti_curve(betti_values, original_sampling, common_sampling):
    """
    Interpolates Betti curve onto a common filtration grid.

    Args:
        betti_values : np.ndarray
            Array of Betti numbers.
        original_sampling : np.ndarray
            The original filtration values associated with the Betti numbers.
        common_sampling : np.ndarray
            The target filtration values for interpolation.

    Returns:
        np.ndarray
            Interpolated Betti curve.
    """
    from scipy.interpolate import interp1d
    interp_func = interp1d(
        original_sampling, betti_values, kind='previous',
        bounds_error=False, fill_value=0.0
    )
    interpolated_values = interp_func(common_sampling)
    return interpolated_values


def compute_betti_statistics(diagram, expected_betti_numbers, n_bins=100):
    """
    Computes Betti statistics given a persistence diagram.

    Args:
        diagram : np.ndarray
            Persistence diagram from Giotto-TDA.
        expected_betti_numbers : np.ndarray
            Expected Betti numbers for different homology dimensions.
        n_bins : int, optional
            Number of bins for the Betti curve computation. Default is 100.

    Returns:
        dict
            A dictionary containing:
            - `'betti_stats'`: Dictionary of Betti statistics.
            - `'observed_betti_numbers'`: Observed Betti numbers.
            - `'expected_betti_numbers'`: Expected Betti numbers.
            - `'l1_distance'`: L1 distance between observed and expected Betti numbers.
            - `'l2_distance'`: L2 distance between observed and expected Betti numbers.
            - `'total_relative_error'`: Total relative error.
    """
    from gtda.diagrams import BettiCurve

    # Collect Sampling Points and Determine Global Filtration Range
    samplings = {}

    # Initialize BettiCurve transformer and compute initial Betti curves
    betti_curve_transformer = BettiCurve(n_bins=n_bins)
    betti_curves = betti_curve_transformer.fit_transform(diagram)
    samplings_raw = betti_curve_transformer.samplings_

    min_filtration = np.inf
    max_filtration = -np.inf

    for dim, sampling in samplings_raw.items():
        samplings[dim] = sampling
        min_filtration = min(min_filtration, sampling.min())
        max_filtration = max(max_filtration, sampling.max())

    # Ensure min_filtration is non-negative
    min_filtration = max(min_filtration, 0.0)

    # Create Common Filtration Grid
    common_sampling = np.linspace(min_filtration, max_filtration, n_bins)

    # Interpolate Betti Curves onto the Common Grid
    interpolated_betti_curves = {}

    for dim in samplings.keys():
        # Extract Betti values and original sampling for the dimension
        betti_values = betti_curves[0][dim, :]
        original_sampling = samplings[dim]
        # Interpolate
        interpolated_values = interpolate_betti_curve(
            betti_values, original_sampling, common_sampling
        )
        interpolated_betti_curves[dim] = interpolated_values

    # Compute Betti Curve Statistics
    homology_dimensions = sorted(interpolated_betti_curves.keys())
    betti_stats = {}
    observed_betti_numbers = []

    for dim in homology_dimensions:
        betti_values = interpolated_betti_curves[dim]
        betti_variance = np.var(betti_values)
        betti_mean = np.mean(betti_values)
        betti_median = compute_betti_median_or_mode(betti_values, statistic="median")
        betti_mode = compute_betti_median_or_mode(betti_values, statistic="mode")
        betti_entropy = compute_betti_entropy(betti_values)
        betti_stab     = betti_stability(betti_values)

        betti_stats[dim] = {
            'variance': betti_variance,
            'mean': betti_mean,
            'median': betti_median,
            'mode': betti_mode,
            'entropy': betti_entropy,
            'stability': betti_stab
        }

        # Use mode as the observed Betti number for distance calculations
        observed_betti_numbers.append(betti_mode)

    observed_betti_numbers = np.array(observed_betti_numbers).astype(int)
    expected_betti_numbers = np.array(expected_betti_numbers).astype(int)

    # Step 6: Compute Distance Metrics
    # Compute L1 distance
    l1_distance = np.sum(np.abs(observed_betti_numbers - expected_betti_numbers))

    # Compute L2 distance
    l2_distance = np.sqrt(np.sum((observed_betti_numbers - expected_betti_numbers) ** 2))

    # Compute Relative Error (handle division by zero)
    with np.errstate(divide='ignore', invalid='ignore'):
        relative_error = np.abs((observed_betti_numbers - expected_betti_numbers) / expected_betti_numbers)
        # Replace infinities and NaNs resulting from division by zero with zeros
        relative_error = np.nan_to_num(relative_error, nan=0.0, posinf=0.0, neginf=0.0)
        total_relative_error = np.sum(relative_error)

    # Compile statistics into a dictionary
    stats_dict = {
        'betti_stats': betti_stats,
        'observed_betti_numbers': observed_betti_numbers,
        'expected_betti_numbers': expected_betti_numbers,
        'l1_distance': l1_distance,
        'l2_distance': l2_distance,
        'total_relative_error': total_relative_error
    }

    return stats_dict




def summarize_betti_statistics(betti_stats):
    """
    Summarizes Betti statistics into pandas DataFrames.

    Args:
        betti_stats : dict
            Dictionary containing Betti statistics for different methods.

    Returns:
        tuple
            - `betti_stats_pivot`: DataFrame of Betti statistics.
            - `distance_metrics_df`: DataFrame of distance metrics.
    """
    import pandas as pd
    # Prepare data for Betti Curve Statistics DataFrame
    methods = []
    dims = []
    stats = []
    values = []

    for method, data in betti_stats.items():
        for dim, stats_dict in data['betti_stats'].items():
            for stat_name, value in stats_dict.items():
                methods.append(method)
                dims.append(f"Dim {dim}")
                stats.append(stat_name.capitalize())
                values.append(value)

    # Create DataFrame for Betti Curve Statistics
    betti_stats_df = pd.DataFrame({
        'Method': methods,
        'Dimension': dims,
        'Statistic': stats,
        'Value': values
    })

    # Pivot the DataFrame to get the desired format
    betti_stats_pivot = betti_stats_df.pivot_table(
        index='Method',
        columns=['Dimension', 'Statistic'],
        values='Value'
    )

    # Prepare data for Distance Metrics DataFrame
    distance_metrics = []

    for method, data in betti_stats.items():
        entry = {
            'Method': method,
            'L1 Distance': data['l1_distance'],
            'L2 Distance': data['l2_distance'],
            'Total Relative Error': data['total_relative_error']
        }
        # Convert observed and expected Betti numbers to strings for display
        observed_betti_numbers_str = ', '.join(map(str, data['observed_betti_numbers']))
        expected_betti_numbers_str = ', '.join(map(str, data['expected_betti_numbers']))
        entry['Observed Betti Numbers'] = observed_betti_numbers_str
        entry['Expected Betti Numbers'] = expected_betti_numbers_str
        distance_metrics.append(entry)

    # Create Distance Metrics DataFrame
    distance_metrics_df = pd.DataFrame(distance_metrics)
    distance_metrics_df.set_index('Method', inplace=True)

    return betti_stats_pivot, distance_metrics_df






