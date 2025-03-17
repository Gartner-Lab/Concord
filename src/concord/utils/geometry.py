# Evaluate the geometric aspects of the methods
import numpy as np

def compute_reconstruction_error(adata, layer1, layer2, metric='mse'):
    """
    Computes pairwise distances for specified embeddings in an AnnData object.

    Args:
        adata (AnnData): AnnData object containing embeddings.
        keys (list): List of keys in adata.obsm or adata.layers for distance computation.
        metric (str, optional): Distance metric to use, e.g., 'cosine' or 'euclidean'. Defaults to 'cosine'.

    Returns:
        dict: Dictionary where keys are embedding names and values are pairwise distances.
    """
    import numpy as np

    # Extract the two layers as numpy arrays
    matrix1 = adata.layers[layer1]
    matrix2 = adata.layers[layer2]

    # Ensure both matrices have the same shape
    if matrix1.shape != matrix2.shape:
        raise ValueError("The matrices must have the same shape.")

    # Compute the error based on the chosen metric
    if metric == 'mse':
        error = np.mean((matrix1 - matrix2) ** 2)
    elif metric == 'mae':
        error = np.mean(np.abs(matrix1 - matrix2))
    else:
        raise ValueError("Invalid metric. Choose 'mse' or 'mae'.")

    return error




def pairwise_distance(adata, keys, metric="cosine"):
    from scipy.spatial.distance import pdist
    distance_result = {}
    for key in keys:
        if key in adata.obsm:
            mtx = adata.obsm[key]
        elif key in adata.layers:
            mtx = adata.layers[key]
        else:
            raise ValueError(f"Key {key} not found in adata.obsm or adata.layers")
        distance_result[key] = pdist(mtx, metric=metric)

    return distance_result
    

    

def local_vs_distal_corr(X_high, X_low, local_percentile=25, distal_percentile=75, method='pearsonr'):
    """
    Computes correlation between local and distal pairwise distances.

    Args:
        X_high (numpy.ndarray): High-dimensional data matrix.
        X_low (numpy.ndarray): Low-dimensional embedding matrix.
        local_percentile (int, optional): Percentile threshold for local distances. Defaults to 25.
        distal_percentile (int, optional): Percentile threshold for distal distances. Defaults to 75.
        method (str, optional): Correlation method; 'pearsonr', 'spearmanr', or 'kendalltau'. Defaults to 'pearsonr'.

    Returns:
        float: Correlation for local distances.
        float: Correlation for distal distances.
    """

    from scipy.stats import spearmanr, pearsonr, kendalltau
    from scipy.spatial.distance import pdist
    # Step 1: Compute pairwise distances
    dist_high = pdist(X_high)  # Original space distances
    dist_low = pdist(X_low)    # Latent space distances

    # Step 2: Define a threshold based on the specified percentile
    distance_threshold_local = np.percentile(dist_high, local_percentile)
    distance_threshold_distal = np.percentile(dist_high, distal_percentile)

    # Step 3: Separate distances into local and distal categories
    local_mask = dist_high <= distance_threshold_local
    distal_mask = dist_high > distance_threshold_distal

    # Step 4: Compute Spearman correlations for local and distal distances
    if method == 'spearmanr':
        local_corr, _ = spearmanr(dist_high[local_mask], dist_low[local_mask])
        distal_corr, _ = spearmanr(dist_high[distal_mask], dist_low[distal_mask])
    elif method == 'pearsonr':
        local_corr, _ = pearsonr(dist_high[local_mask], dist_low[local_mask])
        distal_corr, _ = pearsonr(dist_high[distal_mask], dist_low[distal_mask])
    elif method == 'kendalltau':
        local_corr, _ = kendalltau(dist_high[local_mask], dist_low[local_mask])
        distal_corr, _ = kendalltau(dist_high[distal_mask], dist_low[distal_mask])
    else:
        raise ValueError(f"Method {method} not recognized. Must be one of 'spearmanr', 'pearsonr', or 'kendalltau'.")

    return local_corr, distal_corr


def compute_state_batch_distance_ratio(adata, basis='X_latent', batch_key='batch', state_key='cluster', metric='cosine'):
    """
    Computes the Batch-to-State Distance Ratio using centroids to evaluate batch correction.

    Args:
        adata (AnnData): AnnData object containing latent embeddings.
        basis (str, optional): Key for latent embeddings in adata.obsm. Defaults to 'X_latent'.
        batch_key (str, optional): Key for batch labels in adata.obs. Defaults to 'batch'.
        state_key (str, optional): Key for cell state labels in adata.obs. Defaults to 'cluster'.
        metric (str, optional): Distance metric to use, e.g., 'cosine' or 'euclidean'. Defaults to 'cosine'.

    Returns:
        float: Ratio of average batch distance to average state distance.
    """
    import numpy as np
    from scipy.spatial.distance import pdist
    # Get the latent embeddings
    latent_embeddings = adata.obsm[basis]
    
    # Calculate centroids for each batch within each state
    batch_centroids = {}
    for state in adata.obs[state_key].unique():
        batch_centroids[state] = {}
        for batch in adata.obs[batch_key].unique():
            mask = (adata.obs[state_key] == state) & (adata.obs[batch_key] == batch)
            if np.any(mask):
                batch_centroids[state][batch] = latent_embeddings[mask].mean(axis=0)
    
    # Calculate average distance between centroids of batches within the same state
    batch_distances = []
    for state, centroids in batch_centroids.items():
        centroid_matrix = np.array(list(centroids.values()))
        if centroid_matrix.shape[0] > 1:  # Only if there is more than one batch in the state
            dist_matrix = pdist(centroid_matrix, metric=metric)
            batch_distances.extend(dist_matrix)
    
    avg_batch_distance = np.mean(batch_distances)
    
    # Calculate centroids for each state
    state_centroids = {}
    for state in adata.obs[state_key].unique():
        mask = adata.obs[state_key] == state
        state_centroids[state] = latent_embeddings[mask].mean(axis=0)
    
    # Calculate average distance between centroids of states
    state_centroid_matrix = np.array(list(state_centroids.values()))
    state_distances = pdist(state_centroid_matrix, metric=metric)
    avg_state_distance = np.mean(state_distances)
    
    state_to_batch_ratio = avg_state_distance / avg_batch_distance
    
    return state_to_batch_ratio




def compute_trustworthiness(adata, embedding_keys, groundtruth, metric='euclidean', n_neighbors=10):
    """
    Evaluates trustworthiness of embeddings in an AnnData object.

    Args:
        adata (AnnData): AnnData object containing embeddings in adata.obsm.
        embedding_keys (list): List of keys in adata.obsm to evaluate (e.g., ['X_umap', 'X_tsne']).
        groundtruth (str or numpy.ndarray): Key in adata.obsm or adata.layers for ground truth data, or a precomputed matrix.
        metric (str, optional): Distance metric for trustworthiness calculation, e.g., 'euclidean' or 'cosine'. Defaults to 'euclidean'.
        n_neighbors (int or list, optional): Neighborhood sizes for trustworthiness evaluation. Defaults to 10.

    Returns:
        pandas.DataFrame: Trustworthiness scores for each embedding at each neighborhood size.
        pandas.DataFrame: Summary statistics with average trustworthiness and decay rate.
    """
    import numpy as np
    import pandas as pd
    from sklearn.manifold import trustworthiness
    from scipy.stats import linregress
    # Determine ground truth matrix based on type of groundtruth argument
    if isinstance(groundtruth, str):
        if groundtruth in adata.obsm:
            X = adata.obsm[groundtruth]
        elif groundtruth in adata.layers:
            X = adata.layers[groundtruth]
        else:
            raise ValueError("groundtruth key not found in adata.obsm or adata.layers.")
    elif isinstance(groundtruth, (np.ndarray, np.matrix)):
        X = groundtruth
    else:
        raise ValueError("Invalid groundtruth format. Must be a key in adata or an array/matrix.")

    # Ensure n_neighbors is a list for consistency
    if isinstance(n_neighbors, int):
        n_neighbors = [n_neighbors]

    # Initialize lists to gather data for DataFrames
    trustworthiness_data = []
    summary_stats_data = []

    # Calculate trustworthiness for each embedding and each n_neighbors value
    for key in embedding_keys:
        embedding = adata.obsm[key]
        scores = {}
        
        for k in n_neighbors:
            score = trustworthiness(X, embedding, n_neighbors=k, metric=metric)
            trustworthiness_data.append({'Embedding': key, 'n_neighbors': k, 'Trustworthiness': score})
            scores[k] = score

        avg_trustworthiness = np.mean(list(scores.values()))
        k_values = np.array(n_neighbors)
        t_values = np.array(list(scores.values()))

        if len(n_neighbors) > 1:
            slope, intercept, _, _, _ = linregress(k_values, t_values)  # Decay rate
        else:
            slope = None
            
        summary_stats_data.append({
            'Embedding': key,
            'Average Trustworthiness': avg_trustworthiness,
            'Trustworthiness Decay (100N)': slope * 100
        })

    # Convert collected data to DataFrames
    trustworthiness_df = pd.DataFrame(trustworthiness_data)

    # Summary stats DataFrame, if applicable
    summary_stats_df = pd.DataFrame(summary_stats_data) if summary_stats_data else None
    summary_stats_df.set_index('Embedding', inplace=True)
    return trustworthiness_df, summary_stats_df


def compute_centroid_distance(adata, basis, state_key, dist_metric='cosine'):
    """
    Computes the pairwise distances between centroids of different states.

    Args:
        adata (AnnData): AnnData object containing embeddings.
        basis (str): Key for embedding data in adata.obsm.
        state_key (str): Column in adata.obs defining states or clusters.
        dist_metric (str, optional): Distance metric to use, e.g., 'cosine' or 'euclidean'. Defaults to 'cosine'.

    Returns:
        numpy.ndarray: Pairwise distances between state centroids.
    """
    import pandas as pd
    from scipy.spatial.distance import pdist
    # Convert embedding data to DataFrame and align with observations
    embedding_df = pd.DataFrame(adata.obsm[basis], index=adata.obs.index)
    # Compute centroids by grouping based on the specified state_key
    centroids = embedding_df.groupby(adata.obs[state_key]).mean()
    # Calculate pairwise distances between centroids
    dist = pdist(centroids, metric=dist_metric)
    return dist

def compute_dispersion_across_states(adata, basis, state_key, dispersion_metric='var'):
    """
    Computes dispersion across states for a given embedding.

    Args:
        adata (AnnData): AnnData object containing embeddings.
        basis (str): Key for embedding data in adata.obsm or adata.layers.
        state_key (str): Column in adata.obs defining states or clusters.
        dispersion_metric (str, optional): Metric to compute dispersion; 'var', 'std', 'coefficient_of_variation', or 'fano_factor'. Defaults to 'var'.

    Returns:
        dict: Mean dispersion for each state.
    """
    mean_disp = {}
    data = adata.obsm[basis] if basis in adata.obsm else adata.layers[basis]
    for state in adata.obs[state_key].unique():
        idx = adata.obs[state_key] == state
        if dispersion_metric == 'var':
            mean_disp[state] = np.mean(np.var(data[idx], axis=0))
        elif dispersion_metric == 'std':
            mean_disp[state] = np.mean(np.std(data[idx], axis=0))
        elif dispersion_metric == 'coefficient_of_variation':
            mean_disp[state] = np.mean(np.std(data[idx], axis=0) / np.mean(data[idx], axis=0))
        elif dispersion_metric == 'fano_factor':
            mean_disp[state] = np.mean(np.var(data[idx], axis=0) / np.mean(data[idx], axis=0))
        else:
            raise ValueError(f"Invalid dispersion metric: {dispersion_metric}")
        
    return mean_disp



