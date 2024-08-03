import numpy as np
import pandas as pd
import time
from .knn import initialize_faiss_index, get_knn_indices

def calculate_dataset_coverage(adata, k=128, emb_key='X_pca', dataset_key=None,
                               use_faiss=True, use_ivf=False, ivf_nprobe=10):
    """
    Calculate the neighborhood coverage of each dataset in a k-NN graph.

    Parameters:
    adata: AnnData
        The AnnData object containing the PCA coordinates and dataset identifiers.
    n_neighbors: int
        The number of neighbors to consider in the k-NN graph.
    emb_key: str
        The key for accessing embedding coordinates in adata.obsm.
    dataset_key: str
        The key for accessing dataset identifiers in adata.obs.
    use_faiss: bool
        Whether to use FAISS for k-NN computation.
    use_ivf: bool
        Whether to use IVF FAISS index.
    ivf_nprobe: int
        Number of probes for IVF FAISS index.

    Returns:
    pd.DataFrame
        A DataFrame containing the neighborhood coverage for each dataset.
    """
    emb_coords = adata.obsm[emb_key]
    dataset_ids = adata.obs[dataset_key]
    unique_ids = dataset_ids.unique()

    # Initialize FAISS or sklearn k-NN model
    index, nbrs, use_faiss = initialize_faiss_index(emb_coords, k=k, use_faiss=use_faiss, use_ivf=use_ivf,
                                                    ivf_nprobe=ivf_nprobe)


    # Calculate the indices for each dataset
    dataset_coverage = {}
    total_points = emb_coords.shape[0]

    for dataset in unique_ids:
        dataset_indices = np.where(dataset_ids == dataset)[0]
        dataset_neighbor_indices = get_knn_indices(emb_coords, dataset_indices, k=k, use_faiss=use_faiss,
                                                   index=index, nbrs=nbrs)

        # Flatten and deduplicate indices
        unique_neighbors = set(dataset_neighbor_indices.flatten())

        # Calculate coverage
        coverage = len(unique_neighbors) / total_points
        dataset_coverage[dataset] = coverage

    return dataset_coverage
