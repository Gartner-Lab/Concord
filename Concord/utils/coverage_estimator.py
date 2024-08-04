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



def coverage_to_p_intra(domain_labels, coverage=None, min_p_intra = 0.1, max_p_intra = 1.0,
                                   scale_to_min_max=False):
        """
            Convert coverage values to p_intra values, with optional scaling and capping.

            Args:
                domain_labels (pd.Series or similar): A categorical series of domain labels.
                coverage (dict): Dictionary with domain keys and coverage values.
                min_p_intra (float): Minimum allowed p_intra value.
                max_p_intra (float): Maximum allowed p_intra value.
                scale_to_min_max (bool): Whether to scale the values to the range [min_p_intra, max_p_intra].

            Returns:
                dict: p_intra_domain_dict with domain codes as keys and p_intra values as values.
        """

        unique_domains = domain_labels.cat.categories

        if coverage is None:
            raise ValueError("Coverage dictionary must be provided.")
        missing_domains = set(unique_domains) - set(coverage.keys())
        if missing_domains:
            raise ValueError(f"Coverage values are missing for the following domains: {missing_domains}")

        p_intra_domain_dict = coverage.copy()

        if scale_to_min_max:
            # Linearly scale the values in p_intra_domain_dict to the range between min_p_intra and max_p_intra
            min_coverage = min(p_intra_domain_dict.values())
            max_coverage = max(p_intra_domain_dict.values())
            if min_p_intra < min_coverage:
                raise ValueError(f"Minimum coverage value ({min_coverage:.3f}) is greater than min_p_intra ({min_p_intra:.3f}) when scale_to_min_max is True." 
                                 "Please set min_p_intra to a value greater than or equal to the minimum coverage value.")

            if min_coverage != max_coverage:  # Avoid division by zero
                scale = (max_p_intra - min_p_intra) / (max_coverage - min_coverage)
                p_intra_domain_dict = {
                    domain: min_p_intra + (value - min_coverage) * scale
                    for domain, value in p_intra_domain_dict.items()
                }
            else:
                p_intra_domain_dict = {domain: (min_p_intra + max_p_intra) / 2 for domain in p_intra_domain_dict}
        else:
            # Cap values to the range [min_p_intra, max_p_intra]
            p_intra_domain_dict = {
                domain: max(min(value, max_p_intra), min_p_intra)
                for domain, value in p_intra_domain_dict.items()
            }

        # Convert the domain labels to their corresponding category codes
        domain_codes = {domain: code for code, domain in enumerate(domain_labels.cat.categories)}
        p_intra_domain_dict = {domain_codes[domain]: value for domain, value in p_intra_domain_dict.items()}

        return p_intra_domain_dict
