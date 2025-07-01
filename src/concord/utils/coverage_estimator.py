from __future__ import annotations
import numpy as np
import pandas as pd
import logging

logger = logging.getLogger(__name__)



def calculate_domain_coverage_by_cluster(
    adata, 
    domain_key, 
    cluster_key, 
    enrichment_threshold=0.2
):
    """
    Calculates the manifold coverage for each domain based on its representation across discrete clusters.

    A domain is considered to "cover" a cluster if its presence in that cluster is
    greater than what would be expected by random chance, adjusted for both domain
    size and cluster size (i.e., enrichment > threshold). The final coverage score
    for a domain is the fraction of total clusters it covers.

    Args:
        adata (AnnData): The annotated data matrix.
        domain_key (str): The key in `adata.obs` for the domain/batch labels.
        cluster_key (str): The key in `adata.obs` for the pre-computed cluster labels (e.g., 'leiden').
        enrichment_threshold (float, optional): The minimum enrichment score for a domain
            to be considered as "covering" a cluster. Defaults to 1.0, meaning the domain
            is more present than expected by random chance.

    Returns:
        dict: A dictionary mapping each domain to its coverage score (a float between 0 and 1).
    """
    if cluster_key not in adata.obs.columns:
        raise ValueError(f"Cluster key '{cluster_key}' not found in adata.obs.")
    if domain_key not in adata.obs.columns:
        raise ValueError(f"Domain key '{domain_key}' not found in adata.obs.")

    logger.info(f"Calculating domain coverage using clusters from '{cluster_key}'...")

    # Step 1: Create a contingency table of observed counts (domains vs clusters)
    contingency_table = pd.crosstab(adata.obs[domain_key], adata.obs[cluster_key])
    
    # Step 2: Calculate the expected counts
    domain_counts = contingency_table.sum(axis=1)
    cluster_counts = contingency_table.sum(axis=0)
    total_cells = domain_counts.sum()

    # Use numpy.outer to efficiently calculate the expected matrix
    expected_counts = np.outer(domain_counts, cluster_counts) / total_cells
    
    # Add a small epsilon to avoid division by zero
    epsilon = 1e-9
    
    # Step 3: Calculate the enrichment matrix
    enrichment_matrix = contingency_table.values / (expected_counts + epsilon)
    
    # Step 4: Identify which clusters are "covered" by each domain based on the threshold
    # This results in a boolean matrix
    covered_clusters_matrix = enrichment_matrix > enrichment_threshold
    
    # Step 5: Calculate the final coverage score for each domain
    num_covered_clusters = covered_clusters_matrix.sum(axis=1)
    total_clusters = len(cluster_counts)
    
    coverage_scores_array = num_covered_clusters / total_clusters

    # Convert the NumPy array back to a pandas Series, using the domain names as the index
    coverage_scores_series = pd.Series(coverage_scores_array, index=domain_counts.index)

    # Now, convert the correctly-labeled Series to a dictionary
    domain_coverage = coverage_scores_series.to_dict()
    
    logger.info(f"Domain coverage scores: {domain_coverage}")
    
    return domain_coverage



