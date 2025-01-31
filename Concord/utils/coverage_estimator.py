import numpy as np
from .. import logger

def calculate_domain_coverage(adata, domain_key=None, neighborhood=None, k=100, basis='X_pca', pca_n_comps=50, metric='euclidean'):

    if neighborhood is None:
        from ..model.knn import Neighborhood
        from ..utils.anndata_utils import get_adata_basis
        # Compute neighborhood
        logger.info(f"Computing neighborhood graph for coverage estimation using {basis}.")
        emb = get_adata_basis(adata, basis=basis, pca_n_comps=pca_n_comps)
        neighborhood = Neighborhood(emb=emb, k=k, metric=metric)

    domain_labels = adata.obs[domain_key]
    unique_domains = domain_labels.unique()

    # Calculate the indices for each domain
    domain_coverage = {}
    total_points = adata.n_obs

    for domain in unique_domains:
        domain_indices = np.where(domain_labels == domain)[0]
        domain_neighbor_indices = neighborhood.get_knn(domain_indices)

        # Flatten and deduplicate indices
        unique_neighbors = set(domain_neighbor_indices.flatten())

        # Calculate coverage
        coverage = len(unique_neighbors) / total_points
        domain_coverage[domain] = coverage

    return domain_coverage


def coverage_to_p_intra(domain_labels, coverage=None, min_p_intra_domain = 0.5, max_p_intra_domain = 1.0, scale_to_min_max=True):
        """
            Convert coverage values top_intra values, with optional scaling and capping.

            Args:
                domain_labels (pd.Series or similar): A categorical series of domain labels.
                coverage (dict): Dictionary with domain keys and coverage values.
                min_p_intra_domain (float): Minimum allowed p_intra value.
                max_p_intra_domain (float): Maximum allowed p_intra value.
                scale_to_min_max (bool): Whether to scale the values to the range [min_p_intra_domain, max_p_intra_domain].

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
            p_intra_domain_dict = {
                domain: min_p_intra_domain + (max_p_intra_domain - min_p_intra_domain) * value
                for domain, value in p_intra_domain_dict.items()
            }
        else:
            # Cap values to the range [min_p_intra_domain, max_p_intra_domain]
            p_intra_domain_dict = {
                domain: max(min(value, max_p_intra_domain), min_p_intra_domain)
                for domain, value in p_intra_domain_dict.items()
            }

        return p_intra_domain_dict

