import numpy as np
from .. import logger

def calculate_domain_coverage(adata, domain_key=None, neighborhood=None):

    domain_labels = adata.obs[domain_key]
    unique_domains = domain_labels.unique()

    # Calculate the indices for each domain
    domain_coverage = {}
    total_points = adata.n_obs

    for domain in unique_domains:
        domain_indices = np.where(domain_labels == domain)[0]
        domain_neighbor_indices = neighborhood.get_knn_indices(domain_indices)

        # Flatten and deduplicate indices
        unique_neighbors = set(domain_neighbor_indices.flatten())

        # Calculate coverage
        coverage = len(unique_neighbors) / total_points
        domain_coverage[domain] = coverage

    return domain_coverage


def coverage_to_p_intra(domain_labels, coverage=None, min_p_intra_domain = 0.1, max_p_intra_domain = 1.0,
                                   scale_to_min_max=False):
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
            # Linearly scale the values in p_intra_domain_dict to the range between min_p_intra_domain and max_p_intra_domain
            min_coverage = min(p_intra_domain_dict.values())
            max_coverage = max(p_intra_domain_dict.values())
            if min_p_intra_domain < min_coverage:
                logger.warn(f"Minimum coverage value ({min_coverage:.3f}) is greater than min_p_intra_domain ({min_p_intra_domain:.3f}) when scale_to_min_max is True." 
                                 "Resetting min_p_intra_domain equal to the minimum coverage value.")
                min_p_intra_domain = min_coverage
            if min_coverage != max_coverage:  # Avoid division by zero
                scale = (max_p_intra_domain - min_p_intra_domain) / (max_coverage - min_coverage)
                p_intra_domain_dict = {
                    domain: min_p_intra_domain + (value - min_coverage) * scale
                    for domain, value in p_intra_domain_dict.items()
                }
            else:
                p_intra_domain_dict = {domain: (min_p_intra_domain + max_p_intra_domain) / 2 for domain in p_intra_domain_dict}
        else:
            # Cap values to the range [min_p_intra_domain, max_p_intra_domain]
            p_intra_domain_dict = {
                domain: max(min(value, max_p_intra_domain), min_p_intra_domain)
                for domain, value in p_intra_domain_dict.items()
            }

        # Convert the domain labels to their corresponding category codes
        domain_codes = {domain: code for code, domain in enumerate(domain_labels.cat.categories)}
        p_intra_domain_dict = {domain_codes[domain]: value for domain, value in p_intra_domain_dict.items()}

        return p_intra_domain_dict

