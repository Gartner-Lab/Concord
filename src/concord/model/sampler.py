
import torch
from torch.utils.data import Sampler
import logging
import numpy as np
logger = logging.getLogger(__name__)


class ConcordSampler(Sampler):
    """
    A custom PyTorch sampler that performs probabilistic domain-aware and 
    neighborhood-aware batch sampling for contrastive learning.

    This sampler selects samples from both intra-domain and inter-domain 
    distributions based on configurable probabilities.
    """
    def __init__(self, batch_size, 
                 domain_ids, 
                 neighborhood, 
                 p_intra_knn=0.3, 
                 p_intra_domain=1.0, 
                 min_batch_size=4, 
                 domain_minibatch_strategy='proportional',
                 domain_minibatch_min_count=1,
                 domain_coverage=None,
                 sample_with_replacement=False,
                 device=None):
        """
        Initializes the ConcordSampler.

        Args:
            batch_size (int): Number of samples per batch.
            domain_ids (torch.Tensor): Tensor of domain labels for each sample.
            neighborhood (Neighborhood): Precomputed k-NN index.
            p_intra_knn (float, optional): Probability of selecting samples from k-NN neighborhoods. Default is 0.3.
            p_intra_domain (dict, optional): Probability of selecting samples from the same domain.
            min_batch_size (int, optional): Minimum allowed batch size. Default is 4.
            device (torch.device, optional): Device to store tensors. Defaults to GPU if available.
        """
        self.batch_size = batch_size
        self.p_intra_domain = p_intra_domain
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.domain_ids = domain_ids
        self.min_batch_size = min_batch_size
        self.domain_minibatch_strategy = domain_minibatch_strategy
        self.domain_minibatch_min_count = domain_minibatch_min_count
        self.domain_coverage = domain_coverage 
        self.sample_with_replacement = sample_with_replacement
        if not sample_with_replacement and self.domain_minibatch_strategy != "proportional":
            logger.warning(
                f"sample_with_replacement=False is not supported for "
                f"strategy '{self.domain_minibatch_strategy}'; forcing to True."
            )
        self.sample_with_replacement = (
            sample_with_replacement or self.domain_minibatch_strategy != "proportional"
        )
        self.p_intra_knn = p_intra_knn
        self.neighborhood = neighborhood
        if self.p_intra_knn > 0 and self.neighborhood is None:
            raise ValueError(
                "A neighborhood graph must be provided when p_intra_knn > 0."
            )

        allowed_strategies = ['proportional', 'equal', 'coverage']
        if self.domain_minibatch_strategy not in allowed_strategies:
            raise ValueError(f"domain_minibatch_strategy must be one of {allowed_strategies}")
        
        # Add validation for the coverage strategy
        if self.domain_minibatch_strategy == 'coverage' and self.domain_coverage is None:
            raise ValueError("domain_coverage dictionary must be provided when using the 'coverage' strategy.")
        
        self.unique_domains, self.domain_counts = torch.unique(self.domain_ids, return_counts=True)
        self.valid_batches = None


    def _calculate_num_batches_per_domain(self):
        """
        Calculates the number of minibatches to generate for each domain based on the chosen strategy.
        
        Returns:
            dict: A dictionary mapping each domain ID to its number of batches.
        """
        num_batches_map = {}
        
        if self.domain_minibatch_strategy == 'proportional':
            for i, domain in enumerate(self.unique_domains):
                count = self.domain_counts[i]
                num_batches = max(self.domain_minibatch_min_count, (count // self.batch_size).item())
                num_batches_map[domain.item()] = num_batches

        elif self.domain_minibatch_strategy == 'equal':
            all_proportional_batches = [max(self.domain_minibatch_min_count, (count // self.batch_size).item()) for count in self.domain_counts]
            # Use the median as a robust measure for the "equal" number of batches
            equal_num_batches = max(self.domain_minibatch_min_count, int(np.median(all_proportional_batches)))
            for domain in self.unique_domains:
                num_batches_map[domain.item()] = equal_num_batches

        elif self.domain_minibatch_strategy == 'coverage':
            total_batches_in_epoch = len(self.domain_ids) // self.batch_size
            total_coverage = sum(self.domain_coverage.values())
            
            if total_coverage == 0:
                logger.warning("Total domain coverage is 0. Falling back to 'proportional' minibatch strategy.")
                original_strategy = self.domain_minibatch_strategy
                self.domain_minibatch_strategy = 'proportional'
                num_batches_map = self._calculate_num_batches_per_domain()
                self.domain_minibatch_strategy = original_strategy
                return num_batches_map

            for domain in self.unique_domains:
                domain_id = domain.item()
                coverage_score = self.domain_coverage.get(domain_id, 0)
                proportion = coverage_score / total_coverage
                num_batches = max(self.domain_minibatch_min_count, int(proportion * total_batches_in_epoch))
                num_batches_map[domain_id] = num_batches

        return num_batches_map


    # Function to permute non- -1 values and push -1 values to the end
    @staticmethod
    def _permute_nonneg_and_fill(x, ncol):
        """
        Permutes non-negative values and fills remaining positions with -1.

        Args:
            x (torch.Tensor): Input tensor containing indices.
            ncol (int): Number of columns to keep.

        Returns:
            torch.Tensor: Permuted tensor with -1s filling unused positions.
        """
        result = torch.full((x.size(0), ncol), -1, dtype=x.dtype, device=x.device)
        for i in range(x.size(0)):
            non_negatives = x[i][x[i] >= 0]
            if non_negatives.numel() > 0:
                permuted_non_negatives = non_negatives[torch.randperm(non_negatives.size(0))]
                count = min(ncol, permuted_non_negatives.size(0))
                result[i, :count] = permuted_non_negatives[:count]
        return result


    def _generate_batches(self):
        """
        Generates batches based on intra-domain and intra-neighborhood probabilities.

        Returns:
            list: A list of valid batches.
        """
        all_batches = []
        num_batches_per_domain = self._calculate_num_batches_per_domain()
        for domain in self.unique_domains:
            domain_indices = torch.where(self.domain_ids == domain)[0]
            if len(domain_indices) == 0: continue # Skip domains with no cells

            out_domain_indices = torch.where(self.domain_ids != domain)[0]
            num_batches_domain = num_batches_per_domain[domain.item()]
            
            # Sample within knn neighborhood if p_intra_knn > 0
            if self.p_intra_knn == 0:
                batch_global_in_domain_count = int(self.p_intra_domain * self.batch_size)
                batch_global_out_domain_count = self.batch_size - batch_global_in_domain_count
                batch_knn = torch.empty(num_batches_domain, 0, dtype=torch.long, device=self.device)
            else:
                # Check if neighborhood is available
                if self.neighborhood is None:
                    raise ValueError("Neighborhood must be provided to sample from k-NN neighborhoods.")
                if self.sample_with_replacement:
                    core_sample_indices = torch.randint(len(domain_indices), (num_batches_domain,))
                else:
                    core_sample_indices = torch.randperm(len(domain_indices))[:num_batches_domain]
                
                core_samples = domain_indices[core_sample_indices]

                knn_around_core = self.neighborhood.get_knn(core_samples) # (core_samples, k), contains core + knn around the core samples
                knn_around_core = torch.tensor(knn_around_core, device=self.device, dtype=torch.long)
                knn_domain_ids = self.domain_ids[knn_around_core] # (core_samples, k), shows domain of each knn sample
                domain_mask = knn_domain_ids == domain # mask indicate if sample is in current domain
                knn_in_domain = torch.where(domain_mask, knn_around_core, torch.tensor(-1, device=self.device))
                knn_out_domain = torch.where(~domain_mask, knn_around_core, torch.tensor(-1, device=self.device))

                batch_knn_count = int(self.p_intra_knn * self.batch_size)
                batch_knn_in_domain_count = int(self.p_intra_domain * batch_knn_count)
                batch_knn_out_domain_count = batch_knn_count - batch_knn_in_domain_count
                batch_global_in_domain_count = int(self.p_intra_domain * (self.batch_size - batch_knn_count))
                batch_global_out_domain_count = self.batch_size - batch_knn_count - batch_global_in_domain_count

                #print(f"batch_knn_count: {batch_knn_count}, batch_knn_in_domain_count: {batch_knn_in_domain_count}, batch_knn_out_domain_count: {batch_knn_out_domain_count}, batch_global_in_domain_count: {batch_global_in_domain_count}, batch_global_out_domain_count: {batch_global_out_domain_count}")

                batch_knn_in_domain = self._permute_nonneg_and_fill(knn_in_domain, batch_knn_in_domain_count)
                batch_knn_out_domain = self._permute_nonneg_and_fill(knn_out_domain, batch_knn_out_domain_count)
                batch_knn = torch.cat((batch_knn_in_domain, batch_knn_out_domain), dim=1) 

            if not self.sample_with_replacement and len(domain_indices) >= num_batches_domain * batch_global_in_domain_count:
                batch_global_in_domain = domain_indices[
                    torch.randperm(len(domain_indices))[:num_batches_domain * batch_global_in_domain_count]].view(num_batches_domain, batch_global_in_domain_count)
            else:
                batch_global_in_domain = domain_indices[
                    torch.randint(len(domain_indices), (num_batches_domain, batch_global_in_domain_count))
                ]

            if len(out_domain_indices) > 0:
                if not self.sample_with_replacement and len(out_domain_indices) >= num_batches_domain * batch_global_out_domain_count:
                    batch_global_out_domain = out_domain_indices[
                    torch.randperm(len(out_domain_indices))[:num_batches_domain * batch_global_out_domain_count]].view(
                    num_batches_domain, batch_global_out_domain_count)
                else:
                    batch_global_out_domain = out_domain_indices[
                        torch.randint(len(out_domain_indices), (num_batches_domain, batch_global_out_domain_count))
                    ]
            else:
                batch_global_out_domain = torch.empty(num_batches_domain, 0, dtype=torch.long, device=self.device)

            batch_global = torch.cat((batch_global_in_domain, batch_global_out_domain), dim=1)

            sample_mtx = torch.cat((batch_knn, batch_global), dim=1)

            for _,batch in enumerate(sample_mtx):
                valid_batch = batch[batch >= 0].unique()
                if len(valid_batch) >= self.min_batch_size:
                    all_batches.append(valid_batch)


        # Shuffle all batches to ensure random order of domains
        indices = torch.randperm(len(all_batches)).tolist()
        all_batches = [all_batches[i] for i in indices]

        return all_batches
    

    def __iter__(self):
        """
        Iterator for sampling batches.

        Yields:
            torch.Tensor: A batch of sample indices.
        """
        self.valid_batches = self._generate_batches()
            
        # avg_seen = len(torch.cat(list(self.valid_batches)).unique()) / len(self.domain_ids)
        # print(f"Fraction of cells seen this epoch: {avg_seen:.5f}")
        for batch in self.valid_batches:
            yield batch.tolist()


    def __len__(self):
        """
        Returns the number of batches.

        Returns:
            int: Number of valid batches.
        """
        if self.valid_batches is None:
             self.valid_batches = self._generate_batches()
        return len(self.valid_batches)

