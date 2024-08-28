
import torch
from torch.utils.data import Sampler
import numpy as np
import logging
logger = logging.getLogger(__name__)


class NeighborhoodSampler(Sampler):
    def __init__(self, batch_size, domain_ids, emb, 
                 neighborhood, p_intra_knn=0.3, p_intra_domain_dict=None, device=None):
        self.batch_size = batch_size
        self.p_intra_knn = p_intra_knn
        self.p_intra_domain_dict = p_intra_domain_dict
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.domain_ids = domain_ids
        self.emb = emb
        self.neighborhood = neighborhood
        
        self.unique_domains, self.domain_counts = torch.unique(self.domain_ids, return_counts=True)

        self.valid_batches = self._generate_batches()

    # Function to permute non- -1 values and push -1 values to the end
    @staticmethod
    def permute_nonneg_and_fill(x, ncol):
        result = torch.full((x.size(0), ncol), -1, dtype=x.dtype, device=x.device)
        for i in range(x.size(0)):
            non_negatives = x[i][x[i] >= 0]
            permuted_non_negatives = non_negatives[torch.randperm(non_negatives.size(0))]
            count = min(ncol, permuted_non_negatives.size(0))
            result[i, :count] = permuted_non_negatives[:count]
        return result


    def _generate_batches(self):
        all_batches = []

        for domain in self.unique_domains:
            domain_indices = torch.where(self.domain_ids == domain)[0]
            out_domain_indices = torch.where(self.domain_ids != domain)[0]

            num_core_samples = max(1,(self.domain_counts[domain] // self.batch_size).item()) # number of neighborhoods per domain be proportional to domain cell counts

            core_samples = domain_indices[torch.randperm(len(domain_indices))[:num_core_samples]]

            # Sample within knn neighborhood
            knn_around_core = self.neighborhood.get_knn_indices(core_samples) # (core_samples, k), contains knn around the core samples
            knn_around_core = torch.tensor(knn_around_core).to(self.device)
            knn_domain_ids = self.domain_ids[knn_around_core] # (core_samples, k), shows domain of each knn sample
            domain_mask = knn_domain_ids == domain # mask indicate if sample is in current domain
            knn_in_domain = torch.where(domain_mask, knn_around_core, torch.tensor(-1, device=self.device))
            knn_out_domain = torch.where(~domain_mask, knn_around_core, torch.tensor(-1, device=self.device))

            # Determine p_intra_domain for the current domain
            p_intra_domain = self.p_intra_domain_dict.get(domain.item())
            batch_knn_count = int(self.p_intra_knn * self.batch_size)
            batch_knn_in_domain_count = int(p_intra_domain * batch_knn_count)
            batch_knn_out_domain_count = batch_knn_count - batch_knn_in_domain_count
            batch_global_in_domain_count = int(p_intra_domain * (self.batch_size - batch_knn_count))
            batch_global_out_domain_count = self.batch_size - batch_knn_count - batch_global_in_domain_count

            batch_knn_in_domain = self.permute_nonneg_and_fill(knn_in_domain, batch_knn_in_domain_count)
            batch_knn_out_domain = self.permute_nonneg_and_fill(knn_out_domain, batch_knn_out_domain_count)

            # Sample globally to fill in rest of batch
            if len(domain_indices) < num_core_samples * batch_global_in_domain_count:
                batch_global_in_domain = domain_indices[
                    torch.randint(len(domain_indices), (num_core_samples * batch_global_in_domain_count,))].view(num_core_samples, batch_global_in_domain_count)
            else:
                batch_global_in_domain = domain_indices[
                    torch.randperm(len(domain_indices))[:num_core_samples * batch_global_in_domain_count]].view(num_core_samples, batch_global_in_domain_count)

            if len(out_domain_indices) < num_core_samples * batch_global_out_domain_count:
                batch_global_out_domain = out_domain_indices[
                    torch.randint(len(out_domain_indices), (num_core_samples * batch_global_out_domain_count,))].view(
                    num_core_samples, batch_global_out_domain_count)
            else:
                batch_global_out_domain = out_domain_indices[
                    torch.randperm(len(out_domain_indices))[:num_core_samples * batch_global_out_domain_count]].view(
                    num_core_samples, batch_global_out_domain_count)

            sample_mtx = torch.cat((batch_knn_in_domain, batch_knn_out_domain, batch_global_in_domain, batch_global_out_domain), dim=1) # (num_core_sample, batch_size)

            for batch in sample_mtx:
                all_batches.append(batch[batch>=0])

        # Shuffle all batches to ensure random order of domains
        all_batches = [all_batches[i] for i in torch.randperm(len(all_batches)).tolist()]
        return all_batches

    def __iter__(self):
        self.valid_batches = self._generate_batches()
        for batch in self.valid_batches:
            yield batch

    def __len__(self):
        return len(self.valid_batches)

