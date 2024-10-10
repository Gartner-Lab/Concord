
import torch
from torch.utils.data import Sampler
import logging
import numpy as np
logger = logging.getLogger(__name__)


class ConcordSampler(Sampler):
    def __init__(self, batch_size, domain_ids, 
                 neighborhood, p_intra_knn=0.3, p_intra_domain_dict=None, min_batch_size=4, device=None):
        self.batch_size = batch_size
        self.p_intra_knn = p_intra_knn
        self.p_intra_domain_dict = p_intra_domain_dict
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.domain_ids = domain_ids
        self.neighborhood = neighborhood
        
        self.unique_domains, self.domain_counts = torch.unique(self.domain_ids, return_counts=True)

        #self.valid_batches,_ = self._generate_batches()
        self.valid_batches = None
        self.min_batch_size = min_batch_size

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

            # Determine p_intra_domain for the current domain
            p_intra_domain = self.p_intra_domain_dict.get(domain.item())
            num_batches_domain = max(1,(self.domain_counts[domain] // self.batch_size).item()) # number of batches per domain be proportional to domain cell counts
            
            # Sample within knn neighborhood if p_intra_knn > 0
            if self.p_intra_knn == 0:
                batch_global_in_domain_count = int(p_intra_domain * self.batch_size)
                batch_global_out_domain_count = self.batch_size - batch_global_in_domain_count
            else:
                core_samples = domain_indices[torch.randperm(len(domain_indices))[:num_batches_domain]]
                knn_around_core = self.neighborhood.get_knn_indices(core_samples) # (core_samples, k), contains core + knn around the core samples
                knn_around_core = torch.tensor(knn_around_core).to(self.device)
                knn_domain_ids = self.domain_ids[knn_around_core] # (core_samples, k), shows domain of each knn sample
                domain_mask = knn_domain_ids == domain # mask indicate if sample is in current domain
                knn_in_domain = torch.where(domain_mask, knn_around_core, torch.tensor(-1, device=self.device))
                knn_out_domain = torch.where(~domain_mask, knn_around_core, torch.tensor(-1, device=self.device))

                batch_knn_count = int(self.p_intra_knn * self.batch_size)
                batch_knn_in_domain_count = int(p_intra_domain * batch_knn_count)
                batch_knn_out_domain_count = batch_knn_count - batch_knn_in_domain_count
                batch_global_in_domain_count = int(p_intra_domain * (self.batch_size - batch_knn_count))
                batch_global_out_domain_count = self.batch_size - batch_knn_count - batch_global_in_domain_count

                batch_knn_in_domain = self.permute_nonneg_and_fill(knn_in_domain, batch_knn_in_domain_count)
                batch_knn_out_domain = self.permute_nonneg_and_fill(knn_out_domain, batch_knn_out_domain_count)
                batch_knn = torch.cat((batch_knn_in_domain, batch_knn_out_domain), dim=1) 

            # Sample globally to fill in rest of batch (or all of batch if p_intra_knn == 0)
            if len(domain_indices) < num_batches_domain * batch_global_in_domain_count:
                batch_global_in_domain = domain_indices[
                    torch.randint(len(domain_indices), (num_batches_domain * batch_global_in_domain_count,))].view(num_batches_domain, batch_global_in_domain_count)
            else:
                batch_global_in_domain = domain_indices[
                    torch.randperm(len(domain_indices))[:num_batches_domain * batch_global_in_domain_count]].view(num_batches_domain, batch_global_in_domain_count)

            if len(out_domain_indices) < num_batches_domain * batch_global_out_domain_count:
                batch_global_out_domain = out_domain_indices[
                    torch.randint(len(out_domain_indices), (num_batches_domain * batch_global_out_domain_count,))].view(
                    num_batches_domain, batch_global_out_domain_count)
            else:
                batch_global_out_domain = out_domain_indices[
                    torch.randperm(len(out_domain_indices))[:num_batches_domain * batch_global_out_domain_count]].view(
                    num_batches_domain, batch_global_out_domain_count)

            batch_global = torch.cat((batch_global_in_domain, batch_global_out_domain), dim=1)
            sample_mtx = torch.cat((batch_knn, batch_global), dim=1) if self.p_intra_knn > 0 else batch_global

            for _,batch in enumerate(sample_mtx):
                valid_batch = batch[batch >= 0]
                all_batches.append(valid_batch)


        # Shuffle all batches to ensure random order of domains
        indices = torch.randperm(len(all_batches)).tolist()
        all_batches = [all_batches[i] for i in indices]

        return all_batches
    

    def __iter__(self):
        self.valid_batches = self._generate_batches()
        for batch in self.valid_batches:
            yield batch

    def __len__(self):
        return len(self.valid_batches)



class ConcordMatchNNSampler(Sampler):
    def __init__(self, batch_size, indices, domain_ids, 
                 neighborhood, p_intra_knn=0.3, p_intra_domain_dict=None, 
                 min_batch_size=4, device=None):
        # Use half the batch size for the base sampler
        self.base_sampler = ConcordSampler(
            batch_size // 2, indices, domain_ids, neighborhood, 
            p_intra_knn, p_intra_domain_dict, min_batch_size=min_batch_size, device=device)
        self.neighborhood = neighborhood
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def __iter__(self):
        for batch in self.base_sampler:
            batch = batch.to(self.device)
            # Get the first nearest neighbor for each sample, excluding the sample itself
            nn_indices = self.neighborhood.get_knn_indices(batch, k=1, include_self=False)
            nn_indices = torch.tensor(nn_indices.squeeze(1), device=self.device)
            # Concatenate batch and nn_indices to form the full batch
            full_batch = torch.cat([batch, nn_indices], dim=0)
            yield full_batch


    def __len__(self):
        return len(self.base_sampler)

