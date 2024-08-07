
import torch
from torch.utils.data import Sampler
from sklearn.neighbors import NearestNeighbors
import numpy as np
from ..utils.knn import initialize_faiss_index, get_knn_indices
import logging
logger = logging.getLogger(__name__)


class NeighborhoodSampler(Sampler):
    def __init__(self, dataset, batch_size, emb_key="encoded",
                 local_sampling_method=None,
                 sampler_knn=300, p_intra_knn=0.3,
                 p_intra_domain=1.0,
                 use_faiss=True, use_ivf=False, ivf_nprobe=8,
                 device=None):
        self.dataset = dataset
        self.batch_size = batch_size
        self.local_sampling_method=local_sampling_method
        self.sampler_knn = sampler_knn
        self.p_intra_knn = p_intra_knn
        self.use_faiss = use_faiss
        self.use_ivf = use_ivf
        self.ivf_nprobe = ivf_nprobe
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # Get domain labels from data source
        if isinstance(dataset, torch.utils.data.Subset):
            self.emb = dataset.dataset.data[dataset.indices] if emb_key == "data" else dataset.dataset.get_embedding(emb_key, dataset.indices)
            self.domain_labels = dataset.dataset.get_domain_labels(dataset.indices)
        else:
            self.emb = dataset.data if emb_key == "data" else dataset.adata.obsm[emb_key]
            self.domain_labels = dataset.domain_labels

        if self.domain_labels is None:
            logger.warning("domain/batch information not found, all samples will be treated as from single domain/batch.")
            self.domain_labels = torch.zeros(len(self.emb), dtype=torch.long, device=self.device)

        self.unique_domains, self.domain_counts = torch.unique(self.domain_labels, return_counts=True)
        logger.info(f"Number of unique_domains: {len(self.unique_domains)}")


        if isinstance(p_intra_domain, dict):
            self.p_intra_domain_dict = p_intra_domain
        else:
            if len(self.unique_domains) == 1 and p_intra_domain < 1.0:
                logger.warning(f"You specified p_intra_domain as {p_intra_domain} but you only have one domain. "
                               f"Resetting p_intra_domain to 1.0.")
                p_intra_domain = 1.0

            self.p_intra_domain_dict = {domain.item(): p_intra_domain for domain in self.unique_domains}

        # Initialize FAISS index or NearestNeighbors
        self.index = None
        self.nbrs = None
        if self.use_faiss:
            self.index, self.nbrs, self.use_faiss = initialize_faiss_index(emb=self.emb, k=self.sampler_knn,
                                                                           use_faiss=self.use_faiss, use_ivf=self.use_ivf,
                                                                           ivf_nprobe=self.ivf_nprobe)
        if not self.use_faiss:
            self.nbrs = NearestNeighbors(n_neighbors=self.sampler_knn + 1).fit(self.emb)

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
            domain_indices = torch.where(self.domain_labels == domain)[0]
            out_domain_indices = torch.where(self.domain_labels != domain)[0]

            num_core_samples = max(1,(self.domain_counts[domain] // self.batch_size).item())

            core_samples = domain_indices[torch.randperm(len(domain_indices))[:num_core_samples]]

            # Sample within knn neighborhood
            knn_around_core = get_knn_indices(self.emb, core_samples, k=self.sampler_knn,
                                              use_faiss=self.use_faiss,
                                              index=self.index, nbrs=self.nbrs) # (core_samples, k), contains knn around the core samples
            knn_around_core = torch.tensor(knn_around_core).to(self.device)
            knn_domain_labels = self.domain_labels[knn_around_core] # (core_samples, k), shows domain of each knn sample
            domain_mask = knn_domain_labels == domain # mask indicate if sample is in current domain
            knn_in_domain = torch.where(domain_mask, knn_around_core, torch.tensor(-1, device=self.device))
            knn_out_domain = torch.where(~domain_mask, knn_around_core, torch.tensor(-1, device=self.device))

            # Determine p_intra_domain for the current domain
            p_intra_domain = self.p_intra_domain_dict.get(domain.item())
            batch_knn_count = int(self.p_intra_knn * self.batch_size)
            batch_knn_in_domain_count = int(p_intra_domain * batch_knn_count)
            batch_knn_out_domain_count = batch_knn_count - batch_knn_in_domain_count
            batch_global_in_domain_count = int(p_intra_domain * (self.batch_size - batch_knn_count))
            batch_global_out_domain_count = self.batch_size - batch_knn_count - batch_global_in_domain_count
            # print(f"batch_knn_count:{batch_knn_count}")
            # print(f"batch_knn_in_domain_count{batch_knn_in_domain_count}")
            # print(f"batch_knn_out_domain_count{batch_knn_out_domain_count}")
            # print(f"batch_global_in_domain_count{batch_global_in_domain_count}")
            # print(f"batch_global_out_domain_count{batch_global_out_domain_count}")
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
