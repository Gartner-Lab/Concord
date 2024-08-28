import torch
from torch.utils.data import WeightedRandomSampler, DataLoader, Sampler
import numpy as np
import warnings

class ClassSampler(Sampler):
    def __init__(self, dataset, batch_size, class_weights=None, p_intra=None, drop_last=True, device=None):
        self.dataset = dataset
        self.batch_size = batch_size
        self.drop_last = drop_last
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.class_weights = class_weights
        self.p_intra = p_intra

        # Get domain labels from data source
        if isinstance(dataset, torch.utils.data.Subset):
            self.domain_labels = dataset.dataset.get_domain_labels(dataset.indices)
            self.class_labels = dataset.dataset.get_class_labels(dataset.indices) if class_weights or p_intra else None
        else:
            self.domain_labels = dataset.domain_labels
            self.class_labels = dataset.class_labels if class_weights or p_intra else None

        if self.domain_labels is None:
            warnings.warn("domain/batch information not found, all samples will be treated as from single domain/batch.")
            self.domain_labels = torch.zeros(len(self.emb), dtype=torch.long, device=self.device)

        self.unique_domains = torch.unique(self.domain_labels, return_counts=False)
        self.valid_batches = self._generate_batches()

    def _create_domain_samplers(self):
        samplers = {}
        for domain in self.unique_domains:
            domain_indices = torch.where(self.domain_labels == domain)[0]
            class_weights = np.array([self.class_weights[self.class_labels[i].item()] for i in domain_indices])
            sampler = WeightedRandomSampler(class_weights, len(class_weights), replacement=True)
            samplers[domain.item()] = (domain_indices, sampler)
        return samplers

    def _generate_batches(self):
        all_batches = []

        if self.p_intra is not None:
            samplers = self._create_domain_samplers() if self.class_weights is not None else None

            for domain in self.unique_domains:
                domain_indices = torch.where(self.domain_labels == domain)[0]

                if self.class_weights is not None:
                    sampler = samplers[domain.item()][1]
                    domain_indices = list(DataLoader(domain_indices, batch_size=len(domain_indices), sampler=sampler))[0]

                domain_classes = self.class_labels[domain_indices]
                unique_classes = torch.unique(domain_classes)

                for class_ in unique_classes:
                    class_indices = domain_indices[domain_classes == class_]
                    n_intra = len(class_indices)
                    n_inter = int(n_intra * (1 - self.p_intra) / self.p_intra)

                    intra_class_samples = class_indices
                    non_core_indices = domain_indices[domain_classes != class_]
                    inter_class_samples = self._sample_inter_class(non_core_indices, n_inter)

                    combined_samples = torch.cat((intra_class_samples, inter_class_samples))
                    combined_samples = combined_samples[torch.randperm(len(combined_samples))]  # Shuffle combined samples
                    domain_batches = self._create_batches(combined_samples)
                    all_batches.extend(domain_batches)
        elif self.class_weights is not None:
            samplers = self._create_domain_samplers()
            for domain, (domain_indices, sampler) in samplers.items():
                batch_indices = list(
                    DataLoader(domain_indices, batch_size=self.batch_size, sampler=sampler, drop_last=self.drop_last))
                all_batches.extend(batch_indices)
        else:
            for domain in self.unique_domains:
                domain_indices = torch.where(self.domain_labels == domain)[0]
                shuffled_indices = domain_indices[torch.randperm(len(domain_indices))]
                domain_batches = self._create_batches(shuffled_indices)
                all_batches.extend(domain_batches)

        # Shuffle all batches to ensure random order of domains
        all_batches = [all_batches[i] for i in torch.randperm(len(all_batches)).tolist()]
        return all_batches

    def _create_batches(self, combined_samples):
        return [
            combined_samples[i:i + self.batch_size]
            for i in range(0, len(combined_samples), self.batch_size)
            if len(combined_samples[i:i + self.batch_size]) == self.batch_size or not self.drop_last
        ]

    def _sample_intra_class(self, class_indices, n_intra):
        if n_intra > len(class_indices):
            # If not enough intra-class samples, return all available samples
            return class_indices
        sampled_indices = class_indices[torch.randperm(len(class_indices))[:n_intra]]
        return sampled_indices

    def _sample_inter_class(self, non_core_indices, n_inter):
        if n_inter > len(non_core_indices):
            # If not enough inter-class samples, return all available samples
            return non_core_indices
        sampled_indices = non_core_indices[torch.randperm(len(non_core_indices))[:n_inter]]
        return sampled_indices

    def __iter__(self):
        self.valid_batches = self._generate_batches()
        for batch in self.valid_batches:
            yield batch

    def __len__(self):
        return len(self.valid_batches)





