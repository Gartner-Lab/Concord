import torch
from torch.utils.data import WeightedRandomSampler, DataLoader, Sampler
import numpy as np

class DomainClassSampler(Sampler):
    def __init__(self, dataset, batch_size, device, drop_last=True,
                 sampler_mode='domain', weights=None):
        self.dataset = dataset
        self.batch_size = batch_size
        self.drop_last = drop_last
        self.device = device
        self.sampler_mode = sampler_mode
        self.weights = weights

        # Get domain labels from data source
        if isinstance(dataset, torch.utils.data.Subset):
            self.domain_labels = dataset.dataset.get_domain_labels(dataset.indices)
            if self.sampler_mode == 'domain_and_class':
                self.class_labels = dataset.dataset.get_class_labels(dataset.indices)
        else:
            self.domain_labels = dataset.domain_labels
            if self.sampler_mode == 'domain_and_class':
                self.class_labels = dataset.class_labels

        self.unique_domains = torch.unique(self.domain_labels, return_counts=False)
        self.samplers = self._create_domain_samplers() if self.sampler_mode == 'domain_and_class' else None
        self.valid_batches = self._generate_batches()

    def _create_domain_samplers(self):
        samplers = {}
        for domain in self.unique_domains:
            domain_indices = torch.where(self.domain_labels == domain)[0]
            class_weights = np.array([self.weights[self.class_labels[i].item()] for i in domain_indices])
            sampler = WeightedRandomSampler(class_weights, len(class_weights), replacement=True)
            samplers[domain.item()] = (domain_indices, sampler)
        return samplers

    def _generate_batches(self):
        all_batches = []

        if self.sampler_mode == 'domain':
            for domain in self.unique_domains:
                domain_indices = torch.where(self.domain_labels == domain)[0]
                shuffled_indices = domain_indices[torch.randperm(len(domain_indices))]
                domain_batches = self._create_batches(shuffled_indices)
                all_batches.extend(domain_batches)
        elif self.sampler_mode == 'domain_and_class':
            for domain, (domain_indices, sampler) in self.samplers.items():
                batch_indices = list(
                    DataLoader(domain_indices, batch_size=self.batch_size, sampler=sampler, drop_last=self.drop_last))
                all_batches.extend(batch_indices)
        else:
            raise ValueError(f"Unsupported sampler mode: {self.sampler_mode}")

        # Shuffle all batches to ensure random order of classes
        all_batches = [all_batches[i] for i in torch.randperm(len(all_batches)).tolist()]
        return all_batches

    def _create_batches(self, combined_samples):
        return [
            combined_samples[i:i + self.batch_size]
            for i in range(0, len(combined_samples), self.batch_size)
            if len(combined_samples[i:i + self.batch_size]) == self.batch_size or not self.drop_last
        ]

    def __iter__(self):
        self.valid_batches = self._generate_batches()
        for batch in self.valid_batches:
            yield batch

    def __len__(self):
        return len(self.valid_batches)