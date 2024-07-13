import torch
from .dataloader import anndata_to_dataloader
import numpy as np

class ChunkLoader:
    def __init__(self, adata, input_layer_key, domain_key, class_key=None, extra_keys=None,
                 chunk_size=10000, batch_size=32, train_frac=0.9, sampler_mode="domain", drop_last=True,
                 preprocess=None, device=None):
        self.adata = adata
        self.input_layer_key = input_layer_key
        self.domain_key = domain_key
        self.class_key = class_key
        self.extra_keys = extra_keys
        self.chunk_size = chunk_size
        self.batch_size = batch_size
        self.train_frac = train_frac
        self.sampler_mode = sampler_mode
        self.drop_last = drop_last
        self.preprocess = preprocess
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.total_samples = self.adata.shape[0]
        self.num_chunks = (self.total_samples + chunk_size - 1) // chunk_size
        self.indices = np.arange(self.total_samples)

    def __len__(self):
        return self.num_chunks

    def _shuffle_indices(self):
        np.random.shuffle(self.indices)

    def _load_chunk(self, chunk_idx):
        start_idx = chunk_idx * self.chunk_size
        end_idx = min(start_idx + self.chunk_size, self.total_samples)
        chunk_indices = self.indices[start_idx:end_idx]
        chunk_adata = self.adata[chunk_indices].to_memory()

        train_dataloader, val_dataloader = anndata_to_dataloader(
            chunk_adata, input_layer_key=self.input_layer_key, domain_key=self.domain_key, class_key=self.class_key,
            extra_keys=self.extra_keys, train_frac=self.train_frac, batch_size=self.batch_size,
            sampler_mode=self.sampler_mode, drop_last=self.drop_last, preprocess=self.preprocess,
            device=self.device
        )

        return train_dataloader, val_dataloader, chunk_indices

    def __iter__(self):
        self.current_chunk_idx = 0
        self._shuffle_indices()
        return self

    def __next__(self):
        if self.current_chunk_idx >= self.num_chunks:
            raise StopIteration
        train_dataloader, val_dataloader, chunk_indices = self._load_chunk(self.current_chunk_idx)
        self.current_chunk_idx += 1
        return train_dataloader, val_dataloader, chunk_indices
