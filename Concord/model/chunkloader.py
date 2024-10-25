import torch
import numpy as np
from .dataloader import DataLoaderManager

class ChunkLoader:
    def __init__(self, adata, input_layer_key, domain_key, 
                 class_key=None, covariate_keys=None,
                 chunk_size=10000, batch_size=32, train_frac=0.9,
                 sampler_mode="domain",
                 emb_key=None,
                 sampler_knn=300, p_intra_knn=0.3, p_intra_domain=1.0,
                 use_faiss=True, use_ivf=False, ivf_nprobe=8,
                 class_weights=None, p_intra_class=0.3, drop_last=True,
                 preprocess=None, device=None):
        self.adata = adata
        self.input_layer_key = input_layer_key
        self.domain_key = domain_key
        self.class_key = class_key
        self.covariate_keys = covariate_keys
        self.chunk_size = chunk_size
        self.batch_size = batch_size
        self.train_frac = train_frac
        self.sampler_mode = sampler_mode
        self.sampler_knn = sampler_knn
        self.emb_key = emb_key
        self.use_faiss = use_faiss
        self.use_ivf = use_ivf
        self.ivf_nprobe = ivf_nprobe
        self.class_weights = class_weights
        self.p_intra_knn = p_intra_knn
        self.p_intra_domain = p_intra_domain
        self.p_intra_class = p_intra_class
        self.drop_last = drop_last
        self.preprocess = preprocess
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.total_samples = self.adata.shape[0]
        self.num_chunks = (self.total_samples + chunk_size - 1) // chunk_size
        self.indices = np.arange(self.total_samples)
        self.data_structure = None
        _, _, _ = self._load_chunk(0) # Load first chunk to get data_structure

    def __len__(self):
        return self.num_chunks

    def _shuffle_indices(self):
        np.random.shuffle(self.indices)

    # Future todo: Allow random sampling of indices for each chunk, and allow sampling based on global distance
    def _load_chunk(self, chunk_idx):
        start_idx = chunk_idx * self.chunk_size
        end_idx = min(start_idx + self.chunk_size, self.total_samples)
        chunk_indices = self.indices[start_idx:end_idx]
        chunk_adata = self.adata[chunk_indices].to_memory()

        dataloader_manager = DataLoaderManager(
            chunk_adata, self.input_layer_key, self.domain_key, 
            class_key=self.class_key, covariate_keys=self.covariate_keys, 
            batch_size=self.batch_size, train_frac=self.train_frac,
            sampler_mode=self.sampler_mode, sampler_emb=self.sampler_emb, 
            sampler_knn=self.sampler_knn, p_intra_knn=self.p_intra_knn, 
            p_intra_domain=self.p_intra_domain, use_faiss=self.use_faiss, 
            use_ivf=self.use_ivf, ivf_nprobe=self.ivf_nprobe, 
            class_weights=self.class_weights, p_intra_class=self.p_intra_class, 
            drop_last=self.drop_last, preprocess=self.preprocess, device=self.device
        )
        train_dataloader, val_dataloader, data_structure = dataloader_manager.anndata_to_dataloader()

        if self.data_structure is None:
            self.data_structure = data_structure  # Update data_structure if not initialized

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
