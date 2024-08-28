
import torch
from torch.utils.data import Dataset
from scipy.sparse import issparse
import numpy as np
import logging
logger = logging.getLogger(__name__)


class AnnDataset(Dataset):
    def __init__(self, adata, input_layer_key='X', domain_key='domain', class_key=None, covariate_keys=None, device=None):
        self.adata = adata
        self.input_layer_key = input_layer_key
        self.domain_key = domain_key
        self.class_key = class_key
        self.covariate_keys = covariate_keys if covariate_keys is not None else []
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.data = self._get_data_matrix()
        self.domain_labels = torch.tensor(self.adata.obs[self.domain_key].cat.codes.values, dtype=torch.long).to(
            self.device) if self.domain_key is not None else None
        self.indices = np.arange(len(self.adata))

        if self.class_key:
            self.class_labels = torch.tensor(self.adata.obs[self.class_key].cat.codes.values, dtype=torch.long).to(
                self.device)
        else:
            self.class_labels = None

        self.covariate_tensors = {
            key: torch.tensor(self.adata.obs[key].cat.codes.values, dtype=torch.long).to(self.device)
            for key in self.covariate_keys
        }

        self.data_structure = self._init_data_structure()

        logger.info(f"Initialized dataset with {len(self.indices)} samples. Data structure: {self.data_structure}")

    def _get_data_matrix(self):
        if self.input_layer_key == 'X':
            return self.adata.X.A if issparse(self.adata.X) else self.adata.X
        else:
            return self.adata.layers[self.input_layer_key].A if issparse(self.adata.layers[self.input_layer_key]) else \
            self.adata.layers[self.input_layer_key]

    def get_embedding(self, embedding_key, idx):
        if embedding_key == 'X':
            return self.adata.X.A[idx]
        elif embedding_key in self.adata.obsm.key():
            return self.adata.obsm[embedding_key][idx]
        else:
            raise ValueError(f"Embedding key '{embedding_key}' not found in adata")

    def get_domain_labels(self, idx):
        if self.domain_labels is not None:
            return self.domain_labels[idx]
        return None

    def get_class_labels(self, idx):
        if self.class_labels is not None:
            return self.class_labels[idx]
        return None

    def _init_data_structure(self):
        structure = ['input']
        if self.domain_key is not None:
            structure.append('domain')
        if self.class_key is not None:
            structure.append('class')
        structure.extend(self.covariate_keys)
        structure.append('idx')
        return structure

    def get_data_structure(self):
        return self.data_structure

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx, dynamic_class_labels=None):
        actual_idx = self.indices[idx]
        data_tensor = torch.tensor(self.data[actual_idx], dtype=torch.float32).to(self.device)

        items = []
        for key in self.data_structure:
            if key == 'input':
                items.append(data_tensor)
            elif key == 'domain' and self.domain_labels is not None:
                items.append(self.domain_labels[actual_idx])
            elif key == 'class':
                if dynamic_class_labels is not None:
                    items.append(dynamic_class_labels[actual_idx])
                elif self.class_labels is not None:
                    items.append(self.class_labels[actual_idx])
            elif key in self.covariate_keys:
                items.append(self.covariate_tensors[key][actual_idx])
            elif key == 'idx':
                items.append(actual_idx)

        return tuple(items)

    def shuffle_indices(self):
        np.random.shuffle(self.indices)

    def subset(self, idx):
        # Create a new AnnDataset with only the selected idx
        subset_adata = self.adata[idx].copy()
        return AnnDataset(subset_adata, self.input_layer_key, self.domain_key, self.class_key, self.covariate_keys, self.device)

