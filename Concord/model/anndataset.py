
import torch
from torch.utils.data import Dataset
from scipy.sparse import issparse
import numpy as np
import logging
logger = logging.getLogger(__name__)


class AnnDataset(Dataset):
    def __init__(self, adata, input_layer_key='X', domain_key='domain', class_key=None, extra_keys=None, device=None):
        self.adata = adata
        self.input_layer_key = input_layer_key
        self.domain_key = domain_key
        self.class_key = class_key
        self.extra_keys = extra_keys if extra_keys is not None else []
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

        self.extra_tensors = {
            key: torch.tensor(self.adata.obs[key].values, dtype=torch.float32).to(self.device)
            for key in self.extra_keys
        }

        self.data_structure = self._init_data_structure()

        logger.info(f"Initialized dataset with {len(self.indices)} samples. Data structure: {self.data_structure}")

    def _get_data_matrix(self):
        if self.input_layer_key == 'X':
            return self.adata.X.A if issparse(self.adata.X) else self.adata.X
        else:
            return self.adata.layers[self.input_layer_key].A if issparse(self.adata.layers[self.input_layer_key]) else \
            self.adata.layers[self.input_layer_key]

    def get_embedding(self, embedding_key, indices):
        if embedding_key is not None and embedding_key in self.adata.obsm.key():
            return self.adata.obsm[embedding_key][indices]
        return None

    def get_domain_labels(self, indices):
        if self.domain_labels is not None:
            return self.domain_labels[indices]
        return None

    def get_class_labels(self, indices):
        if self.class_labels is not None:
            return self.class_labels[indices]
        return None

    def _init_data_structure(self):
        structure = ['input']
        if self.domain_key is not None:
            structure.append('domain')
        if self.class_key is not None:
            structure.append('class')
        structure.extend(self.extra_keys)
        structure.append('indices')
        return structure

    def get_data_structure(self):
        return self.data_structure

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        actual_idx = self.indices[idx]
        data_tensor = torch.tensor(self.data[actual_idx], dtype=torch.float32).to(self.device)

        items = []
        for key in self.data_structure:
            if key == 'input':
                items.append(data_tensor)
            elif key == 'domain' and self.domain_labels is not None:
                items.append(self.domain_labels[actual_idx])
            elif key == 'class' and self.class_labels is not None:
                items.append(self.class_labels[actual_idx])
            elif key in self.extra_keys:
                items.append(self.extra_tensors[key][actual_idx])
            elif key == 'indices':
                items.append(actual_idx)

        return tuple(items)

    def shuffle_indices(self):
        np.random.shuffle(self.indices)



