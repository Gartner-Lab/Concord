
import torch
from torch.utils.data import Dataset
from scipy.sparse import issparse
import pandas as pd
import numpy as np


class AnnDataDataset(Dataset):
    def __init__(self, adata, input_layer_key='X', domain_key='domain', class_key=None, extra_keys=None, device=None,
                 keep_indices=True):
        self.adata = adata
        self.input_layer_key = input_layer_key
        self.domain_key = domain_key
        self.class_key = class_key
        self.extra_keys = extra_keys if extra_keys is not None else []
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.keep_indices = keep_indices

        self.data = self._get_data_matrix()
        self.domain_labels = torch.tensor(self.adata.obs[self.domain_key].cat.codes.values, dtype=torch.long).to(
            self.device)
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

        print(f"Initialized dataset with {len(self.indices)} samples.")

    def _get_data_matrix(self):
        if self.input_layer_key == 'X':
            return self.adata.X.A if issparse(self.adata.X) else self.adata.X
        else:
            return self.adata.layers[self.input_layer_key].A if issparse(self.adata.layers[self.input_layer_key]) else \
            self.adata.layers[self.input_layer_key]

    def get_domain_labels(self, indices):
        return self.domain_labels[indices]

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        actual_idx = self.indices[idx]
        data_tensor = torch.tensor(self.data[actual_idx], dtype=torch.float32).to(self.device)
        domain_label_tensor = self.domain_labels[actual_idx]

        items = [data_tensor, domain_label_tensor]

        if self.class_key is not None:
            class_label_tensor = self.class_labels[actual_idx]
            items.append(class_label_tensor)

        for key in self.extra_keys:
            items.append(self.extra_tensors[key][actual_idx])

        if self.keep_indices:
            items.append(actual_idx)

        return tuple(items)

    def shuffle_indices(self):
        np.random.shuffle(self.indices)



