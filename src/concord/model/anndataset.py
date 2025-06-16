
import torch
from torch.utils.data import Dataset
from scipy.sparse import issparse
import numpy as np
import logging
logger = logging.getLogger(__name__)


class AnnDataset(Dataset):
    """
    A PyTorch Dataset class for handling annotated datasets (AnnData).

    This dataset is designed to work with single-cell RNA-seq data stored in 
    AnnData objects. It extracts relevant features, domain labels, class labels, 
    and covariate labels while handling sparse and dense matrices.

    Attributes:
        adata (AnnData): The annotated data matrix.
        input_layer_key (str): The key to retrieve input features from `adata`.
        domain_key (str): The key in `adata.obs` specifying domain labels.
        class_key (str, optional): The key in `adata.obs` specifying class labels.
        covariate_keys (list, optional): A list of keys for covariate labels in `adata.obs`.
        device (torch.device): The device to store tensors (GPU or CPU).
        data (torch.Tensor): Tensor containing input data.
        domain_labels (torch.Tensor): Tensor containing domain labels.
        class_labels (torch.Tensor, optional): Tensor containing class labels if provided.
        covariate_tensors (dict): A dictionary containing tensors for covariate labels.
        indices (np.ndarray): Array of dataset indices.
    """
    def __init__(self, adata, data_structure, input_layer_key='X', domain_key='domain', class_key=None, covariate_keys=None, device=None):
        """
        Initializes the AnnDataset.

        Args:
            adata (AnnData): The annotated dataset.
            input_layer_key (str, optional): Key to extract input data. Defaults to 'X'.
            domain_key (str, optional): Key for domain labels in `adata.obs`. Defaults to 'domain'.
            class_key (str, optional): Key for class labels in `adata.obs`. Defaults to None.
            covariate_keys (list, optional): List of keys for covariate labels in `adata.obs`. Defaults to None.
            device (torch.device, optional): Device to store tensors (GPU or CPU). Defaults to GPU if available.

        Raises:
            ValueError: If domain or class key is not found in `adata.obs`.
        """
        self.adata = adata
        self.data_structure = data_structure
        self.input_layer_key = input_layer_key
        self.domain_key = domain_key
        self.class_key = class_key
        self.covariate_keys = covariate_keys if covariate_keys is not None else []
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.data = torch.tensor(self._get_data_matrix(), dtype=torch.float32)
        self.domain_labels = torch.tensor(self.adata.obs[self.domain_key].cat.codes.values, dtype=torch.long).to(
            self.device)
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

        logger.info(f"Initialized dataset with {len(self.indices)} samples. Data structure: {self.data_structure}")


    def _get_data_matrix(self):
        """
        Retrieves the feature matrix from `adata`.

        Returns:
            np.ndarray: The feature matrix as a NumPy array.

        Raises:
            KeyError: If the specified input layer is not found.
        """
        if self.input_layer_key == 'X':
            return self.adata.X.toarray() if issparse(self.adata.X) else self.adata.X
        else:
            return self.adata.layers[self.input_layer_key].toarray() if issparse(self.adata.layers[self.input_layer_key]) else \
            self.adata.layers[self.input_layer_key]


    def get_embedding(self, embedding_key, idx):
        """
        Retrieves embeddings for a given key and index.

        Args:
            embedding_key (str): The embedding key in `adata.obsm`.
            idx (int or list): Index or indices to retrieve.

        Returns:
            np.ndarray: The embedding matrix.

        Raises:
            ValueError: If the embedding key is not found.
        """
        if embedding_key == 'X':
            return self.adata.X.toarray()[idx]
        elif embedding_key in self.adata.obsm.key():
            return self.adata.obsm[embedding_key][idx]
        else:
            raise ValueError(f"Embedding key '{embedding_key}' not found in adata")


    def get_domain_labels(self, idx):
        """
        Retrieves the domain labels for a given index.

        Args:
            idx (int or list): Index or indices to retrieve.

        Returns:
            torch.Tensor: The domain labels.
        """
        if self.domain_labels is not None:
            return self.domain_labels[idx]
        return None
    

    def get_class_labels(self, idx):
        """
        Retrieves the class labels for a given index.

        Args:
            idx (int or list): Index or indices to retrieve.

        Returns:
            torch.Tensor: The class labels.
        """
        if self.class_labels is not None:
            return self.class_labels[idx]
        return None
    

    def __len__(self):
        """
        Returns the number of samples in the dataset.

        Returns:
            int: The dataset size.
        """
        return len(self.indices)
    

    def __getitem__(self, idx):
        """
        Retrieves the dataset items for the given index.

        Args:
            idx (int or list): The sample index.

        Returns:
            tuple: A tuple containing input data, domain labels, class labels, and covariate tensors.
        """
        if isinstance(idx, torch.Tensor):
            idx = idx.cpu().numpy()
        actual_idx = self.indices[idx]
        data_tensor = self.data[actual_idx]
        items = []
        for key in self.data_structure:
            if key == 'input':
                items.append(data_tensor)
            elif key == 'domain' and self.domain_labels is not None:
                items.append(self.domain_labels[actual_idx])
            elif key == 'class':
                items.append(self.class_labels[actual_idx])
            elif key in self.covariate_keys:
                items.append(self.covariate_tensors[key][actual_idx])
            elif key == 'idx':
                items.append(torch.tensor(actual_idx))

        return tuple(items)


    def shuffle_indices(self):
        """
        Shuffles dataset indices.
        """
        np.random.shuffle(self.indices)


    def subset(self, idx):
        """
        Creates a subset of the dataset with the given indices.

        Args:
            idx (list): Indices of the subset.

        Returns:
            AnnDataset: A new AnnDataset instance containing only the selected indices.
        """
        # Create a new AnnDataset with only the selected idx
        subset_adata = self.adata[idx].copy()
        return AnnDataset(subset_adata, self.input_layer_key, self.domain_key, self.class_key, self.covariate_keys, self.device)

