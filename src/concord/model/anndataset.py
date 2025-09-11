
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
    def __init__(self, adata, domain_key='domain', class_key=None, covariate_keys=None, preload_dense=False):
        """
        Initializes a lightweight AnnDataset that only manages labels and indices.
        """
        self.adata = adata
        self.domain_key = domain_key
        self.class_key = class_key

        self.preload_dense = preload_dense
        self.covariate_keys = list(covariate_keys) if covariate_keys is not None else []
        
        # --- MODE-SPECIFIC INITIALIZATION ---
        if self.preload_dense:
            # Get the data matrix and convert to a dense tensor
            data_matrix = self.adata.X.toarray() if issparse(self.adata.X) else self.adata.X
            self.data = torch.tensor(data_matrix, dtype=torch.float32)
        else:
            # For on-the-fly loading, self.data is not needed
            self.data = None

        # Store labels and covariates as tensors for easy slicing in collate_fn
        self.domain_labels = torch.tensor(self.adata.obs[self.domain_key].cat.codes.values, dtype=torch.long)
        
        if self.class_key:
            self.class_labels = torch.tensor(self.adata.obs[self.class_key].cat.codes.values, dtype=torch.long)
        else:
            self.class_labels = None

        self.covariate_tensors = {
            key: torch.tensor(self.adata.obs[key].cat.codes.values, dtype=torch.long)
            for key in self.covariate_keys # Now safely iterates over a list
        }
        
        self.indices = np.arange(len(self.adata))
        logger.info(f"Initialized lightweight dataset with {len(self.indices)} samples.")
    

        
    @staticmethod
    def _scipy_to_torch_sparse(matrix):
        """
        Converts a Scipy sparse matrix to a PyTorch sparse COO tensor.
        """
        if not issparse(matrix):
            raise TypeError("Input matrix must be a SciPy sparse matrix.")
        
        # Convert to COO format
        coo = matrix.tocoo()
        
        # Create indices and values tensors
        indices = torch.from_numpy(np.vstack((coo.row, coo.col)).astype(np.int64))
        values = torch.from_numpy(coo.data.astype(np.float32))
        
        return torch.sparse_coo_tensor(indices, values, torch.Size(coo.shape))


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
        return self.adata.n_obs
    

    def __getitem__(self, idx):
        if self.preload_dense:
            # Return a complete dictionary for the sample.
            # PyTorch's default collator will stack these dicts into a batch.
            batch = {
                'input': self.data[idx],
                'domain': self.domain_labels[idx],
                'idx': torch.tensor(idx, dtype=torch.long)
            }
            if self.class_labels is not None:
                batch['class'] = self.class_labels[idx]

            for key, tensor in self.covariate_tensors.items():
                batch[key] = tensor[idx]
            
            return batch
        else:
            # Return only the index for the custom collator to handle.
            return idx


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
        return AnnDataset(subset_adata, self.domain_key, self.class_key, self.covariate_keys, self.preload_dense)

