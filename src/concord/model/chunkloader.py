import torch
import numpy as np
from .dataloader import DataLoaderManager

class ChunkLoader:
    """
    A class for handling large datasets in chunks for efficient training.

    This class manages chunk-based data loading for large single-cell datasets, 
    allowing training in smaller subsets without loading the entire dataset into memory.

    Attributes:
        adata (AnnData): The annotated data matrix.
        input_layer_key (str): Key for the input layer in `adata`.
        domain_key (str): Key for domain labels in `adata.obs`.
        class_key (str, optional): Key for class labels in `adata.obs`.
        covariate_keys (list, optional): List of keys for covariates in `adata.obs`.
        chunk_size (int): The number of samples per chunk.
        batch_size (int): The batch size used for training.
        train_frac (float): Fraction of data to be used for training.
        sampler_mode (str): Mode for sampling (e.g., 'domain').
        sampler_knn (int): Number of nearest neighbors for k-NN-based sampling.
        emb_key (str, optional): Key for embedding space used in sampling.
        use_faiss (bool): Whether to use FAISS for k-NN computation.
        use_ivf (bool): Whether to use an IVF FAISS index.
        ivf_nprobe (int): Number of probes for IVF FAISS index.
        class_weights (dict, optional): Weights for balancing class sampling.
        p_intra_knn (float): Probability of sampling within k-NN.
        p_intra_domain (float): Probability of sampling within the same domain.
        p_intra_class (float): Probability of sampling within the same class.
        drop_last (bool): Whether to drop the last batch if it is smaller than batch_size.
        preprocess (callable, optional): Preprocessing function to apply to the dataset.
        device (torch.device): Device on which to load data (CPU or CUDA).
        total_samples (int): Total number of samples in the dataset.
        num_chunks (int): Number of chunks required to load the full dataset.
        indices (np.ndarray): Array of shuffled indices for chunking.
        data_structure (list, optional): Structure of the dataset.

    Methods:
        __len__: Returns the number of chunks.
        _shuffle_indices: Randomly shuffles dataset indices.
        _load_chunk: Loads a specific chunk of data.
        __iter__: Initializes the chunk iterator.
        __next__: Retrieves the next chunk of data.
    """
    def __init__(self, adata, input_layer_key, domain_key, 
                 class_key=None, covariate_keys=None,
                 chunk_size=10000, batch_size=32, train_frac=0.9,
                 sampler_mode="domain",
                 emb_key=None,
                 sampler_knn=300, p_intra_knn=0.3, p_intra_domain=1.0,
                 use_faiss=True, use_ivf=False, ivf_nprobe=8,
                 class_weights=None, p_intra_class=0.3, drop_last=True,
                 preprocess=None, device=None):
        """
        Initializes the ChunkLoader.

        Args:
            adata (AnnData): The annotated data matrix.
            input_layer_key (str): Key for the input layer in `adata`.
            domain_key (str): Key for domain labels in `adata.obs`.
            class_key (str, optional): Key for class labels in `adata.obs`. Default is None.
            covariate_keys (list, optional): List of covariate keys in `adata.obs`. Default is None.
            chunk_size (int, optional): Number of samples per chunk. Default is 10,000.
            batch_size (int, optional): Batch size used in training. Default is 32.
            train_frac (float, optional): Fraction of data for training. Default is 0.9.
            sampler_mode (str, optional): Sampling mode ('domain', etc.). Default is "domain".
            emb_key (str, optional): Key for the embedding space used in sampling. Default is None.
            sampler_knn (int, optional): Number of nearest neighbors for k-NN sampling. Default is 300.
            p_intra_knn (float, optional): Probability of sampling within k-NN. Default is 0.3.
            p_intra_domain (float, optional): Probability of sampling within the same domain. Default is 1.0.
            use_faiss (bool, optional): Whether to use FAISS for k-NN. Default is True.
            use_ivf (bool, optional): Whether to use an IVF FAISS index. Default is False.
            ivf_nprobe (int, optional): Number of probes for IVF FAISS index. Default is 8.
            class_weights (dict, optional): Dictionary of class weights for balancing. Default is None.
            p_intra_class (float, optional): Probability of sampling within the same class. Default is 0.3.
            drop_last (bool, optional): Whether to drop the last batch if it is smaller than `batch_size`. Default is True.
            preprocess (callable, optional): Function to preprocess the dataset. Default is None.
            device (torch.device, optional): Device on which to load data (CPU/GPU). Default is CUDA if available.
        """
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
        """
        Returns the total number of chunks.

        Returns:
            int: Number of chunks.
        """
        return self.num_chunks

    def _shuffle_indices(self):
        """
        Randomly shuffles dataset indices for chunking.
        """
        np.random.shuffle(self.indices)

    # Future todo: Allow random sampling of indices for each chunk, and allow sampling based on global distance
    def _load_chunk(self, chunk_idx):
        """
        Loads a specific chunk of data.

        Args:
            chunk_idx (int): Index of the chunk to load.

        Returns:
            tuple: Training DataLoader, Validation DataLoader, and chunk indices.
        """
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
        """
        Initializes the chunk iterator.

        Returns:
            ChunkLoader: The chunk loader object itself.
        """
        self.current_chunk_idx = 0
        self._shuffle_indices()
        return self

    def __next__(self):
        """
        Retrieves the next chunk of data.

        Returns:
            tuple: Training DataLoader, Validation DataLoader, and chunk indices.

        Raises:
            StopIteration: If all chunks have been iterated over.
        """
        if self.current_chunk_idx >= self.num_chunks:
            raise StopIteration
        train_dataloader, val_dataloader, chunk_indices = self._load_chunk(self.current_chunk_idx)
        self.current_chunk_idx += 1
        return train_dataloader, val_dataloader, chunk_indices
