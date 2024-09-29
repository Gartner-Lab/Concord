

import numpy as np
from sklearn.neighbors import NearestNeighbors
import logging
import math
import torch
logger = logging.getLogger(__name__)


class Neighborhood:
    def __init__(self, emb, k=10, use_faiss=True, use_ivf=False, ivf_nprobe=10):
        """
        Initialize the Neighborhood class.

        Parameters:
        emb: np.ndarray
            The embedding matrix.
        k: int
            The number of nearest neighbors to retrieve.
        use_faiss: bool
            Whether to use FAISS for k-NN computation.
        use_ivf: bool
            Whether to use IVF FAISS index.
        ivf_nprobe: int
            Number of probes for IVF FAISS index.
        """
        if np.isnan(emb).any():
            raise ValueError("There are NaN values in the emb array.")
        self.emb = np.ascontiguousarray(emb).astype(np.float32)
        self.k = k
        self.use_faiss = use_faiss
        if self.use_faiss:
            try:
                import faiss
                if hasattr(faiss, 'StandardGpuResources'):
                    logger.info("Using FAISS GPU index.")
                    self.faiss_gpu = True
                else:
                    logger.info("Using FAISS CPU index.")
                    self.faiss_gpu = False
            except ImportError:
                logger.warning("FAISS not found. Using sklearn for k-NN computation.")
                self.use_faiss = False

        self.use_ivf = use_ivf
        self.ivf_nprobe = ivf_nprobe

        self.index = None
        self.nbrs = None

        self._build_knn_index()

    def _build_knn_index(self):
        """
        Initialize the k-NN index using FAISS or sklearn.
        """
        if self.use_faiss:
            import faiss
            n = self.emb.shape[0]
            d = self.emb.shape[1]
            
            if self.use_ivf:
                if d > 3000:
                    logger.warning("FAISS IVF index is not recommended for data with too many features. Consider set use_ivf=False or set sampler_emb to PCA or other low dimensional embedding.")
                logger.info(f"Building Faiss IVF index. nprobe={self.ivf_nprobe}")
                nlist = int(math.sqrt(n))  # number of clusters
                quantizer = faiss.IndexFlatL2(d)
                index_cpu = faiss.IndexIVFFlat(quantizer, d, nlist, faiss.METRIC_L2)
                index_cpu.train(self.emb)
                index_cpu.nprobe = self.ivf_nprobe
            else:
                logger.info("Building Faiss FlatL2 index.")
                index_cpu = faiss.IndexFlatL2(d)

            # Check if GPU is available and use it if possible
            if self.faiss_gpu:
                res = faiss.StandardGpuResources()
                self.index = faiss.index_cpu_to_gpu(res, 0, index_cpu)
            else:
                logger.info("Using FAISS CPU index.")
                self.index = index_cpu

            self.index.add(self.emb)
        else:
            self.nbrs = NearestNeighbors(n_neighbors=self.k + 1).fit(self.emb)


    def get_knn_indices(self, core_samples, k=None, include_self=True):
        """
        Retrieve k-NN indices for the given samples.

        Parameters:
        core_samples: np.ndarray or torch.Tensor
            The indices of core samples to find k-NN for.
        k: int, optional
            Number of neighbors to retrieve. If None, uses the default self.k
        include_self: bool, default True
            Whether to include the sample itself in the returned neighbors.

        Returns:
        np.ndarray
            The indices of nearest neighbors.
        """
        if k is None:
            k = self.k
        if isinstance(core_samples, torch.Tensor):
            core_samples = core_samples.cpu().numpy()

        if isinstance(core_samples, torch.Tensor) and core_samples.is_cuda:
            core_samples = core_samples.cpu().numpy()
        elif isinstance(core_samples, torch.Tensor):
            core_samples = core_samples.numpy()

        emb_samples = self.emb[core_samples]
        if emb_samples.ndim == 1:
            emb_samples = emb_samples.reshape(1, -1)

        n_neighbors = k
        if not include_self:
            n_neighbors += 1  # Retrieve an extra neighbor to exclude self later

        if self.use_faiss and self.index is not None:
            _, indices = self.index.search(emb_samples.astype(np.float32), n_neighbors)
        else:
            _, indices = self.nbrs.kneighbors(emb_samples, n_neighbors=n_neighbors)

        if not include_self:
            # Exclude the sample itself from the neighbors
            core_samples_expanded = core_samples.reshape(-1, 1)
            mask = indices != core_samples_expanded
            indices_excl_self = indices[mask].reshape(len(core_samples), -1)
            # Return only the top k neighbors excluding the sample itself
            indices = indices_excl_self[:, :k]

        return indices

    
    def update_embedding(self, new_emb):
        """
        Update the embedding matrix and reinitialize the k-NN index.

        Parameters:
        new_emb: np.ndarray
            The new embedding matrix.
        """
        self.emb = new_emb.astype(np.float32)
        self._build_knn_index()


    def average_knn_distance(self, core_samples, mtx, k=None, distance_metric='euclidean'):
        """
        Compute the average distance to the k-th nearest neighbor for each sample.

        Parameters:
        core_samples: np.ndarray
            The indices of core samples.
        mtx: np.ndarray
            The matrix to compute the distance to.
        k: int, optional
            Number of neighbors to retrieve. If None, uses the default self.k
        distance_metric: str, optional
            The distance metric to use: 'euclidean' or 'set_diff'.

        Returns:
        np.ndarray
            The average distance to the k-th nearest neighbor for each sample.
        """
        if k is None:
            k = self.k

        assert(self.emb.shape[0] == mtx.shape[0])

        # Get the indices of k nearest neighbors for the core samples
        indices = self.get_knn_indices(core_samples, k=k, include_self=False)

        # Compute the Euclidean distance
        if distance_metric == 'euclidean':
            return np.mean(np.linalg.norm(mtx[core_samples][:, np.newaxis] - mtx[indices], axis=-1), axis=1)
        # Compute the positive set difference (binary difference) using XOR
        elif distance_metric == 'set_diff':
            return np.mean(np.mean(np.logical_xor(mtx[core_samples][:, np.newaxis] > 0, mtx[indices] > 0), axis=-1), axis=-1)
        elif distance_metric == 'drop_diff':
            core_nonzero = (mtx[core_samples] > 0)  # Boolean array for non-zero genes in core cells
            neighbor_nonzero = (mtx[indices] > 0)   # Boolean array for non-zero genes in neighbors

            # Find where genes are on in core cell but off in nearest neighbor
            turned_off = core_nonzero[:, np.newaxis] & ~neighbor_nonzero

            # Count how many genes are turned off and how many were originally on in the core cells
            num_turned_off = np.sum(turned_off, axis=-1)  # Number of genes turned off
            num_positive_in_core = np.sum(core_nonzero, axis=-1)  # Number of non-zero genes in core cells

            # Compute fraction (turned off / positive genes in core), avoiding division by zero
            fraction_turned_off = np.divide(num_turned_off, num_positive_in_core[:, np.newaxis], where=num_positive_in_core[:, np.newaxis] != 0)

            return np.mean(fraction_turned_off, axis=-1)
        else:
            raise ValueError(f"Unknown distance metric: {distance_metric}")


