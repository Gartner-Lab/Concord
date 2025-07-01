

import numpy as np
from sklearn.neighbors import NearestNeighbors
import logging
import math
import torch
import logging

PKG_ROOT = __package__.split(".")[0]          # → "concord"
logger   = logging.getLogger(f"{PKG_ROOT}")


class Neighborhood:
    """
    A class for k-nearest neighbor (k-NN) computation using either FAISS or sklearn.

    This class constructs a k-NN index, retrieves neighbors, and computes distances
    between embeddings.

    Attributes:
        emb (np.ndarray): The embedding matrix (converted to float32).
        k (int): Number of nearest neighbors to retrieve.
        use_faiss (bool): Whether to use FAISS for k-NN computation.
        use_ivf (bool): Whether to use IVF indexing in FAISS.
        ivf_nprobe (int): Number of probes for FAISS IVF.
        metric (str): Distance metric ('euclidean' or 'cosine').
    """
    def __init__(
        self, 
        emb,
        k=10,
        use_faiss=True,
        use_ivf=False,
        ivf_nprobe=10,
        metric='euclidean'
    ):
        """
        Initializes the Neighborhood class.

        Args:
            emb (np.ndarray): The embedding matrix.
            k (int, optional): Number of nearest neighbors to retrieve. Defaults to 10.
            use_faiss (bool, optional): Whether to use FAISS for k-NN computation. Defaults to True.
            use_ivf (bool, optional): Whether to use IVF FAISS index. Defaults to False.
            ivf_nprobe (int, optional): Number of probes for FAISS IVF. Defaults to 10.
            metric (str, optional): Distance metric ('euclidean' or 'cosine'). Defaults to 'euclidean'.

        Raises:
            ValueError: If there are NaN values in the embedding or if the metric is invalid.
        """
        if np.isnan(emb).any():
            raise ValueError("There are NaN values in the emb array.")

        # Store the original embedding in float32
        self.emb = np.ascontiguousarray(emb).astype(np.float32)

        self.k = k
        self.use_faiss = use_faiss
        self.use_ivf = use_ivf
        self.ivf_nprobe = ivf_nprobe

        # Validate metric
        if metric not in ("euclidean", "cosine"):
            raise ValueError("`metric` must be 'euclidean' or 'cosine'.")
        self.metric = metric
        logger.info(f"Using {metric} distance metric.")

        # Check FAISS availability
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

        # Internal members
        self.index = None
        self.nbrs = None
        self.graph = None

        # Build the index
        self._build_knn_index()


    def _build_knn_index(self):
        """
        Initializes the k-NN index using FAISS or sklearn.
        """
        # If metric is cosine, normalize the embeddings so that
        # L2 distance in normalized space ~ cosine distance
        if self.metric == "cosine":
            # Avoid dividing by zero
            norms = np.linalg.norm(self.emb, axis=1, keepdims=True)
            norms[norms == 0] = 1e-12
            self.emb = self.emb / norms

        if self.use_faiss:
            import faiss
            n, d = self.emb.shape

            if self.use_ivf:
                if d > 3000:
                    logger.warning(
                        "FAISS IVF index is not recommended for data with too many "
                        "features. Consider set use_ivf=False or reduce dimensionality."
                    )
                logger.info(f"Building Faiss IVF index. nprobe={self.ivf_nprobe}")
                nlist = int(math.sqrt(n))  # number of clusters
                quantizer = faiss.IndexFlatL2(d)
                index_cpu = faiss.IndexIVFFlat(quantizer, d, nlist, faiss.METRIC_L2)
                index_cpu.train(self.emb)
                index_cpu.nprobe = self.ivf_nprobe
            else:
                logger.info("Building Faiss FlatL2 index.")
                index_cpu = faiss.IndexFlatL2(d)

            # Use GPU if available
            if self.faiss_gpu:
                res = faiss.StandardGpuResources()
                self.index = faiss.index_cpu_to_gpu(res, 0, index_cpu)
            else:
                logger.info("Using FAISS CPU index.")
                self.index = index_cpu

            self.index.add(self.emb)

        else:
            # Use sklearn NearestNeighbors
            if self.metric == "cosine":
                self.nbrs = NearestNeighbors(n_neighbors=self.k + 1, metric="cosine").fit(self.emb)
            else:
                # Euclidean
                self.nbrs = NearestNeighbors(n_neighbors=self.k + 1, metric="euclidean").fit(self.emb)


    def get_knn(self, core_samples, k=None, include_self=True, return_distance=False):
        """
        Retrieves the k-nearest neighbors for given samples.

        Args:
            core_samples (np.ndarray or torch.Tensor): Indices of samples for which k-NN is retrieved.
            k (int, optional): Number of neighbors. Defaults to self.k.
            include_self (bool, optional): Whether to include the sample itself. Defaults to True.
            return_distance (bool, optional): Whether to return distances. Defaults to False.

        Returns:
            np.ndarray: Indices of nearest neighbors (and distances if return_distance=True).
        """
        if k is None:
            k = self.k

        # Convert torch.Tensor to NumPy if needed
        if isinstance(core_samples, torch.Tensor):
            core_samples = core_samples.cpu().numpy()

        emb_samples = self.emb[core_samples]
        # Reshape if it's a single sample
        if emb_samples.ndim == 1:
            emb_samples = emb_samples.reshape(1, -1)

        # We'll find k neighbors, but if we're excluding self we temporarily get k+1
        n_neighbors = k
        if not include_self:
            n_neighbors += 1

        # FAISS path
        if self.use_faiss and self.index is not None:
            distances, indices = self.index.search(emb_samples.astype(np.float32), n_neighbors)

            # If we are using cosine, convert the L2 distance of normalized vectors into actual "cosine distance"
            #   L2 distance on normalized vectors: dist_l2 = sqrt(2 * (1 - cos))
            #   => dist_l2^2 = 2 * (1 - cos) => 1 - cos = dist_l2^2 / 2
            #   => cos_dist = 1 - cos = dist_l2^2 / 2
            if self.metric == "cosine" and return_distance:
                distances = (distances ** 2) / 2

        # sklearn path
        else:
            distances, indices = self.nbrs.kneighbors(emb_samples, n_neighbors=n_neighbors)

        # Optionally exclude the sample itself from the neighbors
        if not include_self:
            # Exclude the sample's own index
            core_samples_expanded = core_samples.reshape(-1, 1)
            mask = (indices != core_samples_expanded)

            mask_sum = mask.sum(axis=1)
            expected_sum = n_neighbors - 1

            if np.any(mask_sum < expected_sum):
                raise ValueError("Mask inconsistency: fewer neighbors than expected after exclusion.")

            if np.any(mask_sum > expected_sum):
                logger.warning("Mask inconsistency: more than expected neighbors found.")
                rows_to_fix = np.where(mask_sum > expected_sum)[0]
                for row in rows_to_fix:
                    self_pos = np.where(indices[row] == core_samples[row])[0]
                    if len(self_pos) > 0:
                        # set the self position to False in the mask
                        mask[row, self_pos[0]] = False
                    else:
                        # if no self position found, remove a random neighbor
                        mask[row, np.random.choice(n_neighbors)] = False

            indices_excl_self = indices[mask].reshape(len(core_samples), -1)
            distances_excl_self = distances[mask].reshape(len(core_samples), -1)

            # keep top-k neighbors
            indices = indices_excl_self[:, :k]
            distances = distances_excl_self[:, :k]

        if return_distance:
            return indices, distances
        return indices


    def update_embedding(self, new_emb):
        """
        Updates the embedding matrix and rebuilds the k-NN index.

        Args:
            new_emb (np.ndarray): The new embedding matrix.
        """
        self.emb = new_emb.astype(np.float32)
        self._build_knn_index()


    def average_knn_distance(self, core_samples, mtx, k=None, distance_metric='euclidean'):
        """
        Compute the average distance to the k-th nearest neighbor for each sample.

        Parameters
        ----------
        core_samples : np.ndarray
            The indices of core samples.
        mtx : np.ndarray
            The matrix to compute the distance to.
        k : int, optional
            Number of neighbors to retrieve. If None, uses self.k.
        distance_metric : str
            Distance metric to use: 'euclidean', 'set_diff', or 'drop_diff'.

        Returns
        -------
        np.ndarray
            The average distance to the k-th nearest neighbor for each sample.
        """
        if k is None:
            k = self.k

        assert(self.emb.shape[0] == mtx.shape[0])

        # Get KNN indices (excl. self)
        indices = self.get_knn(core_samples, k=k, include_self=False)

        if distance_metric == 'euclidean':
            # L2 distance
            return np.mean(np.linalg.norm(mtx[core_samples][:, np.newaxis] - mtx[indices], axis=-1), axis=1)

        elif distance_metric == 'set_diff':
            # positive set difference (binary difference) using XOR
            return np.mean(
                np.mean(
                    np.logical_xor(
                        mtx[core_samples][:, np.newaxis] > 0, 
                        mtx[indices] > 0
                    ), axis=-1
                ), axis=-1
            )

        elif distance_metric == 'drop_diff':
            # fraction of genes that "turn off" from core cell to neighbors
            core_nonzero = (mtx[core_samples] > 0)
            neighbor_nonzero = (mtx[indices] > 0)

            turned_off = core_nonzero[:, np.newaxis] & ~neighbor_nonzero
            num_turned_off = np.sum(turned_off, axis=-1)
            num_positive_in_core = np.sum(core_nonzero, axis=-1)

            fraction_turned_off = np.divide(
                num_turned_off,
                num_positive_in_core[:, np.newaxis],
                where=num_positive_in_core[:, np.newaxis] != 0
            )
            return np.mean(fraction_turned_off, axis=-1)

        else:
            raise ValueError(f"Unknown distance metric: {distance_metric}")


    def compute_knn_graph(self, k=None):
        """
        Constructs a sparse adjacency matrix for the k-NN graph.

        Args:
            k (int, optional): Number of neighbors. Defaults to self.k.
        """
        from scipy.sparse import csr_matrix

        if k is None:
            k = self.k

        core_samples = np.arange(self.emb.shape[0])
        indices, distances = self.get_knn(
            core_samples, k=k, include_self=False, return_distance=True
        )

        rows = np.repeat(np.arange(self.emb.shape[0]), k)
        cols = indices.flatten()
        weights = distances.flatten()

        self.graph = csr_matrix((weights, (rows, cols)), shape=(self.emb.shape[0], self.emb.shape[0]))


    def get_knn_graph(self):
        """
        Returns the precomputed k-NN graph. Computes it if not available.

        Returns:
            scipy.sparse.csr_matrix: Sparse adjacency matrix of shape (n_samples, n_samples).
        """
        if self.graph is None:
            logger.warning("K-NN graph is not computed. Computing now.")
            self.compute_knn_graph()
        return self.graph
