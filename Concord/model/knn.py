

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
            try:
                import faiss
                if hasattr(faiss, 'StandardGpuResources'):
                    raise ImportError(
                        "FAISS GPU version is installed. Please install FAISS CPU version by running 'pip install faiss-cpu'.")
                else:
                    
                    n = self.emb.shape[0]
                    d = self.emb.shape[1]
                    if self.use_ivf:
                        if d > 3000:
                            logger.warning("FAISS IVF index is not recommended for data with too many features. Consider set use_ivf=False or set sampler_emb to PCA or other low dimensional embedding.")
                        logger.info(f"Building Faiss IVF index. nprobe={self.ivf_nprobe}")
                        nlist = int(math.sqrt(n))  # number of clusters, based on https://github.com/facebookresearch/faiss/issues/112
                        quantizer = faiss.IndexFlatL2(d)
                        self.index = faiss.IndexIVFFlat(quantizer, d, nlist, faiss.METRIC_L2)
                        self.index.train(self.emb)
                        self.index.nprobe = self.ivf_nprobe
                    else:
                        logger.info("Building Faiss FlatL2 index.")
                        self.index = faiss.IndexFlatL2(d)

                    self.index.add(self.emb)
            except ImportError:
                raise ImportError("FAISS is not available. Falling back to sklearn's NearestNeighbors.")
        else:
            self.nbrs = NearestNeighbors(n_neighbors=self.k + 1).fit(self.emb)


    def get_knn_indices(self, core_samples):
        """
        Retrieve k-NN indices for the given samples.

        Parameters:
        core_samples: np.ndarray or torch.Tensor
            The indices of core samples to find k-NN for.

        Returns:
        np.ndarray
            The indices of nearest neighbors.
        """
        if isinstance(core_samples, torch.Tensor) and core_samples.is_cuda:
            core_samples = core_samples.cpu().numpy()
        elif isinstance(core_samples, torch.Tensor):
            core_samples = core_samples.numpy()

        emb_samples = self.emb[core_samples]
        if emb_samples.ndim == 1:
            emb_samples = emb_samples.reshape(1, -1)

        if self.use_faiss and self.index is not None:
            _, indices = self.index.search(emb_samples.astype(np.float32), self.k + 1)
        else:
            _, indices = self.nbrs.kneighbors(emb_samples)
        return indices
    
    def update_embedding(self, new_emb):
        """
        Update the embedding matrix and reinitialize the k-NN index.

        Parameters:
        new_emb: np.ndarray
            The new embedding matrix.
        """
        self.emb = new_emb.astype(np.float32)
        self._initialize_knn_index()
