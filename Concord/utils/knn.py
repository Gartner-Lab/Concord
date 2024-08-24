

import numpy as np
import faiss
from sklearn.neighbors import NearestNeighbors
import logging
import math
import torch
logger = logging.getLogger(__name__)

def initialize_faiss_index(emb, k, use_faiss=True, use_ivf=False, ivf_nprobe=10):
    """
    Initialize FAISS index for k-NN search.

    Parameters:
    emb: np.ndarray
        The embedding matrix.
    use_faiss: bool
        Whether to use FAISS for k-NN computation.
    use_ivf: bool
        Whether to use IVF FAISS index.
    ivf_nprobe: int
        Number of probes for IVF FAISS index.
    batch_size: int
        Batch size for training IVF FAISS index.

    Returns:
    faiss.Index or sklearn.neighbors.NearestNeighbors
        The initialized k-NN index.
    """
    index = None
    nbrs = None

    if np.isnan(emb).any():
        raise ValueError("There are NaN values in the emb array.")

    try:
        if hasattr(faiss, 'StandardGpuResources'):
            logger.warning(
                "FAISS GPU version is installed. Falling back to sklearn's NearestNeighbors. Please install FAISS CPU version by running 'pip install faiss-cpu'.")
            use_faiss = False
        else:
            emb = np.ascontiguousarray(emb)
            n = emb.shape[0]
            d = emb.shape[1]
            if use_ivf:
                logger.info(f"Building Faiss IVF index. nprobe={ivf_nprobe}")
                nlist = int(math.sqrt(n))  # number of clusters, based on https://github.com/facebookresearch/faiss/issues/112
                quantizer = faiss.IndexFlatL2(d)
                index = faiss.IndexIVFFlat(quantizer, d, nlist, faiss.METRIC_L2)
                index.train(emb)
                index.nprobe = ivf_nprobe
            else:
                logger.info("Building Faiss FlatL2 index.")
                index = faiss.IndexFlatL2(d)

            index.add(emb)
    except ImportError:
        logger.warning("FAISS is not available. Falling back to sklearn's NearestNeighbors.")
        use_faiss = False

    if not use_faiss:
        nbrs = NearestNeighbors(n_neighbors=k + 1).fit(emb)

    return index, nbrs, use_faiss


def get_knn_indices(emb, core_samples, k=10, use_faiss=True, index=None, nbrs=None):
    """
    Retrieve k-NN indices for the given samples.

    Parameters:
    emb: np.ndarray
        The embedding matrix.
    core_samples: np.ndarray
        The indices of core samples to find k-NN for.
    manifold_knn: int
        The number of nearest neighbors to retrieve.
    use_faiss: bool
        Whether to use FAISS for k-NN computation.
    index: faiss.Index
        The FAISS index for k-NN search.
    nbrs: sklearn.neighbors.NearestNeighbors
        The sklearn k-NN model for k-NN search.

    Returns:
    np.ndarray
        The indices of nearest neighbors.
    """
    if isinstance(core_samples, torch.Tensor) and core_samples.is_cuda:
        core_samples = core_samples.cpu().numpy()
    elif isinstance(core_samples, torch.Tensor):
        core_samples = core_samples.numpy()

    emb_samples = emb[core_samples]
    if emb_samples.ndim == 1:
        emb_samples = emb_samples.reshape(1, -1)
    if use_faiss and index is not None:
        _, indices = index.search(emb_samples.astype(np.float32), k + 1)
    else:
        _, indices = nbrs.kneighbors(emb_samples)
    return indices
