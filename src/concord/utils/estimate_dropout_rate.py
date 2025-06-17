
import scanpy as sc
import numpy as np
from scipy.sparse import issparse
import matplotlib.pyplot as plt
import seaborn as sns
from ..model.knn import Neighborhood
from .. import logger

def estimate_dropout_rate(adata, k=3, nbr_emb = 'X_pca',  metric='euclidean', n_samples = 3000, use_faiss=False, use_ivf=False, ivf_nprobe=8, return_mean=True, plotting=False):

    if nbr_emb not in adata.obsm:
        raise ValueError(f"Embedding '{nbr_emb}' not found in adata.obsm.")
    else:
        logger.info(f"Using existing embedding '{nbr_emb}' from adata.obsm")
        emb = adata.obsm[nbr_emb].astype(np.float32)

    # Initialize KNN
    neighborhood = Neighborhood(emb=emb, k=k, use_faiss=use_faiss, use_ivf=use_ivf, ivf_nprobe=ivf_nprobe, metric=metric)

    if n_samples >= adata.shape[0]:
        core_samples = np.arange(adata.shape[0])
    else:
        core_samples = np.random.choice(emb.shape[0], min(n_samples, emb.shape[0]), replace=False)

    X = adata.X.toarray() if issparse(adata.X) else adata.X
    avg_distances = neighborhood.average_knn_distance(core_samples, X, k=k, distance_metric='drop_diff')

    if plotting:
        plt.figure(figsize=(6, 5))
        sns.histplot(avg_distances, bins=50, kde=True, color='skyblue', edgecolor='black')
        plt.title('Histogram of Average Feature Drop Rate of Nearest Neighbors')
        plt.xlabel('Average Feature Drop Rate')
        plt.ylabel('Number of Cells')
        plt.grid(True)
        plt.tight_layout()
        plt.show()

    logger.info(f"Average feature drop rate of nearest neighbors: {avg_distances.mean()}")
    
    if return_mean:
        return avg_distances.mean()
    else:
        return avg_distances

