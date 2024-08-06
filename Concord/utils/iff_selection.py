import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time
from ..plotting.pl_enrichment import plot_go_enrichment
from .knn import initialize_faiss_index, get_knn_indices
from .timer import Timer
from .. import logger

# from https://stackoverflow.com/questions/48999542/more-efficient-weighted-gini-coefficient-in-python
def gini_coefficient(x):
    """Compute Gini coefficient of array of values"""
    sorted_x = np.sort(x)
    n = len(x)
    cumx = np.cumsum(sorted_x, dtype=float)
    # The above formula, with all weights equal to 1 simplifies to:
    return (n + 1 - 2 * np.sum(cumx) / cumx[-1]) / n


def iff_select(adata,
               cluster=None,
               cluster_min_cell_num=100,
               min_cluster_expr_fraction=0.1,
               use_knn=False,
               knn_emb_key = 'X_pca',
               k=512,
               knn_samples = 100,
               use_faiss=True,
               use_ivf=True,
               n_top_genes=None,
               gini_cut=None,
               gini_cut_qt=0.75,
               figsize=(10, 3),
               save_path=None):
    if not use_knn and (cluster is None or adata.shape[0] != len(cluster)):
        raise ValueError("Cell number does not match cluster length or cluster is None.")

    if use_knn:
        if knn_emb_key == 'X':
            emb = adata.X
        else:
            if knn_emb_key not in adata.obsm:
                raise ValueError(f"{knn_emb_key} does not exist in adata.obsm.")
            emb = adata.obsm[knn_emb_key]
        with Timer() as timer:
            index, nbrs, use_faiss_flag = initialize_faiss_index(emb, k, use_faiss=use_faiss, use_ivf=use_ivf)
            core_samples = np.random.choice(np.arange(emb.shape[0]), size=min(knn_samples, emb.shape[0]), replace=False)
            knn_indices = get_knn_indices(emb, core_samples, k=k, use_faiss=use_faiss_flag, index=index, nbrs=nbrs)
            expr_clus_frac = pd.DataFrame({
                f'knn_{i}': (adata[knn, :].X > 0).mean(axis=0).A1
                for i, knn in enumerate(knn_indices)
            }, index=adata.var_names)
        logger.info(f"Took {timer.interval:.2f} seconds to compute neighborhood.")
    else:
        cluster_series = pd.Series(cluster)
        use_clus = cluster_series.value_counts()[cluster_series.value_counts() >= cluster_min_cell_num].index.tolist()
        expr_clus_frac = pd.DataFrame({
            cluster: (adata[cluster_series == cluster, :].X > 0).mean(axis=0).A1
            for cluster in use_clus
        }, index=adata.var_names)

    use_g = expr_clus_frac.index[expr_clus_frac.ge(min_cluster_expr_fraction).sum(axis=1) > 0]
    logger.info(f"Selecting informative features from {len(use_g)} robustly detected features.")

    expr_clus_frac = expr_clus_frac.loc[use_g]

    with Timer() as timer:
        gene_clus_gini = expr_clus_frac.apply(gini_coefficient, axis=1)
    logger.info(f"Took {timer.interval:.2f} seconds to compute gini coefficient.")

    if n_top_genes is not None:
        include_g = gene_clus_gini.nlargest(n_top_genes).index.tolist()
        logger.info(f"Selected top {n_top_genes} genes based on gini coefficient.")
    else:
        if gini_cut is None:
            gini_cut = gene_clus_gini.quantile(gini_cut_qt)
            logger.info(f"Cut at gini quantile {gini_cut_qt} with value {gini_cut:.3f}")
        else:
            logger.info(f"Cut at gini value {gini_cut}")

        plt.figure(figsize=figsize)
        gene_clus_gini.hist(bins=100)
        plt.axvline(gini_cut, color='red', linestyle='dashed', linewidth=3)
        if save_path:
            file_suffix = f"{time.strftime('%b%d-%H%M')}"
            plt.savefig(f"{save_path}feature_gini_hist_{file_suffix}.png")
        else:
            plt.show()
        plt.close()

        include_g = gene_clus_gini[gene_clus_gini >= gini_cut].index.tolist()

    logger.info(f"Returning {len(include_g)} informative features.")

    return include_g


