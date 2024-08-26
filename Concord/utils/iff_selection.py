import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time
from .knn import initialize_faiss_index, get_knn_indices
from .timer import Timer
import scanpy as sc
import pandas as pd
from typing import List, Optional, Union
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
               grouping: Union[str, pd.Series, List[str]] = 'cluster',
               cluster_min_cell_num=100,
               min_cluster_expr_fraction=0.1,
               emb_key='X_pca',
               k=512,
               knn_samples=100,
               use_faiss=True,
               use_ivf=True,
               n_top_genes=None,
               gini_cut=None,
               gini_cut_qt=None,
               figsize=(10, 3),
               save_path=None):

    # Check if grouping is a column in adata.obs
    if isinstance(grouping, str) and grouping in adata.obs:
        logger.info(f"Using {grouping} from adata.obs for clustering.")
        cluster_series = adata.obs[grouping]
    elif isinstance(grouping, str):
        if grouping not in ['cluster', 'knn']:
            raise ValueError("grouping must be either a column in adata.obs, 'cluster', 'knn', or a list of cluster labels.")
        else:
            if emb_key == "X_pca" and "X_pca" not in adata.obsm:
                logger.warning("X_pca does not exist in adata.obsm. Computing PCA.")
                sc.pp.highly_variable_genes(adata, n_top_genes=3000, flavor="seurat_v3")
                sc.tl.pca(adata, n_comps=50, use_highly_variable=True)
            elif emb_key is None or emb_key not in adata.obsm:
                raise ValueError(f"{emb_key} does not exist in adata.obsm.")
            else:
                logger.info(f"Using {emb_key} for computing {grouping}.")
            
            if grouping == 'cluster':
                emb = adata.obsm[emb_key]
                with Timer() as timer:
                    sc.pp.neighbors(adata, use_rep=emb_key)
                    sc.tl.leiden(adata)
                    cluster_series = pd.Series(adata.obs['leiden'])
                logger.info(f"Took {timer.interval:.2f} seconds to compute leiden cluster.")
    else:
        if len(grouping) != adata.shape[0]:
            raise ValueError("Length of grouping must match the number of cells.")
        cluster_series = pd.Series(grouping)

    # Compute KNN or clustering if not using an existing grouping from adata.obs
    expr_clus_frac = None
    if isinstance(grouping, str) and grouping == 'knn':
        emb = adata.obsm[emb_key]
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
        use_clus = cluster_series.value_counts()[cluster_series.value_counts() >= cluster_min_cell_num].index.tolist()
        expr_clus_frac = pd.DataFrame({
            cluster: (adata[cluster_series == cluster, :].X > 0).mean(axis=0).A1
            for cluster in use_clus
        }, index=adata.var_names)

    use_g = expr_clus_frac.index[expr_clus_frac.ge(min_cluster_expr_fraction).sum(axis=1) > 0]
    if (len(use_g) < n_top_genes):
        logger.warning(f"Number of features robustly detected is less than number of wanted top features: {n_top_genes}.")
        n_top_genes = len(use_g)
    logger.info(f"Selecting informative features from {len(use_g)} robustly detected features.")

    expr_clus_frac = expr_clus_frac.loc[use_g]

    with Timer() as timer:
        gene_clus_gini = expr_clus_frac.apply(gini_coefficient, axis=1)
    logger.info(f"Took {timer.interval:.2f} seconds to compute gini coefficient.")

    if gini_cut_qt is not None or gini_cut is not None:
        logger.info("Selecting informative features based on gini coefficient ...")
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
    elif n_top_genes is not None:
        include_g = gene_clus_gini.nlargest(n_top_genes).index.tolist()
        logger.info(f"Selected top {n_top_genes} genes based on gini coefficient.")
    else:
        raise ValueError("Either gini_cut_qt, gini_cut, or n_top_genes must be specified.")

    logger.info(f"Returning {len(include_g)} informative features.")

    return include_g

