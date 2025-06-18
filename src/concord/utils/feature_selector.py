import anndata as ad
import numpy as np
from typing import List, Optional, Union
from pathlib import Path
import scanpy as sc
import pandas as pd
import logging
import matplotlib.pyplot as plt
import time
from ..model.knn import Neighborhood


logger = logging.getLogger(__name__)


# from https://stackoverflow.com/questions/48999542/more-efficient-weighted-gini-coefficient-in-python
def gini_coefficient(x):
    """
    Computes the Gini coefficient for a given array of values.

    Args:
        x (array-like): Input array of values.

    Returns:
        float: Gini coefficient, a measure of inequality in the distribution.
    """
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
               metric='euclidean',
               k=512,
               knn_samples=100,
               use_faiss=True,
               use_ivf=True,
               n_top_genes=None,
               gini_cut=None,
               gini_cut_qt=None,
               figsize=(10, 3),
               save_path=None):
    """
    Selects informative features using the Informative Feature Filtering (IFF) approach (Zhu et al. 2020, Blood).

    Args:
        adata (AnnData): AnnData object containing single-cell data.
        grouping (Union[str, pd.Series, List[str]], optional): Grouping strategy for feature selection. Can be:
            - A column name from `adata.obs` (e.g., 'cluster').
            - A list of cluster labels.
            - 'cluster' to compute clustering on the embedding.
            - 'knn' to use k-nearest neighbors.
            Defaults to 'cluster'.
        cluster_min_cell_num (int, optional): Minimum number of cells per cluster to be considered. Defaults to 100.
        min_cluster_expr_fraction (float, optional): Minimum fraction of cells expressing a gene in a cluster. Defaults to 0.1.
        emb_key (str, optional): Key in `adata.obsm` for the embedding used in clustering. Defaults to 'X_pca'.
        metric (str, optional): Distance metric for k-NN search. Defaults to 'euclidean'.
        k (int, optional): Number of neighbors for k-NN search if `grouping='knn'`. Defaults to 512.
        knn_samples (int, optional): Number of samples for k-NN computation. Defaults to 100.
        use_faiss (bool, optional): Whether to use FAISS for k-NN computation. Defaults to True.
        use_ivf (bool, optional): Whether to use Inverted File Index for FAISS. Defaults to True.
        n_top_genes (int, optional): Number of top genes to select. If None, determined by Gini coefficient filtering. Defaults to None.
        gini_cut (float, optional): Gini coefficient cutoff for feature selection. If None, `gini_cut_qt` is used. Defaults to None.
        gini_cut_qt (float, optional): Quantile threshold for selecting genes based on Gini coefficient. Defaults to None.
        figsize (tuple, optional): Size of the Gini coefficient histogram plot. Defaults to (10, 3).
        save_path (str, optional): Path to save the Gini coefficient histogram plot. Defaults to None.

    Returns:
        List[str]: List of selected informative feature names.
    """

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
                sc.pp.neighbors(adata, use_rep=emb_key)
                sc.tl.leiden(adata, resolution=1.0)
                cluster_series = pd.Series(adata.obs['leiden'])
    else:
        if len(grouping) != adata.shape[0]:
            raise ValueError("Length of grouping must match the number of cells.")
        cluster_series = pd.Series(grouping)

    # Compute KNN or clustering if not using an existing grouping from adata.obs
    expr_clus_frac = None
    if isinstance(grouping, str) and grouping == 'knn':
        emb = adata.obsm[emb_key]
        neighborhood = Neighborhood(emb=emb, k=k, use_faiss=use_faiss, use_ivf=use_ivf, metric=metric)
        core_samples = np.random.choice(np.arange(emb.shape[0]), size=min(knn_samples, emb.shape[0]), replace=False)
        knn_indices = neighborhood.get_knn(core_samples)
        expr_clus_frac = pd.DataFrame({
            f'knn_{i}': (adata[knn, :].X > 0).mean(axis=0).A1
            for i, knn in enumerate(knn_indices)
        }, index=adata.var_names)
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
    gene_clus_gini = expr_clus_frac.apply(gini_coefficient, axis=1)

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



def select_features(
    adata: ad.AnnData,
    n_top_features: int = 2000,
    flavor: str = "seurat_v3",
    filter_gene_by_counts: Union[int, bool] = False,
    normalize: bool = False,
    log1p: bool = False,
    grouping: Union[str, pd.Series, List[str]] = 'cluster',
    emb_key: str = 'X_pca',
    k: int = 512,
    knn_samples: int = 100,
    gini_cut_qt: float = None,
    save_path: Optional[Union[str, Path]] = None,
    figsize: tuple = (10, 3),
    subsample_frac: float = 1.0,
    random_state: int = 0
) -> List[str]:
    """
    Selects top informative features from an AnnData object.

    Args:
        adata (AnnData): AnnData object containing gene expression data.
        n_top_features (int, optional): Number of top features to select. Defaults to 2000.
        flavor (str, optional): Feature selection method. Options:
            - 'seurat_v3': Highly variable gene selection based on Seurat v3.
            - 'iff': Uses Informative Feature Filtering (IFF) method.
            Defaults to "seurat_v3".
        filter_gene_by_counts (Union[int, bool], optional): Minimum count threshold for feature filtering. Defaults to False.
        normalize (bool, optional): Whether to normalize the data before feature selection. Defaults to False.
        log1p (bool, optional): Whether to apply log1p transformation before feature selection. Defaults to False.
        grouping (Union[str, pd.Series, List[str]], optional): Clustering/grouping strategy for IFF method. Defaults to 'cluster'.
        emb_key (str, optional): Embedding key in `adata.obsm` used for clustering. Defaults to 'X_pca'.
        k (int, optional): Number of neighbors for k-NN if `grouping='knn'`. Defaults to 512.
        knn_samples (int, optional): Number of k-NN samples if `grouping='knn'`. Defaults to 100.
        gini_cut_qt (float, optional): Quantile threshold for selecting features by Gini coefficient in IFF. Defaults to None.
        save_path (Optional[Union[str, Path]], optional): Path to save Gini coefficient plot. Defaults to None.
        figsize (tuple, optional): Size of Gini coefficient plot. Defaults to (10, 3).
        subsample_frac (float, optional): Fraction of data to subsample for feature selection. Defaults to 1.0.
        random_state (int, optional): Random seed for reproducibility. Defaults to 0.

    Returns:
        List[str]: List of selected feature names.
    """
    # Subsample the data if too large
    sampled_indices=None
    if 0 < subsample_frac < 1.0:
        np.random.seed(random_state)
        sampled_indices = np.random.choice(adata.n_obs, int(subsample_frac * adata.n_obs), replace=False)
        sampled_size = len(sampled_indices)
    else:
        sampled_size = adata.n_obs

    if sampled_size > 100000:
        logger.warning(f"The number of cells for VEG selection ({sampled_size}) exceeds the limit of 100,000. "
                        f"Downsampling cells to 50,000 for VEG selection."
                        f"Note you can set subsample_frac to a value between 0 and 1 to control the number of cells.")
        sampled_indices = np.random.choice(adata.n_obs, 50000, replace=False)
        sampled_size = 50000
    
    # Handle backed mode and subsampling
    if adata.isbacked:
        logger.info("Converting backed AnnData object to memory...")
        sampled_data = adata[sampled_indices].to_memory() if sampled_indices is not None else adata.to_memory()
    else:
        sampled_data = adata[sampled_indices].copy() if sampled_indices is not None else adata.copy()

    # Filter features by counts
    if filter_gene_by_counts:
        logger.info("Filtering features by counts ...")
        sc.pp.filter_genes(
            sampled_data,
            min_counts=filter_gene_by_counts if isinstance(filter_gene_by_counts, int) else None,
        )

    # Normalize and log1p transform
    if normalize:
        logger.info("Normalizing total counts ...")
        sc.pp.normalize_total(sampled_data, target_sum=1e4)
    if log1p:
        logger.info("Log1p transforming for feature selection ...")
        sc.pp.log1p(sampled_data)

    if n_top_features is None or n_top_features > sampled_data.n_vars:
        logger.warning(f"n_top_features is set to {n_top_features}, which is larger than the number of features in the data.")
        n_top_features = sampled_data.n_vars
    
    # Determine features based on the flavor
    if flavor != "iff":
        logger.info(f"Selecting highly variable features with flavor {flavor}...")
        sc.pp.highly_variable_genes(sampled_data, n_top_genes=n_top_features, flavor=flavor)
        feature_list = sampled_data.var[sampled_data.var['highly_variable']].index.tolist()
    else:
        logger.info("Selecting informative features using IFF...")
        feature_list = iff_select(
            adata=sampled_data,
            grouping=grouping,
            emb_key=emb_key,
            k=k,
            knn_samples=knn_samples,
            n_top_genes=n_top_features,
            gini_cut_qt=gini_cut_qt,
            save_path=save_path,
            figsize=figsize
        )
    return feature_list




