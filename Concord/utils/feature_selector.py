import anndata as ad
import numpy as np
from typing import List, Optional, Union
from pathlib import Path
import scanpy as sc
import logging

logger = logging.getLogger(__name__)

def select_features(
    adata: ad.AnnData,
    n_top_genes: int = 2000,
    flavor: str = "seurat_v3",
    filter_gene_by_counts: Union[int, bool] = False,
    cluster: Optional[np.ndarray] = None,
    use_knn: bool = True,
    knn_emb_key: str = 'X_pca',
    k: int = 512,
    knn_samples: int = 100,
    gini_cut_qt: float = 0.75,
    save_path: Optional[Union[str, Path]] = None,
    figsize: tuple = (10, 3),
    subsample_frac: float = 0.1,
    random_state: int = 0
) -> List[str]:
    # Subsample the data if too large
    if subsample_frac < 1.0:
        np.random.seed(random_state)
        sampled_indices = np.random.choice(adata.n_obs, int(subsample_frac * adata.n_obs), replace=False)
        sampled_data = adata[sampled_indices].copy()
    else:
        sampled_data = adata.copy()

    # Filter genes by counts
    if filter_gene_by_counts:
        logger.info("Filtering genes by counts ...")
        sc.pp.filter_genes(
            sampled_data,
            min_counts=filter_gene_by_counts if isinstance(filter_gene_by_counts, int) else None,
        )

    # Normalize and log1p transform
    logger.info("Normalizing total counts for feature selection ...")
    sc.pp.normalize_total(sampled_data, target_sum=1e4)
    logger.info("Log1p transforming for feature selection ...")
    sc.pp.log1p(sampled_data)

    # Determine features based on the flavor
    if flavor != "iff":
        logger.info(f"Selecting highly variable genes with flavor {flavor}...")
        sc.pp.highly_variable_genes(sampled_data, n_top_genes=n_top_genes, flavor=flavor)
        feature_list = sampled_data.var[sampled_data.var['highly_variable']].index.tolist()
    else:
        logger.info("Selecting informative features using IFF...")
        if n_top_genes is not None:
            logger.warning("It is recommended to set n_top_genes to None and use gini_cut_qt for IFF selection.")
        if knn_emb_key == "X_pca" and "X_pca" not in sampled_data.obsm:
            logger.warning("X_pca does not exist in adata.obsm. Computing PCA.")
            sc.pp.highly_variable_genes(sampled_data, n_top_genes=3000, flavor="seurat_v3")
            sc.tl.pca(sampled_data, n_comps=50, use_highly_variable=True)
        feature_list = iff_select(
            adata=sampled_data,
            cluster=cluster,
            use_knn=use_knn,
            knn_emb_key=knn_emb_key,
            k=k,
            knn_samples=knn_samples,
            n_top_genes=n_top_genes,
            gini_cut_qt=gini_cut_qt,
            save_path=save_path,
            figsize=figsize
        )
    return feature_list




