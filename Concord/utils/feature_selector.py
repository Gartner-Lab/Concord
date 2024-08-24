import anndata as ad
import numpy as np
from typing import List, Optional, Union
from pathlib import Path
import scanpy as sc
import pandas as pd
import logging
from . import iff_select

logger = logging.getLogger(__name__)

def select_features(
    adata: ad.AnnData,
    n_top_genes: int = 2000,
    flavor: str = "seurat_v3",
    filter_gene_by_counts: Union[int, bool] = False,
    normalize: bool = True,
    log1p: bool = True,
    grouping: Union[str, pd.Series, List[str]] = 'cluster',
    emb_key: str = 'X_pca',
    k: int = 512,
    knn_samples: int = 100,
    gini_cut_qt: float = 0.75,
    save_path: Optional[Union[str, Path]] = None,
    figsize: tuple = (10, 3),
    subsample_frac: float = 1.0,
    random_state: int = 0
) -> List[str]:
    # Subsample the data if too large
    sampled_indices=None
    if 0 < subsample_frac < 1.0:
        np.random.seed(random_state)
        sampled_indices = np.random.choice(adata.n_obs, int(subsample_frac * adata.n_obs), replace=False)
        sampled_size = len(sampled_indices)
    else:
        sampled_size = adata.n_obs

    if sampled_size > 100000:
        raise ValueError(f"The number of cells for VEG selection ({sampled_data.n_obs}) exceeds the limit of 100,000. "
                        f"Please specify a lower subsample_frac value to downsample cells for VEG calling.")
    
    # Handle backed mode and subsampling
    if adata.isbacked:
        logger.info("Converting backed AnnData object to memory...")
        sampled_data = adata[sampled_indices].to_memory() if sampled_indices is not None else adata.to_memory()
    else:
        sampled_data = adata[sampled_indices].copy() if sampled_indices is not None else adata.copy()

    # Filter genes by counts
    if filter_gene_by_counts:
        logger.info("Filtering genes by counts ...")
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

    if n_top_genes is None or n_top_genes > sampled_data.n_vars:
        logger.warning(f"n_top_genes is set to {n_top_genes}, which is larger than the number of genes in the data.")
        n_top_genes = sampled_data.n_vars
    
    # Determine features based on the flavor
    if flavor != "iff":
        logger.info(f"Selecting highly variable genes with flavor {flavor}...")
        sc.pp.highly_variable_genes(sampled_data, n_top_genes=n_top_genes, flavor=flavor)
        feature_list = sampled_data.var[sampled_data.var['highly_variable']].index.tolist()
    else:
        logger.info("Selecting informative features using IFF...")
        feature_list = iff_select(
            adata=sampled_data,
            grouping=grouping,
            emb_key=emb_key,
            k=k,
            knn_samples=knn_samples,
            n_top_genes=n_top_genes,
            gini_cut_qt=gini_cut_qt,
            save_path=save_path,
            figsize=figsize
        )
    return feature_list




