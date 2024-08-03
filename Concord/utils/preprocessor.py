from typing import Dict, Optional, Union

import scanpy as sc
import anndata as ad
from .. import logger


class Preprocessor:
    def __init__(
        self,
        use_key: Optional[str] = None,
        filter_gene_by_counts: Union[int, bool] = False,
        filter_cell_by_counts: Union[int, bool] = False,
        normalize_total: Union[float, bool] = 1e4,
        result_normed_key: Optional[str] = "X_normed",
        log1p: bool = False,
        result_log1p_key: str = "X_log1p",
        subset_hvg: Union[int, bool] = False,
        hvg_use_key: Optional[str] = None,
        hvg_flavor: str = "seurat_v3",
        domain_key: Optional[str] = None
    ):
        self.use_key = use_key
        self.filter_gene_by_counts = filter_gene_by_counts
        self.filter_cell_by_counts = filter_cell_by_counts
        self.normalize_total = normalize_total
        self.result_normed_key = result_normed_key
        self.log1p = log1p
        self.result_log1p_key = result_log1p_key
        self.subset_hvg = subset_hvg
        self.hvg_use_key = hvg_use_key
        self.hvg_flavor = hvg_flavor
        self.domain_key = domain_key

    def __call__(self, adata) -> Dict:
        key_to_process = self.use_key
        if key_to_process == "X":
            key_to_process = None
        is_logged = self.check_logged(adata, obs_key=key_to_process)

        # filter genes
        if self.filter_gene_by_counts:
            logger.info("Filtering genes by counts ...")
            sc.pp.filter_genes(
                adata,
                min_counts=self.filter_gene_by_counts
                if isinstance(self.filter_gene_by_counts, int)
                else None,
            )

        # filter cells
        if (
            isinstance(self.filter_cell_by_counts, int)
            and self.filter_cell_by_counts > 0
        ):
            logger.info("Filtering cells by counts ...")
            sc.pp.filter_cells(
                adata,
                min_counts=self.filter_cell_by_counts
                if isinstance(self.filter_cell_by_counts, int)
                else None,
            )

        # normalize total
        if self.normalize_total:
            if not is_logged:
                logger.info("Normalizing total counts ...")
                normed_ = sc.pp.normalize_total(
                    adata,
                    target_sum=self.normalize_total
                    if isinstance(self.normalize_total, float)
                    else None,
                    layer=key_to_process,
                    inplace=False,
                )["X"]
                key_to_process = self.result_normed_key or key_to_process
                self._set_obs_rep(adata, normed_, layer=key_to_process)
            else:
                logger.info("Data is already log1p transformed. Skip normalization.")

        # log1p (if not already logged)
        if self.log1p:
            if not is_logged:
                logger.info("Log1p transforming ...")
                if self.result_log1p_key:
                    data_to_transform = self._get_obs_rep(adata, layer=key_to_process).copy()
                    temp_adata = ad.AnnData(data_to_transform)
                    sc.pp.log1p(temp_adata)
                    self._set_obs_rep(adata, temp_adata.X, layer=self.result_log1p_key)
                else:
                    sc.pp.log1p(adata, layer=key_to_process)

            else:
                logger.info("Data is already log1p transformed. Storing in the specified layer.")
                if self.result_log1p_key:
                    self._set_obs_rep(adata, self._get_obs_rep(adata, layer=key_to_process), layer=self.result_log1p_key)

        # subset hvg
        if self.subset_hvg:
            logger.info("Subsetting highly variable genes ...")
            if self.domain_key is None:
                logger.warning(
                    "No domain_key is provided, will use all cells for HVG selection."
                )
            sc.pp.highly_variable_genes(
                adata,
                layer=self.hvg_use_key,
                n_top_genes=self.subset_hvg
                if isinstance(self.subset_hvg, int)
                else None,
                batch_key=self.domain_key,
                flavor=self.hvg_flavor,
                subset=True,
            )

    def _get_obs_rep(self, adata, layer: Optional[str] = None):
        if layer is None:
            return adata.X
        elif layer in adata.layers:
            return adata.layers[layer]
        else:
            raise KeyError(f"Layer '{layer}' not found in AnnData object.")

    def _set_obs_rep(self, adata, data, layer: Optional[str] = None):
        if layer is None:
            adata.X = data
        else:
            adata.layers[layer] = data

    def check_logged(self, adata, obs_key: Optional[str] = None) -> bool:
        data = self._get_obs_rep(adata, layer=obs_key)
        max_, min_ = data.max(), data.min()
        if max_ > 30:
            return False
        if min_ < 0:
            return False

        non_zero_min = data[data > 0].min()
        if non_zero_min >= 1:
            return False

        return True



