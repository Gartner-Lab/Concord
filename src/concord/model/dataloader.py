import torch
from .sampler import ConcordSampler
from .anndataset import AnnDataset
from .knn import Neighborhood
from torch.utils.data import DataLoader
import numpy as np
import scanpy as sc
import pandas as pd
import logging

logger = logging.getLogger(__name__)


class DataLoaderManager:
    """
    Manages data loading for CONCORD, including optional preprocessing and sampling.
    This class handles the standard workflow of total-count normalization and log1p transformation.
    """
    def __init__(self, 
                 domain_key, 
                 class_key=None, 
                 covariate_keys=None,
                 feature_list=None,
                 normalize_total=True,  # Simplified parameter
                 log1p=True,            # Simplified parameter
                 batch_size=32, 
                 train_frac=0.9,
                 use_sampler=True,
                 sampler_emb=None,
                 sampler_knn=300, 
                 sampler_domain_minibatch_strategy='proportional',
                 domain_coverage=None,
                 p_intra_knn=0.3, 
                 p_intra_domain=0.95,
                 dist_metric='euclidean',
                 pca_n_comps=50, 
                 use_faiss=True, 
                 use_ivf=False,
                 ivf_nprobe=8,
                 num_cores=None,
                 device=None):
        """
        Initializes the DataLoaderManager.
        """
        self.domain_key = domain_key
        self.class_key = class_key
        self.covariate_keys = covariate_keys
        self.feature_list = feature_list
        self.normalize_total = normalize_total
        self.log1p = log1p
        self.batch_size = batch_size
        self.train_frac = train_frac
        self.use_sampler = use_sampler
        self.sampler_emb = sampler_emb
        self.sampler_knn = sampler_knn
        self.sampler_domain_minibatch_strategy = sampler_domain_minibatch_strategy
        self.domain_coverage = domain_coverage
        self.p_intra_knn = p_intra_knn
        self.p_intra_domain = p_intra_domain
        self.pca_n_comps = pca_n_comps
        self.use_faiss = use_faiss
        self.use_ivf = use_ivf
        self.ivf_nprobe = ivf_nprobe
        self.num_cores = num_cores
        self.device = device
        self.dist_metric = dist_metric

        # Dynamically set based on adata
        self.adata = None
        self.data_structure = None
        self.knn_index = None
        self.nbrs = None
        self.sampler = None

        self.data_structure = self._get_data_structure()

    
    def _get_data_structure(self):
        """
        Determines the structure of the data to be returned by the dataset.
        This logic is now owned by the manager, not the dataset.
        """
        structure = ['input']
        if self.domain_key is not None:
            structure.append('domain')
        if self.class_key is not None:
            structure.append('class')
        if self.covariate_keys:
            structure.extend(self.covariate_keys)
        structure.append('idx')
        return structure


    def compute_embedding_and_knn(self, emb_key='X_pca'):
        """
        Constructs a k-NN graph based on existing embedding or PCA (of not exist, compute automatically).

        Args:
            emb_key (str, optional): Key for embedding basis. Defaults to 'X_pca'.
        """
        # Get embedding for current adata
        from ..utils.anndata_utils import get_adata_basis
        emb = get_adata_basis(self.adata, basis=emb_key, pca_n_comps=self.pca_n_comps)
        # Initialize KNN
        self.neighborhood = Neighborhood(emb=emb, k=self.sampler_knn, use_faiss=self.use_faiss, use_ivf=self.use_ivf, ivf_nprobe=self.ivf_nprobe, metric=self.dist_metric)


    def anndata_to_dataloader(self, adata):
        """
        Converts an AnnData object to PyTorch DataLoader.

        Args:
            adata (AnnData): The input AnnData object.

        Returns:
            tuple: Train DataLoader, validation DataLoader (if `train_frac < 1.0`), and data structure.
        """
        self.adata = adata

        if self.normalize_total:
            logger.info("Normalizing total counts per cell...")
            sc.pp.normalize_total(self.adata, target_sum=1e4, inplace=True)
        
        if self.log1p:
            logger.info("Log1p transforming data...")
            sc.pp.log1p(self.adata)

        # Subset features if provided
        if self.feature_list:
            logger.info(f"Filtering features with provided list ({len(self.feature_list)} features)...")
            self.adata = self.adata[:, self.feature_list]

        self.domain_labels = self.adata.obs[self.domain_key]
        self.domain_ids = torch.tensor(self.domain_labels.cat.codes.values, dtype=torch.long).to(self.device)
        
        dataset = AnnDataset(self.adata, 
                             data_structure=self.data_structure,
                             input_layer_key='X', 
                             domain_key=self.domain_key, 
                             class_key=self.class_key, 
                             covariate_keys=self.covariate_keys, device=self.device)

        if self.use_sampler:
            if self.p_intra_knn > 0.0:
                self.compute_embedding_and_knn(self.sampler_emb)
            SamplerClass = ConcordSampler
        else:
            SamplerClass = None

        if self.train_frac == 1.0:
            if self.use_sampler:
                self.sampler = SamplerClass(
                    batch_size=self.batch_size, 
                    domain_ids=self.domain_ids, 
                    p_intra_knn=self.p_intra_knn, 
                    p_intra_domain=self.p_intra_domain,
                    domain_minibatch_strategy=self.sampler_domain_minibatch_strategy,
                    domain_coverage=self.domain_coverage,
                    neighborhood=self.neighborhood, 
                    device=self.device
                )
                full_dataloader = DataLoader(dataset, batch_sampler=self.sampler)
            else:
                self.sampler = None
                full_dataloader = DataLoader(dataset, batch_size=self.batch_size)
            return full_dataloader, None, self.data_structure
        else:
            train_size = int(self.train_frac * len(dataset))
            indices = np.arange(len(dataset))
            np.random.shuffle(indices)
            train_indices = indices[:train_size]
            val_indices = indices[train_size:]

            train_dataset = dataset.subset(train_indices)
            val_dataset = dataset.subset(val_indices)

            if self.use_sampler:
                train_sampler = SamplerClass(
                    batch_size=self.batch_size, 
                    domain_ids=self.domain_ids[train_indices],
                    p_intra_knn=self.p_intra_knn, 
                    p_intra_domain=self.p_intra_domain,
                    domain_minibatch_strategy=self.sampler_domain_minibatch_strategy,
                    domain_coverage=self.domain_coverage,
                    neighborhood=None, # Not used if train-val split
                    device=self.device
                )

                val_sampler = SamplerClass(
                    batch_size=self.batch_size, 
                    domain_ids=self.domain_ids[val_indices],
                    p_intra_knn=self.p_intra_knn, 
                    p_intra_domain=self.p_intra_domain,
                    domain_minibatch_strategy=self.sampler_domain_minibatch_strategy,
                    domain_coverage=self.domain_coverage,
                    neighborhood=None, # Not used if train-val split
                    device=self.device
                )
                train_dataloader = DataLoader(train_dataset, batch_sampler=train_sampler)
                val_dataloader = DataLoader(val_dataset, batch_sampler=val_sampler)
            else:
                train_dataloader = DataLoader(train_dataset, batch_size=self.batch_size)
                val_dataloader = DataLoader(val_dataset, batch_size=self.batch_size)

            return train_dataloader, val_dataloader, self.data_structure




