import torch
from .sampler import ConcordSampler
from .anndataset import AnnDataset
from .knn import Neighborhood
from ..utils.anndata_utils import get_adata_basis
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
                 sampler_knn=300, 
                 sampler_emb=None,
                 sampler_domain_minibatch_strategy='proportional',
                 domain_coverage=None,
                 p_intra_knn=0.3, 
                 p_intra_domain=0.95,
                 dist_metric='euclidean',
                 use_faiss=True, 
                 use_ivf=False,
                 ivf_nprobe=8,
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
        self.use_faiss = use_faiss
        self.use_ivf = use_ivf
        self.ivf_nprobe = ivf_nprobe
        self.device = device
        self.dist_metric = dist_metric

        # Dynamically set based on adata
        self.adata = None
        self.data_structure = None
        self.neighborhood = None
        self.train_sampler = None
        self.val_sampler = None

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
    

    def compute_neighborhood(self, adata, emb_key): # Pass adata and emb_key
        logger.info(f"Computing k-NN graph using embedding from '{emb_key}'...")
        emb = get_adata_basis(adata, basis=emb_key)
        self.neighborhood = Neighborhood(
            emb=emb, k=self.sampler_knn, use_faiss=self.use_faiss, 
            use_ivf=self.use_ivf, ivf_nprobe=self.ivf_nprobe, metric=self.dist_metric
        )


    def build_sampler(self, SamplerClass, indices=None, neighborhood=None):
        domain_ids = self.domain_ids if indices is None else self.domain_ids[indices]
        sampler = SamplerClass(
            batch_size=self.batch_size, 
            domain_ids=domain_ids, 
            p_intra_knn=self.p_intra_knn, 
            p_intra_domain=self.p_intra_domain,
            domain_minibatch_strategy=self.sampler_domain_minibatch_strategy,
            domain_coverage=self.domain_coverage,
            neighborhood=neighborhood,
            device=self.device
        )
        return sampler


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
            if self.train_frac == 1.0:
                if self.p_intra_knn > 0.0:
                    self.compute_neighborhood(self.adata, self.sampler_emb)

                self.train_sampler = self.build_sampler(ConcordSampler, neighborhood=self.neighborhood)
                train_dataloader = DataLoader(dataset, batch_sampler=self.train_sampler)
                val_dataloader = None
            else:
                train_size = int(self.train_frac * len(dataset))
                indices = np.arange(len(dataset))
                np.random.shuffle(indices)
                train_indices = indices[:train_size]
                val_indices = indices[train_size:]
                train_dataset = dataset.subset(train_indices)
                val_dataset = dataset.subset(val_indices)

                self.train_sampler = self.build_sampler(ConcordSampler, indices=train_indices, neighborhood=None)
                self.val_sampler = self.build_sampler(ConcordSampler, indices=val_indices, neighborhood=None)
                train_dataloader = DataLoader(train_dataset, batch_sampler=self.train_sampler)
                val_dataloader = DataLoader(val_dataset, batch_sampler=self.val_sampler)
        else: 
            train_dataloader = DataLoader(dataset, batch_size=self.batch_size)
            val_dataloader = None

        return train_dataloader, val_dataloader, self.data_structure




