import torch
from .sampler import ConcordSampler, ConcordMatchNNSampler
from .anndataset import AnnDataset
from .knn import Neighborhood
from ..utils.value_check import validate_probability, validate_probability_dict_compatible
from ..utils.coverage_estimator import calculate_domain_coverage, coverage_to_p_intra
from torch.utils.data import DataLoader
import numpy as np
import scanpy as sc
import pandas as pd
import logging

logger = logging.getLogger(__name__)


class DataLoaderManager:
    def __init__(self, input_layer_key, domain_key, 
                    class_key=None, covariate_keys=None,
                    batch_size=32, train_frac=0.9,
                    use_sampler=True,
                    sampler_emb=None,
                    sampler_knn=300, 
                    p_intra_knn=0.3, p_intra_domain=None,
                    min_p_intra_domain=0.5, max_p_intra_domain=1.0,
                    clr_mode='aug', 
                    pca_n_comps=50, 
                    use_faiss=True, 
                    use_ivf=False,
                    ivf_nprobe=8,
                    preprocess=None, 
                    num_cores=None,
                    device=None):
            
        self.input_layer_key = input_layer_key
        self.domain_key = domain_key
        self.class_key = class_key
        self.covariate_keys = covariate_keys
        self.batch_size = batch_size
        self.train_frac = train_frac
        self.use_sampler = use_sampler
        self.sampler_emb = sampler_emb
        self.sampler_knn = sampler_knn
        self.p_intra_knn = p_intra_knn
        self.p_intra_domain = p_intra_domain
        self.p_intra_domain_dict = None
        self.min_p_intra_domain = min_p_intra_domain
        self.max_p_intra_domain = max_p_intra_domain
        self.clr_mode = clr_mode
        self.pca_n_comps = pca_n_comps
        self.use_faiss = use_faiss
        self.use_ivf = use_ivf
        self.ivf_nprobe = ivf_nprobe
        self.preprocess = preprocess
        self.num_cores = num_cores
        self.device = device

        # Dynamically set based on adata
        self.adata = None
        self.emb = None
        self.data_structure = None
        self.knn_index = None
        self.nbrs = None
        self.sampler = None


    def compute_embedding_and_knn(self):
        # Get embedding for current adata
        if self.sampler_emb not in self.adata.obsm:
            if self.sampler_emb == 'X_pca':
                logger.info("PCA embedding not found in adata.obsm. Running PCA...")
                sc.tl.pca(self.adata, svd_solver='arpack', n_comps=self.pca_n_comps)
                logger.info("PCA completed.")
                self.emb = self.adata.obsm['X_pca'].astype(np.float32)
            else:
                raise ValueError(f"Embedding '{self.sampler_emb}' not found in adata.obsm.")
        else:
            logger.info(f"Using existing embedding '{self.sampler_emb}' from adata.obsm")
            self.emb = self.adata.obsm[self.sampler_emb].astype(np.float32)

        # Initialize KNN
        self.neighborhood = Neighborhood(emb=self.emb, k=self.sampler_knn, use_faiss=self.use_faiss, use_ivf=self.use_ivf, ivf_nprobe=self.ivf_nprobe)


    def compute_p_intra_domain(self):
        # Validate probability values
        validate_probability(self.p_intra_knn, "p_intra_knn")

        unique_domains = self.domain_labels.cat.categories
        logger.info(f"Number of unique_domains: {len(unique_domains)}")

        if self.p_intra_domain is None: 
            # TODO check if chunked mode, if so send error.
            if len(unique_domains) == 1:
                logger.warning(f"Only one domain found in the data. Setting p_intra_domain to 1.0.")
                self.p_intra_domain = {unique_domains[0]: 1.0}
            else:
                if self.min_p_intra_domain >= 1.0:
                    logger.info(f"p_intra_domain is set to 1.0 as min_p_intra_domain >= 1.0.")
                    self.p_intra_domain = {domain: 1.0 for domain in unique_domains}
                else:
                    logger.info(f"Calculating each domain's coverage of the global manifold using {self.sampler_emb}.")
                    domain_coverage = calculate_domain_coverage(
                        adata=self.adata, domain_key=self.domain_key, neighborhood=self.neighborhood
                    )
                    logger.info(f"Converting coverage to p_intra_domain...")
                    self.p_intra_domain = coverage_to_p_intra(
                        self.domain_labels, coverage=domain_coverage, 
                        min_p_intra_domain=self.min_p_intra_domain, 
                        max_p_intra_domain=self.max_p_intra_domain,
                        scale_to_min_max=True # Always true unless user runs himself
                    )
        else:
            validate_probability_dict_compatible(self.p_intra_domain, "p_intra_domain")
            if not isinstance(self.p_intra_domain, dict):
                if len(unique_domains) == 1 and p_intra_domain < 1.0:
                    logger.warning(f"You specified p_intra_domain as {p_intra_domain} but you only have one domain. "
                                f"Resetting p_intra_domain to 1.0.")
                    p_intra_domain = 1.0
                else:
                    p_intra_domain = self.p_intra_domain
                self.p_intra_domain = {domain: p_intra_domain for domain in unique_domains}
            else:
                if len(unique_domains) != len(self.p_intra_domain):
                    raise ValueError(f"Length of p_intra_domain ({len(self.p_intra_domain)}) does not match the number of unique domains ({len(unique_domains)}).")
                for domain in unique_domains:
                    if domain not in self.p_intra_domain:
                        raise ValueError(f"Domain {domain} not found in p_intra_domain dictionary.")
                logger.info(f"Using user-specified p_intra_domain values.")
            
        logger.info(f"Final p_intra_domain values: {', '.join(f'{k}: {v:.2f}' for k, v in self.p_intra_domain.items())}")
        # Convert the domain labels to their corresponding category codes
        domain_codes = {domain: code for code, domain in enumerate(unique_domains)}
        self.p_intra_domain_dict = {domain_codes[domain]: value for domain, value in self.p_intra_domain.items()}


    def anndata_to_dataloader(self, adata):
        self.adata = adata
        
        # Preprocess data if necessary
        if self.preprocess:
            logger.info("Preprocessing adata...")
            self.preprocess(self.adata)

        self.domain_labels = self.adata.obs[self.domain_key]
        self.domain_ids = torch.tensor(self.domain_labels.cat.codes.values, dtype=torch.long).to(self.device)
        
        dataset = AnnDataset(self.adata, input_layer_key=self.input_layer_key, 
                domain_key=self.domain_key, class_key=self.class_key, 
                covariate_keys=self.covariate_keys, device=self.device)
        
        self.data_structure = dataset.get_data_structure()

        if self.use_sampler:
            self.compute_embedding_and_knn()
            self.compute_p_intra_domain()
            SamplerClass = ConcordMatchNNSampler if self.clr_mode == 'nn' else ConcordSampler
        else:
            SamplerClass = None

        if self.train_frac == 1.0:
            indices = torch.arange(len(dataset))
            if self.use_sampler:
                self.sampler = SamplerClass(
                    batch_size=self.batch_size, 
                    indices=indices,
                    domain_ids=self.domain_ids, 
                    p_intra_knn=self.p_intra_knn, 
                    p_intra_domain_dict=self.p_intra_domain_dict,
                    neighborhood=self.neighborhood, 
                    device=self.device
                )
            else:
                self.sampler = None
            full_dataloader = DataLoader(dataset, batch_sampler=self.sampler)
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
                    indices=train_indices,
                    domain_ids=self.domain_ids, 
                    p_intra_knn=self.p_intra_knn, p_intra_domain_dict=self.p_intra_domain_dict,
                    neighborhood=self.neighborhood, 
                    device=self.device
                )

                val_sampler = SamplerClass(
                    batch_size=self.batch_size, 
                    indices=val_indices,
                    domain_ids=self.domain_ids, 
                    p_intra_knn=self.p_intra_knn, p_intra_domain_dict=self.p_intra_domain_dict,
                    neighborhood=self.neighborhood, 
                    device=self.device
                )
            else:
                train_sampler, val_sampler = None, None

            train_dataloader = DataLoader(train_dataset, batch_sampler=train_sampler)
            val_dataloader = DataLoader(val_dataset, batch_sampler=val_sampler)

            return train_dataloader, val_dataloader, self.data_structure




