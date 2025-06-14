import torch
from .sampler import ConcordSampler
from .anndataset import AnnDataset
from .knn import Neighborhood
from ..utils.value_check import validate_probability, validate_probability_dict_compatible
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
    def __init__(self, input_layer_key, domain_key, 
                    class_key=None, covariate_keys=None,
                    feature_list=None,  
                    batch_size=32, train_frac=0.9,
                    use_sampler=True,
                    sampler_emb=None,
                    sampler_knn=300, 
                    sampler_domain_minibatch_strategy='proportional',
                    domain_coverage=None,
                    p_intra_knn=0.3, p_intra_domain=None,
                    dist_metric='euclidean',
                    pca_n_comps=50, 
                    use_faiss=True, 
                    use_ivf=False,
                    ivf_nprobe=8,
                    preprocess=None, 
                    num_cores=None,
                    device=None):
        """
        Initializes the DataLoaderManager.

        Args:
            input_layer_key (str): Key for input layer in `adata`.
            domain_key (str): Key for domain labels in `adata.obs`.
            class_key (str, optional): Key for class labels. Defaults to None.
            covariate_keys (list, optional): List of covariate keys. Defaults to None.
            batch_size (int, optional): Batch size. Defaults to 32.
            train_frac (float, optional): Fraction of data used for training. Defaults to 0.9.
            use_sampler (bool, optional): Whether to use the custom sampler. Defaults to True.
            sampler_emb (str, optional): Key for embeddings used in sampling.
            sampler_knn (int, optional): Number of neighbors for k-NN sampling. Defaults to 300.
            p_intra_knn (float, optional): Probability of intra-cluster sampling. Defaults to 0.3.
            p_intra_domain (float or dict, optional): Probability of intra-domain sampling.
            dist_metric (str, optional): Distance metric for k-NN. Defaults to 'euclidean'.
            pca_n_comps (int, optional): Number of PCA components. Defaults to 50.
            use_faiss (bool, optional): Whether to use FAISS. Defaults to True.
            use_ivf (bool, optional): Whether to use IVF-Faiss indexing. Defaults to False.
            ivf_nprobe (int, optional): Number of probes for IVF-Faiss. Defaults to 8.
            preprocess (callable, optional): Preprocessing function for `adata`.
            num_cores (int, optional): Number of CPU cores. Defaults to None.
            device (torch.device, optional): Device for computation. Defaults to None.
        """
        self.input_layer_key = input_layer_key
        self.domain_key = domain_key
        self.class_key = class_key
        self.covariate_keys = covariate_keys
        self.feature_list = feature_list
        self.batch_size = batch_size
        self.train_frac = train_frac
        self.use_sampler = use_sampler
        self.sampler_emb = sampler_emb
        self.sampler_knn = sampler_knn
        self.sampler_domain_minibatch_strategy = sampler_domain_minibatch_strategy
        self.domain_coverage = domain_coverage
        self.p_intra_knn = p_intra_knn
        self.p_intra_domain = p_intra_domain
        self.p_intra_domain_dict = None
        self.pca_n_comps = pca_n_comps
        self.use_faiss = use_faiss
        self.use_ivf = use_ivf
        self.ivf_nprobe = ivf_nprobe
        self.preprocess = preprocess
        self.num_cores = num_cores
        self.device = device
        self.dist_metric = dist_metric

        # Dynamically set based on adata
        self.adata = None
        self.emb = None
        self.data_structure = None
        self.knn_index = None
        self.nbrs = None
        self.sampler = None


    def compute_embedding_and_knn(self, emb_key='X_pca'):
        """
        Constructs a k-NN graph based on existing embedding or PCA (of not exist, compute automatically).

        Args:
            emb_key (str, optional): Key for embedding basis. Defaults to 'X_pca'.
        """
        # Get embedding for current adata
        from ..utils.anndata_utils import get_adata_basis
        self.emb = get_adata_basis(self.adata, basis=emb_key, pca_n_comps=self.pca_n_comps)
        # Initialize KNN
        self.neighborhood = Neighborhood(emb=self.emb, k=self.sampler_knn, use_faiss=self.use_faiss, use_ivf=self.use_ivf, ivf_nprobe=self.ivf_nprobe, metric=self.dist_metric)


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
                raise ValueError("Multiple domains detected but p_intra_domain is not specified. Please provide p_intra_domain as a float or a dictionary mapping each domain to a probability.")
        else:
            validate_probability_dict_compatible(self.p_intra_domain, "p_intra_domain")
            if not isinstance(self.p_intra_domain, dict):
                if len(unique_domains) == 1:
                    if self.p_intra_domain != 1.0:
                        logger.warning(f"You specified p_intra_domain as {self.p_intra_domain} but you only have one domain. "
                                    f"Resetting p_intra_domain to 1.0.")
                        self.p_intra_domain = 1.0
                self.p_intra_domain = {domain: self.p_intra_domain for domain in unique_domains}
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
        """
        Converts an AnnData object to PyTorch DataLoader.

        Args:
            adata (AnnData): The input AnnData object.

        Returns:
            tuple: Train DataLoader, validation DataLoader (if `train_frac < 1.0`), and data structure.
        """
        self.adata = adata
        # Preprocess data if necessary
        if self.preprocess:
            logger.info("Preprocessing adata...")
            self.preprocess(self.adata)

        # Subset features if provided
        if self.feature_list:
            logger.info(f"Filtering features with provided list ({len(self.feature_list)} features)...")
            self.adata._inplace_subset_var(adata.var_names.isin(self.feature_list))


        self.domain_labels = self.adata.obs[self.domain_key]
        self.domain_ids = torch.tensor(self.domain_labels.cat.codes.values, dtype=torch.long).to(self.device)
        
        dataset = AnnDataset(self.adata, input_layer_key=self.input_layer_key, 
                domain_key=self.domain_key, class_key=self.class_key, 
                covariate_keys=self.covariate_keys, device=self.device)
        
        self.data_structure = dataset.get_data_structure()

        if self.use_sampler:
            self.compute_embedding_and_knn(self.sampler_emb)
            self.compute_p_intra_domain()
            SamplerClass = ConcordSampler
        else:
            SamplerClass = None

        if self.train_frac == 1.0:
            if self.use_sampler:
                self.sampler = SamplerClass(
                    batch_size=self.batch_size, 
                    domain_ids=self.domain_ids, 
                    p_intra_knn=self.p_intra_knn, 
                    p_intra_domain_dict=self.p_intra_domain_dict,
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
                    p_intra_domain_dict=self.p_intra_domain_dict,
                    domain_minibatch_strategy=self.sampler_domain_minibatch_strategy,
                    domain_coverage=self.domain_coverage,
                    neighborhood=None, # Not used if train-val split
                    device=self.device
                )

                val_sampler = SamplerClass(
                    batch_size=self.batch_size, 
                    domain_ids=self.domain_ids[val_indices],
                    p_intra_knn=self.p_intra_knn, 
                    p_intra_domain_dict=self.p_intra_domain_dict,
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




