from pathlib import Path
import torch
import torch.nn.functional as F
from .model.model import ConcordModel
from .model.knn import Neighborhood
from .utils.anndata_utils import ensure_categorical, check_adata_X, get_adata_basis
from .model.dataloader import DataLoaderManager 
from .model.chunkloader import ChunkLoader
from .utils.other_util import add_file_handler, set_seed
from .utils.value_check import validate_probability_dict_compatible
from .model.trainer import Trainer
from .model.augment import MaskNonZerosAugment, FeatureDropAugment
import numpy as np
import scanpy as sc
import pandas as pd
import copy
import json
from . import logger
from . import set_verbose_mode
import torch.nn as nn

class Config:
    def __init__(self, config_dict):
        for key, value in config_dict.items():
            setattr(self, key, value)

    def __getitem__(self, item):
        return getattr(self, item)

    def to_dict(self):
        def serialize(value):
            if isinstance(value, torch.device):
                return str(value)
            if isinstance(value, (pd.Index, pd.CategoricalIndex)):
                return value.tolist()
            if isinstance(value, np.ndarray):
                return value.tolist()
            return value
        return {k: serialize(getattr(self, k))
                for k in dir(self)
                if not k.startswith('__') and not callable(getattr(self, k))}


class Concord:
    """
    A contrastive learning framework for single-cell data analysis.

    CONCORD performs dimensionality reduction, denoising, and batch correction 
    in an unsupervised manner while preserving local and global topological structures.

    Attributes:
        adata (AnnData): Input AnnData object.
        save_dir (Path): Directory to save outputs and logs.
        config (Config): Configuration object storing hyperparameters.
        model (ConcordModel): The main contrastive learning model.
        trainer (Trainer): Handles model training.
        loader (DataLoaderManager or ChunkLoader): Data loading utilities.
    """
    def __init__(self, adata, save_dir='save/', copy_adata=False, verbose=False, **kwargs):
        """
        Initializes the Concord framework.

        Args:
            adata (AnnData): Input single-cell data in AnnData format.
            save_dir (str, optional): Directory to save model outputs. Defaults to 'save/'.
            copy_adata (bool, optional): If True, a copy of `adata` is made before 
                any modifications. The final results are then copied back to the
                original `adata` object. If False (default), operates directly on the 
                provided `adata` object, which is more memory-efficient but may
                alter it (e.g., through feature subsetting).
            verbose (bool, optional): Enable verbose logging. Defaults to False.
            **kwargs: Additional configuration parameters.
        """
        set_verbose_mode(verbose)

        self._adata_original = adata # reference to the original AnnData object
        self._check_adata(adata, copy_adata)

        self.config = None
        self.loader = None
        self.model = None
        self.run = None
        self.sampler_kwargs = {}

        if save_dir is not None:
            self.save_dir = Path(save_dir)
            if not self.save_dir.exists():
                self.save_dir.mkdir(parents=True, exist_ok=True)
            add_file_handler(logger, self.save_dir / "run.log")
        else:
            self.save_dir = None
            logger.warning("save_dir is None. Model and log files will not be saved.")

        self.default_params = dict(
            seed=0,
            input_feature=None,
            normalize_total=False, # default adata.X should be normalized
            log1p=False,
            batch_size=256,
            n_epochs=15,
            lr=1e-2,
            schedule_ratio=0.97,
            train_frac=1.0,
            latent_dim=100,
            encoder_dims=[1000],
            decoder_dims=[1000],
            element_mask_prob=0.4, 
            feature_mask_prob=0.3, 
            domain_key=None,
            class_key=None,
            domain_embedding_dim=8,
            covariate_embedding_dims={},
            use_decoder=False, # Default decoder usage
            decoder_final_activation='relu',
            decoder_weight=1.0,
            clr_temperature=0.4,
            clr_beta=1.0,  # Beta for NT-Xent loss
            clr_weight=1.0,
            use_classifier=False,
            classifier_weight=1.0,
            unlabeled_class=None,
            use_importance_mask=False,
            importance_penalty_weight=0,
            importance_penalty_type='L1',
            dropout_prob=0.0, # Default just SIMCLR augmentation, no internal dropout
            norm_type="layer_norm",  # Default normalization type
            knn_warmup_epochs=2, # Number of epochs to warm up KNN sampling
            sampler_knn=None, # Default neighborhood size, can be adjusted
            sampler_emb=None,
            sampler_domain_minibatch_strategy='proportional', # Strategy for domain minibatch sampling
            domain_coverage = None,
            dist_metric='euclidean',
            p_intra_knn=0.0,
            p_intra_domain=1.0,
            use_faiss=True,
            use_ivf=True,
            ivf_nprobe=10,
            pretrained_model=None,
            preload_dense=False,  # Whether to densify the data in memory
            num_workers=None,  # Number of workers for DataLoader
            chunked=False,
            chunk_size=10000,
            device=torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        )

        self.setup_config(**kwargs)
        set_seed(self.config.seed)

        self._check_input_format(self.adata)
        self._check_input_features()
        self._check_knn_sampler_settings()
        self._check_hcl_sampler_settings()
        self._check_importance_mask()
        self._check_domain_settings(self.adata)
        self._check_class_settings()
        self._check_covariate_settings()

        self.preprocessed = False
        

    def get_default_params(self):
        """
        Returns the default hyperparameters used in CONCORD.

        Returns:
            dict: A dictionary containing default configuration values.
        """
        return self.default_params.copy()
    

    def setup_config(self, **kwargs):
        """
        Sets up the configuration for training.

        Args:
            **kwargs: Key-value pairs to override default parameters.

        Raises:
            ValueError: If an invalid parameter is provided.
        """
        # Start with the default parameters
        initial_params = self.default_params.copy()

        # Check if any of the provided parameters are not in the default parameters
        invalid_params = set(kwargs.keys()) - set(initial_params.keys())
        if invalid_params:
            raise ValueError(f"Invalid parameters provided: {invalid_params}")

        # Update with user-provided values (if any)
        initial_params.update(kwargs)

        self.config = Config(initial_params)


    def init_model(self, input_dim=None):
        """
        Initializes the CONCORD model and loads a pre-trained model if specified.

        Raises:
            FileNotFoundError: If the specified pre-trained model file is missing.
        """
        if self.config.input_feature is not None:
            input_dim = len(self.config.input_feature)
        elif self.adata is not None:
            input_dim = self.adata.shape[1]
        elif input_dim is not None:
            input_dim = input_dim
        else:
            raise ValueError("Input dimension must be specified either through config.input_feature, adata, or input_dim argument.")
        
        hidden_dim = self.config.latent_dim

        self.model = ConcordModel(input_dim, hidden_dim, 
                                  num_domains=self.config.num_domains,
                                  num_classes=self.config.num_classes,
                                  domain_embedding_dim=self.config.domain_embedding_dim,
                                  covariate_embedding_dims=self.config.covariate_embedding_dims,
                                  covariate_num_categories=self.config.covariate_num_categories,
                                  #encoder_append_cov=self.config.encoder_append_cov,
                                  encoder_dims=self.config.encoder_dims,
                                  decoder_dims=self.config.decoder_dims,
                                  decoder_final_activation=self.config.decoder_final_activation,
                                  #augmentation_mask_prob=self.config.augmentation_mask_prob,
                                  dropout_prob=self.config.dropout_prob,
                                  norm_type=self.config.norm_type,
                                  use_decoder=self.config.use_decoder,
                                  use_classifier=self.config.use_classifier,
                                  use_importance_mask=self.config.use_importance_mask).to(self.config.device)

        logger.info(f'Model loaded to device: {self.config.device}')
        total_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        logger.info(f'Total number of parameters: {total_params}')

        if self.config.pretrained_model is not None:
            pretrained_model_path = Path(self.config.pretrained_model)
            if pretrained_model_path.exists():
                logger.info(f"Loading pre-trained model from {pretrained_model_path}")
                self.model.load_model(pretrained_model_path, self.config.device)
            else:
                raise FileNotFoundError(f"Model file not found at {pretrained_model_path}")
            

    def init_trainer(self):
        """
        Initializes the model trainer, setting up loss functions, optimizer, and learning rate scheduler.
        """
        logger.info("Augmentation probabilities:")
        logger.info(f" - Element mask probability: {self.config.element_mask_prob}")
        logger.info(f" - Feature mask probability: {self.config.feature_mask_prob}")

        augment = nn.Sequential(
            MaskNonZerosAugment(p=self.config.element_mask_prob),   # robustness to count noise
            FeatureDropAugment(p=self.config.feature_mask_prob)        # robustness to feature loss
        )

        self.trainer = Trainer(model=self.model,
                               data_structure=self.data_structure,
                               device=self.config.device,
                               logger=logger,
                               lr=self.config.lr,
                               schedule_ratio=self.config.schedule_ratio,
                               augment=augment,
                               use_classifier=self.config.use_classifier, 
                               classifier_weight=self.config.classifier_weight,
                               unique_classes=self.config.unique_classes_code,
                               unlabeled_class=self.config.unlabeled_class_code,
                               use_decoder=self.config.use_decoder,
                               decoder_weight=self.config.decoder_weight,
                               clr_temperature=self.config.clr_temperature,
                               clr_beta=self.config.clr_beta,
                               clr_weight=self.config.clr_weight,
                               importance_penalty_weight=self.config.importance_penalty_weight,
                               importance_penalty_type=self.config.importance_penalty_type)


    def init_dataloader(self, adata=None, train_frac=1.0, use_sampler=True):
        """
        Initializes the data loader for training and evaluation.

        Args:
            train_frac (float, optional): Fraction of data to use for training. Defaults to 1.0.
            use_sampler (bool, optional): Whether to use the probabilistic sampler. Defaults to True.

        Raises:
            ValueError: If `train_frac < 1.0` and contrastive loss mode is 'nn'.
        """
        adata_to_load = adata if adata is not None else self.adata

        if self.preprocessed:
            normalize_total = False
            log1p = False
        else:
            normalize_total = self.config.normalize_total
            log1p = self.config.log1p

        self.data_manager = DataLoaderManager(
            domain_key=self.config.domain_key, 
            class_key=self.config.class_key, covariate_keys=self.config.covariate_embedding_dims.keys(), 
            feature_list=self.config.input_feature,
            normalize_total=normalize_total,
            log1p=log1p,    
            batch_size=self.config.batch_size, train_frac=train_frac,
            use_sampler=use_sampler,
            sampler_emb=self.config.sampler_emb,
            sampler_knn=self.config.sampler_knn,
            sampler_domain_minibatch_strategy=self.config.sampler_domain_minibatch_strategy,
            domain_coverage=self.config.domain_coverage,
            dist_metric=self.config.dist_metric, 
            p_intra_knn=self.config.p_intra_knn, 
            p_intra_domain=self.config.p_intra_domain, 
            use_faiss=self.config.use_faiss, 
            use_ivf=self.config.use_ivf, 
            ivf_nprobe=self.config.ivf_nprobe, 
            preload_dense=self.config.preload_dense,
            device=self.config.device
        )

        self.data_structure = self.data_manager.data_structure

        if self.config.chunked:
            logger.info(f"Using chunked loading with chunk size: {self.config.chunk_size}")
            # Create the ChunkLoader and pass it the persistent data_manager
            self.loader = ChunkLoader(
                adata=adata_to_load,
                chunk_size=self.config.chunk_size,
                data_manager=self.data_manager
            )
        else:
            train_dataloader, val_dataloader, self.data_structure = self.data_manager.anndata_to_dataloader(adata_to_load)
            self.loader = [(train_dataloader, val_dataloader, np.arange(adata_to_load.shape[0]))]
            self.preprocessed = True  # Mark as preprocessed since we loaded the data into memory


    def train(self, save_model=True, patience=2):
        """
        Trains the model on the dataset.

        Args:
            save_model (bool, optional): Whether to save the trained model. Defaults to True.
            patience (int, optional): Number of epochs to wait for improvement before early stopping. Defaults to 2.
        """
        use_knn_sampler = self.config.p_intra_knn > 0
        original_p_intra_knn = self.config.p_intra_knn

        # A warm-up is needed ONLY if using the knn_sampler AND no initial embedding is provided.
        is_warmup_active = (
            use_knn_sampler and
            self.config.sampler_emb is None and
            self.config.knn_warmup_epochs > 0
        )

        if is_warmup_active:
            if self.config.knn_warmup_epochs >= self.config.n_epochs:
                raise ValueError("knn_warmup_epochs must be less than n_epochs.")
            logger.info(f"Starting {self.config.knn_warmup_epochs} warm-up epochs with k-NN sampling disabled.")
            self.config.p_intra_knn = 0.0
        else:
            self.config.knn_warmup_epochs = 0

        self.init_dataloader(train_frac=self.config.train_frac, use_sampler=True)
        self.init_trainer()

        best_val_loss = float('inf')
        best_model_state = None
        epochs_without_improvement = 0

        for epoch in range(self.config.n_epochs):
            logger.info(f'Starting epoch {epoch + 1}/{self.config.n_epochs}')

            # DYNAMIC KNN UPDATE (after warm-up is complete)
            if is_warmup_active and epoch == self.config.knn_warmup_epochs:
                logger.info("Warm-up complete. Computing k-NN graph on learned embeddings.")
                self.init_dataloader(self.adata, train_frac=1.0, use_sampler=False)
                embeddings, *_ = self.predict(self.loader)
                self.adata.obsm['X_concord_warmup'] = embeddings
                self.config.p_intra_knn = original_p_intra_knn
                self.config.sampler_emb = 'X_concord_warmup'
                # reinitiate the dataloader with the updated p_intra_knn
                self.init_dataloader(adata=self.adata, train_frac=self.config.train_frac, use_sampler=True)
                
            for chunk_idx, (train_dataloader, val_dataloader, _) in enumerate(self.loader):
                logger.info(f'Processing chunk {chunk_idx + 1}/{len(self.loader)} for epoch {epoch + 1}')
                if train_dataloader is not None:
                    logger.info(f"Number of samples in train_dataloader: {len(train_dataloader.dataset)}")
                if val_dataloader is not None:
                    logger.info(f"Number of samples in val_dataloader: {len(val_dataloader.dataset)}")

                self.trainer.train_epoch(epoch, train_dataloader)
                
                if val_dataloader is not None:
                    val_loss = self.trainer.validate_epoch(epoch, val_dataloader)
                
                    # Check if the current validation loss is the best we've seen so far
                    if val_loss < best_val_loss:
                        best_val_loss = val_loss
                        best_model_state = copy.deepcopy(self.model.state_dict())
                        logger.info(f"New best model found at epoch {epoch + 1} with validation loss: {best_val_loss:.4f}")
                        epochs_without_improvement = 0  # Reset counter when improvement is found
                    else:
                        epochs_without_improvement += 1
                        logger.info(f"No improvement in validation loss for {epochs_without_improvement} epoch(s).")

                    # Early stopping condition
                    if epochs_without_improvement >= patience:
                        logger.info(f"Stopping early at epoch {epoch + 1} due to no improvement in validation loss.")
                        break

            self.trainer.scheduler.step()

            # Early stopping break condition
            if epochs_without_improvement > patience:
                break

        if best_model_state is not None:
            self.model.load_state_dict(best_model_state)
            logger.info("Best model state loaded into the model before final save.")

        if save_model and self.save_dir is not None:
            import time
            file_suffix = f"{time.strftime('%b%d-%H%M')}"
            model_save_path = self.save_dir / f"final_model_{file_suffix}.pt"
            self.save_model(self.model, model_save_path)

            config_save_path = self.save_dir / f"config_{file_suffix}.json"
            with open(config_save_path, 'w') as f:
                json.dump(self.config.to_dict(), f, indent=4)

            logger.info(f"Final model saved at: {model_save_path}; Configuration saved at: {config_save_path}.")
        elif save_model:
            logger.warning("save_dir is None. Skipping model/config saving.")



    def predict(self, loader, sort_by_indices=False, return_decoded=False, decoder_domain=None, return_class=True, return_class_prob=True):  
        """
        Runs inference on a dataset.

        Args:
            loader (DataLoader or list): Data loader or chunked loader for batch processing.
            sort_by_indices (bool, optional): Whether to return results in original cell order. Defaults to False.
            return_decoded (bool, optional): Whether to return decoded gene expression. Defaults to False.
            decoder_domain (str, optional): Specifies a domain for decoding. Defaults to None.
            return_class (bool, optional): Whether to return predicted class labels. Defaults to True.
            return_class_prob (bool, optional): Whether to return class probabilities. Defaults to True.

        Returns:
            tuple: Encoded embeddings, decoded matrix (if requested), class predictions, class probabilities, true labels, and latent variables.
        """
        self.model.eval()
        class_preds = []
        class_true = []
        class_probs = [] if return_class_prob else None
        embeddings = []
        decoded_mtx = []
        indices = []
        
        if isinstance(loader, list) or type(loader).__name__ == 'ChunkLoader':
            all_embeddings = []
            all_decoded = []
            all_class_preds = []
            all_class_probs = [] if return_class_prob else None
            all_class_true = []
            all_indices = []

            for chunk_idx, (dataloader, _, ck_indices) in enumerate(loader):
                logger.info(f'Predicting for chunk {chunk_idx + 1}/{len(loader)}')
                ck_embeddings, ck_decoded, ck_class_preds, ck_class_probs, ck_class_true = self.predict(dataloader, 
                                                                                        sort_by_indices=True, 
                                                                                        return_decoded=return_decoded, 
                                                                                        decoder_domain=decoder_domain,
                                                                                        return_class=return_class,
                                                                                        return_class_prob=return_class_prob)
                all_embeddings.append(ck_embeddings)
                all_decoded.append(ck_decoded) if return_decoded else None
                all_indices.extend(ck_indices)
                if ck_class_preds is not None:
                    all_class_preds.extend(ck_class_preds)
                if return_class_prob and ck_class_probs is not None:
                    all_class_probs.append(ck_class_probs)
                if ck_class_true is not None:
                    all_class_true.extend(ck_class_true)

            all_indices = np.array(all_indices)
            sorted_indices = np.argsort(all_indices)

            all_embeddings = np.concatenate(all_embeddings, axis=0)[sorted_indices]
            all_decoded = np.concatenate(all_decoded, axis=0)[sorted_indices] if all_decoded else None
            all_class_preds = np.array(all_class_preds)[sorted_indices] if all_class_preds else None
            all_class_true = np.array(all_class_true)[sorted_indices] if all_class_true else None
            if return_class_prob:
                all_class_probs = pd.concat(all_class_probs).iloc[sorted_indices].reset_index(drop=True) if all_class_probs else None
            
            return all_embeddings, all_decoded, all_class_preds, all_class_probs, all_class_true
        else:
            with torch.no_grad():
                if decoder_domain is not None:
                    logger.info(f"Projecting data back to expression space of specified domain: {decoder_domain}")
                    # map domain to actual domain id used in the model
                    decoder_domain_code = self.config.unique_domains.index(decoder_domain)
                    fixed_domain_id = torch.tensor([decoder_domain_code], dtype=torch.long, device=self.config.device)
                else:
                    if self.config.use_decoder:
                        logger.info("No domain specified for decoding. Using the same domain as the input data.")
                    fixed_domain_id = None
                
                for data in loader:
                    # Unpack data based on the provided structure
                    data_dict = {}
                    for key, value in data.items():
                        if isinstance(value, torch.Tensor):
                            data_dict[key] = value.to(self.config.device)
                        else:
                            # Keep non-tensor data as is (e.g., None for class_labels)
                            data_dict[key] = value

                    inputs = data_dict.get('input')
                    # Use fixed domain id if provided, and make it same length as inputs
                    domain_ids = data_dict.get('domain', None) if decoder_domain is None else fixed_domain_id.repeat(inputs.size(0))
                    class_labels = data_dict.get('class', None)
                    original_indices = data_dict.get('idx')
                    covariate_keys = [key for key in data_dict.keys() if key not in ['input', 'domain', 'class', 'idx']]
                    covariate_tensors = {key: data_dict[key] for key in covariate_keys}

                    if class_labels is not None:
                        class_true.extend(class_labels.cpu().numpy())

                    if original_indices is not None:
                        indices.extend(original_indices.cpu().numpy())

                    outputs = self.model(inputs, domain_ids, covariate_tensors)
                    if 'class_pred' in outputs and return_class:
                        class_preds_tensor = outputs['class_pred']
                        class_preds.extend(torch.argmax(class_preds_tensor, dim=1).cpu().numpy()) # TODO May need fix
                        if return_class_prob:
                            class_probs.extend(F.softmax(class_preds_tensor, dim=1).cpu().numpy())
                    if 'encoded' in outputs:
                        embeddings.append(outputs['encoded'].cpu().numpy())
                    if 'decoded' in outputs and return_decoded:
                        decoded_mtx.append(outputs['decoded'].cpu().numpy())
                            

            if not embeddings:
                raise ValueError("No embeddings were extracted. Check the model and dataloader.")

            # Concatenate embeddings
            embeddings = np.concatenate(embeddings, axis=0)

            if decoded_mtx:
                decoded_mtx = np.concatenate(decoded_mtx, axis=0)

            # Convert predictions and true labels to numpy arrays
            class_preds = np.array(class_preds) if class_preds else None
            class_probs = np.array(class_probs) if return_class_prob and class_probs else None
            class_true = np.array(class_true) if class_true else None

            if sort_by_indices and indices:
                # Sort embeddings and predictions back to the original order
                indices = np.array(indices)
                sorted_indices = np.argsort(indices)
                embeddings = embeddings[sorted_indices]
                if return_decoded:
                    decoded_mtx = decoded_mtx[sorted_indices]
                if class_preds is not None:
                    class_preds = class_preds[sorted_indices]
                if return_class_prob and class_probs is not None:
                    class_probs = class_probs[sorted_indices]
                if class_true is not None:
                    class_true = class_true[sorted_indices]

            if return_class and self.config.unique_classes is not None:
                class_preds = self.config.unique_classes[class_preds] if class_preds is not None else None
                if class_true is not None:
                    mask = class_true != self.config.unlabeled_class_code
                    mapped_true = np.empty(len(class_true), dtype=object)
                    mapped_true[~mask] = self.config.unlabeled_class
                    mapped_true[mask] = self.config.unique_classes[class_true[mask]]
                    class_true = mapped_true
                if return_class_prob and class_probs is not None:
                    class_probs = pd.DataFrame(class_probs, columns=self.config.unique_classes)

            return embeddings, decoded_mtx, class_preds, class_probs, class_true



    def _add_results_to_adata(self, adata_to_update, results_tuple, output_key, decoder_domain=None):
        """A helper function to add prediction results to an AnnData object."""
        
        embeddings, decoded, class_preds, class_probs, class_true = results_tuple
        
        adata_to_update.obsm[output_key] = embeddings
        if decoded is not None and len(decoded) > 0:
            if decoder_domain is not None:
                save_key = f"{output_key}_decoded_{decoder_domain}"
            else:
                save_key = f"{output_key}_decoded"
            adata_to_update.layers[save_key] = decoded
        
        if class_true is not None and len(class_true) > 0:
            adata_to_update.obs[f"{output_key}_class_true"] = class_true

        if class_preds is not None and len(class_preds) > 0:
            adata_to_update.obs[f"{output_key}_class_pred"] = class_preds
        
        if class_probs is not None and not class_probs.empty:
            class_probs.index = adata_to_update.obs.index
            # Rename columns and add all at once to avoid fragmentation
            prob_df = class_probs.rename(columns=lambda col: f"{output_key}_class_prob_{col}")
            adata_to_update.obs = pd.concat([adata_to_update.obs, prob_df], axis=1)

        logger.info(f"Predictions added to AnnData object with base key '{output_key}'.")


    def fit_transform(self, output_key="Concord", 
                     return_decoded=False, decoder_domain=None,
                     return_class=True, return_class_prob=True, 
                     save_model=True):
        """
        Encodes an AnnData object using the CONCORD model.

        Args:
            output_key (str, optional): Output key for storing results in AnnData. Defaults to 'Concord'.
            return_decoded (bool, optional): Whether to return decoded gene expression. Defaults to False.
            decoder_domain (str, optional): Specifies domain for decoding. Defaults to None.
            return_class (bool, optional): Whether to return predicted class labels. Defaults to True.
            return_class_prob (bool, optional): Whether to return class probabilities. Defaults to True.
            save_model (bool, optional): Whether to save the model after training. Defaults to True.
        """

        self.init_model()
        self.train(save_model=save_model)
        self.init_dataloader(adata=self.adata, train_frac=1.0, use_sampler=False)
        
        results_tuple = self.predict(
            self.loader, 
            return_decoded=return_decoded, decoder_domain=decoder_domain,
            return_class=return_class, 
            return_class_prob=return_class_prob
        )
        
        # --- 4. SAVE RESULTS ---
        #self._add_results_to_adata(self.adata, results_tuple, output_key, decoder_domain=decoder_domain)

        # if self.copy_adata:
        #     logger.info("Copying results back to the original AnnData object.")
        self._add_results_to_adata(self._adata_original, results_tuple, output_key, decoder_domain=decoder_domain)
            

    def get_domain_embeddings(self):
        """
        Retrieves domain embeddings from the trained model.

        Returns:
            pd.DataFrame: A dataframe containing domain embeddings.
        """
        unique_domain_categories = self.adata.obs[self.config.domain_key].cat.categories.values
        domain_labels = torch.tensor(range(len(unique_domain_categories)), dtype=torch.long).to(self.config.device)
        domain_embeddings = self.model.domain_embedding(domain_labels)
        domain_embeddings = domain_embeddings.cpu().detach().numpy()
        domain_df = pd.DataFrame(domain_embeddings, index=unique_domain_categories)
        return domain_df
    

    def get_covariate_embeddings(self):
        """
        Retrieves covariate embeddings from the trained model.

        Returns:
            dict: A dictionary of DataFrames, each containing embeddings for a covariate.
        """
        covariate_dfs = {}
        for covariate_key in self.config.covariate_embedding_dims.keys():
            if covariate_key in self.model.covariate_embeddings:
                unique_covariate_categories = self.adata.obs[covariate_key].cat.categories.values
                covariate_labels = torch.tensor(range(len(unique_covariate_categories)), dtype=torch.long).to(self.config.device)
                covariate_embeddings = self.model.covariate_embeddings[covariate_key](covariate_labels)
                covariate_embeddings = covariate_embeddings.cpu().detach().numpy()
                covariate_df = pd.DataFrame(covariate_embeddings, index=unique_covariate_categories)
                covariate_dfs[covariate_key] = covariate_df
        return covariate_dfs


    def save_model(self, model, save_path):
        """
        Saves the trained model to a file.

        Args:
            model (torch.nn.Module): The trained model.
            save_path (str or Path): Path to save the model file.

        Returns:
            None
        """
        torch.save(model.state_dict(), save_path)
        logger.info(f"Model saved to {save_path}")



    @classmethod
    def load(cls, model_dir: str, device=None) -> 'Concord':
        """
        Loads a pre-trained CONCORD model from a directory.

        This method finds the config.json and final_model.pt files within the
        specified directory, initializes the Concord object, and loads the model weights.

        Args:
            model_dir (str): Path to the directory where the model and config were saved.
            device (torch.device, optional): The device to load the model onto. 
                                             If None, uses the device from the saved config.

        Returns:
            A pre-trained, ready-to-use Concord object.
            
        Raises:
            FileNotFoundError: If the model or config file cannot be found.
        """
        from concord.utils import load_json
        model_dir = Path(model_dir)
        
        # --- 1. Find the config and model files ---
        config_files = list(model_dir.glob("config_*.json"))
        if not config_files:
            raise FileNotFoundError(f"No 'config_*.json' file found in {model_dir}")
        # Use the most recently created file if multiple exist
        config_file = max(config_files, key=lambda p: p.stat().st_mtime)

        model_files = list(model_dir.glob("final_model_*.pt"))
        if not model_files:
            raise FileNotFoundError(f"No 'final_model_*.pt' file found in {model_dir}")
        model_file = max(model_files, key=lambda p: p.stat().st_mtime)

        logger.info(f"Loading configuration from: {config_file}")
        logger.info(f"Loading model weights from: {model_file}")

        # --- 2. Load the configuration ---
        with open(config_file, 'r') as f:
            concord_args = load_json(str(config_file))
            concord_args['pretrained_model'] = model_file

        # infer input_dim from the checkpoint
        sd = torch.load(model_file, map_location='cpu')
        if "encoder.0.weight" in sd:
            input_dim = sd["encoder.0.weight"].shape[1]
        else:
            raise KeyError("'encoder.0.weight' not found in checkpoint")
        
        # Override device if specified by user
        if device:
            concord_args['device'] = device

        if 'unique_classes' in concord_args and isinstance(concord_args['unique_classes'], list):
            concord_args['unique_classes'] = pd.Index(concord_args['unique_classes'])
        if 'unique_classes_code' in concord_args and isinstance(concord_args['unique_classes_code'], list):
            concord_args['unique_classes_code'] = np.array(concord_args['unique_classes_code'])
        concord_args['device'] = torch.device(concord_args['device'])
        
        # --- 3. Create and configure the Concord object ---
        # We create a "skeletal" object without an AnnData object, as we are
        # loading a pre-trained model and will predict on new data later.
        # We bypass the complex __init__ by using __new__ and manually setting attributes.
        instance = cls.__new__(cls)
        instance.config = Config(concord_args)
        instance.save_dir = model_dir
        instance.adata = None  # No adata object at load time
        
        instance.pretrained_model = model_file
        # --- 4. Initialize the model with the loaded config and load weights ---
        instance.init_model(input_dim=input_dim)
        
        logger.info("Pre-trained Concord model loaded successfully.")
        
        return instance


    def predict_adata(self, adata_new: 'AnnData', 
                    output_key="Concord_pred",
                    return_decoded=False, 
                    domain_key=None,
                    decoder_domain=None,
                    return_class=True, return_class_prob=True):
        """
        Generates predictions for a new, unseen AnnData object using the loaded model.

        Args:
            adata_new (AnnData): The new data to predict on.
            (See `predict` method for other arguments)

        Returns:
            dict: A dictionary containing the requested outputs ('embeddings', 'decoded', etc.).
        """
        if self.model is None:
            raise RuntimeError("Model has not been loaded or initialized. Call `Concord.load()` first.")

        # Validate that new data has the required features
        if self.config.input_feature is not None:
            missing_features = set(self.config.input_feature) - set(adata_new.var_names)
            if missing_features:
                raise ValueError(f"The new anndata object is missing {len(missing_features)} features "
                                f"required by the model. Missing features: {list(missing_features)[:5]}...")
            
        self.preprocessed = False # Use the same preprocessing as during training
        
        self.config.domain_key = domain_key
        self._check_domain_settings(adata_new)

        self.init_dataloader(adata=adata_new, train_frac=1.0, use_sampler=False)

        results_tuple = self.predict(
            loader=self.loader, # Pass the dataloader for the new data
            sort_by_indices=False, # Prediction is already in order
            return_decoded=return_decoded,
            decoder_domain=decoder_domain,
            return_class=return_class,
            return_class_prob=return_class_prob
        )
        
        self._add_results_to_adata(adata_new, results_tuple, output_key, decoder_domain=decoder_domain)


    # Helper check methods:

    def _check_adata(self, adata: sc.AnnData, copy_adata: bool = True):
        if adata.isbacked:
            logger.info("Running CONCORD on a backed AnnData object. ")
            if not copy_adata:
                logger.warning("Input AnnData object is in backed mode. Preprocessing will not modify the file on disk.")
            
            if adata.is_view:
                raise ValueError(
                    "CONCORD does not support operating on a view of a backed AnnData object. "
                    "This is due to limitations in modifying on-disk data structures. \n"
                    "Please either use the full backed AnnData object directly, or load your "
                    "desired subset into memory first using `adata_view.to_memory()`."
                )
            
            self.adata = adata
        else:
            if copy_adata:
                logger.info("Creating a copy of the AnnData object to work on. Final results will be copied back.")
                self.adata = adata.copy()
            else:
                logger.info("Operating directly on the provided AnnData object. Object may be modified.")
                self.adata = adata

    def _check_input_format(self, adata: sc.AnnData):
        """Warn if counts look raw vs. normalized/log‑transformed."""
        detected_format = check_adata_X(adata)
        if detected_format == 'raw' and not self.config.normalize_total:
            logger.warning("Input data in adata.X appears to be raw counts. "
                           "CONCORD performs best on normalized and log-transformed data. "
                           "Consider setting normalize_total=True and log1p=True.")
        elif detected_format == 'normalized' and (self.config.normalize_total or self.config.log1p):
            logger.warning("Input data in adata.X appears to be already normalized, "
                           "but preprocessing flags (normalize_total or log1p) are set to True. "
                           "This may lead to unexpected results. Please ensure this is intended.")

    def _check_knn_sampler_settings(self):
        """Sanity‑check k‑NN sampling related parameters."""
        if self.config.p_intra_knn <= 0:
            return
        else: 
            logger.info("KNN sampling mode is enabled.")

        ncells = self.adata.n_obs
        if ncells > 100_000:
            logger.warning(f"Dataset contains {ncells} cells, which is large. "
                               "Using k-NN sampling may be computationally expensive and non-optimal. "
                               "We recommend using the HCL mode by setting `clr_beta = 1.0 or 2.0`, and setting `p_intra_knn = 0.0` to disable k-NN sampling.")
        if self.config.sampler_knn is None:
            self.config.sampler_knn = min(1000, ncells // 10)
            logger.info(f"Setting sampler_knn to {self.config.sampler_knn} to be 1/10 the number of cells in the dataset. You can change this value by setting sampler_knn in the configuration.")

        validate_probability_dict_compatible(self.config.p_intra_knn, "p_intra_knn")
        if self.config.train_frac < 1.0 and self.config.p_intra_knn > 0:
            logger.warning("kNN mode is currently not supported for training fraction less than 1.0, consider run in HCL mode. Setting p_intra_knn to 0.")
            self.config.p_intra_knn = 0
    
    def _check_hcl_sampler_settings(self):
        """Sanity-check HCL sampling related parameters."""
        if self.config.clr_beta > 0:
            logger.info(f"Using NT-Xent loss with beta={self.config.clr_beta}. This will apply hard-negative weighting to the contrastive loss.")
            if self.config.p_intra_domain < .95:
                logger.warning("Using NT-Xent loss with beta > 0 and p_intra_domain < 0.95 may lead to non-ideal batch correction. Consider setting p_intra_domain to 1.0 for best integration results.")
            logger.info("HCL (Contrastive learning with hard negative samples) mode is enabled.")

        if self.config.clr_temperature <= 0 or self.config.clr_temperature > 1:
            raise ValueError("clr_temperature must be in the range (0, 1]. "
                             "This is a scaling factor for the contrastive loss. "
                             "Consider setting it to a value between 0.2 to 0.6 for best results.")
        
    def _check_input_features(self):
        if self.config.input_feature is None:
            logger.warning("No input feature list provided. It is recommended to first select features using the command `concord.ul.select_features()`.")
            logger.info(f"Proceeding with all {self.adata.shape[1]} features in the dataset.")

    def _check_importance_mask(self):
        if self.config.use_importance_mask:
            logger.warning("Importance mask is enabled. This will apply differential weighting to features based on their importance. Note this feature is experimental.")
            if self.config.importance_penalty_weight == 0.0:
                logger.warning("Importance mask is enabled but importance_penalty_weight is set to 0.0. This will still cause differential weighting of features, but without penalty.")

    def _check_domain_settings(self, adata: sc.AnnData = None):
        if self.config.domain_key is not None:
            if(self.config.domain_key not in adata.obs.columns):
                raise ValueError(f"Domain key {self.config.domain_key} not found in adata.obs. Please provide a valid domain key.")
            ensure_categorical(adata, obs_key=self.config.domain_key, drop_unused=True)
        else:
            logger.warning("domain/batch information not found, all samples will be treated as from single domain/batch.")
            self.config.domain_key = 'tmp_domain_label'
            adata.obs[self.config.domain_key] = pd.Series(data='single_domain', index=adata.obs_names).astype('category')
            self.config.p_intra_domain = 1.0

        self.config.unique_domains = adata.obs[self.config.domain_key].cat.categories.tolist()
        self.config.num_domains = len(adata.obs[self.config.domain_key].cat.categories)

        validate_probability_dict_compatible(self.config.p_intra_domain, "p_intra_domain")
        
        if self.config.num_domains == 1:
            logger.warning(f"Only one domain found in the data. Setting p_intra_domain to 1.0.")
            self.config.p_intra_domain = 1.0

    def _check_class_settings(self):
        if self.config.use_classifier:
            if self.config.class_key is None:
                raise ValueError("Cannot use classifier without providing a class key.")
            if(self.config.class_key not in self.adata.obs.columns):
                raise ValueError(f"Class key {self.config.class_key} not found in adata.obs. Please provide a valid class key.")
            ensure_categorical(self.adata, obs_key=self.config.class_key, drop_unused=True)

            cats = list(self.adata.obs[self.config.class_key].cat.categories)
            if self.config.unlabeled_class is not None:
                if self.config.unlabeled_class in cats:
                    # Reorder: remove unlabeled and append to end
                    cats.remove(self.config.unlabeled_class)
                    cats.append(self.config.unlabeled_class)
                    # Update the categorical with new order
                    self.adata.obs[self.config.class_key] = pd.Categorical(
                        self.adata.obs[self.config.class_key].values,  # Use .values to avoid index issues
                        categories=cats
                    )
                else:
                    raise ValueError(f"Unlabeled class {self.config.unlabeled_class} not found in the class key.")
            
            # Now set configs based on (possibly reordered) categories
            cats = self.adata.obs[self.config.class_key].cat.categories  # Reload after update
            self.config.unique_classes = pd.Index(cats)            
            self.config.unique_classes_code = np.arange(len(cats), dtype=int)
            if self.config.unlabeled_class is not None:
                self.config.unlabeled_class_code = len(cats) - 1
                self.config.unique_classes_code = self.config.unique_classes_code[:-1]
                self.config.unique_classes = self.config.unique_classes[:-1]
            else:
                self.config.unlabeled_class_code = None

            self.config.num_classes = len(self.config.unique_classes_code)
        else:
            self.config.unique_classes = None
            self.config.unique_classes_code = None
            self.config.unlabeled_class_code = None
            self.config.num_classes = None

    def _check_covariate_settings(self):
        # Compute the number of categories for each covariate
        self.config.covariate_num_categories = {}
        for covariate_key in self.config.covariate_embedding_dims.keys():
            if covariate_key in self.adata.obs:
                ensure_categorical(self.adata, obs_key=covariate_key, drop_unused=True)
                self.config.covariate_num_categories[covariate_key] = len(self.adata.obs[covariate_key].cat.categories)
