from pathlib import Path
import torch
import torch.nn.functional as F
from .model.model import ConcordModel
from .utils.preprocessor import Preprocessor
from .utils.anndata_utils import ensure_categorical
from .model.dataloader import DataLoaderManager 
from .model.chunkloader import ChunkLoader
from .utils.other_util import add_file_handler, set_seed
from .model.trainer import Trainer
import numpy as np
import scanpy as sc
import pandas as pd
import copy
import json
from . import logger
from . import set_verbose_mode

class Config:
    def __init__(self, config_dict):
        for key, value in config_dict.items():
            setattr(self, key, value)

    def __getitem__(self, item):
        return getattr(self, item)

    def to_dict(self):
        def serialize(value):
            if isinstance(value, (torch.device,)):
                return str(value)
            # Add more cases if needed for other non-serializable types
            return value

        return {key: serialize(getattr(self, key)) for key in dir(self)
                if not key.startswith('__') and not callable(getattr(self, key))}


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
    def __init__(self, adata, save_dir='save/', inplace=True, verbose=False, **kwargs):
        """
        Initializes the Concord framework.

        Args:
            adata (AnnData): Input single-cell data in AnnData format.
            save_dir (str, optional): Directory to save model outputs. Defaults to 'save/'.
            inplace (bool, optional): If True, modifies `adata` in place. Defaults to True.
            verbose (bool, optional): Enable verbose logging. Defaults to False.
            **kwargs: Additional configuration parameters.

        Raises:
            ValueError: If `inplace` is set to True on a backed AnnData object.
        """
        set_verbose_mode(verbose)
        if adata.isbacked:
            logger.warning("Input AnnData object is backed. With same amount of epochs, Concord will perform better when adata is loaded into memory.")
            if inplace:
                raise ValueError("Inplace mode is not supported for backed AnnData object. Set inplace to False.")
            self.adata = adata
        else:
            self.adata = adata if inplace else adata.copy()

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
            project_name="concord",
            input_feature=None,
            batch_size=64,
            n_epochs=10,
            lr=1e-2,
            schedule_ratio=0.97,
            train_frac=1.0,
            latent_dim=100,
            encoder_dims=[512],
            decoder_dims=[512],
            augmentation_mask_prob=0.4,  
            domain_key=None,
            class_key=None,
            domain_embedding_dim=8,
            covariate_embedding_dims={},
            use_decoder=False, # Default decoder usage
            decoder_final_activation='relu',
            decoder_weight=1.0,
            clr_mode="aug", # Consider fix
            clr_temperature=0.5,
            clr_weight=1.0,
            use_classifier=False,
            classifier_weight=1.0,
            unlabeled_class=None,
            use_importance_mask=False,
            importance_penalty_weight=0,
            importance_penalty_type='L1',
            dropout_prob=0.1,
            norm_type="layer_norm",  # Default normalization type
            sampler_emb="X_pca",
            sampler_knn=None, # Default neighborhood size, can be adjusted
            sampler_domain_minibatch_strategy='proportional', # Strategy for domain minibatch sampling
            domain_coverage = None,
            dist_metric='euclidean',
            p_intra_knn=0.3,
            p_intra_domain=0.95,
            pca_n_comps=50,
            use_faiss=True,
            use_ivf=True,
            ivf_nprobe=10,
            pretrained_model=None,
            classifier_freeze_param=False,
            chunked=False,
            chunk_size=10000,
            #encoder_append_cov=False, # Should always be False for now
            device=torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        )

        self.setup_config(**kwargs)
        set_seed(self.config.seed)

        if self.config.sampler_knn is None:
            self.config.sampler_knn = self.adata.n_obs // 10 
            logger.info(f"Setting sampler_knn to {self.config.sampler_knn} to be 1/10 the number of cells in the dataset. You can change this value by setting sampler_knn in the configuration.")

        # Checks to convert None values to default values
        if self.config.input_feature is None:
            logger.warning("No input feature list provided. It is recommended to first select features using the command `concord.ul.select_features()`.")
            logger.info(f"Proceeding with all {self.adata.shape[1]} features in the dataset.")
            self.config.input_feature = self.adata.var_names.tolist()

        if self.config.importance_penalty_weight == 0 and self.config.use_importance_mask:
            logger.warning("Importance mask is enabled but importance_penalty_weight is set to 0.0. This will still cause differential weighting of features, but without penalty.")

        if self.config.domain_key is not None:
            if(self.config.domain_key not in self.adata.obs.columns):
                raise ValueError(f"Domain key {self.config.domain_key} not found in adata.obs. Please provide a valid domain key.")
            ensure_categorical(self.adata, obs_key=self.config.domain_key, drop_unused=True)
        else:
            logger.warning("domain/batch information not found, all samples will be treated as from single domain/batch.")
            self.config.domain_key = 'tmp_domain_label'
            self.adata.obs[self.config.domain_key] = pd.Series(data='single_domain', index=self.adata.obs_names).astype('category')
            self.p_intra_domain = 1.0

        self.num_domains = len(self.adata.obs[self.config.domain_key].cat.categories)

        if self.config.train_frac < 1.0 and self.config.p_intra_knn > 0:
            logger.warning("Nearest neighbor contrastive loss is currently not supported for training fraction less than 1.0. Setting p_intra_knn to 0.")
            self.config.p_intra_knn = 0

        # Check if batch_size conflicts with p_intra_knn
        batch_knn_count = int(self.config.p_intra_knn * self.config.batch_size)
        if self.config.p_intra_knn > 0 and batch_knn_count > self.config.sampler_knn:
            logger.warning(f"p_intra_knn * batch_size ({batch_knn_count}) is greater than sampler_knn ({self.config.sampler_knn}). This will cause actual sampling ratio not matching specified p_intra_knn.")
            logger.warning(f"You should either set batch_size to be smaller than sampler_knn/p_intra_knn ({int(self.config.sampler_knn/self.config.p_intra_knn)})")
            logger.warning(f"or set sampler_knn to be greater than p_intra_knn * batch_size ({batch_knn_count}).")
            #self.config.p_intra_knn = 0

        if self.config.use_classifier:
            if self.config.class_key is None:
                raise ValueError("Cannot use classifier without providing a class key.")
            if(self.config.class_key not in self.adata.obs.columns):
                raise ValueError(f"Class key {self.config.class_key} not found in adata.obs. Please provide a valid class key.")
            ensure_categorical(self.adata, obs_key=self.config.class_key, drop_unused=True)

            self.unique_classes = self.adata.obs[self.config.class_key].cat.categories
            self.unique_classes_code = [code for code, _ in enumerate(self.unique_classes)]
            if self.config.unlabeled_class is not None:
                if self.config.unlabeled_class in self.unique_classes:
                    self.unlabeled_class_code = self.unique_classes.get_loc(self.config.unlabeled_class)
                    self.unique_classes_code = self.unique_classes_code[self.unique_classes_code != self.unlabeled_class_code]
                    self.unique_classes = self.unique_classes[self.unique_classes != self.config.unlabeled_class]
                else:
                    raise ValueError(f"Unlabeled class {self.config.unlabeled_class} not found in the class key.")
            else:
                self.unlabeled_class_code = None

            self.num_classes = len(self.unique_classes_code)
        else:
            self.unique_classes = None
            self.unique_classes_code = None
            self.unlabeled_class_code = None
            self.num_classes = None

        # Compute the number of categories for each covariate
        self.covariate_num_categories = {}
        for covariate_key in self.config.covariate_embedding_dims.keys():
            if covariate_key in self.adata.obs:
                ensure_categorical(self.adata, obs_key=covariate_key, drop_unused=True)
                self.covariate_num_categories[covariate_key] = len(self.adata.obs[covariate_key].cat.categories)
        
        # to be used by chunkloader for data transformation
        self.preprocessor = Preprocessor(
            use_key="X",
            normalize_total=1e4,
            result_normed_key="X_normed",
            log1p=True,
            result_log1p_key="X_log1p"
        )


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


    def init_model(self):
        """
        Initializes the CONCORD model and loads a pre-trained model if specified.

        Raises:
            FileNotFoundError: If the specified pre-trained model file is missing.
        """
        input_dim = len(self.config.input_feature)
        hidden_dim = self.config.latent_dim

        self.model = ConcordModel(input_dim, hidden_dim, 
                                  num_domains=self.num_domains,
                                  num_classes=self.num_classes,
                                  domain_embedding_dim=self.config.domain_embedding_dim,
                                  covariate_embedding_dims=self.config.covariate_embedding_dims,
                                  covariate_num_categories=self.covariate_num_categories,
                                  #encoder_append_cov=self.config.encoder_append_cov,
                                  encoder_dims=self.config.encoder_dims,
                                  decoder_dims=self.config.decoder_dims,
                                  decoder_final_activation=self.config.decoder_final_activation,
                                  augmentation_mask_prob=self.config.augmentation_mask_prob,
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
        self.trainer = Trainer(model=self.model,
                               data_structure=self.data_structure,
                               device=self.config.device,
                               logger=logger,
                               lr=self.config.lr,
                               schedule_ratio=self.config.schedule_ratio,
                               use_classifier=self.config.use_classifier, 
                               classifier_weight=self.config.classifier_weight,
                               unique_classes=self.unique_classes_code,
                               unlabeled_class=self.unlabeled_class_code,
                               use_decoder=self.config.use_decoder,
                               decoder_weight=self.config.decoder_weight,
                               clr_mode=self.config.clr_mode, 
                               clr_temperature=self.config.clr_temperature,
                               clr_weight=self.config.clr_weight,
                               importance_penalty_weight=self.config.importance_penalty_weight,
                               importance_penalty_type=self.config.importance_penalty_type)


    def init_dataloader(self, input_layer_key='X_log1p', preprocess=True, train_frac=1.0, use_sampler=True):
        """
        Initializes the data loader for training and evaluation.

        Args:
            input_layer_key (str, optional): Key in `adata.layers` to use as input. Defaults to 'X_log1p'.
            preprocess (bool, optional): Whether to apply preprocessing. Defaults to True.
            train_frac (float, optional): Fraction of data to use for training. Defaults to 1.0.
            use_sampler (bool, optional): Whether to use the probabilistic sampler. Defaults to True.

        Raises:
            ValueError: If `train_frac < 1.0` and contrastive loss mode is 'nn'.
        """
        if train_frac < 1.0 and self.config.clr_mode == 'nn':
            raise ValueError("Nearest neighbor contrastive loss is not supported for training fraction less than 1.0.")
        self.data_manager = DataLoaderManager(
            input_layer_key=input_layer_key, domain_key=self.config.domain_key, 
            class_key=self.config.class_key, covariate_keys=self.config.covariate_embedding_dims.keys(), 
            feature_list=self.config.input_feature,
            batch_size=self.config.batch_size, train_frac=train_frac,
            use_sampler=use_sampler,
            sampler_emb=self.config.sampler_emb, 
            sampler_knn=self.config.sampler_knn,
            sampler_domain_minibatch_strategy=self.config.sampler_domain_minibatch_strategy,
            domain_coverage=self.config.domain_coverage,
            
            dist_metric=self.config.dist_metric, 
            p_intra_knn=self.config.p_intra_knn, 
            p_intra_domain=self.config.p_intra_domain, 
            clr_mode=self.config.clr_mode, 
            pca_n_comps=self.config.pca_n_comps,
            use_faiss=self.config.use_faiss, 
            use_ivf=self.config.use_ivf, 
            ivf_nprobe=self.config.ivf_nprobe, 
            preprocess=self.preprocessor if preprocess else None,
            num_cores=self.num_classes, 
            device=self.config.device
        )

        if self.config.chunked:
            self.loader = ChunkLoader(
                adata=self.adata,
                chunk_size=self.config.chunk_size,
                data_manager=self.data_manager
            )
            self.data_structure = self.loader.data_structure  # Retrieve data_structure
        else:
            train_dataloader, val_dataloader, self.data_structure = self.data_manager.anndata_to_dataloader(self.adata)
            self.loader = [(train_dataloader, val_dataloader, np.arange(self.adata.shape[0]))]


    def train(self, save_model=True, patience=2):
        """
        Trains the model on the dataset.

        Args:
            save_model (bool, optional): Whether to save the trained model. Defaults to True.
            patience (int, optional): Number of epochs to wait for improvement before early stopping. Defaults to 2.
        """
        best_val_loss = float('inf')
        best_model_state = None
        epochs_without_improvement = 0

        for epoch in range(self.config.n_epochs):
            logger.info(f'Starting epoch {epoch + 1}/{self.config.n_epochs}')
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



    def predict(self, loader, sort_by_indices=False, return_decoded=False, decoder_domain=None, return_latent=False, return_class=True, return_class_prob=True):  
        """
        Runs inference on a dataset.

        Args:
            loader (DataLoader or list): Data loader or chunked loader for batch processing.
            sort_by_indices (bool, optional): Whether to return results in original cell order. Defaults to False.
            return_decoded (bool, optional): Whether to return decoded gene expression. Defaults to False.
            decoder_domain (str, optional): Specifies a domain for decoding. Defaults to None.
            return_latent (bool, optional): Whether to return latent variables. Defaults to False.
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

        latent_matrices = {}
        
        if isinstance(loader, list) or type(loader).__name__ == 'ChunkLoader':
            all_embeddings = []
            all_decoded = []
            all_class_preds = []
            all_class_probs = [] if return_class_prob else None
            all_class_true = []
            all_indices = []
            all_latent_matrices = {}

            for chunk_idx, (dataloader, _, ck_indices) in enumerate(loader):
                logger.info(f'Predicting for chunk {chunk_idx + 1}/{len(loader)}')
                ck_embeddings, ck_decoded, ck_class_preds, ck_class_probs, ck_class_true, ck_latent = self.predict(dataloader, 
                                                                                        sort_by_indices=True, 
                                                                                        return_decoded=return_decoded, 
                                                                                        decoder_domain=decoder_domain,
                                                                                        return_latent=return_latent,
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
                if return_latent:
                    for key in ck_latent.keys():
                        if key not in all_latent_matrices:
                            all_latent_matrices[key] = []
                        all_latent_matrices[key].append(ck_latent[key])

            all_indices = np.array(all_indices)
            sorted_indices = np.argsort(all_indices)

            all_embeddings = np.concatenate(all_embeddings, axis=0)[sorted_indices]
            all_decoded = np.concatenate(all_decoded, axis=0)[sorted_indices] if all_decoded else None
            all_class_preds = np.array(all_class_preds)[sorted_indices] if all_class_preds else None
            all_class_true = np.array(all_class_true)[sorted_indices] if all_class_true else None
            if return_class_prob:
                all_class_probs = pd.concat(all_class_probs).iloc[sorted_indices].reset_index(drop=True) if all_class_probs else None
            
            if return_latent:
                for key in all_latent_matrices.keys():
                    all_latent_matrices[key] = np.concatenate(all_latent_matrices[key], axis=0)[sorted_indices]
            return all_embeddings, all_decoded, all_class_preds, all_class_probs, all_class_true, all_latent_matrices
        else:
            with torch.no_grad():
                if decoder_domain is not None:
                    logger.info(f"Projecting data back to expression space of specified domain: {decoder_domain}")
                    # map domain to actual domain id used in the model
                    fixed_domain_id = torch.tensor([self.adata.obs[self.config.domain_key].cat.categories.get_loc(decoder_domain)], dtype=torch.long).to(
                        self.config.device)
                else:
                    if self.config.use_decoder:
                        logger.info("No domain specified for decoding. Using the same domain as the input data.")
                    fixed_domain_id = None
                
                for data in loader:
                    # Unpack data based on the provided structure
                    data_dict = {key: value.to(self.config.device) for key, value in zip(self.data_structure, data)}

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

                    outputs = self.model(inputs, domain_ids, covariate_tensors, return_latent=return_latent)
                    if 'class_pred' in outputs and return_class:
                        class_preds_tensor = outputs['class_pred']
                        class_preds.extend(torch.argmax(class_preds_tensor, dim=1).cpu().numpy()) # TODO May need fix
                        if return_class_prob:
                            class_probs.extend(F.softmax(class_preds_tensor, dim=1).cpu().numpy())
                    if 'encoded' in outputs:
                        embeddings.append(outputs['encoded'].cpu().numpy())
                    if 'decoded' in outputs and return_decoded:
                        decoded_mtx.append(outputs['decoded'].cpu().numpy())
                    if 'latent' in outputs and return_latent:
                        latent = outputs['latent']
                        for key, val in latent.items():
                            if key not in latent_matrices:
                                latent_matrices[key] = []
                            latent_matrices[key].append(val.cpu().numpy())   
                            

            if not embeddings:
                raise ValueError("No embeddings were extracted. Check the model and dataloader.")

            # Concatenate embeddings
            embeddings = np.concatenate(embeddings, axis=0)

            if decoded_mtx:
                decoded_mtx = np.concatenate(decoded_mtx, axis=0)

            if return_latent:
                for key in latent_matrices.keys():
                    latent_matrices[key] = np.concatenate(latent_matrices[key], axis=0)

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
                if return_latent:
                    for key in latent_matrices.keys():
                        latent_matrices[key] = latent_matrices[key][sorted_indices]

            if return_class and self.unique_classes is not None:
                class_preds = self.unique_classes[class_preds] if class_preds is not None else None
                class_true = self.unique_classes[class_true] if class_true is not None else None
                if return_class_prob and class_probs is not None:
                    class_probs = pd.DataFrame(class_probs, columns=self.unique_classes)

            return embeddings, decoded_mtx, class_preds, class_probs, class_true, latent_matrices


    def encode_adata(self, input_layer_key="X_log1p", output_key="Concord", preprocess=True, 
                     return_decoded=False, decoder_domain=None,
                     return_latent=False, 
                     return_class=True, return_class_prob=True, 
                     save_model=True):
        """
        Encodes an AnnData object using the CONCORD model.

        Args:
            input_layer_key (str, optional): Input layer key. Defaults to 'X_log1p'.
            output_key (str, optional): Output key for storing results in AnnData. Defaults to 'Concord'.
            preprocess (bool, optional): Whether to apply preprocessing. Defaults to True.
            return_decoded (bool, optional): Whether to return decoded gene expression. Defaults to False.
            decoder_domain (str, optional): Specifies domain for decoding. Defaults to None.
            return_latent (bool, optional): Whether to return latent variables. Defaults to False.
            return_class (bool, optional): Whether to return predicted class labels. Defaults to True.
            return_class_prob (bool, optional): Whether to return class probabilities. Defaults to True.
            save_model (bool, optional): Whether to save the model after training. Defaults to True.
        """

        # Initialize the model
        self.init_model()
        # Initialize the dataloader
        self.init_dataloader(input_layer_key=input_layer_key, preprocess=preprocess, train_frac=self.config.train_frac, use_sampler=True)
        # Initialize the trainer
        self.init_trainer()
        # Train the model
        self.train(save_model=save_model)
        # Reinitialize the dataloader without using the sampler
        self.init_dataloader(input_layer_key=input_layer_key, preprocess=preprocess, train_frac=1.0, use_sampler=False)
        
        # Predict and store the results
        encoded, decoded, class_preds, class_probs, class_true, latent_matrices = self.predict(self.loader, 
                                                                                               return_decoded=return_decoded, decoder_domain=decoder_domain,
                                                                                               return_latent=return_latent,
                                                                                               return_class=return_class, return_class_prob=return_class_prob)
        self.adata.obsm[output_key] = encoded
        if return_decoded:
            if decoder_domain is not None:
                save_key = f"{output_key}_decoded_{decoder_domain}"
            else:
                save_key = f"{output_key}_decoded"
            self.adata.layers[save_key] = decoded
        if return_latent:
            for key, val in latent_matrices.items():
                self.adata.obsm[output_key+'_'+key] = val
        if class_true is not None:
            self.adata.obs[output_key+'_class_true'] = class_true
        if class_preds is not None:
            self.adata.obs[output_key+'_class_pred'] = class_preds
        if class_probs is not None:
            class_probs.index = self.adata.obs.index
            for col in class_probs.columns:
                self.adata.obs[f'class_prob_{col}'] = class_probs[col]

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
        
        # Override device if specified by user
        if device:
            concord_args['device'] = device
        
        # --- 3. Create and configure the Concord object ---
        # We create a "skeletal" object without an AnnData object, as we are
        # loading a pre-trained model and will predict on new data later.
        # We bypass the complex __init__ by using __new__ and manually setting attributes.
        instance = cls.__new__(cls)
        instance.config = Config(concord_args)
        instance.save_dir = model_dir
        instance.adata = None  # No adata object at load time
        
        # Manually set attributes that are derived in __init__ from the adata object,
        # but are now fixed by the saved configuration.
        instance.num_domains = len(instance.config.to_dict().get('domain_key', [])) # A bit of a placeholder
        instance.num_classes = len(instance.config.to_dict().get('unique_classes', []))
        instance.unique_classes = instance.config.to_dict().get('unique_classes')
        instance.covariate_num_categories = instance.config.to_dict().get('covariate_num_categories', {})
        
        instance.pretrained_model = model_file
        # --- 4. Initialize the model with the loaded config and load weights ---
        instance.init_model()
        
        logger.info("Pre-trained Concord model loaded successfully.")
        
        return instance



    def predict_new(self, adata_new: 'AnnData', input_layer_key="X", preprocess=True,
                    return_decoded=False, decoder_domain=None, return_latent=False,
                    return_class=True, return_class_prob=True):
        """
        Generates predictions for a new, unseen AnnData object using the loaded model.

        Args:
            adata_new (AnnData): The new data to predict on.
            input_layer_key (str): The key in adata.layers to use as input.
            preprocess (bool): Whether to apply preprocessing.
            (See `predict` method for other arguments)

        Returns:
            dict: A dictionary containing the requested outputs ('embeddings', 'decoded', etc.).
        """
        if self.model is None:
            raise RuntimeError("Model has not been loaded or initialized. Call `Concord.load()` first.")

        # --- 1. Validate that new data has the required features ---
        missing_features = set(self.config.input_feature) - set(adata_new.var_names)
        if missing_features:
            raise ValueError(f"The new anndata object is missing {len(missing_features)} features "
                             f"required by the model. Missing features: {list(missing_features)[:5]}...")
        
        self.adata = adata_new  # Set the new data as the current adata object
        self.init_dataloader(input_layer_key=input_layer_key, preprocess=preprocess, train_frac=1.0, use_sampler=False)
        # --- 3. Run prediction ---
        logger.info("Generating predictions for the new data...")
        embeddings, decoded, class_preds, class_probs, class_true, latent_matrices = self.predict(
            loader=self.loader, # Pass the dataloader for the new data
            sort_by_indices=False, # Prediction is already in order
            return_decoded=return_decoded,
            decoder_domain=decoder_domain,
            return_latent=return_latent,
            return_class=return_class,
            return_class_prob=return_class_prob
        )
        
        # --- 4. Package and return results ---
        results = {
            'embeddings': embeddings,
            'decoded': decoded,
            'class_predictions': class_preds,
            'class_probabilities': class_probs,
            'class_true': class_true, # Ground truth from the new data, if available
            'latent_variables': latent_matrices
        }
        
        return results


    

