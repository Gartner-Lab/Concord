import wandb
from pathlib import Path
import torch
import torch.nn.functional as F
from .model.model import ConcordModel
from .utils.preprocessor import Preprocessor
from .utils.anndata_utils import ensure_categorical
from .model.dataloader import anndata_to_dataloader
from .model.chunkloader import ChunkLoader
from .utils.other_util import add_file_handler, update_wandb_params
from .utils.value_check import validate_probability, validate_probability_dict_compatible, check_dict_condition
from .utils.coverage_estimator import calculate_dataset_coverage, coverage_to_p_intra
from .model.trainer import Trainer
import numpy as np
import scanpy as sc
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
    def __init__(self, adata, save_dir='save/', inplace=True, use_wandb=False, verbose=True, **kwargs):
        set_verbose_mode(verbose)
        self.adata = adata if inplace else adata.copy()
        self.save_dir = Path(save_dir)
        self.config = None
        self.loader = None
        self.model = None
        self.use_wandb = use_wandb
        self.run = None
        self.sampler_kwargs = {}

        if not self.save_dir.exists():
            self.save_dir.mkdir(parents=True, exist_ok=True)

        add_file_handler(logger, self.save_dir / "run.log")
        self.setup_config(**kwargs)

        if self.config.input_feature is None:
            logger.warning("No input feature list provided. It is recommended to first select features using the command `concord.ul.select_features()`.")
            logger.info(f"Proceeding with all {self.adata.shape[1]} features in the dataset.")
            self.config.input_feature = self.adata.var_names.tolist()
        
        # to be used by chunkloader for data transformation
        self.preprocessor = Preprocessor(
            use_key="X",
            feature_list=self.config.input_feature,
            normalize_total=1e4,
            result_normed_key="X_normed",
            log1p=True,
            result_log1p_key="X_log1p"
        )


    def setup_config(self, 
                     project_name="concord",
                     input_feature=None,
                     batch_size=64, 
                     n_epochs=5,
                     lr=1e-3,
                     schedule_ratio=0.9, 
                     latent_dim=32, 
                     encoder_dims=[128],
                     decoder_dims=[128],
                     augmentation_mask_prob=0.6,
                     use_decoder=True, # Consider fix
                     decoder_final_activation='leaky_relu',
                     decoder_weight=1.0,
                     use_clr=True, # Consider fix
                     clr_temperature=0.5,
                     use_classifier=False,
                     classifier_weight=1.0,
                     unlabeled_class=None,
                     use_importance_mask = False,
                     importance_penalty_weight=0,
                     importance_penalty_type='L1',
                     use_dab=False,
                     use_domain_encoding=True, # Consider fix
                     dropout_prob=0.1,
                     norm_type="layer_norm", # Consider fix
                     domain_key=None,
                     class_key=None,
                     extra_keys=None,
                     sampler_mode="neighborhood",
                     sampler_emb="X_pca",
                     sampler_knn=256, 
                     p_intra_knn=0.1,
                     p_intra_domain=None,
                     min_p_intra_domain=0.6,
                     max_p_intra_domain=1.0,
                     use_faiss=True, 
                     use_ivf=True, 
                     ivf_nprobe=10,
                     pretrained_model=None,
                     classifier_freeze_param=False,
                     doublet_synth_ratio=0.4,
                     chunked=False,
                     chunk_size=10000,
                     wandb_reinit=True,
                     device='cpu',
                     seed=0):
        initial_params = dict(
            seed=seed,
            project_name=project_name,
            input_feature=input_feature,
            batch_size=batch_size,
            n_epochs=n_epochs,
            lr=lr,
            schedule_ratio=schedule_ratio,
            latent_dim=latent_dim,
            encoder_dims=encoder_dims,
            decoder_dims=decoder_dims,
            augmentation_mask_prob=augmentation_mask_prob,
            domain_key=domain_key,
            class_key=class_key,
            extra_keys=extra_keys,
            use_decoder=use_decoder,
            decoder_final_activation=decoder_final_activation,
            decoder_weight=decoder_weight,
            use_clr=use_clr,
            clr_temperature=clr_temperature,
            use_classifier=use_classifier,
            classifier_weight=classifier_weight,
            unlabeled_class=unlabeled_class,
            use_importance_mask=use_importance_mask,
            importance_penalty_weight=importance_penalty_weight,
            importance_penalty_type=importance_penalty_type,
            use_dab=use_dab, # Does improve based on testing, should be False, not deleted for future improvements
            use_domain_encoding=use_domain_encoding,
            dropout_prob=dropout_prob,
            norm_type=norm_type,
            sampler_mode=sampler_mode,
            sampler_emb=sampler_emb,
            sampler_knn=sampler_knn,
            p_intra_knn=p_intra_knn,
            p_intra_domain=p_intra_domain,
            min_p_intra_domain=min_p_intra_domain,
            max_p_intra_domain=max_p_intra_domain,
            use_faiss=use_faiss,
            use_ivf=use_ivf,
            ivf_nprobe=ivf_nprobe,
            pretrained_model=pretrained_model,
            classifier_freeze_param=classifier_freeze_param,
            doublet_synth_ratio=doublet_synth_ratio,
            chunked=chunked,  # Add chunked parameter
            chunk_size=chunk_size,
            device=device
        )

        if self.use_wandb:
            config, run = update_wandb_params(initial_params, project_name=self.config.project_name, reinit=wandb_reinit)
            self.config = config
            self.run = run
        else:
            self.config = Config(initial_params)


    def init_model(self):
        input_dim = len(self.config.input_feature)
        hidden_dim = self.config.latent_dim

        num_domains = len(self.adata.obs[self.config.domain_key].cat.categories) if self.config.domain_key is not None else 0

        if self.config.class_key is not None:
            all_classes = self.adata.obs[self.config.class_key].cat.categories
            if self.config.unlabeled_class is not None:
                if self.config.unlabeled_class in all_classes:
                    all_classes = all_classes.drop(self.config.unlabeled_class)
                else:
                    raise ValueError(f"Unlabeled class {self.config.unlabeled_class} not found in the class key.")
            num_classes = len(all_classes) 
        else:
            num_classes = 0

        self.model = ConcordModel(input_dim, hidden_dim, 
                                  num_domains=num_domains,
                                  num_classes=num_classes,
                                  encoder_dims=self.config.encoder_dims,
                                  decoder_dims=self.config.decoder_dims,
                                  decoder_final_activation=self.config.decoder_final_activation,
                                  augmentation_mask_prob=self.config.augmentation_mask_prob,
                                  dropout_prob=self.config.dropout_prob,
                                  norm_type=self.config.norm_type,
                                  use_decoder=self.config.use_decoder,
                                  use_classifier=self.config.use_classifier,
                                  use_importance_mask=self.config.use_importance_mask,
                                  use_dab=self.config.use_dab,
                                  use_domain_encoding=self.config.use_domain_encoding).to(self.config.device)

        total_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        logger.info(f'Total number of parameters: {total_params}')
        if self.use_wandb:
            wandb.log({"total_parameters": total_params})

        if self.config.pretrained_model is not None:
            best_model_path = Path(self.config.pretrained_model)
            if best_model_path.exists():
                logger.info(f"Loading pre-trained model from {best_model_path}")
                self.model.load_model(best_model_path, self.config.device)
            else:
                raise FileNotFoundError(f"Model file not found at {best_model_path}")


    def init_trainer(self):
        # Convert unlabeled_class from name to code
        if self.config.unlabeled_class is not None and self.config.class_key is not None:
            class_categories = self.adata.obs[self.config.class_key].cat.categories
            unlabeled_class_code = class_categories.get_loc(self.config.unlabeled_class)
        else:
            unlabeled_class_code = None
        self.trainer = Trainer(model=self.model,
                               data_structure=self.data_structure,
                               device=self.config.device,
                               logger=logger,
                               lr=self.config.lr,
                               schedule_ratio=self.config.schedule_ratio,
                               use_classifier=self.config.use_classifier, 
                               classifier_weight=self.config.classifier_weight,
                               unlabeled_class=unlabeled_class_code,
                               use_decoder=self.config.use_decoder,
                               decoder_weight=self.config.decoder_weight,
                               use_clr=self.config.use_clr, 
                               clr_temperature=self.config.clr_temperature,
                               use_wandb=self.use_wandb,
                               importance_penalty_weight=self.config.importance_penalty_weight,
                               importance_penalty_type=self.config.importance_penalty_type,
                               use_dab=self.config.use_dab)


    def init_sampler_params(self, sampler_mode, 
                            sampler_emb, 
                            sampler_knn=256, 
                            p_intra_knn=0.3,  
                            p_intra_class=None,
                            p_intra_domain=None,
                            class_weights=None, 
                            coverage_knn=256, 
                            min_p_intra_domain=0.5, max_p_intra_domain=1.0, scale_to_min_max=True,
                            use_faiss=True, use_ivf=True, ivf_nprobe=10):
        if sampler_mode and self.config.domain_key is not None:
            ensure_categorical(self.adata, obs_key=self.config.domain_key, drop_unused=True)
        if class_weights and self.config.class_key is not None:
            ensure_categorical(self.adata, obs_key=self.config.class_key, drop_unused=True)
            weights_show = {k: f"{v:.2e}" for k, v in class_weights.items()}
            logger.info(f"Creating weighted samplers with specified weights: {weights_show}")
        
        # Validate probability values
        validate_probability(p_intra_knn, "p_intra_knn")
        validate_probability(p_intra_class, "p_intra_class")

        # Additional checks
        if p_intra_knn is not None and p_intra_knn > 0.5:
            raise ValueError("p_intra_knn should not exceed 0.5 as it can lead to deteriorating performance.")
        if p_intra_class is not None and p_intra_class > 0.5:
            raise ValueError("p_intra_class should not exceed 0.5 as it can lead to deteriorating performance.")
        
        if (isinstance(p_intra_domain, dict) and check_dict_condition(p_intra_domain, lambda x: x < 0.1)) or \
           (p_intra_domain is not None and not isinstance(p_intra_domain, dict) and p_intra_domain < 0.1):
            logger.warning("It is recommended to set p_intra_domain values above 0.1 for good batch-correction performance.")

        if p_intra_domain is None:
            if sampler_emb not in self.adata.obsm:
                if sampler_emb == "X_pca":
                    logger.warning("PCA embeddings are not found in adata.obsm. Computing PCA...")
                    adata_copy = self.adata.copy() # Prevent filtering features of original adata, for large datasets consider do a subsample in future
                    self.preprocessor(adata=adata_copy)
                    sc.tl.pca(adata_copy, n_comps=50)
                    self.adata.obsm["X_pca"] = adata_copy.obsm["X_pca"]
                else:
                    raise ValueError(f"Embedding {sampler_emb} is not found in adata.obsm. Please provide a valid embedding key.")
            logger.info(f"Calculating each domain's coverage of the global manifold with knn (k={coverage_knn}) constructed on {sampler_emb}.")
            dataset_coverage = calculate_dataset_coverage(self.adata,
                                                            k=coverage_knn,
                                                            emb_key=sampler_emb,
                                                            dataset_key=self.config.domain_key,
                                                            use_faiss=use_faiss,
                                                            use_ivf=use_ivf,
                                                            ivf_nprobe=ivf_nprobe)
            logger.info(f"Converting coverage to p_intra_domain with specified min: {min_p_intra_domain:.2f}, max: {max_p_intra_domain:.2f}, rescale: {scale_to_min_max}.")
            p_intra_domain_dict = coverage_to_p_intra(self.adata.obs[self.config.domain_key], 
                                                    coverage=dataset_coverage, 
                                                    min_p_intra_domain=min_p_intra_domain, 
                                                    max_p_intra_domain=max_p_intra_domain, 
                                                    scale_to_min_max=scale_to_min_max)
            logger.info(f"Final p_intra_domain values: {', '.join(f'{k}: {v:.2f}' for k, v in p_intra_domain_dict.items())}")
        else:
            if isinstance(p_intra_domain, dict):
                logger.info(f"Using provided p_intra_domain values: {', '.join(f'{k}: {v:.2f}' for k, v in p_intra_domain.items())}.")
            else:
                logger.info(f"Using provided p_intra_domain values: {p_intra_domain:.2f}.")
            p_intra_domain_dict = p_intra_domain

        validate_probability_dict_compatible(p_intra_domain_dict, "p_intra_domain")

        self.sampler_kwargs = {
            'sampler_mode': sampler_mode,
            'emb_key': sampler_emb,
            'sampler_knn': sampler_knn,
            'p_intra_knn': p_intra_knn,
            'p_intra_domain': p_intra_domain_dict,
            'use_faiss': use_faiss,
            'use_ivf': use_ivf,
            'ivf_nprobe': ivf_nprobe,
            'class_weights': class_weights,
            'p_intra_class': p_intra_class
        }


    def init_dataloader(self, input_layer_key='X_log1p',
                        train_frac=1.0, use_sampler=True):
        if use_sampler:
            if not self.sampler_kwargs:
                raise ValueError("Sampler parameters are not initialized. Please call init_sampler_params() before initializing the dataloader.")
            else:
                sampler_kwargs = self.sampler_kwargs
        else:
            sampler_kwargs={}
        
        kwargs = {
            'adata': self.adata,
            'batch_size': self.config.batch_size,
            'input_layer_key': input_layer_key,
            'domain_key': self.config.domain_key,
            'class_key': self.config.class_key,
            'extra_keys': self.config.extra_keys,
            'train_frac': train_frac,
            'drop_last': False,
            'preprocess': self.preprocessor,
            'device': self.config.device,
            **sampler_kwargs
        }

        if self.config.chunked:
            self.loader = ChunkLoader(
                chunk_size=self.config.chunk_size,
                **kwargs
            )
            self.data_structure = self.loader.data_structure  # Retrieve data_structure
        else:
            train_dataloader, val_dataloader, self.data_structure = anndata_to_dataloader(
                **kwargs
            )
            self.loader = [(train_dataloader, val_dataloader, np.arange(self.adata.shape[0]))]


    def train(self, save_model=True):
        for epoch in range(self.config.n_epochs):
            logger.info(f'Starting epoch {epoch + 1}/{self.config.n_epochs}')
            for chunk_idx, (train_dataloader, val_dataloader, _) in enumerate(self.loader):
                logger.info(f'Processing chunk {chunk_idx + 1}/{len(self.loader)} for epoch {epoch + 1}')
                if train_dataloader is not None:
                    logger.info(f"Number of samples in train_dataloader: {len(train_dataloader.dataset)}")
                if val_dataloader is not None:
                    logger.info(f"Number of samples in val_dataloader: {len(val_dataloader.dataset)}")

                # Run training and validation for the current epoch
                if self.config.use_classifier:
                    if self.config.class_key is None:
                        raise ValueError("Class key is not provided. Please provide a valid class key for training the classifier.")
                    if self.config.class_key not in self.adata.obs.columns:
                        raise ValueError(f"Class key {self.config.class_key} not found in adata.obs. Please provide a valid class key.")
                    unique_classes = self.adata.obs[self.config.class_key].cat.categories
                    if self.config.unlabeled_class is not None and self.config.unlabeled_class in unique_classes:
                        unique_classes = unique_classes.drop(self.config.unlabeled_class)
                else:
                    unique_classes = None

                self.trainer.train_epoch(epoch, train_dataloader, unique_classes=unique_classes, n_epoch=self.config.n_epochs)
                if val_dataloader is not None:
                    self.trainer.validate_epoch(epoch, val_dataloader, unique_classes=unique_classes)

            self.trainer.scheduler.step()

        if save_model:
            model_save_path = self.save_dir / "final_model.pth"
            self.save_model(self.model, model_save_path)


    def predict(self, loader, sort_by_indices=False, return_decoded=False):  
        self.model.eval()
        class_preds = []
        class_true = []
        embeddings = []
        decoded_mtx = []
        indices = []

        if isinstance(loader, list) and all(isinstance(item, tuple) for item in loader):
            all_embeddings = []
            all_decoded = []
            all_class_preds = []
            all_class_true = []
            all_indices = []

            for chunk_idx, (dataloader, _, ck_indices) in enumerate(loader):
                logger.info(f'Predicting for chunk {chunk_idx + 1}/{len(loader)}')
                ck_embeddings, ck_decoded, ck_class_preds, ck_class_true = self.predict(dataloader, sort_by_indices=True, return_decoded=return_decoded)
                all_embeddings.append(ck_embeddings)
                all_decoded.append(ck_decoded) if return_decoded else None
                all_indices.extend(ck_indices)
                if ck_class_preds is not None:
                    all_class_preds.extend(ck_class_preds)
                if ck_class_true is not None:
                    all_class_true.extend(ck_class_true)

            all_embeddings = np.vstack(all_embeddings)
            all_decoded = np.vstack(all_decoded) if all_decoded else None
            all_indices = np.array(all_indices)
            sorted_indices = np.argsort(all_indices)
            all_embeddings = all_embeddings[sorted_indices]
            if all_class_preds:
                all_class_preds = np.array(all_class_preds)[sorted_indices]
            else:
                all_class_preds = None
            if all_class_true:
                all_class_true = np.array(all_class_true)[sorted_indices]
            else:
                all_class_true = None
            return all_embeddings, all_decoded, all_class_preds, all_class_true
        
        else:
            with torch.no_grad():
                for data in loader:
                    # Unpack data based on the provided structure
                    data_dict = {key: value.to(self.config.device) for key, value in zip(self.data_structure, data)}

                    inputs = data_dict.get('input')
                    domain_labels = data_dict.get('domain', None)
                    class_labels = data_dict.get('class', None)
                    original_indices = data_dict.get('indices')

                    if class_labels is not None:
                        class_true.extend(class_labels.cpu().numpy())

                    if original_indices is not None:
                        indices.extend(original_indices.cpu().numpy())

                    if self.model.use_domain_encoding and domain_labels is not None:
                        domain_labels_one_hot = F.one_hot(domain_labels, num_classes=self.model.domain_dim).float().to(self.config.device)
                    else:
                        domain_labels_one_hot = None

                    outputs = self.model(inputs, domain_labels_one_hot)
                    if 'class_pred' in outputs:
                        class_preds.extend(torch.argmax(outputs['class_pred'], dim=1).cpu().numpy())
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
                if class_true is not None:
                    class_true = class_true[sorted_indices]

            return embeddings, decoded_mtx, class_preds, class_true


    def encode_adata(self, input_layer_key="X_log1p", output_key="X_concord", return_decoded=False, save_model=True):
        # Initialize sampler parameters
        self.init_sampler_params(
            sampler_mode=self.config.sampler_mode, 
            sampler_emb=self.config.sampler_emb, 
            sampler_knn=self.config.sampler_knn, 
            p_intra_knn=self.config.p_intra_knn, 
            p_intra_domain=self.config.p_intra_domain, 
            min_p_intra_domain=self.config.min_p_intra_domain, 
            max_p_intra_domain=self.config.max_p_intra_domain,
            use_faiss=self.config.use_faiss,
            use_ivf=self.config.use_ivf,
            ivf_nprobe=self.config.ivf_nprobe
        )
        
        # Initialize the model
        self.init_model()
        # Initialize the dataloader
        self.init_dataloader(input_layer_key=input_layer_key)
        # Initialize the trainer
        self.init_trainer()
        # Train the model
        self.train(save_model=save_model)
        # Reinitialize the dataloader without using the sampler
        self.init_dataloader(input_layer_key=input_layer_key, use_sampler=False)
        
        # Predict and store the results
        encoded, decoded, _, _ = self.predict(self.loader, return_decoded=return_decoded)
        self.adata.obsm[output_key] = encoded
        if decoded is not None:
            self.adata.layers[output_key+'_decoded'] = decoded # Store decoded values in adata.layers


    def save_model(self, model, save_path):
        torch.save(model.state_dict(), save_path)
        logger.info(f"Model saved to {save_path}")


    @staticmethod
    def calculate_class_weights(class_labels, mode, heterogeneity_scores=None, enhancing_classes=None):
        unique_classes, class_counts = np.unique(class_labels.cat.codes, return_counts=True)

        if mode == "heterogeneity" and heterogeneity_scores:
            weights = {cls: 1 / heterogeneity_scores.get(cls, 1) for cls in unique_classes}
        elif mode == "count":
            weights = {cls: 1 / count for cls, count in zip(unique_classes, class_counts)}
        else:
            weights = {cls: 1 for cls in unique_classes}  # Initialize weights to 1

        if mode == "enhancing" and enhancing_classes:
            enhancing_class_codes = class_labels.cat.categories.get_indexer(enhancing_classes.keys())
            for cls, multiplier in zip(enhancing_class_codes, enhancing_classes.values()):
                if cls in weights:
                    weights[cls] *= multiplier

        return weights


    

