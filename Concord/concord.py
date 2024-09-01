import wandb
from pathlib import Path
import torch
import torch.nn.functional as F
from .model.model import ConcordModel
from .utils.preprocessor import Preprocessor
from .utils.anndata_utils import ensure_categorical
from .model.dataloader import DataLoaderManager 
from .model.chunkloader import ChunkLoader
from .utils.other_util import add_file_handler, update_wandb_params
from .model.trainer import Trainer
import numpy as np
import scanpy as sc
import pandas as pd
import copy
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
    def __init__(self, adata, save_dir='save/', inplace=False, use_wandb=False, verbose=True, **kwargs):
        set_verbose_mode(verbose)
        if adata.isbacked:
            logger.warning("Input AnnData object is backed. With same amount of epochs, Concord will perform better when adata is loaded into memory.")
            if inplace:
                raise ValueError("Inplace mode is not supported for backed AnnData object. Set inplace to False.")
            self.adata = adata
        else:
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

        self.num_classes = None
        self.num_domains = None

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
                     train_frac=1.0,
                     latent_dim=32, 
                     encoder_dims=[128],
                     decoder_dims=[128],
                     augmentation_mask_prob=0.6,
                     use_decoder=True, # Consider fix
                     decoder_final_activation='leaky_relu',
                     decoder_weight=1.0,
                     use_clr_aug=True, # Consider fix
                     clr_aug_temperature=0.5,
                     clr_aug_weight=1.0,
                     use_clr_knn=False,
                     clr_knn_temperature=0.5,
                     clr_knn_weight=1.0,
                     use_classifier=False,
                     classifier_weight=1.0,
                     unlabeled_class=None,
                     use_importance_mask = True,
                     importance_penalty_weight=0,
                     importance_penalty_type='L1',
                     dropout_prob=0.1,
                     norm_type="layer_norm", # Consider fix
                     domain_key=None,
                     class_key=None,
                     domain_embedding_dim=8,
                     covariate_embedding_dims={},
                     sampler_emb="X_pca",
                     sampler_knn=256, 
                     p_intra_knn=0.3,
                     p_intra_domain=None,
                     min_p_intra_domain=0.6,
                     max_p_intra_domain=1.0,
                     pca_n_comps=50,
                     use_faiss=True, 
                     use_ivf=True, 
                     ivf_nprobe=10,
                     pretrained_model=None,
                     classifier_freeze_param=False,
                     doublet_synth_ratio=0.4,
                     chunked=False,
                     chunk_size=10000,
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
            train_frac=train_frac,
            latent_dim=latent_dim,
            encoder_dims=encoder_dims,
            decoder_dims=decoder_dims,
            augmentation_mask_prob=augmentation_mask_prob,
            domain_key=domain_key,
            class_key=class_key,
            domain_embedding_dim=domain_embedding_dim,
            covariate_embedding_dims=covariate_embedding_dims,
            use_decoder=use_decoder,
            decoder_final_activation=decoder_final_activation,
            decoder_weight=decoder_weight,
            use_clr_aug=use_clr_aug,
            clr_aug_temperature=clr_aug_temperature,
            clr_aug_weight=clr_aug_weight,
            use_clr_knn=use_clr_knn,
            clr_knn_temperature=clr_knn_temperature,
            clr_knn_weight=clr_knn_weight,
            use_classifier=use_classifier,
            classifier_weight=classifier_weight,
            unlabeled_class=unlabeled_class,
            use_importance_mask=use_importance_mask,
            importance_penalty_weight=importance_penalty_weight,
            importance_penalty_type=importance_penalty_type,
            dropout_prob=dropout_prob,
            norm_type=norm_type,
            sampler_emb=sampler_emb,
            sampler_knn=sampler_knn,
            p_intra_knn=p_intra_knn,
            p_intra_domain=p_intra_domain,
            min_p_intra_domain=min_p_intra_domain,
            max_p_intra_domain=max_p_intra_domain,
            pca_n_comps=pca_n_comps,
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
            config, run = update_wandb_params(initial_params, project_name=self.config.project_name, reinit=True)
            self.config = config
            self.run = run
        else:
            self.config = Config(initial_params)


    def init_model(self):
        input_dim = len(self.config.input_feature)
        hidden_dim = self.config.latent_dim

        if self.config.domain_key is not None:
            if(self.config.domain_key not in self.adata.obs.columns):
                raise ValueError(f"Domain key {self.config.domain_key} not found in adata.obs. Please provide a valid domain key.")
            ensure_categorical(self.adata, obs_key=self.config.domain_key, drop_unused=True)
            self.num_domains = len(self.adata.obs[self.config.domain_key].cat.categories)
        else:
            self.num_domains = 0

        if self.config.class_key is not None:
            if(self.config.class_key not in self.adata.obs.columns):
                raise ValueError(f"Class key {self.config.class_key} not found in adata.obs. Please provide a valid class key.")
            ensure_categorical(self.adata, obs_key=self.config.class_key, drop_unused=True)
            all_classes = self.adata.obs[self.config.class_key].cat.categories
            if self.config.unlabeled_class is not None:
                if self.config.unlabeled_class in all_classes:
                    all_classes = all_classes.drop(self.config.unlabeled_class)
                else:
                    raise ValueError(f"Unlabeled class {self.config.unlabeled_class} not found in the class key.")

            self.num_classes = len(all_classes) 
        else:
            self.num_classes = 0

        # Compute the number of categories for each covariate
        covariate_num_categories = {}
        for covariate_key in self.config.covariate_embedding_dims.keys():
            if covariate_key in self.adata.obs:
                ensure_categorical(self.adata, obs_key=covariate_key, drop_unused=True)
                covariate_num_categories[covariate_key] = len(self.adata.obs[covariate_key].cat.categories)

        self.model = ConcordModel(input_dim, hidden_dim, 
                                  num_domains=self.num_domains,
                                  num_classes=self.num_classes,
                                  domain_embedding_dim=self.config.domain_embedding_dim,
                                  covariate_embedding_dims=self.config.covariate_embedding_dims,
                                  covariate_num_categories=covariate_num_categories,
                                  encoder_dims=self.config.encoder_dims,
                                  decoder_dims=self.config.decoder_dims,
                                  decoder_final_activation=self.config.decoder_final_activation,
                                  augmentation_mask_prob=self.config.augmentation_mask_prob,
                                  dropout_prob=self.config.dropout_prob,
                                  norm_type=self.config.norm_type,
                                  use_decoder=self.config.use_decoder,
                                  use_classifier=self.config.use_classifier,
                                  use_importance_mask=self.config.use_importance_mask).to(self.config.device)

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
                               use_clr_aug=self.config.use_clr_aug, 
                               clr_aug_temperature=self.config.clr_aug_temperature,
                               clr_aug_weight=self.config.clr_aug_weight,
                               use_clr_knn=self.config.use_clr_knn,
                               clr_knn_temperature=self.config.clr_knn_temperature,
                               clr_knn_weight=self.config.clr_knn_weight,
                               use_wandb=self.use_wandb,
                               importance_penalty_weight=self.config.importance_penalty_weight,
                               importance_penalty_type=self.config.importance_penalty_type)


    def init_dataloader(self, input_layer_key='X_log1p', train_frac=1.0, use_sampler=True):
        data_manager = DataLoaderManager(
            input_layer_key=input_layer_key, domain_key=self.config.domain_key, 
            class_key=self.config.class_key, covariate_keys=self.config.covariate_embedding_dims.keys(), 
            batch_size=self.config.batch_size, train_frac=train_frac,
            use_sampler=use_sampler,
            sampler_emb=self.config.sampler_emb, 
            sampler_knn=self.config.sampler_knn, p_intra_knn=self.config.p_intra_knn, 
            p_intra_domain=self.config.p_intra_domain, 
            min_p_intra_domain=self.config.min_p_intra_domain,
            max_p_intra_domain=self.config.max_p_intra_domain,
            pca_n_comps=self.config.pca_n_comps,
            use_faiss=self.config.use_faiss, 
            use_ivf=self.config.use_ivf, 
            ivf_nprobe=self.config.ivf_nprobe, 
            preprocess=self.preprocessor,
            num_cores=self.num_classes, 
            use_clr_knn=self.config.use_clr_knn,
            device=self.config.device
        )

        if self.config.chunked:
            self.loader = ChunkLoader(
                adata=self.adata,
                chunk_size=self.config.chunk_size,
                data_manager=data_manager
            )
            self.data_structure = self.loader.data_structure  # Retrieve data_structure
        else:
            train_dataloader, val_dataloader, self.data_structure = data_manager.anndata_to_dataloader(self.adata)
            self.loader = [(train_dataloader, val_dataloader, np.arange(self.adata.shape[0]))]


    def train(self, save_model=True, patience=2):
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

        if save_model:
            model_save_path = self.save_dir / "final_model.pth"
            self.save_model(self.model, model_save_path)
            logger.info(f"Final model saved at: {model_save_path}")


    def predict(self, loader, sort_by_indices=False, return_decoded=False, return_class=True, return_class_prob=True):  
        self.model.eval()
        class_preds = []
        class_true = []
        class_probs = [] if return_class_prob else None
        embeddings = []
        decoded_mtx = []
        indices = []

        # Get the original class categories
        class_categories = self.adata.obs[self.config.class_key].cat.categories if self.config.class_key is not None else None
        
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
            all_embeddings = np.vstack(all_embeddings)[sorted_indices]
            all_decoded = np.vstack(all_decoded)[sorted_indices] if all_decoded else None
            all_class_preds = np.array(all_class_preds)[sorted_indices] if all_class_preds else None
            all_class_true = np.array(all_class_true)[sorted_indices] if all_class_true else None
            if return_class_prob:
                all_class_probs = pd.concat(all_class_probs).iloc[sorted_indices].reset_index(drop=True) if all_class_probs else None
            return all_embeddings, all_decoded, all_class_preds, all_class_probs, all_class_true
        else:
            with torch.no_grad():
                for data in loader:
                    # Unpack data based on the provided structure
                    data_dict = {key: value.to(self.config.device) for key, value in zip(self.data_structure, data)}

                    inputs = data_dict.get('input')
                    domain_ids = data_dict.get('domain', None)
                    class_labels = data_dict.get('class', None)
                    original_indices = data_dict.get('idx')
                    covariate_keys = [key for key in data_dict.keys() if key not in ['input', 'domain', 'class', 'idx']]
                    covariate_tensors = {key: data_dict[key] for key in covariate_keys}

                    if class_labels is not None:
                        class_true.extend(class_labels.cpu().numpy())

                    if original_indices is not None:
                        indices.extend(original_indices.cpu().numpy())

                    outputs = self.model(inputs, domain_ids, covariate_tensors)
                    if 'class_pred' in outputs:
                        class_preds_tensor = outputs['class_pred']
                        class_preds.extend(torch.argmax(class_preds_tensor, dim=1).cpu().numpy())
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

            if return_class and class_categories is not None:
                class_preds = class_categories[class_preds] if class_preds is not None else None
                class_true = class_categories[class_true] if class_true is not None else None
                if return_class_prob and class_probs is not None:
                    class_probs = pd.DataFrame(class_probs, columns=class_categories)

            return embeddings, decoded_mtx, class_preds, class_probs, class_true


    def encode_adata(self, input_layer_key="X_log1p", output_key="Concord", return_decoded=False, return_class=True, return_class_prob=True, save_model=True):
        # Initialize the model
        self.init_model()
        # Initialize the dataloader
        self.init_dataloader(input_layer_key=input_layer_key, train_frac=self.config.train_frac, use_sampler=True)
        # Initialize the trainer
        self.init_trainer()
        # Train the model
        self.train(save_model=save_model)
        # Reinitialize the dataloader without using the sampler
        self.init_dataloader(input_layer_key=input_layer_key, train_frac=1.0, use_sampler=False)
        
        # Predict and store the results
        encoded, decoded, class_preds, class_probs, class_true = self.predict(self.loader, return_decoded=return_decoded, return_class=return_class, return_class_prob=return_class_prob)
        self.adata.obsm[output_key] = encoded
        if return_decoded:
            self.adata.layers[output_key+'_decoded'] = decoded
        if class_true is not None:
            self.adata.obs[output_key+'_class_true'] = class_true
        if class_preds is not None:
            self.adata.obs[output_key+'_class_pred'] = class_preds
        if class_probs is not None:
            class_probs.index = self.adata.obs.index
            for col in class_probs.columns:
                self.adata.obs[f'class_prob_{col}'] = class_probs[col]
        if decoded is not None:
            self.adata.layers[output_key+'_decoded'] = decoded # Store decoded values in adata.layers


    def save_model(self, model, save_path):
        torch.save(model.state_dict(), save_path)
        logger.info(f"Model saved to {save_path}")




    

