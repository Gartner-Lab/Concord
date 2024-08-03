import wandb
from pathlib import Path
import torch
from .model import ConcordModel
from ..utils.preprocessor import Preprocessor
from ..utils.anndata_utils import ensure_categorical
from .dataloader import anndata_to_dataloader
from .chunkloader import ChunkLoader
from ..utils.other_util import add_file_handler, update_wandb_params
from .evaluator import eval_scib_metrics
from .. import logger
from .trainer import Trainer
import numpy as np


class Config:
    def __init__(self, config_dict):
        for key, value in config_dict.items():
            setattr(self, key, value)

    def __getitem__(self, item):
        return getattr(self, item)

    def to_dict(self):
        return {key: getattr(self, key) for key in dir(self) if
                not key.startswith('__') and not callable(getattr(self, key))}


class Concord:
    def __init__(self, adata, proj_name, save_dir='save/', use_wandb=False, **kwargs):
        self.adata = adata
        self.proj_name = proj_name
        self.save_dir = Path(save_dir)
        self.use_wandb = use_wandb
        self.config = None
        self.run = None
        self.model = None
        self.n_epochs_run = 0

        if not self.save_dir.exists():
            self.save_dir.mkdir(parents=True, exist_ok=True)

        add_file_handler(logger, self.save_dir / "run.log")
        self.setup_config(**kwargs)

        self.preprocessor = Preprocessor(
            use_key="X",
            normalize_total=1e4,
            result_normed_key="X_normed",
            log1p=True,
            result_log1p_key="X_log1p",
            domain_key=self.config.domain_key
        )


    def setup_config(self, seed=0, data_path="data/",
                     batch_size=64, schedule_ratio=0.9, latent_dim=32, encoder_dims=[128],decoder_dims=[128],
                     augmentation_mask_prob=0.6,
                     domain_key=None,
                     class_key=None,
                     extra_keys=None,
                     use_decoder=True,
                     use_dab=False,
                     use_clr=True,
                     use_classifier=False,
                     use_importance_mask = False,
                     importance_penalty_weight=0.1,
                     importance_penalty_type='L1',
                     dab_nlayers=1,
                     dab_lambd=1.0,
                     dropout_prob=0.1,
                     norm_type="layer_norm",
                     min_epochs_for_best_model=0,
                     pretrained_model=None,
                     classifier_freeze_param=False,
                     doublet_synth_ratio=0.4,
                     chunked=False,
                     chunk_size=10000,
                     wandb_reinit=True,
                     eval_epoch_interval=5,
                     device='cpu'):
        initial_params = dict(
            seed=seed,
            project_name=self.proj_name,
            data_path=data_path,
            batch_size=batch_size,
            schedule_ratio=schedule_ratio,
            latent_dim=latent_dim,
            encoder_dims=encoder_dims,
            decoder_dims=decoder_dims,
            domain_key=domain_key,
            class_key=class_key,
            extra_keys=extra_keys,
            use_decoder=use_decoder,
            use_dab=use_dab,
            use_clr=use_clr,
            use_classifier=use_classifier,
            use_importance_mask=use_importance_mask,
            importance_penalty_weight=importance_penalty_weight,
            importance_penalty_type=importance_penalty_type,
            dab_nlayers=dab_nlayers,
            dab_lambd=dab_lambd,
            augmentation_mask_prob=augmentation_mask_prob,
            dropout_prob=dropout_prob,
            norm_type=norm_type,
            min_epochs_for_best_model=min_epochs_for_best_model,
            pretrained_model=pretrained_model,
            classifier_freeze_param=classifier_freeze_param,
            doublet_synth_ratio=doublet_synth_ratio,
            chunked=chunked,  # Add chunked parameter
            chunk_size=chunk_size,
            eval_epoch_interval=eval_epoch_interval,
            device=device
        )

        if self.use_wandb:
            config, run = update_wandb_params(initial_params, project_name=self.proj_name, reinit=wandb_reinit)
            self.config = config
            self.run = run
            print(config)
        else:
            self.config = Config(initial_params)




    def init_model(self):
        input_dim = self.adata.shape[1]
        hidden_dim = self.config.latent_dim
        num_domain = len(self.adata.obs[self.config.domain_key].unique()) if self.config.domain_key is not None else 0
        num_classes = 0  # No classification for this run

        self.model = ConcordModel(input_dim, hidden_dim, num_domain, num_classes,
                               encoder_dims=self.config.encoder_dims,
                               decoder_dims=self.config.decoder_dims,
                               dab_lambd=self.config.dab_lambd,
                               augmentation_mask_prob=self.config.augmentation_mask_prob,
                               dropout_prob=self.config.dropout_prob,
                               norm_type=self.config.norm_type,
                               use_decoder=self.config.use_decoder,
                               use_dab=self.config.use_dab,
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

        self.n_epochs_run = 0

    def init_trainer(self, lr=1e-3):
        self.trainer = Trainer(model=self.model,
                               data_structure=self.data_structure,
                               device=self.config.device,
                               logger=logger,
                               lr=lr,
                               schedule_ratio=self.config.schedule_ratio,
                               use_classifier=self.config.use_classifier, use_decoder=self.config.use_decoder,
                               use_dab=self.config.use_dab, use_clr=self.config.use_clr, use_wandb=self.use_wandb,
                               importance_penalty_weight=self.config.importance_penalty_weight,
                               importance_penalty_type=self.config.importance_penalty_type)



    def init_dataloader(self, input_layer_key='X_log1p',
                        train_frac=1.0,
                        sampler_mode=None, emb_key=None,
                        manifold_knn=300, p_intra_knn=0.3, p_intra_domain=1.0,
                        use_faiss=True, use_ivf=False, ivf_nprobe=8, class_weights=None, p_intra_class=None):
        if sampler_mode and self.config.domain_key is not None:
            ensure_categorical(self.adata, obs_key=self.config.domain_key, drop_unused=True)
        if class_weights and self.config.class_key is not None:
            ensure_categorical(self.adata, obs_key=self.config.class_key, drop_unused=True)
            weights_show = {k: f"{v:.2e}" for k, v in class_weights.items()}
            print(f"Creating weighted samplers with specified weights: {weights_show}")

        kwargs = {
            'adata': self.adata,
            'batch_size': self.config.batch_size,
            'input_layer_key': input_layer_key,
            'domain_key': self.config.domain_key,
            'class_key': self.config.class_key,
            'extra_keys': self.config.extra_keys,
            'train_frac': train_frac,
            'sampler_mode': sampler_mode,
            'emb_key': emb_key,
            'manifold_knn': manifold_knn,
            'p_intra_knn': p_intra_knn,
            'p_intra_domain': p_intra_domain,
            'use_faiss': use_faiss,
            'use_ivf': use_ivf,
            'ivf_nprobe': ivf_nprobe,
            'class_weights': class_weights,
            'p_intra_class': p_intra_class,
            'drop_last': False,
            'preprocess': self.preprocessor,
            'device': self.config.device
        }

        if self.config.chunked:
            chunk_loader = ChunkLoader(
                chunk_size=self.config.chunk_size,
                **kwargs
            )
            self.data_structure = chunk_loader.data_structure  # Retrieve data_structure
        else:
            train_dataloader, val_dataloader, self.data_structure = anndata_to_dataloader(
                **kwargs
            )
            chunk_loader = [(train_dataloader, val_dataloader, np.arange(self.adata.shape[0]))]

        return chunk_loader


    def train_and_eval(self, chunk_loader, n_epochs=3):
        for epoch in range(n_epochs):
            logger.info(f'Starting epoch {self.n_epochs_run + epoch + 1}/{self.n_epochs_run + n_epochs}')
            for chunk_idx, (train_dataloader, val_dataloader, _) in enumerate(chunk_loader):
                logger.info(f'Processing chunk {chunk_idx + 1}/{len(chunk_loader)} for epoch {self.n_epochs_run + epoch + 1}')
                if train_dataloader is not None:
                    print(f"Number of samples in train_dataloader: {len(train_dataloader.dataset)}")
                if val_dataloader is not None:
                    print(f"Number of samples in val_dataloader: {len(val_dataloader.dataset)}")

                # Run training and validation for the current epoch
                self.trainer.train_epoch(epoch, train_dataloader, None)
                if val_dataloader is not None:
                    self.trainer.validate_epoch(epoch, val_dataloader, None)

            self.trainer.scheduler.step()

            if self.config.eval_epoch_interval > 0 and (self.n_epochs_run + epoch + 1) % self.config.eval_epoch_interval == 0:
                full_embeddings = self.get_full_embeddings()
                self.adata.obsm['encoded'] = full_embeddings
                logger.info(f'Evaluating scib metrics at epoch {self.n_epochs_run + epoch + 1}')
                eval_results = eval_scib_metrics(self.adata, batch_key=self.config.domain_key,
                                                 label_key=self.config.class_key, )
                logger.info(f"Evaluation results at epoch {self.n_epochs_run + epoch + 1}: {eval_results}")
                if self.use_wandb:
                    wandb.log(eval_results)

        self.n_epochs_run += n_epochs  # Update epoch counter
        model_save_path = self.save_dir / "final_model.pth"
        self.save_model(self.model, model_save_path)


    def predict_with_model(self, model, dataloader, device, data_structure, sort_by_indices=False):  # Added data_structure to function parameters
        model.eval()
        class_preds = []
        class_true = []
        embeddings = []
        indices = []

        with torch.no_grad():
            for data in dataloader:
                # Unpack data based on the provided structure
                data_dict = {key: value.to(device) for key, value in zip(data_structure, data)}

                inputs = data_dict.get('input')
                domain_labels = data_dict.get('domain', None)
                class_labels = data_dict.get('class', None)
                original_indices = data_dict.get('indices')

                if class_labels is not None:
                    class_true.extend(class_labels.cpu().numpy())

                if original_indices is not None:
                    indices.extend(original_indices.cpu().numpy())

                domain_idx = domain_labels[0].item() if domain_labels is not None else None
                outputs = model(inputs, domain_idx)
                class_pred = outputs.get('class_pred')

                if class_pred is not None:
                    class_preds.extend(torch.argmax(class_pred, dim=1).cpu().numpy())

                if 'encoded' in outputs:
                    embeddings.append(outputs['encoded'].cpu().numpy())
                else:
                    raise ValueError("Model output does not contain 'encoded' embeddings.")

        if not embeddings:
            raise ValueError("No embeddings were extracted. Check the model and dataloader.")

        # Concatenate embeddings
        embeddings = np.concatenate(embeddings, axis=0)

        # Convert predictions and true labels to numpy arrays
        class_preds = np.array(class_preds) if class_preds else None
        class_true = np.array(class_true) if class_true else None

        if sort_by_indices and indices:
            # Sort embeddings and predictions back to the original order
            indices = np.array(indices)
            sorted_indices = np.argsort(indices)
            embeddings = embeddings[sorted_indices]
            if class_preds is not None:
                class_preds = class_preds[sorted_indices]
            if class_true is not None:
                class_true = class_true[sorted_indices]

        return embeddings, class_preds, class_true


    def get_full_embeddings(self, chunk_loader):
        all_embeddings = []
        all_indices = []

        for chunk_idx, (train_dataloader, _, chunk_indices) in enumerate(chunk_loader):
            logger.info(f'Predicting embeddings for chunk {chunk_idx + 1}/{len(chunk_loader)}')
            embeddings, _, _ = self.predict_with_model(self.model, train_dataloader, self.config.device, self.data_structure,
                                                  sort_by_indices=True)
            all_embeddings.append(embeddings)
            all_indices.extend(chunk_indices)

        all_embeddings = np.vstack(all_embeddings)
        all_indices = np.array(all_indices)
        sorted_indices = np.argsort(all_indices)
        return all_embeddings[sorted_indices]


    def encode_adata(self, input_layer_key='X_log1p', n_epochs = 3, lr=1e-3, class_weights=None):
        # Model Training
        self.init_model()
        chunk_loader = self.init_dataloader(input_layer_key, class_weights=class_weights)
        self.init_trainer(lr=lr)
        self.train_and_eval(chunk_loader, n_epochs=n_epochs)
        # Predict Embeddings after all epochs have been run
        full_data_loader = self.init_dataloader(input_layer_key)
        self.adata.obsm['encoded'] = self.get_full_embeddings(full_data_loader)



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

    @staticmethod
    def coverage_to_p_intra(domain_labels, coverage=None, min_p_intra = 0.1, max_p_intra = 1.0,
                                   scale_to_min_max=False):
        """
            Convert coverage values to p_intra values, with optional scaling and capping.

            Args:
                domain_labels (pd.Series or similar): A categorical series of domain labels.
                coverage (dict): Dictionary with domain keys and coverage values.
                min_p_intra (float): Minimum allowed p_intra value.
                max_p_intra (float): Maximum allowed p_intra value.
                scale_to_min_max (bool): Whether to scale the values to the range [min_p_intra, max_p_intra].

            Returns:
                dict: p_intra_domain_dict with domain codes as keys and p_intra values as values.
        """

        unique_domains = domain_labels.cat.categories

        if coverage is None:
            raise ValueError("Coverage dictionary must be provided.")
        missing_domains = set(unique_domains) - set(coverage.keys())
        if missing_domains:
            raise ValueError(f"Coverage values are missing for the following domains: {missing_domains}")

        p_intra_domain_dict = coverage.copy()

        if scale_to_min_max:
            # Linearly scale the values in p_intra_domain_dict to the range between min_p_intra and max_p_intra
            min_coverage = min(p_intra_domain_dict.values())
            max_coverage = max(p_intra_domain_dict.values())
            if min_coverage != max_coverage:  # Avoid division by zero
                scale = (max_p_intra - min_p_intra) / (max_coverage - min_coverage)
                p_intra_domain_dict = {
                    domain: min_p_intra + (value - min_coverage) * scale
                    for domain, value in p_intra_domain_dict.items()
                }
            else:
                p_intra_domain_dict = {domain: (min_p_intra + max_p_intra) / 2 for domain in p_intra_domain_dict}
        else:
            # Cap values to the range [min_p_intra, max_p_intra]
            p_intra_domain_dict = {
                domain: max(min(value, max_p_intra), min_p_intra)
                for domain, value in p_intra_domain_dict.items()
            }

        # Convert the domain labels to their corresponding category codes
        domain_codes = {domain: code for code, domain in enumerate(domain_labels.cat.categories)}
        p_intra_domain_dict = {domain_codes[domain]: value for domain, value in p_intra_domain_dict.items()}

        return p_intra_domain_dict


