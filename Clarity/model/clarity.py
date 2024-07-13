import wandb
from pathlib import Path
import torch
import umap
from .model import ClarityModel
from ..utils.preprocessor import Preprocessor
from ..utils.util import ensure_categorical
from .dataloader import anndata_to_dataloader
from .chunkloader import ChunkLoader
from ..utils.util import add_file_handler, update_wandb_params
#from .predict_functions import predict_with_model
from .evaluator import eval_scib_metrics
from .. import logger
from .trainer import Trainer
import numpy as np
import matplotlib.pyplot as plt


class Config:
    def __init__(self, config_dict):
        for key, value in config_dict.items():
            setattr(self, key, value)

    def __getitem__(self, item):
        return getattr(self, item)

    def to_dict(self):
        return {key: getattr(self, key) for key in dir(self) if
                not key.startswith('__') and not callable(getattr(self, key))}


class Clarity:
    def __init__(self, adata, proj_name, save_dir='save/', use_wandb=False, **kwargs):
        self.adata = adata
        self.proj_name = proj_name
        self.save_dir = Path(save_dir)
        self.use_wandb = use_wandb
        self.config = None
        self.run = None
        self.model = None
        self.chunk_loader = None

        if not self.save_dir.exists():
            self.save_dir.mkdir(parents=True, exist_ok=True)

        add_file_handler(logger, self.save_dir / "run.log")
        self.setup_config(**kwargs)

        if self.config.sampler_mode == 'domain_and_class':
            self.data_structure = ['input', 'domain', 'class', 'indices']
        elif self.config.sampler_mode == 'domain':
            self.data_structure = ['input', 'domain', 'indices']
        else:
            self.data_structure = ['input', 'indices']

        ensure_categorical(self.adata, obs_key=self.config.domain_key, drop_unused=True)
        self.preprocessor = Preprocessor(
            use_key="X",
            normalize_total=1e4,
            result_normed_key="X_normed",
            log1p=True,
            result_log1p_key="X_log1p",
            domain_key=self.config.domain_key
        )

    def setup_config(self, seed=0, data_path="data/", epochs=10, lr=1e-3,
                     batch_size=64, schedule_ratio=0.9, latent_dim=32, encoder_dims=[128],decoder_dims=[128],
                     augmentation_mask_prob=0.6,
                     domain_key=None,
                     class_key=None,
                     sampler_mode="domain",
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
            epochs=epochs,
            lr=lr,
            batch_size=batch_size,
            schedule_ratio=schedule_ratio,
            latent_dim=latent_dim,
            encoder_dims=encoder_dims,
            decoder_dims=decoder_dims,
            domain_key=domain_key,
            class_key=class_key,
            sampler_mode=sampler_mode,
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
        unique_domains = self.adata.obs[self.config.domain_key].unique()
        num_domain = len(unique_domains)
        num_classes = 0  # No classification for this run

        self.model = ClarityModel(input_dim, hidden_dim, num_domain, num_classes,
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

    def init_trainer(self):
        self.trainer = Trainer(self.model, self.data_structure, self.config.device, logger, self.config.lr, self.config.schedule_ratio,
                               use_classifier=self.config.use_classifier, use_decoder=self.config.use_decoder,
                               use_dab=self.config.use_dab, use_clr=self.config.use_clr, use_wandb=self.use_wandb,
                               importance_penalty_weight=self.config.importance_penalty_weight,
                               importance_penalty_type=self.config.importance_penalty_type)

    def init_dataloader(self, input_layer_key='X_log1p'):
        if self.config.sampler_mode in ['domain', 'domain_and_class']:
            ensure_categorical(self.adata, obs_key=self.config.domain_key)
        if self.config.sampler_mode == 'domain_and_class':
            ensure_categorical(self.adata, obs_key=self.config.class_key)

        if self.config.chunked:
            self.chunk_loader = ChunkLoader(
                self.adata, input_layer_key=input_layer_key, domain_key=self.config.domain_key, class_key=None,
                extra_keys=None, chunk_size=self.config.chunk_size, batch_size=self.config.batch_size,
                train_frac=1.0, sampler_mode=self.config.sampler_mode, drop_last=False, preprocess=self.preprocessor,
                device=self.config.device
            )
        else:
            train_dataloader, val_dataloader = anndata_to_dataloader(
                adata=self.adata,
                input_layer_key=input_layer_key,
                domain_key=self.config.domain_key,
                class_key=None,
                train_frac=1.0,
                batch_size=self.config.batch_size,
                sampler_mode=self.config.sampler_mode,
                drop_last=False,
                keep_indices=True,
                preprocess=self.preprocessor,
                device=self.config.device
            )
            self.chunk_loader = [(train_dataloader, val_dataloader, np.arange(self.adata.shape[0]))]


    def train_and_eval(self):
        for epoch in range(self.config.epochs):
            logger.info(f'Starting epoch {epoch + 1}/{self.config.epochs}')
            for chunk_idx, (train_dataloader, val_dataloader, _) in enumerate(self.chunk_loader):
                logger.info(f'Processing chunk {chunk_idx + 1}/{len(self.chunk_loader)} for epoch {epoch + 1}')
                if train_dataloader is not None:
                    print(f"Number of samples in train_dataloader: {len(train_dataloader.dataset)}")
                if val_dataloader is not None:
                    print(f"Number of samples in val_dataloader: {len(val_dataloader.dataset)}")

                # Run training and validation for the current epoch
                self.trainer.train_epoch(epoch, train_dataloader, None)
                if val_dataloader is not None:
                    self.trainer.validate_epoch(epoch, val_dataloader, None)

            self.trainer.scheduler.step()

            if self.config.eval_epoch_interval > 0 and (epoch + 1) % self.config.eval_epoch_interval == 0:
                full_embeddings = self.get_full_embeddings(self.chunk_loader, self.data_structure)
                self.adata.obsm['encoded'] = full_embeddings
                logger.info(f'Evaluating scib metrics at epoch {epoch + 1}')
                eval_results = eval_scib_metrics(self.adata, batch_key=self.config.domain_key,
                                                 label_key=self.config.class_key, )
                logger.info(f"Evaluation results at epoch {epoch + 1}: {eval_results}")
                if self.use_wandb:
                    wandb.log(eval_results)

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

                inputs = data_dict['input']
                domain_labels = data_dict['domain']
                class_labels = data_dict.get('class')
                original_indices = data_dict.get('indices')

                if class_labels is not None:
                    class_true.extend(class_labels.cpu().numpy())

                if original_indices is not None:
                    indices.extend(original_indices.cpu().numpy())

                domain_idx = domain_labels[0].item()
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


    def encode_adata(self, input_layer_key='X_log1p'):
        # Model Training
        self.init_model()
        self.init_trainer()
        self.init_dataloader(input_layer_key)
        self.train_and_eval()
        # Predict Embeddings after all epochs have been run
        self.adata.obsm['encoded'] = self.get_full_embeddings(self.chunk_loader)
        return self.adata



    def run_umap(self, use_cuml=False, umap_key='encoded_UMAP', n_components=2, n_neighbors=30, min_dist=0.1, metric='euclidean', spread=1.0, n_epochs=500, random_state=None):
        if random_state is None:
            random_state = self.config.seed

        if use_cuml:
            try:
                from cuml.manifold import UMAP as cumlUMAP
                umap_model = cumlUMAP(n_components=n_components, n_neighbors=n_neighbors, min_dist=min_dist, metric=metric, spread=spread, n_epochs=n_epochs, random_state=random_state)
            except ImportError:
                logger.warning("cuML is not available. Falling back to standard UMAP.")
                umap_model = umap.UMAP(n_components=n_components, n_neighbors=n_neighbors, min_dist=min_dist, metric=metric, spread=spread, n_epochs=n_epochs, random_state=random_state)
        else:
            umap_model = umap.UMAP(n_components=n_components, n_neighbors=n_neighbors, min_dist=min_dist, metric=metric, spread=spread, n_epochs=n_epochs, random_state=random_state)

        self.adata.obsm[umap_key] = umap_model.fit_transform(self.adata.obsm['encoded'])
        logger.info(f"UMAP embedding stored in self.adata.obsm['{umap_key}']")

        return self.adata

    def save_model(self, model, save_path):
        torch.save(model.state_dict(), save_path)
        logger.info(f"Model saved to {save_path}")




