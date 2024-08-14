
import torch
from torch import nn, optim
import torch.nn.functional as F
from tqdm import tqdm
import wandb
import copy
import numpy as np
from ..utils.evaluator import log_classification
from .loss import ContrastiveLoss, importance_penalty_loss

class Trainer:
    def __init__(self, model, data_structure, device, logger, lr, schedule_ratio,
                 use_classifier=False, classifier_weight=1.0, unlabeled_class=None,
                 use_decoder=True, decoder_weight=1.0, 
                 use_clr=True, clr_temperature=0.5,
                 importance_penalty_weight=0, importance_penalty_type='L1',
                 use_dab=False, 
                 use_wandb=False):
        self.model = model
        self.data_structure = data_structure
        self.device = device
        self.logger = logger
        self.use_classifier = use_classifier
        self.classifier_weight = classifier_weight
        self.unlabeled_class = unlabeled_class
        self.use_decoder = use_decoder
        self.decoder_weight = decoder_weight
        self.use_clr = use_clr
        self.use_dab = use_dab
        self.use_wandb = use_wandb
        self.importance_penalty_weight = importance_penalty_weight
        self.importance_penalty_type = importance_penalty_type

        self.classifier_criterion = nn.CrossEntropyLoss() if use_classifier else None
        self.mse_criterion = nn.MSELoss() if use_decoder else None
        self.contrastive_criterion = ContrastiveLoss(temperature=clr_temperature) if use_clr else None
        self.dab_criterion = nn.CrossEntropyLoss() if use_dab else None

        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)
        self.scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size=1, gamma=schedule_ratio)

    def forward_pass(self, inputs, class_labels, domain_labels, alpha=None):
        if self.model.use_domain_encoding and domain_labels is not None:
            domain_labels_one_hot = F.one_hot(domain_labels, num_classes=self.model.domain_dim).float().to(self.device)
        else:
            domain_labels_one_hot = None
    
        outputs = self.model(inputs, domain_labels_one_hot, alpha)
        class_pred = outputs.get('class_pred')
        decoded = outputs.get('decoded')
        dab_pred = outputs.get('dab_pred')

        loss_classifier = torch.tensor(0.0, device=self.device)
        if class_pred is not None and class_labels is not None:
            labeled_mask = (class_labels != self.unlabeled_class).to(self.device) if self.unlabeled_class is not None else torch.ones_like(class_labels, dtype=torch.bool, device=self.device)
            if labeled_mask.sum() > 0:
                class_labels = class_labels[labeled_mask]
                class_pred = class_pred[labeled_mask] 
                loss_classifier = self.classifier_criterion(class_pred, class_labels) * self.classifier_weight
            else:
                class_labels = None
                class_pred = None

        loss_mse = self.mse_criterion(decoded, inputs) * self.decoder_weight if decoded is not None else torch.tensor(0.0)
        loss_clr = torch.tensor(0.0)
        loss_dab = self.dab_criterion(dab_pred, domain_labels) if dab_pred is not None else torch.tensor(0.0)

        if self.contrastive_criterion is not None:
            outputs_aug = self.model(inputs, domain_labels_one_hot, alpha)
            loss_clr = self.contrastive_criterion(outputs['encoded'], outputs_aug['encoded'])

        # Importance penalty loss
        if self.model.use_importance_mask:
            importance_weights = self.model.get_importance_weights()
            loss_penalty = importance_penalty_loss(importance_weights, self.importance_penalty_weight, self.importance_penalty_type)
        else:
            loss_penalty = torch.tensor(0.0)

        loss = loss_classifier + loss_mse + loss_clr + loss_dab + loss_penalty

        return loss, loss_classifier, loss_mse, loss_clr, loss_dab, loss_penalty, class_labels, class_pred

    def train_epoch(self, epoch, train_dataloader, unique_classes, n_epoch):
        self.model.train()
        total_loss, total_mse, total_clr, total_classifier, total_dab, total_importance_penalty = 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
        train_preds, train_y = [], []
        len_dataloader = len(train_dataloader)
        data_iter = iter(train_dataloader)
        progress_bar = tqdm(range(len_dataloader), desc=f"Epoch {epoch} Training", position=0, leave=True, dynamic_ncols=True)

        for i in progress_bar:
            self.optimizer.zero_grad()

            data = next(data_iter)

            # Unpack data based on the provided structure
            data_dict = {key: value.to(self.device) for key, value in zip(self.data_structure, data)}

            inputs = data_dict['input']
            domain_labels = data_dict.get('domain', None)
            class_labels = data_dict.get('class', None)

            # For DAB
            if self.use_dab:
                p = float(i + epoch * len_dataloader) / (n_epoch * len_dataloader)
                alpha = 2. / (1. + np.exp(-10 * p)) - 1
            else:
                alpha = None

            loss, loss_classifier, loss_mse, loss_clr, loss_dab, loss_penalty, class_labels, class_pred = self.forward_pass(
                inputs, class_labels, domain_labels, alpha
            )

            # Backward pass and optimization
            loss.backward()
            self.optimizer.step()

            # Logging
            if self.use_wandb:
                if self.use_classifier:
                    wandb.log({"train/classifier": loss_classifier.item()})
                if self.use_decoder:
                    wandb.log({"train/mse": loss_mse.item()})
                if self.use_clr:
                    wandb.log({"train/clr": loss_clr.item()})
                if self.use_dab:
                    wandb.log({"train/dab": loss_dab.item()})
                if self.model.use_importance_mask:
                    wandb.log({"train/importance_penalty": loss_penalty.item()})

            total_loss += loss.item()
            total_mse += loss_mse.item()
            total_classifier += loss_classifier.item() if self.use_classifier else 0
            total_clr += loss_clr.item() if self.use_clr else 0
            total_dab += loss_dab.item() if self.use_dab else 0
            total_importance_penalty += loss_penalty.item() if self.model.use_importance_mask else 0

            if class_pred is not None and class_labels is not None:
                train_preds.extend(torch.argmax(class_pred, dim=1).cpu().numpy())
                train_y.extend(class_labels.cpu().numpy())

            progress_bar.set_postfix({"loss": loss.item()})
            progress_bar.update(1)

        self.scheduler.step()

        avg_loss = total_loss / len(train_dataloader)
        avg_mse = total_mse / len(train_dataloader)
        avg_clr = total_clr / len(train_dataloader)
        avg_classifier = total_classifier / len(train_dataloader)
        avg_dab = total_dab / len(train_dataloader)
        avg_importance_penalty = total_importance_penalty / len(train_dataloader)

        self.logger.info(
            f'epoch {epoch:3d} | Train Loss:{avg_loss:5.2f}, MSE:{avg_mse:5.2f}, CLASS:{avg_classifier:5.2f}, '
            f'CLR:{avg_clr:5.2f}, DAB:{avg_dab:5.2f}, IMPORTANCE:{avg_importance_penalty:5.2f}')

        if self.use_classifier:
            log_classification(epoch, "train", train_preds, train_y, self.logger, unique_classes)

        return avg_loss

    def validate_epoch(self, epoch, val_dataloader, unique_classes):
        self.model.eval()
        total_loss, total_mse, total_clr, total_classifier, total_dab, total_importance_penalty = 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
        val_preds, val_y = [], []

        progress_bar = tqdm(val_dataloader, desc=f"Epoch {epoch} Validation", position=0, leave=True, dynamic_ncols=True)

        with torch.no_grad():
            for data in progress_bar:
                # Unpack data based on the provided structure
                data_dict = {key: value.to(self.device) for key, value in zip(self.data_structure, data)}

                inputs = data_dict['input']
                domain_labels = data_dict['domain']
                class_labels = data_dict.get('class')

                loss, loss_classifier, loss_mse, loss_clr, loss_dab, loss_penalty, class_labels, class_pred = self.forward_pass(
                    inputs, class_labels, domain_labels
                )

                # Logging losses
                if self.use_wandb:
                    if self.use_classifier:
                        wandb.log({"val/classifier": loss_classifier.item()})
                    if self.use_decoder:
                        wandb.log({"val/mse": loss_mse.item()})
                    if self.use_clr:
                        wandb.log({"val/clr": loss_clr.item()})
                    if self.use_dab:
                        wandb.log({"val/dab": loss_dab.item()})
                    if self.model.use_importance_mask:
                        wandb.log({"val/importance_penalty": loss_penalty.item()})

                total_loss += loss.item()
                total_mse += loss_mse.item()
                total_classifier += loss_classifier.item() if self.use_classifier else 0
                total_dab += loss_dab.item() if self.use_dab else 0
                total_clr += loss_clr.item() if self.use_clr else 0
                total_importance_penalty += loss_penalty.item() if self.model.use_importance_mask else 0

                if class_pred is not None and class_labels is not None:
                    val_preds.extend(torch.argmax(class_pred, dim=1).cpu().numpy())
                    val_y.extend(class_labels.cpu().numpy())

                progress_bar.set_postfix({"loss": loss.item()})
                progress_bar.update(1)

        avg_loss = total_loss / len(val_dataloader)
        avg_mse = total_mse / len(val_dataloader)
        avg_dab = total_dab / len(val_dataloader)
        avg_clr = total_clr / len(val_dataloader)
        avg_classifier = total_classifier / len(val_dataloader)
        avg_importance_penalty = total_importance_penalty / len(val_dataloader)

        self.logger.info(
            f'epoch {epoch:3d} | Val Loss:{avg_loss:5.2f}, MSE:{avg_mse:5.2f}, CLASS:{avg_classifier:5.2f}, '
            f'CLR:{avg_clr:5.2f}, DAB:{avg_dab:5.2f}, IMPORTANCE:{avg_importance_penalty:5.2f}')

        if self.use_classifier:
            log_classification(epoch, "val", val_preds, val_y, self.logger, unique_classes)

        return avg_loss

    # This is not used currently
    def main_training_loop(self, train_dataloader, val_dataloader, num_epochs, unique_classes):
        best_model = None
        best_loss = float("inf")
        best_model_epoch = 0

        for epoch in range(num_epochs):
            self.logger.info(f'Epoch {epoch + 1}/{num_epochs}')

            # Training step
            avg_train_loss = self.train_epoch(epoch, train_dataloader, unique_classes)

            # Validation step
            if val_dataloader is not None:
                avg_val_loss = self.validate_epoch(epoch, val_dataloader, unique_classes)
                judging_loss = avg_val_loss  # Should be preferred
            else:
                judging_loss = avg_train_loss

            if judging_loss < best_loss:
                best_loss = judging_loss
                best_model = copy.deepcopy(self.model)
                best_model_epoch = epoch
                self.logger.info(f"Best model with loss {best_loss:5.4f}")

            self.scheduler.step()

        return best_model, best_model_epoch
