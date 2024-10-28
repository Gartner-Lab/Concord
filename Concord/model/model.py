import torch
import torch.nn as nn
from .build_layer import get_normalization_layer, build_layers
from .. import logger

class ConcordModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_domains, num_classes,  
                 domain_embedding_dim=0, 
                 covariate_embedding_dims={},
                 covariate_num_categories={},
                 encoder_dims=[], decoder_dims=[], 
                 augmentation_mask_prob: float = 0.3, dropout_prob: float = 0.1, norm_type='layer_norm', 
                 #encoder_append_cov=False, 
                 use_decoder=True, decoder_final_activation='leaky_relu',
                 use_classifier=False, use_importance_mask=False):
        super().__init__()

        # Encoder
        self.domain_embedding_dim = domain_embedding_dim 
        self.input_dim = input_dim
        self.augmentation_mask = nn.Dropout(augmentation_mask_prob)
        self.use_classifier = use_classifier
        self.use_decoder = use_decoder
        self.use_importance_mask = use_importance_mask

        total_embedding_dim = 0
        if domain_embedding_dim > 0:
            self.domain_embedding = nn.Embedding(num_embeddings=num_domains, embedding_dim=domain_embedding_dim)
            total_embedding_dim += domain_embedding_dim

        self.covariate_embeddings = nn.ModuleDict()
        for key, dim in covariate_embedding_dims.items():
            if dim > 0:
                self.covariate_embeddings[key] = nn.Embedding(num_embeddings=covariate_num_categories[key], embedding_dim=dim)
                total_embedding_dim += dim

        # self.encoder_append_cov = encoder_append_cov
        # if self.encoder_append_cov :
        #     encoder_input_dim = input_dim + total_embedding_dim
        # else:
        #     encoder_input_dim = input_dim

        encoder_input_dim = input_dim

        logger.info(f"Encoder input dim: {encoder_input_dim}")
        if self.use_decoder:
            decoder_input_dim = hidden_dim + total_embedding_dim
            logger.info(f"Decoder input dim: {decoder_input_dim}")
        if self.use_classifier:
            classifier_input_dim = hidden_dim # decoder_input_dim 
            logger.info(f"Classifier input dim: {classifier_input_dim}")

        # Encoder
        if encoder_dims:
            self.encoder = build_layers(encoder_input_dim, hidden_dim, encoder_dims, dropout_prob, norm_type)
        else:
            self.encoder = nn.Sequential(
                nn.Linear(encoder_input_dim, hidden_dim),
                get_normalization_layer(norm_type, hidden_dim),
                nn.LeakyReLU(0.1)
            )

        # Decoder
        if self.use_decoder:
            if decoder_dims:
                self.decoder = build_layers(decoder_input_dim, input_dim, decoder_dims, dropout_prob, norm_type, 
                                            final_layer_norm=False, final_layer_dropout=False, final_activation=decoder_final_activation)
            else:
                self.decoder = nn.Sequential(
                    nn.Linear(decoder_input_dim, input_dim)
                )

        # Classifier head
        if self.use_classifier:
            self.classifier = nn.Sequential(
                nn.Linear(classifier_input_dim, hidden_dim),
                get_normalization_layer(norm_type, hidden_dim),
                nn.LeakyReLU(0.1),
                nn.Linear(hidden_dim, num_classes)
            )

        self._initialize_weights()

        # Learnable mask for feature importance
        if self.use_importance_mask:
            self.importance_mask = nn.Parameter(torch.ones(input_dim))

    def forward(self, x, domain_labels=None, covariate_tensors=None, return_latent=False):
        out = {}

        embeddings = []
        if self.domain_embedding_dim > 0 and domain_labels is not None:
            domain_embeddings = self.domain_embedding(domain_labels)
            embeddings.append(domain_embeddings)

        # Use covariate embeddings if available
        if covariate_tensors is not None:
            for key, tensor in covariate_tensors.items():
                if key in self.covariate_embeddings:
                    embeddings.append(self.covariate_embeddings[key](tensor))

        if self.use_importance_mask:
            importance_weights = self.get_importance_weights()
            x = x * importance_weights

        x = self.augmentation_mask(x)
        # if self.encoder_append_cov and embeddings:
        #     x = torch.cat([x] + embeddings, dim=1)

        if return_latent:
            out['latent'] = dict()

        for i in range(len(self.encoder)):
            layer = self.encoder[i]
            x = layer(x)
            if return_latent:
                if layer.__class__.__name__ in ['LeakyReLU', 'ReLU']:
                    out['latent'][f'encoder_{i}'] = x                

        out['encoded'] = x

        if self.use_decoder:
            x = out['encoded']
            if embeddings:
                x = torch.cat([x] + embeddings, dim=1)
            for i in range(len(self.decoder)):
                layer = self.decoder[i]
                x = layer(x)
                if return_latent:
                    if layer.__class__.__name__ in ['LeakyReLU', 'ReLU']:
                        out['latent'][f'decoder_{i}'] = x

            out['decoded'] = x


        if self.use_classifier:
            x = out['encoded']
            # if embeddings:
            #     x = torch.cat([x] + embeddings, dim=1)
            out['class_pred'] = self.classifier(x)

        return out

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def freeze_encoder(self):
        for param in self.encoder.parameters():
            param.requires_grad = False

    def load_model(self, path, device):
        state_dict = torch.load(path, map_location=device)
        model_state_dict = self.state_dict()

        # Filter out layers with mismatched sizes
        filtered_state_dict = {k: v for k, v in state_dict.items() if
                               k in model_state_dict and v.size() == model_state_dict[k].size()}
        model_state_dict.update(filtered_state_dict)
        self.load_state_dict(model_state_dict, strict=False)

    def get_importance_weights(self):
        if self.use_importance_mask:
            #return torch.softmax(self.importance_mask, dim=0) * self.input_dim
            return torch.relu(self.importance_mask)
        else:
            raise ValueError("Importance mask is not used in this model.")

