import torch
import torch.nn as nn
from .dab import AdversarialDiscriminator
from .build_layer import get_normalization_layer, build_layers


class ConcordModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_domains, num_classes, encoder_dims=[], decoder_dims=[], 
                 augmentation_mask_prob: float = 0.3, dropout_prob: float = 0.1, norm_type='layer_norm', 
                 use_decoder=True, decoder_final_activation='leaky_relu',
                 use_classifier=False, use_importance_mask=False,
                 use_dab=False, dab_lambd=1.0):
        super().__init__()

        # Encoder
        self.input_dim = input_dim
        self.augmentation_mask = nn.Dropout(augmentation_mask_prob)
        self.use_classifier = use_classifier
        self.use_decoder = use_decoder
        self.use_importance_mask = use_importance_mask
        self.use_dab = use_dab

        if encoder_dims:
            self.encoder = build_layers(input_dim, hidden_dim, encoder_dims, dropout_prob, norm_type)
        else:
            self.encoder = nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                get_normalization_layer(norm_type, hidden_dim),
                nn.LeakyReLU(0.1)
            )

        # Decoder
        if self.use_decoder:
            if decoder_dims:
                self.decoder = build_layers(hidden_dim, input_dim, decoder_dims, dropout_prob, norm_type, 
                                            final_layer_norm=False, final_layer_dropout=False, final_activation=decoder_final_activation)
            else:
                self.decoder = nn.Sequential(
                    nn.Linear(hidden_dim, input_dim)
                )
            
        if self.use_dab:
            self.dab_decoder = AdversarialDiscriminator(hidden_dim, num_domains, reverse_grad=True, 
                                                        norm_type=norm_type, lambd = dab_lambd)

        # Classifier head
        if self.use_classifier:
            self.classifier = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                get_normalization_layer(norm_type, hidden_dim),
                nn.LeakyReLU(0.1),
                nn.Linear(hidden_dim, num_classes)
            )

        self._initialize_weights()

        # Learnable mask for feature importance
        if self.use_importance_mask:
            self.importance_mask = nn.Parameter(torch.ones(input_dim))

    def forward(self, x):
        out = {}

        if self.use_importance_mask:
            importance_weights = self.get_importance_weights()
            x = x * importance_weights

        x = self.augmentation_mask(x)

        for layer in self.encoder:
            x = layer(x)
        out['encoded'] = x

        if self.use_decoder:
            x = out['encoded']
            for layer in self.decoder:
                x = layer(x)
            if self.use_importance_mask:
                out['decoded'] = x * importance_weights
            else:
                out['decoded'] = x

        if self.use_dab:
            out['dab_pred'] = self.dab_decoder(out['encoded'])

        if self.use_classifier:
            out['class_pred'] = self.classifier(out['encoded'])

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

