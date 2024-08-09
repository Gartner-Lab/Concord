import torch
import torch.nn as nn


def get_normalization_layer(norm_type, num_features):
    if norm_type == 'batch_norm':
        return nn.BatchNorm1d(num_features)
    elif norm_type == 'layer_norm':
        return nn.LayerNorm(num_features)
    elif norm_type == 'none':
        return nn.Identity()
    else:
        raise ValueError(f"Unknown normalization type: {norm_type}")

def build_layers(input_dim, output_dim, layer_dims, dropout_prob, norm_type,final_layer_norm=True, final_layer_dropout=True, final_activation='leaky_relu'):
    layers = [
        nn.Linear(input_dim, layer_dims[0]),

        get_normalization_layer(norm_type, layer_dims[0]),
        nn.LeakyReLU(0.1),
        nn.Dropout(dropout_prob)
    ]
    for i in range(len(layer_dims) - 1):
        layers.extend([
            nn.Linear(layer_dims[i], layer_dims[i + 1]),

            get_normalization_layer(norm_type, layer_dims[i + 1]),
            nn.LeakyReLU(0.1),
            nn.Dropout(dropout_prob)
        ])
    
    layers.append(nn.Linear(layer_dims[-1], output_dim))
    if final_layer_norm:
        layers.append(get_normalization_layer(norm_type, output_dim))
    if final_activation == 'relu':
        layers.append(nn.ReLU())
    else:
        layers.append(nn.LeakyReLU(0.1))
    if final_layer_dropout:
        layers.append(nn.Dropout(dropout_prob))
    return nn.Sequential(*layers)


class ConcordModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_classes, encoder_dims=[], decoder_dims=[], 
                 augmentation_mask_prob: float = 0.3, dropout_prob: float = 0.1, norm_type='layer_norm', use_decoder=True,
                 use_classifier=False, use_importance_mask=False):
        super().__init__()

        # Encoder
        self.input_dim = input_dim
        self.augmentation_mask = nn.Dropout(augmentation_mask_prob)
        self.norm_type = norm_type
        self.use_classifier = use_classifier
        self.use_decoder = use_decoder
        self.use_importance_mask = use_importance_mask

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
                                            final_layer_norm=False, final_layer_dropout=False, final_activation='relu')
            else:
                self.decoder = nn.Sequential(
                    nn.Linear(hidden_dim, input_dim)
                )

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

    def forward(self, x, domain_idx=None):
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

