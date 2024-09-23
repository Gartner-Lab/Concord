
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
    elif final_activation == 'leaky_relu':
        layers.append(nn.LeakyReLU(0.1))
    else:
        raise ValueError(f"Unknown final activation function: {final_activation}")
    if final_layer_dropout:
        layers.append(nn.Dropout(dropout_prob))
    return nn.Sequential(*layers)