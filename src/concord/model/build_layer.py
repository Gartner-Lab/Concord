
import torch.nn as nn

def get_normalization_layer(norm_type, num_features):
    """
    Returns the appropriate normalization layer.

    Args:
        norm_type (str): Type of normalization ('batch_norm', 'layer_norm', or 'none').
        num_features (int): Number of features for normalization.

    Returns:
        nn.Module: The corresponding normalization layer.

    Raises:
        ValueError: If an unknown normalization type is provided.
    """
    if norm_type == 'batch_norm':
        return nn.BatchNorm1d(num_features)
    elif norm_type == 'layer_norm':
        return nn.LayerNorm(num_features)
    elif norm_type == 'none':
        return nn.Identity()
    else:
        raise ValueError(f"Unknown normalization type: {norm_type}")

def build_layers(input_dim, output_dim, layer_dims, dropout_prob, norm_type,final_layer_norm=True, final_layer_dropout=True, final_activation='leaky_relu'):
    """
    Constructs a fully connected feedforward network with optional normalization, dropout, and activation.

    Args:
        input_dim (int): Number of input features.
        output_dim (int): Number of output features.
        layer_dims (list[int]): List of hidden layer sizes.
        dropout_prob (float): Dropout probability.
        norm_type (str): Type of normalization ('batch_norm', 'layer_norm', or 'none').
        final_layer_norm (bool, optional): Whether to apply normalization to the final layer. Defaults to True.
        final_layer_dropout (bool, optional): Whether to apply dropout to the final layer. Defaults to True.
        final_activation (str, optional): Final activation function ('relu' or 'leaky_relu'). Defaults to 'leaky_relu'.

    Returns:
        nn.Sequential: A PyTorch sequential model containing the specified layers.

    Raises:
        ValueError: If an unknown activation function is provided.
    """
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