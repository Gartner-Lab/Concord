import torch
import pandas as pd

def compute_feature_importance(model, input_tensors, layer_index):
    """
    Compute the importance of each input feature to the specified layer's neurons.

    Parameters:
    - model (nn.Module): The model containing the layers.
    - input_tensors (torch.Tensor): The input data.
    - layer_index (int): The index of the layer to visualize.

    Returns:
    - importance_matrix (torch.Tensor): The computed importance matrix.
    """

    if layer_index >= len(model.encoder):
        raise ValueError(f"layer_index {layer_index} is greater than the total number of layers in the encoder {len(model.encoder)}")

    input_tensors.requires_grad = True
    x = input_tensors

    if model.use_importance_mask:
        importance_weights = model.get_importance_weights()
        x = x * importance_weights

    # Forward pass until the specified layer
    for idx, layer in enumerate(model.encoder):
        x = layer(x)
        if idx == layer_index:
            break

    encoded_output = x
    n_input_features = input_tensors.size(1)
    n_encoded_neurons = encoded_output.size(1)

    importance_matrix = torch.zeros((n_input_features, n_encoded_neurons))

    for i in range(n_encoded_neurons):  # Iterate over each encoded neuron
        model.zero_grad()
        encoded_output[:, i].sum().backward(retain_graph=True)
        importance_matrix[:, i] = input_tensors.grad.mean(dim=0)
        input_tensors.grad.zero_()

    return importance_matrix.cpu().numpy()






def prepare_ranked_list(importance_matrix, adata, expr_level=False):
    """
    Prepare a ranked list of genes based on their importance weights for each neuron.

    Parameters:
    - importance_matrix (torch.Tensor): The importance matrix with shape (n_input_features, n_encoded_neurons).
    - adata (anndata.AnnData): The AnnData object containing the input features.

    Returns:
    - ranked_lists (dict): A dictionary with neuron names as keys and ranked gene lists as values.
    """
    # Convert the importance matrix to a DataFrame
    encoded_neuron_names = [f'Neuron {i}' for i in range(importance_matrix.shape[1])]
    input_features = adata.var_names
    df_importance = pd.DataFrame(importance_matrix, index=input_features, columns=encoded_neuron_names)
    if expr_level:
        # Compute expression level of each gene
        expr_levels = adata.X.mean(axis=0).A1
        nonzero_levels = (adata.X > 0).mean(axis=0).A1
        df_importance['Expression Level'] = expr_levels
        df_importance['Nonzero Fraction'] = nonzero_levels
    
    # Prepare ranked lists
    ranked_lists = {}
    for neuron in encoded_neuron_names:
        ranked_list = df_importance[neuron].sort_values(ascending=False).reset_index()
        ranked_list.columns = ['Gene', 'Importance']
        if expr_level:
            ranked_list = ranked_list.merge(df_importance[['Expression Level', 'Nonzero Fraction']], left_on='Gene', right_index=True)
        ranked_lists[neuron] = ranked_list

    return ranked_lists

