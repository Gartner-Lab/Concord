# Evaluate the geometric aspects of the methods


def compute_reconstruction_error(adata, layer1, layer2, metric='mse'):
    """
    Computes the reconstruction error between two layers in an AnnData object.

    Parameters:
    - adata: AnnData object containing the layers.
    - layer1: Name of the first layer (e.g., 'original').
    - layer2: Name of the second layer (e.g., 'reconstructed').
    - metric: Error metric to use; 'mse' for Mean Squared Error or 'mae' for Mean Absolute Error.

    Returns:
    - error: The computed reconstruction error.
    """
    import numpy as np

    # Extract the two layers as numpy arrays
    matrix1 = adata.layers[layer1]
    matrix2 = adata.layers[layer2]

    # Ensure both matrices have the same shape
    if matrix1.shape != matrix2.shape:
        raise ValueError("The matrices must have the same shape.")

    # Compute the error based on the chosen metric
    if metric == 'mse':
        error = np.mean((matrix1 - matrix2) ** 2)
    elif metric == 'mae':
        error = np.mean(np.abs(matrix1 - matrix2))
    else:
        raise ValueError("Invalid metric. Choose 'mse' or 'mae'.")

    return error

