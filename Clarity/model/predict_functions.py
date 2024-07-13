import torch
import numpy as np


def predict_with_model(model, dataloader, device, data_structure, sort_by_indices=False):
    """
    Predict using the model and handle different data structures.

    Parameters:
    - model: The model to use for prediction.
    - dataloader: The DataLoader providing the data.
    - device: The device to run the model on.
    - data_structure: A list indicating the structure of the data. Example: ['input', 'class', 'domain', 'indices']
    - sort_by_indices: Boolean, if True, sort the results by the indices.

    Returns:
    - embeddings: The encoded embeddings from the model.
    - class_preds: The predicted classes (if available).
    - class_true: The true labels (if available).
    """
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

    # Check length consistency
    if len(embeddings) != len(dataloader.dataset):
        raise ValueError(
            f"Mismatch in number of embeddings: {len(embeddings)} vs dataset size: {len(dataloader.dataset)}")

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

