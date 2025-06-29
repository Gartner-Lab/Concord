

from __future__ import annotations

def get_attribute_from_dataloader(dataloader, batch_indices=None, data_structure=None, attribute='class'):
    """
    Get the attribute data from selected batches in the dataloader.

    Parameters:
    - chunk_loader: Chunk loader object containing the dataloaders.
    - data_structure: Structure of the data in the dataloader.
    - attribute: 'class' or 'domain' to specify the attribute to extract.
    - max_batches: Maximum number of batches to extract data from.

    Returns:
    - attribute_data: Dictionary containing the attribute data for each batch.
    """
    if not dataloader:
        raise ValueError("Data loader is cannot be None.")
    if batch_indices is None or len(batch_indices) == 0:
        raise ValueError("batch_index must be specified.")
    if attribute not in data_structure:
        raise ValueError("Attribute must be in data structure.")

    attribute_data = []
    found_indices = set()

    for batch_idx, batch in enumerate(dataloader):
        if batch_idx in batch_indices:
            attr_data = batch[data_structure.index(attribute)].cpu()
            attribute_data.append(attr_data)
            found_indices.add(batch_idx)
            if len(found_indices) == len(batch_indices):
                break

    if len(found_indices) != len(batch_indices):
        raise ValueError("Some batch indices were not found in the dataloader.")

    return attribute_data
