
import matplotlib.pyplot as plt
import numpy as np
import torch
import seaborn as sns
from ..utils.batch_analysis import get_attribute_from_dataloader

def visualize_batch_composition(dataloader, adata, batch_indices=None, data_structure=None, attribute='class', attribute_key=None, figsize=(12, 5), save_path=None):
    """
    Visualize the composition of specified batches in terms of class or domain.

    Parameters:
    - chunk_loader: Chunk loader object containing the dataloaders.
    - data_structure: Structure of the data in the dataloader.
    - attribute: 'class' or 'domain' to specify the attribute to visualize.
    - attribute_key: Key to access the attribute names in adata.
    - save_path: Optional path to save the plot.
    - max_batches: Maximum number of batches to visualize.
    """

    attribute_data = get_attribute_from_dataloader(dataloader, batch_indices, data_structure, attribute)

    # Flatten each batch's data and stack them into a single tensor
    flat_attribute_data = [batch.view(-1) for batch in attribute_data]
    flat_attribute_data = torch.cat(flat_attribute_data).numpy()

    # Prepare data for plotting
    unique_attributes = sorted(set(flat_attribute_data))
    attribute_names = [adata.obs[attribute_key].cat.categories[attr] for attr in unique_attributes]
    num_batches = len(attribute_data)
    num_attributes = len(unique_attributes)
    counts = np.zeros((num_batches, num_attributes), dtype=int)

    for i, batch in enumerate(attribute_data):
        batch_flat = batch.view(-1).numpy()
        counts[i, :] = np.array([(batch_flat == attr).sum() for attr in unique_attributes])

    # Use seaborn color palette
    palette = sns.color_palette('tab20', num_attributes) if num_attributes <= 20 else sns.color_palette('husl', num_attributes)

    # Plotting
    fig, ax = plt.subplots(figsize=figsize)
    bar_width = 0.35 / num_attributes  # Adjusted bar width to prevent overlapping
    bar_positions = np.arange(num_batches)

    for attr_idx, attr_name in enumerate(attribute_names):
        color = palette[attr_idx]
        ax.bar(bar_positions + attr_idx * bar_width, counts[:, attr_idx], bar_width, label=attr_name, color=color)

    ax.set_xlabel('Batch')
    ax.set_ylabel('Counts')
    ax.set_title(f'Batch Composition by {attribute.capitalize()}')
    ax.set_xticks(bar_positions + bar_width * (num_attributes - 1) / 2)
    ax.set_xticklabels(range(num_batches))

    # Place the legend outside the plot
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)

    if save_path:
        plt.savefig(save_path)
    else:
        plt.show()