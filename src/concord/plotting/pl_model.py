

import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from .. import logger
import math
import time

from ..utils.importance_analysis import compute_feature_importance

def visualize_importance_weights(model, adata, top_n=20, mode='histogram', fontsize=12, figsize=(5, 3), save_path=None):
    """
    Visualizes feature importance weights from a trained model.

    This function plots either a histogram of all importance weights or a bar chart
    of the top features based on their importance values.

    Args:
        model (torch.nn.Module): 
            The trained model containing feature importance weights.
        adata (AnnData): 
            The AnnData object containing gene expression data.
        top_n (int, optional): 
            Number of top features to plot when `mode` is not 'histogram'. Defaults to `20`.
        mode (str, optional): 
            Visualization mode. Options:
            - `'histogram'`: Plots the distribution of all importance weights.
            - `'highest'`: Shows top `top_n` features with highest importance.
            - `'lowest'`: Shows `top_n` features with lowest importance.
            - `'absolute'`: Shows `top_n` features with highest absolute importance.
            Defaults to `'histogram'`.
        fontsize (int, optional): 
            Font size for axis labels and titles. Defaults to `12`.
        figsize (tuple, optional): 
            Figure size `(width, height)`. Defaults to `(5, 3)`.
        save_path (str, optional): 
            If provided, saves the figure at the specified path. Defaults to `None`.

    Raises:
        ValueError: If `mode` is not one of `'histogram'`, `'highest'`, `'lowest'`, `'absolute'`.

    Returns:
        None
            Displays or saves the importance weights plot.

    Example:
        ```python
        visualize_importance_weights(model, adata, mode='highest', top_n=30)
        ```
    """
    if not model.use_importance_mask:
        logger.warning("Importance mask is not used in this model.")
        return

    importance_weights = model.get_importance_weights().detach().cpu().numpy()

    if mode == 'histogram':
        plt.figure(figsize=figsize)
        plt.hist(importance_weights, bins=30, edgecolor='k')
        plt.xlabel('Importance Weight', fontsize=fontsize)
        plt.ylabel('Frequency', fontsize=fontsize)
        plt.title('Distribution of Importance Weights', fontsize=fontsize)
        plt.xticks(fontsize=fontsize)
        plt.yticks(fontsize=fontsize)

        if save_path:
            plt.savefig(save_path)
        else:
            plt.show()
    else:
        if mode == 'highest':
            top_indices = np.argsort(importance_weights)[-top_n:]
        elif mode == 'lowest':
            top_indices = np.argsort(importance_weights)[:top_n]
        elif mode == 'absolute':
            top_indices = np.argsort(np.abs(importance_weights))[-top_n:]
        else:
            raise ValueError("Mode must be one of ['highest', 'lowest', 'absolute', 'histogram']")

        top_weights = importance_weights[top_indices]
        feature_names = adata.var_names[top_indices]

        plt.figure(figsize=figsize)
        plt.barh(range(top_n), top_weights, align='center')
        plt.yticks(range(top_n), feature_names, fontsize=fontsize)
        plt.xlabel('Importance Weight', fontsize=fontsize)
        plt.ylabel('Feature Names', fontsize=fontsize)
        plt.title(f'Top Features by Importance Weight ({mode})', fontsize=fontsize)
        plt.xticks(fontsize=fontsize)
        plt.yticks(fontsize=fontsize)

        if save_path:
            plt.savefig(save_path)
        else:
            plt.show()




def plot_importance_heatmap(importance_matrix, input_feature=None, figsize=(20, 15), save_path=None):
    """
    Plots a heatmap of feature importance across encoded neurons.

    This function visualizes the importance of each input feature for different
    encoded neurons using hierarchical clustering.

    Args:
        importance_matrix (numpy.ndarray or torch.Tensor): 
            The importance matrix with shape `(n_input_features, n_encoded_neurons)`.
        input_feature (list of str, optional): 
            List of input feature names (e.g., gene names). If `None`, generic feature names are used.
        figsize (tuple, optional): 
            Figure size `(width, height)`. Defaults to `(20, 15)`.
        save_path (str, optional): 
            If provided, saves the heatmap at the specified path. Defaults to `None`.

    Returns:
        None
            Displays or saves the importance heatmap.

    Example:
        ```python
        plot_importance_heatmap(importance_matrix, input_feature=adata.var_names)
        ```
    """

    # Extract input feature names from adata
    input_feature_names = input_feature
    encoded_neuron_names = [f'Neuron {i}' for i in range(importance_matrix.shape[1])]

    # Create a DataFrame for the heatmap
    df_importance = pd.DataFrame(importance_matrix.T, index=encoded_neuron_names, columns=input_feature_names)

    # Plot the heatmap with hierarchical clustering
    cluster_grid = sns.clustermap(
        df_importance,
        cmap='viridis',
        annot=False,
        cbar=True,
        figsize=figsize,
        #xticklabels=True,
        yticklabels=True
    )

    # Adjust the x-axis labels for better readability
    plt.setp(cluster_grid.ax_heatmap.xaxis.get_majorticklabels(), rotation=90, fontsize=8)
    plt.setp(cluster_grid.ax_heatmap.yaxis.get_majorticklabels(), fontsize=10)

    # Adjust the overall plot to make room for labels
    cluster_grid.figure.subplots_adjust(bottom=0.3, right=0.8)

    plt.title('Feature Importance Heatmap with Hierarchical Clustering')
    plt.xlabel('Input Features')
    plt.ylabel('Encoded Neurons')

    if save_path:
        plt.savefig(save_path)
    else:
        plt.show()


def plot_top_genes_per_neuron(ranked_gene_lists, show_neurons=None, top_n=10, ncols=4, figsize=(4, 4), save_path=None):
    """
    Plots bar charts of the top contributing genes for each neuron.

    This function generates bar plots showing the most important genes contributing
    to each encoded neuron in a compact grid layout.

    Args:
        ranked_gene_lists (dict): 
            Dictionary where keys are neuron names and values are DataFrames containing ranked genes.
        show_neurons (list, optional): 
            List of neurons to plot. If `None`, plots all neurons available in `ranked_gene_lists`. Defaults to `None`.
        top_n (int, optional): 
            Number of top contributing genes to display for each neuron. Defaults to `10`.
        ncols (int, optional): 
            Number of columns in the subplot grid. Defaults to `4`.
        figsize (tuple, optional): 
            Size of each subplot `(width, height)`. Defaults to `(4, 4)`.
        save_path (str, optional): 
            If provided, saves the plot at the specified path. Defaults to `None`.

    Returns:
        None
            Displays or saves the bar charts.

    Example:
        ```python
        plot_top_genes_per_neuron(ranked_gene_lists, top_n=15, ncols=3, save_path="top_genes.png")
        ```
    """

    # If `show_neurons` is None, use all available neurons
    if show_neurons is None:
        show_neurons = list(ranked_gene_lists.keys())
    else:
        # Filter the provided neurons to only those in `ranked_gene_lists`
        show_neurons = [neuron for neuron in show_neurons if neuron in ranked_gene_lists]

    # Determine the number of rows needed
    nrows = math.ceil(len(show_neurons) / ncols)

    # Create subplots
    fig, axes = plt.subplots(
        nrows=nrows,
        ncols=ncols,
        figsize=(ncols * figsize[0], nrows * figsize[1]),
        constrained_layout=True,
    )
    axes = axes.flatten()  # Flatten the 2D array of axes for easy iteration

    # Plot top genes for each neuron
    for idx, neuron in enumerate(show_neurons):
        if neuron in ranked_gene_lists:
            top_genes = ranked_gene_lists[neuron].head(top_n)
            sns.barplot(
                x=top_genes["Importance"].values,
                y=top_genes["Gene"].values,
                palette="viridis_r",
                ax=axes[idx],
            )
            axes[idx].set_title(f"Top {top_n} Contributing Genes for {neuron}")
            axes[idx].set_xlabel("Importance")
            axes[idx].set_ylabel("Genes")

    # Remove empty subplots
    for i in range(len(show_neurons), len(axes)):
        fig.delaxes(axes[i])

    # Save or display the plot
    if save_path:
        plt.savefig(save_path)
    else:
        plt.show()

