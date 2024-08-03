

import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from .. import logger
import math
import time

from ..utils.importance_analysis import compute_feature_importance

def visualize_importance_weights(model, adata, top_n=20, mode='histogram', fontsize=12, figsize=(5, 3), save_path=None):
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




def plot_importance_heatmap(importance_matrix, adata, figsize=(20, 15), save_path=None):
    """
    Plots a heatmap of the importance matrix with adjustments for label readability.

    Parameters:
    - importance_matrix (torch.Tensor): The importance matrix with shape (n_input_features, n_encoded_neurons).
    - adata (anndata.AnnData): The AnnData object containing the input features.
    """
    # Convert the importance matrix to a NumPy array
    importance_matrix = importance_matrix.detach().numpy()

    # Extract input feature names from adata
    input_feature_names = adata.var.index.tolist()
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
    cluster_grid.fig.subplots_adjust(bottom=0.3, right=0.8)

    plt.title('Feature Importance Heatmap with Hierarchical Clustering')
    plt.xlabel('Input Features')
    plt.ylabel('Encoded Neurons')

    if save_path:
        plt.savefig(save_path)
    else:
        plt.show()



def plot_top_genes_per_neuron(importance_matrix, adata, top_n=10, ncols=4, figsize=(4, 4), save_path=None):
    """
    Plots bar charts of the top contributing genes for each neuron in a compact grid layout.

    Parameters:
    - importance_matrix (torch.Tensor): The importance matrix with shape (n_input_features, n_encoded_neurons).
    - adata (anndata.AnnData): The AnnData object containing the input features.
    - top_n (int): Number of top contributing genes to display for each neuron.
    - ncols (int): Number of columns in the grid layout.
    """
    # Convert the importance matrix to a NumPy array
    importance_matrix = importance_matrix.detach().numpy()

    # Extract input feature names from adata
    input_feature_names = adata.var.index.tolist()
    encoded_neuron_names = [f'Neuron {i}' for i in range(importance_matrix.shape[1])]

    # Create a DataFrame for the importance matrix
    df_importance = pd.DataFrame(importance_matrix, index=input_feature_names, columns=encoded_neuron_names)

    # Determine the number of rows needed
    nrows = math.ceil(len(encoded_neuron_names) / ncols)

    # Create subplots
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(ncols * figsize[0], nrows * figsize[1]), constrained_layout=True)
    axes = axes.flatten()  # Flatten the 2D array of axes for easy iteration

    # Plot top genes for each neuron
    for idx, neuron in enumerate(df_importance.columns):
        top_genes = df_importance[neuron].nlargest(top_n)
        sns.barplot(x=top_genes.values, y=top_genes.index, palette="viridis_r", ax=axes[idx])
        axes[idx].set_title(f'Top {top_n} Contributing Genes for {neuron}')
        axes[idx].set_xlabel('Importance')
        axes[idx].set_ylabel('Genes')

    # Remove empty subplots
    for i in range(len(encoded_neuron_names), len(axes)):
        fig.delaxes(axes[i])

    if save_path:
        file_suffix = f"{time.strftime('%b%d-%H%M')}"
        save_path = f"{save_path}_{file_suffix}.png"
        plt.savefig(save_path)
    else:
        plt.show()


