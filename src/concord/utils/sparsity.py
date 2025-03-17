import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from anndata import AnnData
from matplotlib.lines import Line2D

def plot_sparsity_by_batch(adata: AnnData, batch_column: str = 'batch'):
    # Ensure batch_column exists in the metadata
    if batch_column not in adata.obs.columns:
        raise ValueError(f"'{batch_column}' not found in adata.obs.columns")
    
    # Compute the fraction of non-zero values for each cell
    non_zero_fraction = np.array((adata.X > 0).sum(axis=1)).flatten() / adata.shape[1]
    
    # Add this as a new column in the obs DataFrame
    adata.obs['non_zero_fraction'] = non_zero_fraction
    
    # Create a DataFrame for plotting
    plot_df = pd.DataFrame({
        'Non-Zero Fraction': adata.obs['non_zero_fraction'],
        'Batch': adata.obs[batch_column]
    })
    
    # Plot histogram using seaborn
    plt.figure(figsize=(10, 6))
    sns.histplot(data=plot_df, x='Non-Zero Fraction', hue='Batch', element='step', stat='density', common_norm=False)
    plt.title('Sparsity of Cells by Batch')
    plt.xlabel('Fraction of Non-Zero Values')
    plt.ylabel('Density')
    
    # Manually create the legend
    batches = plot_df['Batch'].unique()
    handles = [Line2D([0], [0], color=sns.color_palette()[i], lw=3) for i in range(len(batches))]
    plt.legend(handles=handles, labels=batches, title='Batch', loc='upper right', bbox_to_anchor=(1.15, 1))
    
    plt.grid(True)
    plt.show()
