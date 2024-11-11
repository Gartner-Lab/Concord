import matplotlib.pyplot as plt

def plot_trustworthiness(trustworthiness_df, fontsize=5, figsize=(6,4), dpi=300, save_path=None):
    plt.figure(figsize=figsize, dpi=dpi)

    # Plot trustworthiness for each embedding
    for embedding_key in trustworthiness_df['Embedding'].unique():
        # Select data for the current embedding
        embedding_data = trustworthiness_df[trustworthiness_df['Embedding'] == embedding_key]
        
        # Plot trustworthiness over n_neighbors for the embedding
        plt.plot(embedding_data['n_neighbors'], embedding_data['Trustworthiness'], label=embedding_key)
        
        # Add text label at the last point for each embedding
        plt.text(
            embedding_data['n_neighbors'].values[-1], 
            embedding_data['Trustworthiness'].values[-1], 
            embedding_key, 
            fontsize=fontsize
        )

    # Add plot details
    plt.title('Trustworthiness of Latent Embeddings')
    plt.xlabel('Number of Neighbors')
    plt.ylabel('Trustworthiness')
    
    # Add legend at the bottom left
    plt.legend(fontsize='small', loc='lower left')

    # Save and show the plot
    if save_path:
        plt.savefig(save_path)
    plt.show()