import matplotlib.pyplot as plt

def plot_trustworthiness(trustworthiness_df, text_label=True, text_shift=1, legend=False, fontsize=8, legend_fontsize=8, figsize=(6,4), dpi=300, save_path=None):
    plt.figure(figsize=figsize, dpi=dpi)

    # Plot trustworthiness for each embedding
    for embedding_key in trustworthiness_df['Embedding'].unique():
        # Select data for the current embedding
        embedding_data = trustworthiness_df[trustworthiness_df['Embedding'] == embedding_key]
        
        # Plot trustworthiness over n_neighbors for the embedding
        plt.plot(embedding_data['n_neighbors'], embedding_data['Trustworthiness'], label=embedding_key)
        
        # Add text label at the last point for each embedding
        if text_label:
            plt.text(
                embedding_data['n_neighbors'].values[-1]+text_shift, 
                embedding_data['Trustworthiness'].values[-1], 
                embedding_key, 
                fontsize=fontsize
            )

    # Add plot details
    plt.title('Trustworthiness of Latent Embeddings', fontsize=9)
    plt.xlabel('Number of Neighbors', fontsize=8)
    plt.ylabel('Trustworthiness', fontsize=8)
    plt.xticks(fontsize=7)
    plt.yticks(fontsize=7)
    
    # Add legend at right margin
    if legend:
        plt.legend(
            title=None, 
            loc='center left', 
            bbox_to_anchor=(1, 0.5),
            markerscale=1.5,
            handletextpad=0.2,
            fontsize=legend_fontsize,
            title_fontsize=legend_fontsize
        )

    # Save and show the plot
    if save_path:
        plt.savefig(save_path)
    plt.show()


def plot_distance_heatmap(distances, n_cols=3, annot_value=False, figsize=(2, 1.6), cbar=True, fontsize=10, dpi=300, save_path=None):
    # Visualize the distance matrices in a more compact layout
    from scipy.spatial.distance import squareform
    import matplotlib.pyplot as plt
    import seaborn as sns
    import numpy as np
    
    keys = list(distances.keys())
    n_plots = len(keys)
    n_rows = int(np.ceil(n_plots / n_cols))
    base_width = figsize[0]
    base_height = figsize[1]
    
    fig, axes = plt.subplots(
        n_rows, n_cols, figsize=(base_width * n_cols, base_height * n_rows), dpi=dpi
    )
    axes = np.atleast_2d(axes).flatten() 

    cbar_kws = {"shrink": 0.8, "label": None, "format": "%.2f", "pad": 0.02} if cbar else None

    for i, key in enumerate(keys):
        ax = axes[i]
        sns.heatmap(
            squareform(distances[key]),
            ax=ax,
            cmap="viridis",
            xticklabels=False,
            yticklabels=False,
            annot=annot_value, fmt=".2f", annot_kws={"size": 6},
            cbar=cbar,  # Pass the value of cbar here to toggle the color bar
            cbar_kws=cbar_kws
        )
        ax.set_title(key, fontsize=fontsize)  # Increase the title font size

    # Hide empty subplots if n_plots < n_cols * n_rows
    for j in range(n_plots, n_cols * n_rows):
        fig.delaxes(axes[j])

    # Set compact layout
    fig.tight_layout(pad=0.5, h_pad=0.8, w_pad=0.3)  # Adjust padding

    # Save the plot
    if save_path:
        plt.savefig(save_path, bbox_inches="tight")

    plt.show()


def plot_geometry_scatter(data_dict, correlation = None, ground_key='PCA_no_noise', s=1, alpha=0.5, n_cols=3, figsize=(4, 4), dpi=300, save_path=None):
    import matplotlib.pyplot as plt
    import numpy as np

    keys = list(data_dict.keys())
    n_plots = len(keys)
    n_rows = int(np.ceil(n_plots / n_cols))
    base_width = figsize[0]
    base_height = figsize[1]
    
    fig, axes = plt.subplots(
        n_rows, n_cols, figsize=(base_width * n_cols, base_height * n_rows), dpi=dpi
    )
    axes = np.atleast_2d(axes).flatten() 
    for i, key in enumerate(keys):
        # avoid plotting empty subplots
        if i >= n_plots:
            break
        ax = axes[i]

        # flat distance[ground_key] to np array if dict
        if isinstance(data_dict[ground_key], dict):
            ground_val = np.array([data_dict[ground_key][k] for k in data_dict[ground_key].keys()])
            latent_val = np.array([data_dict[key][k] for k in data_dict[key].keys()])
        else:        
            ground_val = data_dict[ground_key] 
            latent_val = data_dict[key]

        ax.scatter(ground_val, latent_val, s=s, alpha=alpha, edgecolors='none')
        if correlation is not None:
            corr_text = '\n' + '\n'.join([f'{col}:{correlation.loc[key, col]:.2f}' for col in correlation.columns])
        else:
            corr_text = ''
        ax.set_title(f'{key}{corr_text}', fontsize=6)
        ax.set_xlabel(f'{ground_key}', fontsize=6)
        ax.set_ylabel(f'{key}', fontsize=6)
        # ax set tick label font
        ax.tick_params(axis='both', which='major', labelsize=6)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, bbox_inches="tight")
    
    plt.show()



