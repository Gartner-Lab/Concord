
import pandas as pd
import matplotlib.pyplot as plt


def plot_coverage(coverage_dict, fontsize=10, figsize=(8,5), dpi=300, save_path=None):
    """
    Plot the neighborhood coverage for each dataset.

    Parameters:
    coverage_df: pd.DataFrame
        A DataFrame containing the neighborhood coverage for each dataset.
    """
    # Display the coverage DataFrame

    coverage_df = pd.DataFrame.from_dict(coverage_dict, orient='index', columns=['Neighborhood_Coverage'])
    coverage_df.index.name = 'Dataset'
    coverage_df = coverage_df.reset_index()

    # Visualize the coverage
    plt.figure(figsize=figsize, dpi=dpi)
    plt.bar(coverage_df['Dataset'], coverage_df['Neighborhood_Coverage'])
    plt.xlabel('Datasets', fontsize=fontsize)
    plt.ylabel('Neighborhood Coverage', fontsize=fontsize)
    plt.title('Neighborhood Coverage of Each Dataset in the k-NN Graph', fontsize=fontsize)
    plt.xticks(rotation=90, fontsize=fontsize)
    plt.yticks(fontsize=fontsize)

    if save_path:
        plt.savefig(save_path)
    else:
        plt.show()
