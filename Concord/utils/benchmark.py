
import pandas as pd
from .timer import Timer
import logging
import numpy as np
from .. import logger


def count_total_runs(param_grid):
    total_runs = 0
    for key, values in param_grid.items():
        if isinstance(values, list):
            total_runs += len(values)
    return total_runs

def run_hyperparameter_tests(adata, base_params, param_grid, output_key = "X_concord", return_decoded=False, trace_memory=False, trace_gpu_memory=False, save_dir="./"):
    import time
    import json
    from pathlib import Path
    from copy import deepcopy
    from .timer import Timer
    from .anndata_utils import save_obsm_to_hdf5
    import tracemalloc

    total_runs = count_total_runs(param_grid)
    logger.info(f"Total number of runs: {total_runs}")

    log_df = pd.DataFrame(columns=["param_name", "value", "time_minutes", "peak_memory_MB", "peak_gpu_memory_MB"])

    if trace_gpu_memory:
        try:
            import pynvml
            pynvml.nvmlInit()
            handle = pynvml.nvmlDeviceGetHandleByIndex(0)  # Assuming a single GPU setup
        except ImportError:
            raise ImportError("pynvml is not available.")

    for param_name, values in param_grid.items():
        for value in values:
            # Create a deep copy of base_params to avoid modifying the original dictionary
            params_copy = deepcopy(base_params)
            
            with Timer() as timer:

                if trace_memory:
                    tracemalloc.start()

                if trace_gpu_memory:
                    # Get initial GPU memory usage
                    initial_gpu_memory = pynvml.nvmlDeviceGetMemoryInfo(handle).used

                logger.info(f"Encoding adata with {param_name}={value}.")
                params_copy[param_name] = value

                # Initialize Concord model with updated parameters
                from ..concord import Concord
                ccd = Concord(adata=adata, **params_copy)

                # Define the output key and file suffix including param_name and value
                output_key_final = f"{output_key}_{param_name}_{str(value).replace(' ', '')}"
                file_suffix = f"{param_name}_{str(value).replace(' ', '')}_{time.strftime('%b%d-%H%M')}"


                # Encode adata and store the results in adata.obsm
                ccd.encode_adata(input_layer_key="X_log1p", output_key=output_key_final, return_decoded=return_decoded)
                adata.obsm[output_key_final] = ccd.adata.obsm[output_key_final]

                if trace_memory:
                    current, peak = tracemalloc.get_traced_memory()
                    tracemalloc.stop()
                    peak_memory_MB = peak / (1024 * 1024)  # Convert to MB
                else:
                    peak_memory_MB = None

                if trace_gpu_memory:
                    # Get final GPU memory usage and calculate peak usage
                    final_gpu_memory = pynvml.nvmlDeviceGetMemoryInfo(handle).used
                    peak_gpu_memory_MB = (final_gpu_memory - initial_gpu_memory) / (1024 * 1024)  # Convert to MB
                else:
                    peak_gpu_memory_MB = None


                # Save the parameter settings
                config_filename = Path(save_dir) / f"config_{file_suffix}.json"
                with open(config_filename, 'w') as f:
                    json.dump(ccd.config.to_dict(), f, indent=4)

            time_minutes = timer.interval / 60
            logger.info(f"Took {time_minutes:.2f} minutes to encode adata with {param_name}={value}, result saved to adata.obsm['{output_key}'], config saved to {config_filename}.")

            log_data = pd.DataFrame([{
                "param_name": param_name,
                "value": value,
                "time_minutes": time_minutes,
                "peak_memory_MB": peak_memory_MB,
                "peak_gpu_memory_MB": peak_gpu_memory_MB
            }])
            log_df = pd.concat([log_df, log_data], ignore_index=True)

    if trace_gpu_memory:
        pynvml.nvmlShutdown()

    # Save the entire adata.obsm to file after all tests
    obsm_filename =  Path(save_dir) / f"final_obsm_{file_suffix}.h5"
    save_obsm_to_hdf5(adata, obsm_filename)

    log_filename = Path(save_dir) / f"hyperparameter_test_log_{file_suffix}.csv"
    log_df.to_csv(log_filename, index=False)

    return adata




def compute_correlation(data_dict, corr_types=['pearsonr', 'spearmanr', 'kendalltau'], groundtruth_key="PCA_no_noise"):
    from scipy.stats import pearsonr, spearmanr, kendalltau
    import pandas as pd

    pr_result, sr_result, kt_result = {}, {}, {}
    
    # Calculate correlations based on requested types
    for key in data_dict.keys():
        ground_val = data_dict[groundtruth_key]
        ground_val = np.array(list(ground_val.values())) if isinstance(ground_val, dict) else ground_val

        latent_val = data_dict[key]
        latent_val = np.array(list(latent_val.values())) if isinstance(latent_val, dict) else latent_val

        if 'pearsonr' in corr_types:
            pr_result[key] = pearsonr(ground_val, latent_val)[0]
        if 'spearmanr' in corr_types:
            sr_result[key] = spearmanr(ground_val, latent_val)[0]
        if 'kendalltau' in corr_types:
            kt_result[key] = kendalltau(ground_val, latent_val)[0]
    
    # Collect correlation values for each type
    corr_values = {}
    for key in data_dict.keys():
        corr_values[key] = [
            pr_result.get(key, None),
            sr_result.get(key, None),
            kt_result.get(key, None)
        ]
    
    # Create DataFrame with correlation types as row indices and keys as columns
    corr_df = pd.DataFrame(corr_values, index=['pearsonr', 'spearmanr', 'kendalltau'])
    
    # Filter only for requested correlation types
    corr_df = corr_df.loc[corr_types].T

    return corr_df


def compare_graph_connectivity(adata, emb1, emb2, k=30, use_faiss=False, use_ivf=False, ivf_nprobe=10, metric=['jaccard', 'frobenius', 'hamming'], dist_metric='euclidean'):
    """
    Compare the graph connectivity of two embeddings by computing their k-NN graphs
    and comparing their adjacency matrices using specified metrics.

    Parameters:
    - adata: AnnData
        AnnData object containing embeddings in `adata.obsm`.
    - emb1: str
        Key for the first embedding in `adata.obsm`.
    - emb2: str
        Key for the second embedding in `adata.obsm`.
    - k: int
        Number of nearest neighbors for the k-NN graph.
    - use_faiss: bool
        Whether to use FAISS for nearest neighbor computation.
    - use_ivf: bool
        Whether to use IVF FAISS index.
    - ivf_nprobe: int
        Number of probes for IVF FAISS index.
    - metric: list of str
        List of metrics to use for graph comparison: ['jaccard', 'frobenius', 'hamming'].

    Returns:
    - graph_distance: dict
        Dictionary with keys as metric names and values as similarity scores.
    """
    from scipy.sparse import csr_matrix
    import numpy as np
    from ..model.knn import Neighborhood  # Adjust import based on your directory structure

    # Check if embeddings exist in adata.obsm
    if emb1 not in adata.obsm or emb2 not in adata.obsm:
        raise ValueError(f"Embedding keys {emb1} and {emb2} not found in adata.obsm.")
    
    emb1 = adata.obsm[emb1]
    emb2 = adata.obsm[emb2]

    # Initialize Neighborhood objects for both embeddings
    neighborhood1 = Neighborhood(emb1, k=k, use_faiss=use_faiss, use_ivf=use_ivf, ivf_nprobe=ivf_nprobe, metric=dist_metric)
    neighborhood2 = Neighborhood(emb2, k=k, use_faiss=use_faiss, use_ivf=use_ivf, ivf_nprobe=ivf_nprobe, metric=dist_metric)

    # Compute k-NN indices for all points
    core_samples1 = np.arange(emb1.shape[0])
    core_samples2 = np.arange(emb2.shape[0])

    indices1 = neighborhood1.get_knn(core_samples1, k=k, include_self=False)
    indices2 = neighborhood2.get_knn(core_samples2, k=k, include_self=False)

    # Create adjacency matrices
    rows1 = np.repeat(core_samples1, k)
    cols1 = indices1.flatten()
    graph1 = csr_matrix((np.ones_like(cols1), (rows1, cols1)), shape=(emb1.shape[0], emb1.shape[0]))

    rows2 = np.repeat(core_samples2, k)
    cols2 = indices2.flatten()
    graph2 = csr_matrix((np.ones_like(cols2), (rows2, cols2)), shape=(emb2.shape[0], emb2.shape[0]))

    # Compare graphs based on the chosen metric
    graph_distance = {}
    if 'jaccard' in metric:
        graph1_binary = graph1 > 0
        graph2_binary = graph2 > 0
        intersection = graph1_binary.multiply(graph2_binary).sum()
        union = (graph1_binary + graph2_binary > 0).sum()
        graph_distance['jaccard'] = intersection / union
    if 'hamming' in metric:
        graph1_binary = graph1 > 0
        graph2_binary = graph2 > 0
        graph_distance['hamming'] = 1 - (graph1_binary != graph2_binary).sum() / graph1_binary.nnz
    if 'frobenius' in metric:
        graph_distance['frobenius'] = np.linalg.norm((graph1 - graph2).toarray())

    return graph_distance


def benchmark_graph_connectivity(adata, emb_keys, k=30, use_faiss=False, use_ivf=False, ivf_nprobe=10, metric=['jaccard', 'hamming'], 
                                 groundtruth_keys = {'(nn)': 'PCA_no_noise','(wn)': 'PCA_wt_noise'}, dist_metric='cosine'):

    connectivity_df = pd.DataFrame()
    for gname,gemb in groundtruth_keys.items():
        results = []
        for key in emb_keys:
            similarity_scores = compare_graph_connectivity(
                adata,
                emb1=key,
                emb2=gemb,
                k=k,
                metric=metric,
                dist_metric=dist_metric,
                use_faiss=use_faiss,
                use_ivf=use_ivf,
                ivf_nprobe=ivf_nprobe
            )
            results.append(similarity_scores)

        df = pd.DataFrame(results, index=emb_keys)
        # Add a second level index to the column named 'metric'
        df.columns = pd.MultiIndex.from_tuples([(f'Graph connectivity', col + gname) for col in df.columns])
        connectivity_df = pd.concat([connectivity_df, df], axis=1)
    
    return connectivity_df



def benchmark_topology(diagrams, expected_betti_numbers=[1,0,0], n_bins=100, save_dir=None, file_suffix=None):
    """
    Benchmark the topological properties of persistence diagrams.

    Args:
        diagrams : dict
            A dictionary where keys are method names and values are persistence diagrams.
        expected_betti_numbers : list, optional
            A list specifying the expected Betti numbers for different homology dimensions. Default is [1, 0, 0].
        n_bins : int, optional
            Number of bins to use for Betti curve calculations. Default is 100.
        save_dir : str, optional
            Directory to save benchmarking results as CSV files. If None, results are not saved.
        file_suffix : str, optional
            Suffix to append to saved filenames.

    Returns:
        dict
            A dictionary containing:
            - `'betti_stats'`: DataFrame summarizing Betti statistics.
            - `'distance_metrics'`: DataFrame of computed distances between Betti curves.
            - `'combined_metrics'`: DataFrame of entropy, variance, and L1 distance metrics.
    """
    import pandas as pd
    from .tda import compute_betti_statistics, summarize_betti_statistics

    results = {}
    betti_stats = {}    
    # Compute betti stats for all keys
    for key in diagrams.keys():
        betti_stats[key] = compute_betti_statistics(diagram=diagrams[key], expected_betti_numbers=expected_betti_numbers, n_bins=n_bins)

    betti_stats_pivot, distance_metrics_df = summarize_betti_statistics(betti_stats)
    results['betti_stats'] = betti_stats_pivot
    results['distance_metrics'] = distance_metrics_df

    entropy_columns = betti_stats_pivot.loc[:, pd.IndexSlice[:, 'Entropy']]
    average_entropy = entropy_columns.mean(axis=1)
    variance_columns = betti_stats_pivot.loc[:, pd.IndexSlice[:, 'Variance']]
    average_variance = variance_columns.mean(axis=1)
    final_metrics = pd.concat([average_entropy, average_variance], axis=1)
    final_metrics.columns = pd.MultiIndex.from_tuples([('Betti curve', 'Entropy'), ('Betti curve', 'Variance')])
    final_metrics[('Betti number', 'L1 distance')] = distance_metrics_df['L1 Distance']
    results['combined_metrics'] = final_metrics

    if save_dir is not None and file_suffix is not None:
        for key, result in results.items():
            if isinstance(result, pd.DataFrame):
                result.to_csv(save_dir / f"{key}_{file_suffix}.csv")
            else:
                continue

    return results



def benchmark_geometry(adata, keys, 
                       eval_metrics=['pseudotime', 'cell_distance_corr', 'local_distal_corr', 'trustworthiness', 'state_distance_corr', 'state_dispersion_corr', 'state_batch_distance_ratio'],
                       dist_metric='cosine', 
                       groundtruth_key = 'PCA_no_noise', 
                       state_key = 'cluster',
                       batch_key = 'batch',
                       groundtruth_dispersion = None,
                       ground_truth_dispersion_key = 'wt_noise',
                       corr_types = ['pearsonr', 'spearmanr', 'kendalltau'], 
                       trustworthiness_n_neighbors = np.arange(10, 101, 10),
                       dispersion_metric='var',
                       return_type='dataframe',
                       local_percentile=0.1,
                       distal_percentile=0.9,
                       start_point=0,
                       end_point=None,
                       pseudotime_k = 30,
                       truetime_key = 'time',
                       verbose=True,
                       save_dir=None, 
                       file_suffix=None):
    """
    Benchmark the geometric properties of different embeddings.

    Args:
        adata : anndata.AnnData
            The AnnData object containing cell embeddings.
        keys : list
            List of embeddings (keys in `adata.obsm`) to evaluate.
        eval_metrics : list, optional
            Metrics to compute, such as 'pseudotime', 'cell_distance_corr', etc. Default includes multiple metrics.
        dist_metric : str, optional
            Distance metric for computing cell distances. Default is 'cosine'.
        groundtruth_key : str, optional
            Key in `adata.obsm` containing the ground truth embedding. Default is 'PCA_no_noise'.
        state_key : str, optional
            Key in `adata.obs` representing cell states or clusters.
        batch_key : str, optional
            Key in `adata.obs` representing batch information.
        groundtruth_dispersion : dict, optional
            Precomputed dispersion values for ground truth, if available.
        ground_truth_dispersion_key : str, optional
            Key used when computing dispersion correlations. Default is 'wt_noise'.
        corr_types : list, optional
            List of correlation methods to compute. Default includes 'pearsonr', 'spearmanr', and 'kendalltau'.
        trustworthiness_n_neighbors : np.ndarray, optional
            Range of neighborhood sizes for trustworthiness computation. Default is `np.arange(10, 101, 10)`.
        dispersion_metric : str, optional
            Metric to compute dispersion, e.g., 'var' (variance). Default is 'var'.
        return_type : str, optional
            If 'dataframe', returns summary statistics; if 'full', returns additional details. Default is 'dataframe'.
        local_percentile : float, optional
            Percentile threshold for local distance correlations. Default is 0.1.
        distal_percentile : float, optional
            Percentile threshold for distal distance correlations. Default is 0.9.
        start_point : int, optional
            Index of the starting cell for pseudotime computation. Must be specified.
        end_point : int, optional
            Index of the ending cell for pseudotime computation. Must be specified.
        pseudotime_k : int, optional
            Number of neighbors used in k-NN graph for pseudotime computation. Default is 30.
        truetime_key : str, optional
            Key in `adata.obs` representing ground truth time. Default is 'time'.
        verbose : bool, optional
            Whether to enable logging. Default is True.
        save_dir : str, optional
            Directory to save benchmarking results. If None, results are not saved.
        file_suffix : str, optional
            Suffix for saved filenames.

    Returns:
        pd.DataFrame or tuple
            If `return_type='dataframe'`, returns a DataFrame summarizing benchmark results.
            If `return_type='full'`, returns both the DataFrame and a detailed results dictionary.
    """
    import pandas as pd
    from .geometry import pairwise_distance, local_vs_distal_corr, compute_trustworthiness, compute_centroid_distance, compute_state_batch_distance_ratio, compute_dispersion_across_states
    results_df = {}
    results_full = {}
    if verbose:
        logger.setLevel(logging.INFO)


    # Pseudotime correlation
    if 'pseudotime' in eval_metrics:
        from .path_analysis import shortest_path_on_knn_graph, compute_pseudotime_from_shortest_path
        logger.info("Computing pseudotime correlation")
        if start_point is None or end_point is None:
            raise ValueError("start_point and end_point must be specified for pseudotime computation.")
        if truetime_key not in adata.obs:
            raise ValueError(f"Groundtruth time key '{truetime_key}' not found in adata.obs.")
        # Compute pseudotime for each method post integration
        time_dict={}
        path_dict = {}
        for basis in keys:
            logger.info(f"Computing pseudotime for {basis}")
            if basis not in adata.obsm:
                continue
            
            pseudotime_key = f"{basis}_pseudotime"
            try:
                path, _ = shortest_path_on_knn_graph(adata, emb_key=basis, k=pseudotime_k, point_a=start_point, point_b=end_point, use_faiss=True)
                time_dict[basis] = compute_pseudotime_from_shortest_path(adata, path=path, basis=basis, pseudotime_key=pseudotime_key)
                path_dict[basis] = path
            except:
                logger.info(f"Failed to compute shortest path for {basis}")
                continue

        time_dict[truetime_key] = adata.obs[truetime_key]
        pseudotime_result = compute_correlation(time_dict, corr_types=corr_types, groundtruth_key=truetime_key)
        pseudotime_result.columns = [f'{col}(pt)' for col in pseudotime_result.columns]
        results_df['Pseudotime'] = pseudotime_result.drop(truetime_key, inplace=False)
        results_full['Pseudotime'] = {
            'path': path_dict,
            'pseudotime': time_dict,
            'correlation': pseudotime_result
        }

    

    # Global distance correlation
    if 'cell_distance_corr' in eval_metrics:
        logger.info("Computing cell distance correlation")
        distance_result = pairwise_distance(adata, keys = keys, metric=dist_metric)            
        corr_result = compute_correlation(distance_result, corr_types=corr_types, groundtruth_key=groundtruth_key)
        results_df['cell_distance_corr'] = corr_result
        results_df['cell_distance_corr'].columns = [f'{col}(cd)' for col in corr_result.columns]
        results_full['cell_distance_corr'] = {
            'distance': distance_result,
            'correlation': corr_result
        }

    # Local vs distal correlation
    if 'local_distal_corr' in eval_metrics:
        logger.info("Computing local vs distal correlation")
        local_cor = {}
        distal_cor = {}
        corr_method = 'spearmanr' # Default to spearmanr
        for key in keys:
            local_cor[key], distal_cor[key] = local_vs_distal_corr(adata.obsm[groundtruth_key], adata.obsm[key], method=corr_method, local_percentile=local_percentile, distal_percentile=distal_percentile)

        local_cor_df = pd.DataFrame(local_cor, index = [f'Local {corr_method}']).T
        distal_cor_df = pd.DataFrame(distal_cor, index = [f'Distal {corr_method}']).T
        local_distal_corr_df = pd.concat([local_cor_df, distal_cor_df], axis=1)
        results_df['local_distal_corr'] = local_distal_corr_df
        results_full['local_distal_corr'] = local_distal_corr_df

    # Trustworthiness
    if 'trustworthiness' in eval_metrics:
        logger.info("Computing trustworthiness")
        trustworthiness_scores, trustworthiness_stats = compute_trustworthiness(adata, embedding_keys = keys, groundtruth=groundtruth_key, metric=dist_metric, n_neighbors=trustworthiness_n_neighbors)
        results_df['trustworthiness'] = trustworthiness_stats
        results_full['trustworthiness'] = {
            'scores': trustworthiness_scores,
            'stats': trustworthiness_stats
        }
        
    # Cluster centroid distances correlation
    if 'state_distance_corr' in eval_metrics:
        logger.info("Computing cluster centroid distances correlation")
        cluster_centroid_distances = {}
        for key in keys:
            cluster_centroid_distances[key] = compute_centroid_distance(adata, key, state_key)
            
        corr_dist_result = compute_correlation(cluster_centroid_distances, corr_types=corr_types, groundtruth_key=groundtruth_key)
        corr_dist_result.columns = [f'{col}(sd)' for col in corr_dist_result.columns]
        results_df['state_distance_corr'] = corr_dist_result
        results_full['state_distance_corr'] = {
            'distance': cluster_centroid_distances,
            'correlation': corr_dist_result
        }

    if 'state_dispersion_corr' in eval_metrics:
        logger.info("Computing state dispersion correlation")
        state_dispersion = {}
        for key in keys:
            state_dispersion[key] = compute_dispersion_across_states(adata, basis = key, state_key=state_key, dispersion_metric=dispersion_metric)
            
        if groundtruth_dispersion is not None:
            state_dispersion['Groundtruth'] = groundtruth_dispersion

        corr_dispersion_result = compute_correlation(state_dispersion, corr_types=corr_types, groundtruth_key='Groundtruth' if groundtruth_dispersion is not None else ground_truth_dispersion_key)
        corr_dispersion_result.columns = [f'{col}(sv)' for col in corr_dispersion_result.columns]
        results_df['state_dispersion_corr'] = corr_dispersion_result
        results_full['state_dispersion_corr'] = {
            'dispersion': state_dispersion,
            'correlation': corr_dispersion_result
        }

    # Batch-to-State Distance Ratio for all latent embeddings
    if 'state_batch_distance_ratio' in eval_metrics:
        logger.info("Computing state-batch distance ratio")
        state_batch_distance_ratios = {}
        for key in keys:
            state_batch_distance_ratios[key] = compute_state_batch_distance_ratio(adata, basis=key, batch_key=batch_key, state_key=state_key, metric='cosine')

        state_batch_distance_ratio_df = pd.DataFrame(state_batch_distance_ratios, index=[f'State-Batch Distance Ratio']).T
        state_batch_distance_ratio_df = np.log10(state_batch_distance_ratio_df)
        state_batch_distance_ratio_df.columns = [f'State-Batch Distance Ratio (log10)']
        # Set groundtruth to Nan
        if groundtruth_key in state_batch_distance_ratio_df.index:
            state_batch_distance_ratio_df.loc[groundtruth_key] = np.nan
        results_df['state_batch_distance_ratio'] = state_batch_distance_ratio_df
        results_full['state_batch_distance_ratio'] = state_batch_distance_ratio_df
    
    if save_dir is not None and file_suffix is not None:
        for key, result in results_df.items():
            result.to_csv(save_dir / f"{key}_{file_suffix}.csv")
    
    combined_results_df = pd.concat(results_df, axis=1)

    colname_mapping = {
        'cell_distance_corr': 'Cell distance correlation',
        'local_distal_corr': 'Cell distance correlation',
        'trustworthiness': 'Trustworthiness',
        'state_distance_corr': 'State distance',
        'state_dispersion_corr': 'Dispersion',
        'state_batch_distance_ratio': 'State/batch',
        'Average Trustworthiness': 'Mean',
        'Trustworthiness Decay (100N)': 'Decay',
        'State-Batch Distance Ratio (log10)': 'Distance ratio (log10)',
    }
    combined_results_df = combined_results_df.rename(columns=colname_mapping)

    if return_type == 'full':
        return combined_results_df, results_full
    else :
        return combined_results_df
    

def simplify_geometry_benchmark_table(df):
    # Simplify the dataframe by computing average for each metric
    if "Cell distance correlation" in df.columns.get_level_values(0):
        df[("Geometric metrics", "Cell distance correlation")] = df["Cell distance correlation"][
            ["pearsonr(cd)", "spearmanr(cd)", "kendalltau(cd)"]
        ].mean(axis=1)

        df.drop(
            columns=[
                ("Cell distance correlation", "pearsonr(cd)"),
                ("Cell distance correlation", "spearmanr(cd)"),
                ("Cell distance correlation", "kendalltau(cd)"),
                ("Cell distance correlation", "Local spearmanr"),
                ("Cell distance correlation", "Distal spearmanr")
            ],
            inplace=True
        )

    if "Trustworthiness" in df.columns.get_level_values(0):
        df[("Geometric metrics", "Trustworthiness")] = df["Trustworthiness"][["Mean"]]
        df.drop(
            columns=[
                ("Trustworthiness", "Decay"),
                ("Trustworthiness", "Mean")
            ],
            inplace=True
        )

    if "State distance" in df.columns.get_level_values(0):
        df[("Geometric metrics", "State distance correlation")] = df["State distance"][
            ["pearsonr(sd)", "spearmanr(sd)", "kendalltau(sd)"]
        ].mean(axis=1)
        df.drop(
            columns=[
                ("State distance", "pearsonr(sd)"),
                ("State distance", "spearmanr(sd)"),
                ("State distance", "kendalltau(sd)"),
            ],
            inplace=True
        )

    if "Dispersion" in df.columns.get_level_values(0):
        df[("Geometric metrics", "State dispersion correlation")] = df["Dispersion"][
            ["pearsonr(sv)", "spearmanr(sv)", "kendalltau(sv)"]
        ].mean(axis=1)
        df.drop(
            columns=[
                ("Dispersion", "pearsonr(sv)"),
                ("Dispersion", "spearmanr(sv)"),
                ("Dispersion", "kendalltau(sv)"),
            ],
            inplace=True
        )
    return df


# Convert benchmark table to scores
def benchmark_stats_to_score(df, fillna=None, min_max_scale=True, one_minus=False, aggregate_score=False, aggregate_score_name1 = 'Aggregate score', aggregate_score_name2='', name_exact=False, rank=False, rank_col=None):
    import pandas as pd
    from sklearn.preprocessing import MinMaxScaler

    df = df.copy()
    if fillna is not None:
        df = df.fillna(fillna)

    if min_max_scale:
        df = pd.DataFrame(
            MinMaxScaler().fit_transform(df),
            columns=df.columns,
            index=df.index,
        )
        if name_exact:
            df.columns = pd.MultiIndex.from_tuples([(col[0], f"{col[1]}(min-max)") for col in df.columns])

    if one_minus:
        df = 1 - df
        if name_exact:
            df.columns = pd.MultiIndex.from_tuples([(col[0], f"1-{col[1]}") for col in df.columns])

    if aggregate_score:
        aggregate_df = pd.DataFrame(
            df.mean(axis=1),
            columns=pd.MultiIndex.from_tuples([(aggregate_score_name1, aggregate_score_name2)]),
        )
        df = pd.concat([df, aggregate_df], axis=1)

    if rank:
        if rank_col is None:
            raise ValueError("rank_col must be specified when rank=True.")
        # Reorder the rows based on the aggregate score
        df = df.sort_values(by=rank_col, ascending=False)

    return df




# Recompute nmi and ari using the approach described in paper, with resolution range from 0.1 to 1.0 step 0.1
def compute_nmi_ari(adata, emb_key, label_key, resolution_range = np.arange(0.1, 1.1, 0.1), n_neighbors=30, metric='euclidean', verbose=True):
    import scanpy as sc
    import scib
    if verbose:
        logger.setLevel(logging.INFO)

    sc.pp.neighbors(adata, n_neighbors=n_neighbors, use_rep=emb_key, metric=metric)
    cluster_key = f'leiden_{emb_key}'
    nmi_vals = []
    ari_vals = []
    for resolution in resolution_range:
        logger.info(f"Computing NMI and ARI for resolution {resolution}")
        sc.tl.leiden(adata, resolution=resolution, key_added=cluster_key)
        nmi_vals.append(scib.metrics.nmi(adata, cluster_key, label_key))
        ari_vals.append(scib.metrics.ari(adata, cluster_key, label_key))
    
    return nmi_vals, ari_vals



def benchmark_nmi_ari(adata, emb_keys, label_key='cell_type', resolution_range = np.arange(0.1, 1.1, 0.1), n_neighbors=30, metric='euclidean', verbose=True):
    import pandas as pd
    if verbose:
        logger.setLevel(logging.INFO)
    nmi_vals = {}
    ari_vals = {}
    for key in emb_keys:
        logger.info(f"Computing NMI and ARI for {key}")
        nmi_vals[key], ari_vals[key] = compute_nmi_ari(adata, key, label_key, resolution_range=resolution_range, n_neighbors=n_neighbors, metric=metric, verbose=verbose)
    
    nmi_df = pd.DataFrame(nmi_vals, index=resolution_range)
    ari_df = pd.DataFrame(ari_vals, index=resolution_range)

    return nmi_df, ari_df





