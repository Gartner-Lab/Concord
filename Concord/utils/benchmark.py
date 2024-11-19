
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


def benchmark_topology(diagrams, expected_betti_numbers=[1,0,0], n_bins=100, save_dir=None, file_suffix=None):
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
                       eval_metrics=['cell_distance_corr', 'local_distal_corr', 'trustworthiness', 'state_distance_corr', 'state_dispersion_corr', 'state_batch_distance_ratio'],
                       dist_metric='cosine', 
                       groundtruth_key = 'PCA_no_noise', 
                       state_key = 'cluster',
                       batch_key = 'batch',
                       groundtruth_dispersion = None,
                       corr_types = ['pearsonr', 'spearmanr', 'kendalltau'], 
                       trustworthiness_n_neighbors = np.arange(10, 101, 10),
                       dispersion_metric='var',
                       return_type='dataframe',
                       local_percentile=0.1,
                       distal_percentile=0.9,
                       verbose=True,
                       save_dir=None, 
                       file_suffix=None):
    import pandas as pd
    from .geometry import pairwise_distance, local_vs_distal_corr, compute_trustworthiness, compute_centroid_distance, compute_state_batch_distance_ratio, compute_dispersion_across_states
    results_df = {}
    results_full = {}
    if verbose:
        logger.setLevel(logging.INFO)

    # Global distance correlation
    if 'cell_distance_corr' in eval_metrics:
        logger.info("Computing cell distance correlation")
        distance_result = pairwise_distance(adata, keys = keys, metric=dist_metric)            
        corr_result = compute_correlation(distance_result, corr_types=corr_types, groundtruth_key=groundtruth_key)
        results_df['cell_distance_corr'] = corr_result
        results_df['cell_distance_corr'].columns = [f'cell_{col}' for col in corr_result.columns]
        results_full['cell_distance_corr'] = {
            'distance': distance_result,
            'correlation': corr_result
        }

    # Local vs distal correlation
    if 'local_distal_corr' in eval_metrics:
        logger.info("Computing local vs distal correlation")
        local_cor = {}
        distal_cor = {}
        corr_method = 'pearsonr'
        for key in keys:
            local_cor[key], distal_cor[key] = local_vs_distal_corr(adata.obsm[groundtruth_key], adata.obsm[key], method=corr_method, local_percentile=local_percentile, distal_percentile=distal_percentile)

        local_cor_df = pd.DataFrame(local_cor, index = [f'Local correlation']).T
        distal_cor_df = pd.DataFrame(distal_cor, index = [f'Distal correlation']).T
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
        corr_dist_result.columns = [f'State_distance_{col}' for col in corr_dist_result.columns]
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

        # Fix
        corr_dispersion_result = compute_correlation(state_dispersion, corr_types=corr_types, groundtruth_key='Groundtruth' if groundtruth_dispersion is not None else groundtruth_key)
        corr_dispersion_result.columns = [f'State_dispersion_{col}' for col in corr_dispersion_result.columns]
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
        'cell_pearsonr': 'Global',
        'cell_kendalltau': 'Kendall(c)',
        'Local correlation': 'Local',
        'Distal correlation': 'Distal',
        'Average Trustworthiness': 'Mean',
        'Trustworthiness Decay (100N)': 'Decay',
        'State_distance_pearsonr': 'Pearson(s)',
        'State_distance_kendalltau': 'Kendall(s)',
        'State_dispersion_pearsonr': 'Correlation',
        'State_dispersion_kendalltau': 'Kendall(s)',
        'State-Batch Distance Ratio (log10)': 'Distance ratio (log10)',
    }
    combined_results_df = combined_results_df.rename(columns=colname_mapping)

    if return_type == 'full':
        return combined_results_df, results_full
    else :
        return combined_results_df
    

# Convert benchmark table to scores
def benchmark_stats_to_score(df, min_max_scale=True, one_minus=False, aggregate_score=False, aggregate_score_name1 = 'Aggregate score', aggregate_score_name2='', name_exact=False, rank=False, rank_col=None):
    import pandas as pd
    from sklearn.preprocessing import MinMaxScaler

    df = df.copy()
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


