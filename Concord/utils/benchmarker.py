
import pandas as pd
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
