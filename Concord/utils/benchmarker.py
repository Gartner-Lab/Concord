import time
import json
from pathlib import Path
from copy import deepcopy
from .timer import Timer
from .anndata_utils import save_obsm_to_hdf5
from .. import logger

def count_total_runs(param_grid):
    total_runs = 0
    for key, values in param_grid.items():
        if isinstance(values, list):
            total_runs += len(values)
    return total_runs

def run_hyperparameter_tests(adata, base_params, param_grid, save_dir):
    total_runs = count_total_runs(param_grid)
    logger.info(f"Total number of runs: {total_runs}")
    for param_name, values in param_grid.items():
        for value in values:
            # Create a deep copy of base_params to avoid modifying the original dictionary
            params_copy = deepcopy(base_params)
            
            with Timer() as timer:
                logger.info(f"Encoding adata with {param_name}={value}.")
                params_copy[param_name] = value

                # Initialize Concord model with updated parameters
                from ..concord import Concord
                ccd = Concord(adata=adata, **params_copy)

                # Define the output key and file suffix including param_name and value
                output_key = f"encoded_{param_name}_{value}"
                file_suffix = f"{param_name}_{value}_{time.strftime('%b%d-%H%M')}"

                # Encode adata and store the results in adata.obsm
                ccd.encode_adata(input_layer_key="X_log1p", output_key=output_key)

                # Save the parameter settings
                config_filename = Path(save_dir) / f"config_{file_suffix}.json"
                with open(config_filename, 'w') as f:
                    json.dump(ccd.config.to_dict(), f, indent=4)

            logger.info(f"Took {timer.interval/60:.2f} minutes to encode adata with {param_name}={value}, result saved to adata.obsm['{output_key}'], config saved to {config_filename}.")

    # Save the entire adata.obsm to file after all tests
    obsm_filename = Path(save_dir) / "final_obsm.h5"
    save_obsm_to_hdf5(adata, obsm_filename)

    return adata
