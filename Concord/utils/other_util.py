import logging
import random
from pathlib import Path
import numpy as np
import torch
import pandas as pd
from . import io


def set_seed(seed):
    """set random seed."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    # if n_gpu > 0:
    #     torch.cuda.manual_seed_all(seed)




def add_file_handler(logger: logging.Logger, log_file_path: Path):
    """
    Add a file handler to the logger.
    """
    h = logging.FileHandler(log_file_path)

    # format showing time, name, function, and message
    formatter = logging.Formatter(
        "%(asctime)s-%(name)s-%(levelname)s-%(funcName)s: %(message)s",
        datefmt="%H:%M:%S",
    )
    h.setFormatter(formatter)
    h.setLevel(logger.level)
    logger.addHandler(h)



def update_wandb_params(params, project_name=None, reinit=True, start_method="thread"):
    import wandb
    if reinit:
        run = wandb.init(
            config=params,
            project=project_name,
            reinit=True,
            settings=wandb.Settings(start_method=start_method),
        )
    else:
        run = wandb.run

    config = wandb.config
    config.update(params)
    return config, run


def wandb_log_plot(plt, plot_name, plot_path=None):
    """
    Logs a confusion matrix plot to W&B. If plot_path is provided, saves the plot to the specified path before logging.
    Otherwise, logs the plot directly from an in-memory buffer.

    Args:
        plt (matplotlib.pyplot): The Matplotlib pyplot object with the confusion matrix plot.
        plot_path (str, optional): The file path to save the plot. If None, the plot is logged from an in-memory buffer.
    """
    import wandb
    if plot_path is not None:
        plt.savefig(str(plot_path))
        wandb.log({plot_name: wandb.Image(str(plot_path))})
    else:
        buf = io.BytesIO()
        plt.savefig(buf, format='png')
        buf.seek(0)
        wandb.log({plot_name: wandb.Image(buf)})


def log_scib_results(config_dicts, eval_results, save_dir, file_name):
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)  # Ensure save_dir is created if it does not exist

    config_df = pd.DataFrame(config_dicts)
    eval_df = pd.DataFrame(eval_results)

    with pd.ExcelWriter(save_dir / file_name) as writer:
        config_df.to_excel(writer, sheet_name='Hyperparameters', index=False)
        eval_df.to_excel(writer, sheet_name='Results', index=False)

    return config_df, eval_df




def natural_key(string_):
    import re
    return [int(s) if s.isdigit() else s for s in re.split(r'(\d+)', string_)]


def sort_string_list(string_list):
    return sorted(string_list, key=natural_key)

