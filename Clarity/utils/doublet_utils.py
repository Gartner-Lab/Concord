import numpy as np
import torch
import matplotlib.pyplot as plt
import anndata as ad

def generate_synthetic_doublets(adata_true_singlet, doublet_synth_ratio, seed, batch_key, droplet_type_key, mean=0.5, var=0.25, clip_range=(0.2, 0.8), plot_histogram=True):
    """
    Generate synthetic doublets from singlet data in an AnnData object within each batch.

    Parameters:
    - adata_true_singlet: AnnData object containing the true singlet data
    - doublet_synth_ratio: float, the ratio of synthetic doublets to true singlets
    - seed: int, random seed for reproducibility
    - batch_key: str, the key in .obs indicating batch information
    - droplet_type_key: str, the key in .obs indicating droplet type
    - mean: float, mean of the normal distribution for generating fractions (default: 0.5)
    - var: float, variance of the normal distribution for generating fractions (default: 0.25)
    - clip_range: tuple, range to clip the generated fractions (default: (0.2, 0.8))
    - plot_histogram: bool, whether to plot the histogram of synthetic doublet fractions

    Returns:
    - adata_synthetic_doublets: AnnData object containing the synthetic doublets
    """

    np.random.seed(seed)
    all_synthetic_doublets_expr = []
    all_batches = []
    batches = adata_true_singlet.obs[batch_key].unique()

    total_num_doublets = sum(int(len(adata_true_singlet[adata_true_singlet.obs[batch_key] == batch]) * doublet_synth_ratio) for batch in batches)

    # Generate random fractions
    fractions = np.random.normal(mean, var, size=(total_num_doublets, 2))
    fractions = np.clip(fractions, *clip_range)
    fractions = fractions / fractions.sum(axis=1, keepdims=True)

    if plot_histogram:
        plt.figure(figsize=(4, 3))
        plt.hist(fractions[:, 0], bins=30, alpha=0.5, label='Fraction 1')
        plt.hist(fractions[:, 1], bins=30, alpha=0.5, label='Fraction 2')
        plt.xlabel('Fraction')
        plt.ylabel('Count')
        plt.title('Histogram of Synthetic Doublet Fractions')
        plt.legend()
        plt.show()

    fraction_idx = 0

    for batch in batches:
        batch_data = adata_true_singlet[adata_true_singlet.obs[batch_key] == batch]
        num_doublets = int(len(batch_data) * doublet_synth_ratio)
        singlet_expr = batch_data.X

        # Generate random indices
        indices = np.random.choice(singlet_expr.shape[0], size=(num_doublets, 2), replace=True)

        # Filter out indices where the two values are the same
        filtered_indices = indices[indices[:, 0] != indices[:, 1]]

        # Extract the corresponding fractions
        batch_fractions = fractions[fraction_idx:fraction_idx + len(filtered_indices)]
        fraction_idx += len(filtered_indices)

        # Extract singlet expressions using the filtered indices
        expr1 = singlet_expr[filtered_indices[:, 0]].todense()
        expr2 = singlet_expr[filtered_indices[:, 1]].todense()

        # Convert numpy arrays to torch tensors
        batch_fractions_tensor = torch.tensor(batch_fractions, dtype=torch.float32)
        expr1_tensor = torch.tensor(expr1, dtype=torch.float32)
        expr2_tensor = torch.tensor(expr2, dtype=torch.float32)

        # Generate synthetic doublets using matrix multiplication
        synthetic_doublets_expr = batch_fractions_tensor[:, 0].unsqueeze(1) * expr1_tensor + batch_fractions_tensor[:, 1].unsqueeze(1) * expr2_tensor

        all_synthetic_doublets_expr.append(synthetic_doublets_expr)
        all_batches.extend([batch] * synthetic_doublets_expr.shape[0])

    all_synthetic_doublets_expr = torch.cat(all_synthetic_doublets_expr).numpy()

    # Create a new AnnData object for synthetic doublets
    adata_synthetic_doublets = ad.AnnData(X=all_synthetic_doublets_expr)

    # Create new columns in .obs for the synthetic doublets
    adata_synthetic_doublets.obs[droplet_type_key] = ['synthetic_doublet'] * adata_synthetic_doublets.shape[0]
    adata_synthetic_doublets.obs[batch_key] = all_batches
    adata_synthetic_doublets.var = adata_true_singlet.var.copy()

    return adata_synthetic_doublets



def split_combined_adata(adata_combined, droplet_col, train_frac=0.9, seed=42, logger=None,
                         singlet_key = 'singlet',
                         synthetic_doublet_key = 'synthetic_doublet',
                         multiplet_key = 'multiplet'
                         ):
    np.random.seed(seed)

    # Identify indices for each type
    singlet_indices = np.where(adata_combined.obs[droplet_col] == singlet_key)[0]
    synthetic_doublet_indices = np.where(adata_combined.obs[droplet_col] == synthetic_doublet_key)[0]
    multiplet_indices = np.where(adata_combined.obs[droplet_col] == multiplet_key)[0] if multiplet_key is not None else np.array([])

    # Shuffle indices
    np.random.shuffle(singlet_indices)
    np.random.shuffle(synthetic_doublet_indices)

    # Split indices
    num_singlet_train = int(train_frac * len(singlet_indices))
    num_synthetic_doublet_train = int(train_frac * len(synthetic_doublet_indices))

    train_indices = np.concatenate([
        singlet_indices[:num_singlet_train],
        synthetic_doublet_indices[:num_synthetic_doublet_train],
    ])

    val_indices = np.concatenate([
        singlet_indices[num_singlet_train:],
        synthetic_doublet_indices[num_synthetic_doublet_train:],
        multiplet_indices
    ])
    if logger is not None:
        logger.info(f"num_singlet_train:{num_singlet_train} | num_synthetic_doublet_train:{num_synthetic_doublet_train}")
        logger.info(f"num_singlet_val:{len(singlet_indices)-num_singlet_train} | num_synthetic_doublet_val:{len(synthetic_doublet_indices)-num_synthetic_doublet_train} | num_true_multiplet:{len(multiplet_indices)}")
    return train_indices, val_indices
