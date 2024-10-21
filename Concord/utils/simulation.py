import numpy as np
import pandas as pd
import anndata as ad

class Simulation:
    def __init__(self, n_cells=1000, n_genes=1000, 
                 n_batches=2, n_states=3, 
                 state_type='group', 
                 batch_type='batch_specific_features', 
                 state_distribution='normal',
                 state_strength=1.0, 
                 state_noise=0.1, 
                 batch_distribution='normal',
                 batch_strength=1.0, 
                 batch_noise=0.1, 
                 batch_cell_proportion=None,
                 batch_feature_frac=0.1,
                 seed=0):
        self.n_cells = n_cells
        self.n_genes = n_genes
        self.n_batches = n_batches
        self.n_states = n_states
        self.state_type = state_type
        self.state_distribution = state_distribution
        self.state_strength = state_strength
        self.state_noise = state_noise

        # Batch parameters, if multiple batches, allow list of values, if not list, use the same value for all batches
        self.batch_type = batch_type if isinstance(batch_type, list) else [batch_type] * n_batches
        self.batch_strength = batch_strength if isinstance(batch_strength, list) else [batch_strength] * n_batches
        self.batch_noise = batch_noise if isinstance(batch_noise, list) else [batch_noise] * n_batches
        if batch_cell_proportion is not None:
            self.batch_cell_proportion = batch_cell_proportion if isinstance(batch_cell_proportion, list) else [batch_cell_proportion] * n_batches
        else:
            self.batch_cell_proportion = [1 / self.n_batches] * n_batches

        self.batch_distribution = batch_distribution if isinstance(batch_distribution, list) else [batch_distribution] * n_batches
        self.batch_feature_frac = batch_feature_frac if isinstance(batch_feature_frac, list) else [batch_feature_frac] * n_batches

        self.seed = seed
        np.random.seed(seed)

    def simulate_data(self):
        import re
        from scipy import sparse as sp
        adata_state = self.simulate_state()
        batch_list = []
        for i in range(self.n_batches):
            batch_adata = self.simulate_batch(adata_state, batch_name=f"batch_{i+1}", effect_type=self.batch_type[i], 
                                              effect_strength=self.batch_strength[i], noise_std=self.batch_noise[i], 
                                              cell_proportion=self.batch_cell_proportion[i], batch_feature_frac=self.batch_feature_frac[i], seed=self.seed+i)
            batch_list.append(batch_adata)

        adata = ad.concat(batch_list, join='outer')
        adata.X = adata.X.toarray() if sp.issparse(adata.X) else adata.X
        adata.X = np.nan_to_num(adata.X, nan=0.0)

        gene_names = adata.var_names
        def natural_key(string_):
            return [int(s) if s.isdigit() else s for s in re.split(r'(\d+)', string_)]

        sorted_gene_names = sorted(gene_names, key=natural_key)
        adata = adata[:, sorted_gene_names]
        return adata

    def simulate_state(self):
        if self.state_type == 'group':
            return self.simulate_expression_groups(num_genes=self.n_genes, num_cells=self.n_cells, num_groups=self.n_states, 
                                                   distribution = self.state_distribution,
                                                   mean_expression=self.state_strength, noise_std=self.state_noise, 
                                                   p_gene_nonspecific=0, seed=self.seed)
        elif self.state_type == 'trajectory_gradual':
            return self.simulate_gradual_changes(num_genes=self.n_genes, num_cells=self.n_cells, direction='both', 
                                                 max_expression=self.state_strength, noise_level=self.state_noise, seed=self.seed)
        elif self.state_type == 'trajectory_dimension_shift':
            return self.simulate_dimension_shift(num_genes=self.n_genes, num_cells=self.n_cells, direction='increase', 
                                                 group_size=3, gap_size=5, mean_expression=10, noise_std=self.state_noise, seed=self.seed)
        else:
            raise ValueError(f"Unknown state_type '{self.state_type}'.")

    def simulate_batch(self, adata, batch_name='batch_1', effect_type='mean_shift', effect_strength=1.0, noise_std = 0.1, cell_proportion=0.3, batch_feature_frac=0.1, seed=42):
        from scipy import sparse as sp
        np.random.seed(seed)
        if not (0 < cell_proportion <= 1):
            raise ValueError("cell_proportion must be between 0 and 1.")
        n_cells = int(adata.n_obs * cell_proportion)
        cell_indices = np.random.choice(adata.n_obs, size=n_cells, replace=False)
        batch_adata = adata[cell_indices].copy()
        batch_adata.obs['batch'] = batch_name

        if effect_type == 'variance_inflation':
            scale_vector = 1 + np.random.normal(loc=0, scale=effect_strength, size=batch_adata.n_vars * batch_adata.n_obs).reshape(batch_adata.n_obs, batch_adata.n_vars)
            batch_adata.X = batch_adata.X.toarray() * scale_vector if sp.issparse(batch_adata.X) else batch_adata.X * scale_vector
        elif effect_type == 'uniform_dropout':
            dropout_prob = effect_strength
            dropout_mask = np.random.rand(batch_adata.n_obs, batch_adata.n_vars) < dropout_prob
            batch_adata.X = batch_adata.X.toarray() if sp.issparse(batch_adata.X) else batch_adata.X
            batch_adata.X[dropout_mask] = 0
        elif effect_type == 'value_dependent_dropout':
            batch_adata.X[batch_adata.X < 0] = 0
            batch_adata.X = Simulation.simulate_dropout(batch_adata.X, dropout_lambda=effect_strength, seed=seed)
        elif effect_type == "downsampling":
            batch_adata.X[batch_adata.X < 0] = 0
            batch_adata.X = Simulation.downsample_mtx_umi(mtx = batch_adata.X.astype(int), ratio=effect_strength, seed=seed)
        elif effect_type == 'scaling_factor':
            scaling_vector = effect_strength
            batch_adata.X = batch_adata.X @ sp.diags(scaling_vector) if sp.issparse(batch_adata.X) else batch_adata.X * scaling_vector
        elif effect_type == 'batch_specific_expression':
            num_genes_batch_specific = int(batch_feature_frac * batch_adata.n_vars)
            batch_specific_genes = np.random.choice(batch_adata.n_vars, num_genes_batch_specific, replace=False)
            batch_adata.X[:, batch_specific_genes] += np.random.normal(loc=effect_strength, scale=noise_std, size=(batch_adata.n_obs, num_genes_batch_specific))
        elif effect_type == 'batch_specific_features':
            num_new_features = int(batch_feature_frac * batch_adata.n_vars)
            batch_expr = np.random.normal(loc=effect_strength, scale=noise_std, size=(batch_adata.n_obs, num_new_features))
            new_vars = pd.DataFrame(index=[f"{batch_name}_Gene_{i+1}" for i in range(num_new_features)])
            batch_adata = ad.AnnData(X=np.hstack([batch_adata.X, batch_expr]), obs=batch_adata.obs, var=pd.concat([batch_adata.var, new_vars]))
        else:
            raise ValueError(f"Unknown effect_type '{effect_type}'.")

        return batch_adata

    def simulate_expression_groups(self, num_genes=6, num_cells=12, num_groups=2, distribution="normal", mean_expression=10, noise_std=1.0, p_gene_nonspecific=0.1, group_key='group', permute=False, seed=42):
        np.random.seed(seed)
        genes_per_group = num_genes // num_groups
        cells_per_group = num_cells // num_groups
        num_nonspecific_genes = int(num_genes * p_gene_nonspecific)
        expression_matrix = np.zeros((num_cells, num_genes))
        cell_groups = []

        for group in range(num_groups):
            cell_start, cell_end = group * cells_per_group, (group + 1) * cells_per_group if group < num_groups - 1 else num_cells
            gene_start, gene_end = group * genes_per_group, (group + 1) * genes_per_group if group < num_groups - 1 else num_genes
            if distribution == "normal":
                expression_matrix[cell_start:cell_end, gene_start:gene_end] = mean_expression + np.random.normal(0, noise_std, (cell_end - cell_start, gene_end - gene_start))
            elif distribution == 'poisson':
                expression_matrix[cell_start:cell_end, gene_start:gene_end] = np.random.poisson(mean_expression, (cell_end - cell_start, gene_end - gene_start))
            elif distribution == 'negative_binomial': # TODO - check if this is correct
                expression_matrix[cell_start:cell_end, gene_start:gene_end] = np.random.negative_binomial(1, 1 / (1 + mean_expression), (cell_end - cell_start, gene_end - gene_start))
            elif distribution == 'lognormal':
                expression_matrix[cell_start:cell_end, gene_start:gene_end] = np.random.lognormal(mean_expression, noise_std, (cell_end - cell_start, gene_end - gene_start))
            else:
                raise ValueError(f"Unknown distribution '{distribution}'.")  

            other_genes = np.setdiff1d(np.arange(num_genes), np.arange(gene_start, gene_end))
            nonspecific_gene_indices = np.random.choice(other_genes, num_nonspecific_genes, replace=False)
            expression_matrix[cell_start:cell_end, nonspecific_gene_indices] = mean_expression + np.random.normal(0, noise_std, (cell_end - cell_start, num_nonspecific_genes))
            cell_groups.extend([f"{group_key}_{group+1}"] * (cell_end - cell_start))

        obs = pd.DataFrame({f"{group_key}": cell_groups}, index=[f"Cell_{i+1}" for i in range(num_cells)])
        var = pd.DataFrame(index=[f"Gene_{i+1}" for i in range(num_genes)])
        adata = ad.AnnData(X=expression_matrix, obs=obs, var=var)
        if permute:
            adata = adata[np.random.permutation(adata.obs_names), :]
        return adata

    def simulate_gradual_changes(self, num_genes=10, num_cells=10, direction='both', time_key='time', max_expression=1, noise_level=0.1):
        expression_matrix = np.zeros((num_cells, num_genes))
        for i in range(num_genes):
            noise = np.random.normal(0, noise_level, num_cells)
            if direction == 'increase':
                expression_matrix[:, i] = np.linspace(0, max_expression, num_cells) + noise
            elif direction == 'decrease':
                expression_matrix[:, i] = np.linspace(max_expression, 0, num_cells) + noise
            elif direction == 'both':
                expression_matrix[:, i] = np.linspace(0, max_expression, num_cells) + noise if i >= num_genes // 2 else np.linspace(max_expression, 0, num_cells) + noise
        obs = pd.DataFrame(index=[f"Cell_{i+1}" for i in range(num_cells)])
        obs[time_key] = np.linspace(0, 1, num_cells)
        var = pd.DataFrame(index=[f"Gene_{i+1}" for i in range(num_genes)])
        return ad.AnnData(X=expression_matrix, obs=obs, var=var)

    def simulate_dimension_shift(self, num_genes=10, num_cells=100, direction='increase', group_size=3, gap_size=5, mean_expression=10, noise_std=1.0, seed=42):
        np.random.seed(seed)
        assert group_size <= num_genes
        assert gap_size <= num_cells
        step_size = num_cells // gap_size
        expression_matrix = np.zeros((num_cells, num_genes))
        pseudotime = np.arange(num_cells)
        ngroups = num_genes // group_size + 1
        mid = num_genes // 2 // group_size

        for i in range(ngroups):
            gene_idx = np.arange(min(i * group_size, num_genes), min((i + 1) * group_size, num_genes))
            start, end = min(i * gap_size, num_cells), min(start + step_size, num_cells)
            if direction == 'increase':
                expression_matrix[start:end, gene_idx] = np.linspace(0, mean_expression, end - start)[:, None]
                expression_matrix[end:, gene_idx] = mean_expression
            elif direction == 'decrease':
                expression_matrix[:start, gene_idx] = mean_expression
                expression_matrix[start:end, gene_idx] = np.linspace(mean_expression, 0, end - start)[:, None]
            elif direction == 'both':
                if i >= mid:
                    expression_matrix[start:end, gene_idx] = np.linspace(0, mean_expression, end - start)[:, None]
                    expression_matrix[end:, gene_idx] = mean_expression
                else:
                    expression_matrix[:start, gene_idx] = mean_expression
                    expression_matrix[start:end, gene_idx] = np.linspace(mean_expression, 0, end - start)[:, None]
            expression_matrix[:, gene_idx] += np.random.normal(0, noise_std, (num_cells, len(gene_idx)))

        obs = pd.DataFrame({'time': pseudotime}, index=[f"Cell_{i+1}" for i in range(num_cells)])
        var = pd.DataFrame(index=[f"Gene_{i+1}" for i in range(num_genes)])
        return ad.AnnData(X=expression_matrix, obs=obs, var=var)
    
    @staticmethod
    def downsample_mtx_umi(mtx, ratio=0.1, seed=1):
        """
        Simulates downsampling of a gene expression matrix (UMI counts) by a given ratio.

        Parameters:
            mtx (numpy.ndarray): The input matrix where rows represent genes and columns represent cells.
            ratio (float): The downsampling ratio (default 0.1).
            seed (int): Random seed for reproducibility (default 1).

        Returns:
            numpy.ndarray: The downsampled matrix.
        """
        np.random.seed(seed)

        # Initialize the downsampled matrix
        downsampled_mtx = np.zeros_like(mtx, dtype=int)

        # Loop over each gene (row in the matrix)
        for i, x in enumerate(mtx):
            total_reads = int(np.sum(x))
            n = int(np.floor(total_reads * ratio))  # Number of reads to sample

            if n == 0 or total_reads == 0:
                continue  # Skip genes with no reads or no reads after downsampling

            # Sample n read indices without replacement
            ds_reads = np.sort(np.random.choice(np.arange(1, total_reads + 1), size=n, replace=False))

            # Create read breaks using cumulative sum of original counts
            read_breaks = np.concatenate(([0], np.cumsum(x)))

            # Use histogram to count the number of reads per cell after downsampling
            counts, _ = np.histogram(ds_reads, bins=read_breaks)
            downsampled_mtx[i, :] = counts

        return downsampled_mtx
    

    @staticmethod
    def simulate_dropout(mtx, dropout_lambda=1.0, seed=None):
        """
        Simulates dropout in UMI counts based on the specified dropout lambda.

        Parameters:
            mtx (numpy.ndarray): The actual UMI counts matrix (genes x cells).
            dropout_lambda (float): The lambda parameter controlling the dropout probability.
            seed (int, optional): Seed for the random number generator for reproducibility.

        Returns:
            numpy.ndarray: The UMI counts matrix after applying dropout.
        """
        # Set the random seed if provided
        if seed is not None:
            np.random.seed(seed)

        # Check if any negative values are present in the matrix
        if np.any(mtx < 0):
            raise ValueError("Input matrix contains negative values.")
        
        # Compute dropout probability matrix
        # Equivalent to exp(-dropout_lambda * mtx^2)
        dropout_prob_mtx = np.exp(-dropout_lambda * mtx ** 2)
        
        # Generate dropout indicators using a binomial distribution
        # Each element is 1 with probability dropout_prob_mtx, otherwise 0
        # We need to flatten the matrix for random.binomial and then reshape back
        flat_prob = dropout_prob_mtx.flatten()
        dropout_indicator_flat = np.random.binomial(n=1, p=flat_prob)
        dropout_indicator = dropout_indicator_flat.reshape(mtx.shape)

        print(f"Percent 0s: {np.mean(dropout_indicator):.2f}")
        # Apply dropout to the UMI counts
        final_umi_mtx = mtx * (1 - dropout_indicator)
        
        return final_umi_mtx

