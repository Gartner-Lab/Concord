import numpy as np
import pandas as pd
import anndata as ad
import logging

logger = logging.getLogger(__name__)


class Simulation:
    def __init__(self, n_cells=1000, n_genes=1000, 
                 n_batches=2, n_states=3, 
                 state_type='group', 
                 batch_type='batch_specific_features', 
                 state_distribution='normal',
                 state_level=1.0, 
                 state_dispersion=0.1, 
                 trajectory_trend='both',
                 trajectory_program_feature_size=3,
                 trajectory_program_transition_time=0.3,
                 trajectory_program_retention_time=0.2,
                 batch_distribution='normal',
                 batch_level=1.0, 
                 batch_dispersion=0.1, 
                 batch_cell_proportion=None,
                 batch_feature_frac=0.1,
                 non_neg = False,
                 to_int = False,
                 seed=0):
        self.n_cells = n_cells
        self.n_genes = n_genes
        self.n_batches = n_batches
        self.n_states = n_states
        self.state_type = state_type
        self.state_distribution = state_distribution
        self.state_level = state_level
        self.state_dispersion = state_dispersion

        self.trajectory_trend = trajectory_trend
        self.trajectory_program_feature_size = trajectory_program_feature_size
        self.trajectory_program_retention_time = trajectory_program_retention_time
        self.trajectory_program_transition_time = trajectory_program_transition_time

        # Batch parameters, if multiple batches, allow list of values, if not list, use the same value for all batches
        self.batch_type = batch_type if isinstance(batch_type, list) else [batch_type] * n_batches
        self.batch_level = batch_level if isinstance(batch_level, list) else [batch_level] * n_batches
        self.batch_dispersion = batch_dispersion if isinstance(batch_dispersion, list) else [batch_dispersion] * n_batches
        if batch_cell_proportion is not None:
            self.batch_cell_proportion = batch_cell_proportion if isinstance(batch_cell_proportion, list) else [batch_cell_proportion] * n_batches
        else:
            self.batch_cell_proportion = [1 / self.n_batches] * n_batches

        self.batch_distribution = batch_distribution if isinstance(batch_distribution, list) else [batch_distribution] * n_batches
        self.batch_feature_frac = batch_feature_frac if isinstance(batch_feature_frac, list) else [batch_feature_frac] * n_batches

        self.non_neg = non_neg
        self.to_int = to_int
        self.seed = seed
        np.random.seed(seed)

    def simulate_data(self):
        import re
        from scipy import sparse as sp
        adata_state = self.simulate_state()
        batch_list = []
        for i in range(self.n_batches):
            batch_adata = self.simulate_batch(adata_state, batch_name=f"batch_{i+1}", effect_type=self.batch_type[i], 
                                              distribution = self.batch_distribution[i],
                                              level=self.batch_level[i], dispersion=self.batch_dispersion[i], 
                                              cell_proportion=self.batch_cell_proportion[i], batch_feature_frac=self.batch_feature_frac[i], seed=self.seed+i)
            batch_list.append(batch_adata)

        adata = ad.concat(batch_list, join='outer')
        adata.X = adata.X.toarray() if sp.issparse(adata.X) else adata.X
        adata.X = np.nan_to_num(adata.X, nan=0.0)

        if self.non_neg:
            adata.X[adata.X < 0] = 0
        if self.to_int:
            adata.X = adata.X.astype(int)

        gene_names = adata.var_names
        def natural_key(string_):
            return [int(s) if s.isdigit() else s for s in re.split(r'(\d+)', string_)]

        sorted_gene_names = sorted(gene_names, key=natural_key)
        adata = adata[:, sorted_gene_names]
        return adata, adata_state

    def simulate_state(self):
        if self.state_type == 'group':
            logger.info(f"Simulating group state with {self.n_states} groups, distribution: {self.state_distribution} with mean expression {self.state_level} and dispersion {self.state_dispersion}.")
            return self.simulate_expression_groups(num_genes=self.n_genes, num_cells=self.n_cells, num_groups=self.n_states, 
                                                   distribution = self.state_distribution,
                                                   mean_expression=self.state_level, dispersion=self.state_dispersion, 
                                                   p_gene_nonspecific=0, seed=self.seed)
        elif self.state_type == 'trajectory_gradual':
            logger.info(f"Simulating gradual trajectory with expression trend={self.trajectory_trend}, max expression {self.state_level}, distribution {self.state_distribution} and dispersion level {self.state_dispersion}.")
            return self.simulate_gradual_changes(num_genes=self.n_genes, num_cells=self.n_cells,
                                                  trend=self.trajectory_trend, distribution=self.state_distribution, 
                                                 max_expression=self.state_level, dispersion=self.state_dispersion, seed=self.seed)
        elif self.state_type == 'trajectory_dimension_shift':
            logger.info(f"Simulating dimension shift trajectory with expression trend={self.trajectory_trend}, gene group size {self.trajectory_program_feature_size}, "
                        f"expression gap size {self.trajectory_program_retention_time}, distribution {self.state_distribution} with mean expression {self.state_level} and dispersion {self.state_dispersion}.")
            return self.simulate_dimension_shift(num_genes=self.n_genes, num_cells=self.n_cells, trend=self.trajectory_trend, 
                                                 group_size=self.trajectory_program_feature_size, program_retention_time=self.trajectory_program_retention_time, 
                                                 mean_expression=self.state_level, dispersion=self.state_dispersion, seed=self.seed)
        elif self.state_type == "trajectory":
            logger.info(f"Simulating trajectory with {self.n_states} states, distribution: {self.state_distribution} with mean expression {self.state_level} and dispersion {self.state_dispersion}.")
            return self.simulate_trajectory(num_genes=self.n_genes, num_cells=self.n_cells, 
                                            program_feature_size=self.trajectory_program_feature_size, 
                                            program_retention_time=self.trajectory_program_retention_time,
                                            program_transition_time=self.trajectory_program_transition_time, 
                                           mean_expression=self.state_level, dispersion=self.state_dispersion, seed=self.seed)
        else:
            raise ValueError(f"Unknown state_type '{self.state_type}'.")

    def simulate_batch(self, adata, batch_name='batch_1', effect_type='batch_specific_features', distribution='normal', 
                       level=1.0, dispersion = 0.1, cell_proportion=0.3, batch_feature_frac=0.1, seed=42):
        from scipy import sparse as sp
        np.random.seed(seed)
        if not (0 < cell_proportion <= 1):
            raise ValueError("cell_proportion must be between 0 and 1.")
        n_cells = int(adata.n_obs * cell_proportion)
        cell_indices = np.random.choice(adata.n_obs, size=n_cells, replace=False)
        batch_adata = adata[cell_indices].copy()
        batch_adata.obs['batch'] = batch_name

        if effect_type == 'variance_inflation':
            logger.info(f"Simulating variance inflation effect on {batch_name} by multiplying original data with a normal distributed scaling factor with level 1 and std {dispersion}.")
            scale_vector = 1 + np.random.normal(loc=0, scale=dispersion, size=batch_adata.n_vars * batch_adata.n_obs).reshape(batch_adata.n_obs, batch_adata.n_vars)
            batch_adata.X = batch_adata.X.toarray() * scale_vector if sp.issparse(batch_adata.X) else batch_adata.X * scale_vector
        elif effect_type == 'uniform_dropout':
            logger.info(f"Simulating uniform dropout effect on {batch_name} by setting a random fraction of the data to 0. Fraction is determined by level={level}.")
            dropout_prob = level
            dropout_mask = np.random.rand(batch_adata.n_obs, batch_adata.n_vars) < dropout_prob
            batch_adata.X = batch_adata.X.toarray() if sp.issparse(batch_adata.X) else batch_adata.X
            batch_adata.X[dropout_mask] = 0
        elif effect_type == 'value_dependent_dropout':
            logger.info(f"Simulating value-dependent dropout effect on {batch_name}. Dropout probability p=1-exp(-lambda*x^2) where x is the original value and lambda={level}.")
            batch_adata.X[batch_adata.X < 0] = 0
            batch_adata.X = Simulation.simulate_dropout(batch_adata.X, dropout_lambda=level, seed=seed)
        elif effect_type == "downsampling":
            logger.info(f"Simulating downsampling effect on {batch_name} by downsampling the data by a factor of {level}.")
            batch_adata.X[batch_adata.X < 0] = 0
            batch_adata.X = Simulation.downsample_mtx_umi(mtx = batch_adata.X.astype(int), ratio=level, seed=seed)
        elif effect_type == 'scaling_factor':
            logger.info(f"Simulating scaling factor effect on {batch_name} by multiplying original data with a scaling factor of {level}.")
            batch_adata.X = batch_adata.X @ sp.diags(level) if sp.issparse(batch_adata.X) else batch_adata.X * level
        elif effect_type == 'batch_specific_expression':
            logger.info(f"Simulating batch-specific expression effect on {batch_name} by adding a {distribution} distributed value with level {level} and dispersion {dispersion} to a random subset of batch-affected genes.")
            num_genes_batch_specific = int(batch_feature_frac * batch_adata.n_vars)
            batch_specific_genes = np.random.choice(batch_adata.n_vars, num_genes_batch_specific, replace=False)
            batch_adata.X[:, batch_specific_genes] += Simulation.simulate_distribution(distribution, level, dispersion, (batch_adata.n_obs, num_genes_batch_specific))
        elif effect_type == 'batch_specific_features':
            logger.info(f"Simulating batch-specific features effect on {batch_name} by appending a set of batch-specific genes with {distribution} distributed value with level {level} and dispersion {dispersion}.")
            num_new_features = int(batch_feature_frac * batch_adata.n_vars)
            batch_expr = Simulation.simulate_distribution(distribution, level, dispersion, (batch_adata.n_obs, num_new_features))
            new_vars = pd.DataFrame(index=[f"{batch_name}_Gene_{i+1}" for i in range(num_new_features)])
            batch_adata = ad.AnnData(X=np.hstack([batch_adata.X, batch_expr]), obs=batch_adata.obs, var=pd.concat([batch_adata.var, new_vars]))
        else:
            raise ValueError(f"Unknown batch effect type '{effect_type}'.")

        return batch_adata

    def simulate_expression_groups(self, num_genes=6, num_cells=12, num_groups=2, distribution="normal", mean_expression=10, dispersion=1.0, p_gene_nonspecific=0.1, group_key='group', permute=False, seed=42):
        np.random.seed(seed)
        genes_per_group = num_genes // num_groups
        cells_per_group = num_cells // num_groups
        num_nonspecific_genes = int(num_genes * p_gene_nonspecific)
        expression_matrix = np.zeros((num_cells, num_genes))
        cell_groups = []

        for group in range(num_groups):
            cell_start, cell_end = group * cells_per_group, (group + 1) * cells_per_group if group < num_groups - 1 else num_cells
            gene_start, gene_end = group * genes_per_group, (group + 1) * genes_per_group if group < num_groups - 1 else num_genes

            expression_matrix[cell_start:cell_end, gene_start:gene_end] = Simulation.simulate_distribution(distribution, mean_expression, dispersion, (cell_end - cell_start, gene_end - gene_start))

            other_genes = np.setdiff1d(np.arange(num_genes), np.arange(gene_start, gene_end))
            nonspecific_gene_indices = np.random.choice(other_genes, num_nonspecific_genes, replace=False)
            expression_matrix[cell_start:cell_end, nonspecific_gene_indices] = Simulation.simulate_distribution(distribution, mean_expression, dispersion, (cell_end - cell_start, num_nonspecific_genes))
            cell_groups.extend([f"{group_key}_{group+1}"] * (cell_end - cell_start))

        obs = pd.DataFrame({f"{group_key}": cell_groups}, index=[f"Cell_{i+1}" for i in range(num_cells)])
        var = pd.DataFrame(index=[f"Gene_{i+1}" for i in range(num_genes)])
        adata = ad.AnnData(X=expression_matrix, obs=obs, var=var)
        if permute:
            adata = adata[np.random.permutation(adata.obs_names), :]
        return adata

    def simulate_gradual_changes(self, num_genes=10, num_cells=10, trend='both', distribution='normal', time_key='time', max_expression=1, dispersion=0.1, seed=42):
        expression_matrix = np.zeros((num_cells, num_genes))
        for i in range(num_genes):
            if trend == 'increase':
                expression_matrix[:, i] = np.linspace(0, max_expression, num_cells)
            elif trend == 'decrease':
                expression_matrix[:, i] = np.linspace(max_expression, 0, num_cells)
            elif trend == 'both':
                expression_matrix[:, i] = np.linspace(0, max_expression, num_cells) if i >= num_genes // 2 else np.linspace(max_expression, 0, num_cells)
            expression_matrix[:, i] = Simulation.simulate_distribution(distribution, expression_matrix[:, i], dispersion, num_cells)

        obs = pd.DataFrame(index=[f"Cell_{i+1}" for i in range(num_cells)])
        obs[time_key] = np.linspace(0, 1, num_cells)
        var = pd.DataFrame(index=[f"Gene_{i+1}" for i in range(num_genes)])
        return ad.AnnData(X=expression_matrix, obs=obs, var=var)

    def simulate_dimension_shift(self, num_genes=10, num_cells=100, trend='increase', group_size=3, program_retention_time=5, mean_expression=10, dispersion=1.0, seed=42):
        np.random.seed(seed)
        assert group_size <= num_genes
        assert program_retention_time <= num_cells
        step_size = num_cells // program_retention_time
        expression_matrix = np.zeros((num_cells, num_genes))
        pseudotime = np.arange(num_cells)
        ngroups = num_genes // group_size + 1
        mid = num_genes // 2 // group_size

        for i in range(ngroups):
            gene_idx = np.arange(min(i * group_size, num_genes), min((i + 1) * group_size, num_genes))
            start = min(i * program_retention_time, num_cells)  
            end = min(start + step_size, num_cells) 
            if trend == 'increase':
                expression_matrix[start:end, gene_idx] = np.linspace(0, mean_expression, end - start)[:, None]
                expression_matrix[end:, gene_idx] = mean_expression
            elif trend == 'decrease':
                expression_matrix[:start, gene_idx] = mean_expression
                expression_matrix[start:end, gene_idx] = np.linspace(mean_expression, 0, end - start)[:, None]
            elif trend == 'both':
                if i >= mid:
                    expression_matrix[start:end, gene_idx] = np.linspace(0, mean_expression, end - start)[:, None]
                    expression_matrix[end:, gene_idx] = mean_expression
                else:
                    expression_matrix[:start, gene_idx] = mean_expression
                    expression_matrix[start:end, gene_idx] = np.linspace(mean_expression, 0, end - start)[:, None]
            expression_matrix[:, gene_idx] += np.random.normal(0, dispersion, (num_cells, len(gene_idx)))

        obs = pd.DataFrame({'time': pseudotime}, index=[f"Cell_{i+1}" for i in range(num_cells)])
        var = pd.DataFrame(index=[f"Gene_{i+1}" for i in range(num_genes)])
        return ad.AnnData(X=expression_matrix, obs=obs, var=var)
    
    def simulate_trajectory(self, num_genes=10, num_cells=100, 
                            program_feature_size=3, program_retention_time=0.2, program_transition_time=0.3,
                            mean_expression=10, dispersion=1.0, seed=42):
        np.random.seed(seed)

        # Simulate a transitional gene program from off to on and back to off

        pseudotime = np.arange(num_cells)
        program_transition_time = int(num_cells * program_transition_time)
        program_retention_time = int(num_cells * program_retention_time)
        program_on_time = program_transition_time * 2 + program_retention_time
        
        num_programs = num_genes // program_feature_size + 1

        ncells_sim = num_cells + program_transition_time * 2
        expression_matrix = np.zeros((ncells_sim, num_genes))
        gap_size = (ncells_sim - program_on_time) / (num_programs - 1)
        for i in range(num_programs):
            gene_idx = np.arange(min(i * program_feature_size, num_genes), min((i + 1) * program_feature_size, num_genes))
            cell_start = int(i * gap_size)
            if cell_start >= ncells_sim or gene_idx.size == 0:
                break
            cell_end = min(cell_start + program_on_time, ncells_sim)
            
            cell_on_start = min(cell_start + program_transition_time, ncells_sim)
            cell_on_end = min(cell_start + program_transition_time + program_retention_time, ncells_sim)
            cell_off_start = min(cell_start + program_transition_time + program_retention_time, ncells_sim)

            expression_matrix[cell_start:cell_on_start, gene_idx] = np.linspace(0, mean_expression, cell_on_start - cell_start).reshape(-1, 1)
            expression_matrix[cell_on_start:cell_on_end, gene_idx] = mean_expression
            expression_matrix[cell_off_start:cell_end, gene_idx] = np.linspace(mean_expression, 0, cell_end - cell_off_start).reshape(-1, 1)
        
        # Add noise to the expression matrix
        expression_matrix += np.random.normal(0, dispersion, (ncells_sim, num_genes))

        # Cut the expression matrix to the original number of cells
        expression_matrix = expression_matrix[program_transition_time:program_transition_time + num_cells, :]
        
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
    def simulate_distribution(distribution, mean, dispersion, size):
        if distribution == "normal":
            return mean + np.random.normal(0, dispersion, size)
        elif distribution == 'poisson':
            return np.random.poisson(mean, size)
        elif distribution == 'negative_binomial':
            return Simulation.rnegbin(mean, dispersion, size)
        elif distribution == 'lognormal':
            return np.random.lognormal(mean, dispersion, size)
        else:
            raise ValueError(f"Unknown distribution '{distribution}'.")


    @staticmethod
    def rnegbin(mu, theta, size):
        """
        Generate random numbers from a negative binomial distribution.

        Parameters:
        n: Number of random numbers to generate.
        mu: Mean of the distribution.
        theta: Dispersion parameter.
        """
        import numpy as np
        from scipy.stats import nbinom
        mu = np.array(mu)
        p = theta / (theta + mu)
        r = theta
        return nbinom.rvs(r, p, size=size)


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

