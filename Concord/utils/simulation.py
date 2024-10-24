import numpy as np
import pandas as pd
import anndata as ad
import scanpy as sc
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
                 trajectory_program_num=3,
                 trajectory_program_transition_time=0.3,
                 trajectory_program_retention_time=0.2,
                 trajectory_loop_to = None,
                 tree_branching_factor=2,
                 tree_depth=3,
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

        self.trajectory_program_num = trajectory_program_num
        self.trajectory_program_retention_time = trajectory_program_retention_time
        self.trajectory_program_transition_time = trajectory_program_transition_time
        self.trajectory_loop_to = trajectory_loop_to

        self.tree_branching_factor = tree_branching_factor
        self.tree_depth = tree_depth

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

    def sort_adata_genes(self, adata):
        import re
        gene_names = adata.var_names
        def natural_key(string_):
            return [int(s) if s.isdigit() else s for s in re.split(r'(\d+)', string_)]

        sorted_gene_names = sorted(gene_names, key=natural_key)
        adata = adata[:, sorted_gene_names]
        return adata

    def simulate_data(self):
        from scipy import sparse as sp
        adata_state = self.simulate_state()
        adata_state = self.sort_adata_genes(adata_state)

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

        adata = self.sort_adata_genes(adata)

        return adata, adata_state

    def simulate_state(self):
        if self.state_type == 'group':
            logger.info(f"Simulating group state with {self.n_states} groups, distribution: {self.state_distribution} with mean expression {self.state_level} and dispersion {self.state_dispersion}.")
            adata = self.simulate_expression_groups(num_genes=self.n_genes, num_cells=self.n_cells, num_groups=self.n_states, 
                                                   distribution = self.state_distribution,
                                                   mean_expression=self.state_level, dispersion=self.state_dispersion, 
                                                   p_gene_nonspecific=0, seed=self.seed)
        elif self.state_type == "trajectory":
            logger.info(f"Simulating trajectory with {self.n_states} states, distribution: {self.state_distribution} with mean expression {self.state_level} and dispersion {self.state_dispersion}.")
            adata = self.simulate_trajectory(num_genes=self.n_genes, num_cells=self.n_cells, 
                                            program_num=self.trajectory_program_num,
                                            program_retention_time=self.trajectory_program_retention_time,
                                            program_transition_time=self.trajectory_program_transition_time, 
                                            loop_to=self.trajectory_loop_to,
                                           mean_expression=self.state_level, dispersion=self.state_dispersion, seed=self.seed)
        elif self.state_type == "tree":
            logger.info(f"Simulating tree with branching factor {self.tree_branching_factor}, depth {self.tree_depth}, distribution: {self.state_distribution} with mean expression {self.state_level} and dispersion {self.state_dispersion}.")
            adata = self.simulate_tree(num_genes=self.n_genes, num_cells=self.n_cells, 
                                      branching_factor=self.tree_branching_factor, depth=self.tree_depth,
                                      mean_expression=self.state_level, dispersion=self.state_dispersion, seed=self.seed)
        else:
            raise ValueError(f"Unknown state_type '{self.state_type}'.")
        
        # Fill in na values with 0
        adata.X = np.nan_to_num(adata.X, nan=0.0)
        return adata

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


    
    def simulate_trajectory(self, num_genes=10, num_cells=100, 
                            program_num=3, program_retention_time=0.2, program_transition_time=0.3,
                            distribution='normal', mean_expression=10, dispersion=1.0, seed=42,
                            loop_to = None):
        np.random.seed(seed)

        # Simulate a transitional gene program from off to on and back to off

        pseudotime = np.arange(num_cells)
        program_transition_time = int(num_cells * program_transition_time)
        program_retention_time = int(num_cells * program_retention_time)
        program_on_time = program_transition_time * 2 + program_retention_time
        
        program_feature_size = num_genes // program_num + (1 if num_genes % program_num != 0 else 0)
        ncells_sim = num_cells + program_on_time
        expression_matrix = np.zeros((ncells_sim, num_genes))

        if loop_to is not None:
            if isinstance(loop_to, int):
                loop_to = [loop_to]

            if isinstance(loop_to, list):
                assert max(loop_to) < program_num-1, "loop_to should be less than program_num-1"
                cellgroup_num = program_num + len(loop_to)
            else:
                raise ValueError("loop_to should be an integer or a list of integers")

        gap_size = num_cells / (cellgroup_num - 1)

        for i in range(cellgroup_num):
            if loop_to is not None and i >= program_num:
                loop_to_program_idx = loop_to[i - program_num]
                gene_idx = np.arange(min(loop_to_program_idx * program_feature_size, num_genes), min((loop_to_program_idx + 1) * program_feature_size, num_genes))
            else:
                gene_idx = np.arange(min(i * program_feature_size, num_genes), min((i + 1) * program_feature_size, num_genes))
            
            cell_start = int(i * gap_size)
            if cell_start >= ncells_sim or gene_idx.size == 0:
                break
            cell_end = min(cell_start + program_on_time, ncells_sim)
            
            expression_matrix = self.simulate_expression_block(expression_matrix, gene_idx, cell_start, cell_end, mean_expression, program_transition_time, program_retention_time)

        # Add noise to the expression matrix
        expression_matrix = Simulation.simulate_distribution(distribution, expression_matrix, dispersion, (ncells_sim, num_genes))

        # Cut the expression matrix to the original number of cells
        expression_matrix = expression_matrix[(program_transition_time + program_retention_time//2):(program_transition_time + program_retention_time//2 + num_cells), :]
        
        obs = pd.DataFrame({'time': pseudotime}, index=[f"Cell_{i+1}" for i in range(num_cells)])
        var = pd.DataFrame(index=[f"Gene_{i+1}" for i in range(num_genes)])
        return ad.AnnData(X=expression_matrix, obs=obs, var=var)
    

    def simulate_expression_block(self, expression_matrix, gene_idx, cell_start, cell_end, mean_expression, program_transition_time, program_retention_time):
        ncells_sim = expression_matrix.shape[0]
        
        cell_on_start = min(cell_start + program_transition_time, ncells_sim)
        cell_on_end = min(cell_start + program_transition_time + program_retention_time, ncells_sim)
        cell_off_start = min(cell_start + program_transition_time + program_retention_time, ncells_sim)

        expression_matrix[cell_start:cell_on_start, gene_idx] = np.linspace(0, mean_expression, cell_on_start - cell_start).reshape(-1, 1)
        expression_matrix[cell_on_start:cell_on_end, gene_idx] = mean_expression
        expression_matrix[cell_off_start:cell_end, gene_idx] = np.linspace(mean_expression, 0, cell_end - cell_off_start).reshape(-1, 1)

        return expression_matrix
    


    def simulate_tree(self, num_genes=10, num_cells=100, 
                        branching_factor=2, depth=3, 
                        mean_expression=10, dispersion=1.0, seed=42):
        np.random.seed(seed)
        
        # Simulate a tree-like gene program
        num_cells_per_branch = num_cells // (branching_factor ** (depth+1))
        num_genes_per_branch = num_genes // (branching_factor ** (depth+1))
        
        # Keep track of the cell and gene indices
        cell_counter = 0
        gene_counter = 0

        # Recursive function to simulate gene expression for each branch
        def simulate_branch(depth, branch_path, inherited_genes=None):
            nonlocal cell_counter, gene_counter
            branch_str = '_'.join(map(str, branch_path)) if branch_path else 'root'
            print("Simulating depth:", depth, "branch:", branch_str)

            # Determine the number of genes and cells for this branch
            cur_num_genes = num_genes_per_branch
            cur_num_cells = num_cells_per_branch
            
            cur_branch_genes = [f"gene_{gene_counter + i}" for i in range(cur_num_genes)]
            cur_branch_cells = [f"cell_{cell_counter + i}" for i in range(cur_num_cells)]

            gene_counter += cur_num_genes
            cell_counter += cur_num_cells

            #print("depth", depth, "branch_path", branch_path, "gene_counter", gene_counter, "cell_counter", cell_counter)

            # Simulate linear increasing gene expression for branch-specific genes
            cur_branch_expression = np.tile(np.linspace(0, mean_expression, cur_num_cells).reshape(-1, 1), (1, cur_num_genes))
            cur_branch_expression = Simulation.simulate_distribution('normal', cur_branch_expression, dispersion, (cur_num_cells, cur_num_genes))
            
            if inherited_genes is not None:
                # Simulate linear decreasing gene expression for inherited genes
                inherited_expression = Simulation.simulate_distribution('normal', mean_expression, dispersion, (cur_num_cells, len(inherited_genes)))
                cur_branch_expression = np.concatenate([inherited_expression, cur_branch_expression], axis=1)
                cur_branch_genes = inherited_genes + cur_branch_genes

            adata = sc.AnnData(cur_branch_expression)
            adata.obs_names = cur_branch_cells
            adata.var_names = cur_branch_genes
            adata.obs['branch'] = branch_str
            adata.obs['depth'] = depth

            # Base case: if depth is 0, return the adata
            if depth == 0:
                return adata
            
            #expression_matrix[cell_idx, gene_idx] = np.random.normal(mean_expression, dispersion, (num_cells_per_branch, num_gene_per_branch))[0]
            # Recursively simulate sub-branches
            for i in range(branching_factor):
                new_branch_path = branch_path + [i]
                new_adata = simulate_branch(depth - 1, new_branch_path, inherited_genes=cur_branch_genes)
                adata = sc.concat([adata, new_adata], join='outer')

            return adata
            
        adata = simulate_branch(depth, branch_path=[])
        return adata


        


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

