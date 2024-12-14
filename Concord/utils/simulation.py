import numpy as np
import pandas as pd
import anndata as ad
import scanpy as sc
import logging

logger = logging.getLogger(__name__)


class Simulation:
    def __init__(self, n_cells=1000, n_genes=1000, 
                 n_batches=2, n_states=3, 
                 state_type='cluster', 
                 batch_type='batch_specific_features', 
                 state_distribution='normal',
                 state_level=1.0, 
                 state_min_level=0.0,
                 state_dispersion=0.1, 
                 program_structure="linear",
                 program_on_time_fraction=0.3,
                 trajectory_program_num=3,
                 trajectory_cell_block_size_ratio=0.3,
                 trajectory_loop_to = None,
                 tree_branching_factor=2,
                 tree_depth=3,
                 tree_program_decay=0.5,
                 batch_distribution='normal',
                 batch_level=1.0, 
                 batch_dispersion=0.1, 
                 batch_cell_proportion=None,
                 batch_feature_frac=0.1,
                 global_non_specific_gene_fraction=0.1,
                 pairwise_non_specific_gene_fraction=None,
                 universal_gene_fraction=0.0,
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
        self.state_min_level = state_min_level
        self.state_dispersion = state_dispersion
        self.program_structure = program_structure
        self.program_on_time_fraction = program_on_time_fraction

        self.trajectory_program_num = trajectory_program_num
        self.trajectory_cell_block_size_ratio = trajectory_cell_block_size_ratio
        self.trajectory_loop_to = trajectory_loop_to

        self.tree_branching_factor = tree_branching_factor
        self.tree_depth = tree_depth
        self.tree_program_decay = tree_program_decay

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

        self.global_non_specific_gene_fraction = global_non_specific_gene_fraction
        self.pairwise_non_specific_gene_fraction = pairwise_non_specific_gene_fraction
        self.universal_gene_fraction = universal_gene_fraction
        self.non_neg = non_neg
        self.to_int = to_int
        self.seed = seed
        np.random.seed(seed)


    def simulate_data(self):
        from scipy import sparse as sp
        from .other_util import sort_string_list
        adata_state = self.simulate_state()

        adata_state = adata_state[:, sort_string_list(adata_state.var_names)]

        batch_list = []
        state_list = []
        for i in range(self.n_batches):
            batch_adata, batch_adata_pre = self.simulate_batch(
                adata_state, 
                cell_proportion=self.batch_cell_proportion[i], 
                batch_name=f"batch_{i+1}", effect_type=self.batch_type[i], 
                distribution = self.batch_distribution[i],
                level=self.batch_level[i], dispersion=self.batch_dispersion[i], 
                batch_feature_frac=self.batch_feature_frac[i], seed=self.seed+i)
            batch_list.append(batch_adata)
            state_list.append(batch_adata_pre)

        adata = ad.concat(batch_list, join='outer')
        adata.X = adata.X.toarray() if sp.issparse(adata.X) else adata.X
        adata.X = np.nan_to_num(adata.X, nan=0.0)

        adata = adata[:, sort_string_list(adata.var_names)]

        adata_pre = ad.concat(state_list, join='outer')
        adata_pre = adata_pre[:, sort_string_list(adata_pre.var_names)]

        # Concatenate batch name to cell names to make them unique
        adata.obs_names = [f"{batch}_{cell}" for batch, cell in zip(adata.obs['batch'], adata.obs_names)]
        adata_pre.obs_names = [f"{batch}_{cell}" for batch, cell in zip(adata_pre.obs['batch'], adata_pre.obs_names)]

        adata.obs['batch'] = adata.obs['batch'].astype('category')
        adata_pre.obs['batch'] = adata_pre.obs['batch'].astype('category')

        if self.non_neg:
            adata.X[adata.X < 0] = 0
            adata_pre.X[adata_pre.X < 0] = 0
        if self.to_int:
            adata.X = adata.X.astype(int)
            adata_pre.X = adata_pre.X.astype(int)

        adata_pre.layers['wt_noise'] = adata_pre.X
        adata.layers['counts'] = adata.X.copy()

        adata.layers['no_noise'] = np.zeros_like(adata.X)
        adata.layers['wt_noise'] = np.zeros_like(adata.X)
        common_genes = adata.var_names.intersection(adata_pre.var_names)
        adata_indices = adata.var_names.get_indexer(common_genes)
        adata_pre_indices = adata_pre.var_names.get_indexer(common_genes)

        # Copy data from `adata_pre` to `adata` for these common genes
        adata.layers['no_noise'][:, adata_indices] = adata_pre.layers['no_noise'][:, adata_pre_indices].copy()
        adata.layers['wt_noise'][:, adata_indices] = adata_pre.layers['wt_noise'][:, adata_pre_indices].copy()

        return adata, adata_pre


    def simulate_state(self):
        if self.state_type == 'cluster':
            logger.info(f"Simulating {self.n_states} clusters with distribution: {self.state_distribution} with mean expression {self.state_level} and dispersion {self.state_dispersion}.")
            adata = self.simulate_clusters(n_genes=self.n_genes, n_cells=self.n_cells, num_clusters=self.n_states, 
                                           program_structure=self.program_structure, program_on_time_fraction=self.program_on_time_fraction,
                                           distribution = self.state_distribution,
                                           mean_expression=self.state_level, min_expression=self.state_min_level, dispersion=self.state_dispersion, 
                                           global_non_specific_gene_fraction=self.global_non_specific_gene_fraction,
                                           pairwise_non_specific_gene_fraction=self.pairwise_non_specific_gene_fraction,
                                           seed=self.seed)
        elif self.state_type == "trajectory":
            logger.info(f"Simulating trajectory with {self.n_states} states, distribution: {self.state_distribution} with mean expression {self.state_level} and dispersion {self.state_dispersion}.")
            adata = self.simulate_trajectory(n_genes=self.n_genes, n_cells=self.n_cells, 
                                             cell_block_size_ratio=self.trajectory_cell_block_size_ratio,
                                            program_num=self.trajectory_program_num,
                                            program_structure=self.program_structure,
                                            program_on_time_fraction=self.program_on_time_fraction,
                                            loop_to=self.trajectory_loop_to,
                                            distribution=self.state_distribution,
                                            mean_expression=self.state_level, 
                                            min_expression=self.state_min_level,
                                            dispersion=self.state_dispersion, seed=self.seed)
        elif self.state_type == "tree":
            logger.info(f"Simulating tree with branching factor {self.tree_branching_factor}, depth {self.tree_depth}, distribution: {self.state_distribution} with mean expression {self.state_level} and dispersion {self.state_dispersion}.")
            adata = self.simulate_tree(n_genes=self.n_genes, n_cells=self.n_cells, 
                                      branching_factor=self.tree_branching_factor, depth=self.tree_depth,
                                      program_structure=self.program_structure,
                                      program_on_time_fraction=self.program_on_time_fraction,
                                      program_decay=self.tree_program_decay,
                                      distribution=self.state_distribution,
                                      mean_expression=self.state_level, min_expression=self.state_min_level,
                                      dispersion=self.state_dispersion, seed=self.seed)
        elif self.state_type == 'gatto':
            logger.info(f"Simulating dataset in Gatto et al. 2023 with distribution: {self.state_distribution} with background expression {self.state_level} and dispersion {self.state_dispersion}.")
            adata = self.simulate_gatto(n_genes=self.n_genes, n_cells=self.n_cells, 
                                      distribution=self.state_distribution, background_shift=self.state_level, dispersion=self.state_dispersion, seed=self.seed)
        elif self.state_type == 's_curve':
            logger.info(f"Simulating S-curve with distribution: {self.state_distribution} with mean shift {self.state_level} and dispersion {self.state_dispersion}.")
            adata = self.simulate_s_curve(n_genes=self.n_genes, n_cells=self.n_cells, 
                                      distribution=self.state_distribution, mean_expression=self.state_level, dispersion=self.state_dispersion, 
                                      universal_gene_fraction=self.universal_gene_fraction, seed=self.seed)
            
        elif self.state_type == 'swiss_roll':
            logger.info(f"Simulating Swiss roll with distribution: {self.state_distribution} with mean shift {self.state_level} and dispersion {self.state_dispersion}.")
            adata = self.simulate_swiss_roll(n_genes=self.n_genes, n_cells=self.n_cells, 
                                      distribution=self.state_distribution, mean_expression=self.state_level, dispersion=self.state_dispersion, 
                                      hole=False,
                                      universal_gene_fraction=self.universal_gene_fraction, seed=self.seed)
        elif self.state_type == 'swiss_roll_hole':
            logger.info(f"Simulating Swiss roll with a hole with distribution: {self.state_distribution} with mean shift {self.state_level} and dispersion {self.state_dispersion}.")
            adata = self.simulate_swiss_roll(n_genes=self.n_genes, n_cells=self.n_cells, 
                                      distribution=self.state_distribution, mean_expression=self.state_level, dispersion=self.state_dispersion, 
                                      hole=True,
                                      universal_gene_fraction=self.universal_gene_fraction, seed=self.seed)
        elif self.state_type == 'intersecting_circle':
            logger.info(f"Simulating intersecting circles with distribution: {self.state_distribution} with mean shift {self.state_level} and dispersion {self.state_dispersion}.")
            adata = self.simulate_intersecting_circles(n_genes=self.n_genes, n_cells=self.n_cells, 
                                      distribution=self.state_distribution, mean_expression=self.state_level, dispersion=self.state_dispersion, 
                                      universal_gene_fraction=self.universal_gene_fraction, seed=self.seed)
        elif self.state_type == 'nonintersecting_loops':
            logger.info(f"Simulating nonintersecting loops with distribution: {self.state_distribution} with mean shift {self.state_level} and dispersion {self.state_dispersion}.")
            adata = self.simulate_nonintersecting_loops(n_genes=self.n_genes, n_cells=self.n_cells, 
                                      distribution=self.state_distribution, mean_expression=self.state_level, dispersion=self.state_dispersion, 
                                      universal_gene_fraction=self.universal_gene_fraction, seed=self.seed)
        else:
            raise ValueError(f"Unknown state_type '{self.state_type}'.")
        
        # Fill in na values with 0
        adata.X = np.nan_to_num(adata.X, nan=0.0)
        return adata

    def simulate_batch(self, adata, cell_indices=None, cell_proportion=0.3, batch_name='batch_1', effect_type='batch_specific_features', distribution='normal', 
                       level=1.0, dispersion = 0.1,batch_feature_frac=0.1, seed=42):
        from scipy import sparse as sp
        np.random.seed(seed)
        if not (0 < cell_proportion <= 1):
            raise ValueError("cell_proportion must be between 0 and 1.")
        
        if cell_indices is None:
            n_cells = int(adata.n_obs * cell_proportion)
            cell_indices = np.random.choice(adata.n_obs, size=n_cells, replace=False)
            # Sort the cell indices
            cell_indices = np.sort(cell_indices)
        
        batch_adata_pre = adata[cell_indices].copy()
        batch_adata_pre.obs['batch'] = batch_name
        batch_adata = batch_adata_pre.copy()

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
            n_genes_batch_specific = int(batch_feature_frac * batch_adata.n_vars)
            batch_specific_genes = np.random.choice(batch_adata.n_vars, n_genes_batch_specific, replace=False)
            batch_adata.X[:, batch_specific_genes] += Simulation.simulate_distribution(distribution, level, dispersion, (batch_adata.n_obs, n_genes_batch_specific))
        elif effect_type == 'batch_specific_features':
            logger.info(f"Simulating batch-specific features effect on {batch_name} by appending a set of batch-specific genes with {distribution} distributed value with level {level} and dispersion {dispersion}.")
            num_new_features = int(batch_feature_frac * batch_adata.n_vars)
            batch_expr = Simulation.simulate_distribution(distribution, level, dispersion, (batch_adata.n_obs, num_new_features))
            new_vars = pd.DataFrame(index=[f"{batch_name}_Gene_{i+1}" for i in range(num_new_features)])
            batch_adata = ad.AnnData(X=np.hstack([batch_adata.X, batch_expr]), obs=batch_adata.obs, var=pd.concat([batch_adata.var, new_vars]))
        else:
            raise ValueError(f"Unknown batch effect type '{effect_type}'.")

        return batch_adata, batch_adata_pre
    

    def simulate_clusters(self, n_genes=6, n_cells=12, num_clusters=2, program_structure="uniform", program_on_time_fraction=0.3,
                      distribution="normal", mean_expression=10, min_expression=1, dispersion=1.0, 
                      global_non_specific_gene_fraction=0.1, 
                      pairwise_non_specific_gene_fraction=None,
                      cluster_key='cluster', permute=False, seed=42):
        np.random.seed(seed)
        
        # Check if n_genes is a list or integer
        if isinstance(n_genes, list):
            if len(n_genes) != num_clusters:
                raise ValueError("Length of n_genes list must match num_clusters.")
            genes_per_cluster_list = n_genes
        else:
            genes_per_cluster = n_genes // num_clusters
            genes_per_cluster_list = [genes_per_cluster] * num_clusters
        
        # Check if n_cells is a list or integer
        if isinstance(n_cells, list):
            if len(n_cells) != num_clusters:
                raise ValueError("Length of n_cells list must match num_clusters.")
            cells_per_cluster_list = n_cells
        else:
            cells_per_cluster = n_cells // num_clusters
            cells_per_cluster_list = [cells_per_cluster] * num_clusters
        
        total_n_genes = sum(genes_per_cluster_list)
        total_n_cells = sum(cells_per_cluster_list)
        expression_matrix = np.zeros((total_n_cells, total_n_genes))
        cell_clusters = []
        gene_offset = 0  # Tracks the starting gene index for each cluster
        cell_offset = 0  # Tracks the starting cell index for each cluster

        cluster_gene_indices = {}
        cluster_cell_indices = {}
        for cluster in range(num_clusters):
            # Define cell range for this cluster based on the supplied list
            cell_start = cell_offset
            cell_end = cell_offset + cells_per_cluster_list[cluster]
            cell_indices = np.arange(cell_start, cell_end)
            cell_offset = cell_end  # Update offset for the next cluster
            #print(cell_indices)
            
            # Define gene range for this cluster based on the supplied list
            gene_start = gene_offset
            gene_end = gene_offset + genes_per_cluster_list[cluster]
            gene_offset = gene_end  # Update offset for the next cluster

            # Combine the specific and nonspecific gene indices
            gene_indices = np.concatenate([np.arange(gene_start, gene_end)])

            cluster_gene_indices[cluster] = gene_indices
            cluster_cell_indices[cluster] = cell_indices
            # Simulate expression for the current cluster
            expression_matrix = self.simulate_expression_block(
                expression_matrix, program_structure, gene_indices, cell_indices, mean_expression, min_expression, program_on_time_fraction
            )
            
            cell_clusters.extend([f"{cluster_key}_{cluster+1}"] * len(cell_indices))

        # Add non-specific genes to the expression matrix

        if pairwise_non_specific_gene_fraction is not None:
            logger.info("Adding non-specific genes to the expression matrix. Note this will increase gene count compared to the specified value.")
            for cluster_pairs in pairwise_non_specific_gene_fraction.keys():
                cluster1, cluster2 = cluster_pairs
                cluster1_genes = cluster_gene_indices[cluster1]
                cluster2_genes = cluster_gene_indices[cluster2]
                union_genes = np.union1d(cluster1_genes, cluster2_genes)
                num_nonspecific_genes = int(len(union_genes) * pairwise_non_specific_gene_fraction[cluster_pairs])
                nonspecific_gene_indices = np.random.choice(union_genes, num_nonspecific_genes, replace=False)
                cluster1_cells = cluster_cell_indices[cluster1]
                cluster2_cells = cluster_cell_indices[cluster2]
                union_cells = np.union1d(cluster1_cells, cluster2_cells)
                expression_matrix = self.simulate_expression_block(
                    expression_matrix, program_structure, nonspecific_gene_indices, union_cells, mean_expression, min_expression, program_on_time_fraction
                )
            
        if global_non_specific_gene_fraction is not None and global_non_specific_gene_fraction > 0:
            logger.info("Adding non-specific genes to the expression matrix. Note this will increase gene count compared to the specified value.")
            num_nonspecific_genes = int(expression_matrix.shape[1] * global_non_specific_gene_fraction)
            all_genes = np.arange(expression_matrix.shape[1])
            nonspecific_gene_indices = np.random.choice(all_genes, num_nonspecific_genes, replace=False)
            expression_matrix = self.simulate_expression_block(
                expression_matrix, program_structure, nonspecific_gene_indices, np.arange(total_n_cells), mean_expression, min_expression, program_on_time_fraction
            )
            # Sort these genes to the end
            specific_gene_indices = np.setdiff1d(all_genes, nonspecific_gene_indices)
            expression_matrix = expression_matrix[:, np.concatenate([specific_gene_indices, nonspecific_gene_indices])]
            
        # Apply distribution to simulate realistic expression values
        if not isinstance(dispersion, list):
            expression_matrix_wt_noise = Simulation.simulate_distribution(distribution, expression_matrix, dispersion)
        else:
            expression_matrix_wt_noise = expression_matrix.copy()
            # For each cluster, apply a different dispersion value
            for i, disp in enumerate(dispersion):
                cell_indices = cluster_cell_indices[i]
                expression_matrix_wt_noise[cell_indices, :] = Simulation.simulate_distribution(distribution, expression_matrix[cell_indices, :], disp)


        # Create AnnData object
        obs = pd.DataFrame({f"{cluster_key}": cell_clusters}, index=[f"Cell_{i+1}" for i in range(total_n_cells)])
        var = pd.DataFrame(index=[f"Gene_{i+1}" for i in range(total_n_genes)])
        adata = ad.AnnData(X=expression_matrix_wt_noise, obs=obs, var=var)
        adata.layers['no_noise'] = expression_matrix
        
        if permute:
            adata = adata[np.random.permutation(adata.obs_names), :]
        
        return adata



    
    def simulate_trajectory(self, n_genes=10, n_cells=100, cell_block_size_ratio=0.3,
                            program_num=3, program_structure="linear", program_on_time_fraction=0.3,
                            distribution='normal', mean_expression=10, min_expression=0, dispersion=1.0, seed=42,
                            loop_to = None):
        np.random.seed(seed)

        # Simulate a transitional gene program from off to on and back to off

        pseudotime = np.arange(n_cells)
        cell_block_size = int(n_cells * cell_block_size_ratio)
        
        program_feature_size = n_genes // program_num + (1 if n_genes % program_num != 0 else 0)
        ncells_sim = n_cells + cell_block_size
        expression_matrix = np.zeros((ncells_sim, n_genes))

        if loop_to is not None:
            if isinstance(loop_to, int):
                loop_to = [loop_to]

            if isinstance(loop_to, list):
                assert max(loop_to) < program_num-1, "loop_to should be less than program_num-1"
                cellgroup_num = program_num + len(loop_to)
            else:
                raise ValueError("loop_to should be an integer or a list of integers")
        else:
            cellgroup_num = program_num

        gap_size = n_cells / (cellgroup_num - 1)

        for i in range(cellgroup_num):
            if loop_to is not None and i >= program_num:
                loop_to_program_idx = loop_to[i - program_num]
                gene_idx = np.arange(min(loop_to_program_idx * program_feature_size, n_genes), min((loop_to_program_idx + 1) * program_feature_size, n_genes))
            else:
                gene_idx = np.arange(min(i * program_feature_size, n_genes), min((i + 1) * program_feature_size, n_genes))
            
            cell_start = int(i * gap_size)
            if cell_start >= ncells_sim or gene_idx.size == 0:
                break
            cell_end = min(cell_start + cell_block_size, ncells_sim)
            cell_indices = np.arange(cell_start, cell_end)
            
            expression_matrix = self.simulate_expression_block(expression_matrix, program_structure, gene_idx, cell_indices, mean_expression, min_expression, program_on_time_fraction)


        # Cut the expression matrix to the original number of cells
        expression_matrix = expression_matrix[(cell_block_size//2):(cell_block_size//2 + n_cells), :]

        # Add noise to the expression matrix
        expression_matrix_wt_noise = Simulation.simulate_distribution(distribution, expression_matrix, dispersion)

        obs = pd.DataFrame({'time': pseudotime}, index=[f"Cell_{i+1}" for i in range(n_cells)])
        var = pd.DataFrame(index=[f"Gene_{i+1}" for i in range(n_genes)])
        adata = ad.AnnData(X=expression_matrix_wt_noise, obs=obs, var=var)
        adata.layers['no_noise'] = expression_matrix
        return adata
    

    def simulate_expression_block(self, expression_matrix, structure, gene_idx, cell_idx, mean_expression, min_expression, on_time_fraction = 0.3):
        ncells = len(cell_idx)
        assert(on_time_fraction <= 1), "on_time_fraction should be less than or equal to 1"

        cell_start = cell_idx[0]
        cell_end = cell_idx[-1]+1

        program_on_time = int(ncells * on_time_fraction)
        program_transition_time = int(ncells - program_on_time) // 2 if "bidirectional" in structure else int(ncells - program_on_time)

        transition_end = cell_start if "decreasing" in structure else min(cell_start + program_transition_time, cell_end)
        on_end = min(transition_end + program_on_time, cell_end) 

        #print("program_on_time", program_on_time, "program_transition_time", program_transition_time)
        #print("cell_start", cell_start, "transition_end", transition_end, "on_end", on_end, "cell_end", cell_end)

        if "linear" in structure:
            expression_matrix[cell_start:transition_end, gene_idx] = np.linspace(min_expression, mean_expression, transition_end-cell_start).reshape(-1, 1)
            expression_matrix[transition_end:on_end, gene_idx] = mean_expression
            expression_matrix[on_end:cell_end, gene_idx] = np.linspace(mean_expression, min_expression, cell_end-on_end).reshape(-1, 1)
        elif "dimension_increase" in structure:
            gap_size = max((ncells - program_on_time) // len(gene_idx), 1)
            #print("ncells", ncells, "len(gene_idx)", len(gene_idx), "gap_size", gap_size)
            # Simulate a gene program that has each of its genes gradually turning on
            for i, gene in enumerate(gene_idx):
                cur_gene_start = min(cell_start + i * gap_size, cell_end)
                #print("cur_gene_start", cur_gene_start, "transition_end", transition_end, "cell_end", cell_end, "gene", gene)
                expression_matrix[cur_gene_start:transition_end, gene] = np.linspace(min_expression, mean_expression, transition_end-cur_gene_start)
                expression_matrix[transition_end:cell_end, gene] = mean_expression
        elif structure == "uniform":
            expression_matrix[cell_start:cell_end, gene_idx] = mean_expression
        else:
            raise ValueError(f"Unknown structure '{structure}'.")        

        return expression_matrix
    


    def simulate_tree(self, n_genes=10, n_cells=100, 
                        branching_factor=2, depth=3, 
                        program_structure="linear_increasing",
                        program_on_time_fraction=0.3,
                        program_decay=0.5,
                        distribution='normal',
                        mean_expression=10, min_expression=0, 
                        dispersion=1.0, seed=42):
        np.random.seed(seed)
        
        # Simulate a tree-like gene program
        n_cells_per_branch = n_cells // (branching_factor ** (depth+1))
        n_genes_per_branch = n_genes // (branching_factor ** (depth+1))
        
        # Keep track of the cell and gene indices
        cell_counter = 0
        gene_counter = 0
        total_depth = depth

        if(program_decay != 1):
            logger.warning("Total number of genes will not be equal to n_genes due to program_decay < 1.")

        # Recursive function to simulate gene expression for each branch
        def simulate_branch(depth, branch_path, inherited_genes=None, start_time=0):
            nonlocal cell_counter, gene_counter
            branch_str = '_'.join(map(str, branch_path)) if branch_path else 'root'
            #print("Simulating depth:", depth, "branch:", branch_str)

            # Determine the number of genes and cells for this branch
            cur_n_genes = max(int(n_genes_per_branch * program_decay ** (total_depth-depth)),1)
            cur_n_cells = n_cells_per_branch
            
            cur_branch_genes = [f"gene_{gene_counter + i}" for i in range(cur_n_genes)]
            cur_branch_cells = [f"cell_{cell_counter + i}" for i in range(cur_n_cells)]

            gene_counter += cur_n_genes
            cell_counter += cur_n_cells

            #print("depth", depth, "branch_path", branch_path, "gene_counter", gene_counter, "cell_counter", cell_counter)
            #print("cur_n_genes", cur_n_genes, "cur_n_cells", cur_n_cells)
            # Simulate linear increasing gene expression for branch-specific genes
            cur_branch_expression = self.simulate_expression_block(
                np.zeros((cur_n_cells, cur_n_genes)), 
                program_structure, 
                np.arange(cur_n_genes), 
                np.arange(cur_n_cells), 
                mean_expression, 
                min_expression, 
                on_time_fraction=program_on_time_fraction
            )
            
            if inherited_genes is not None:
                # Simulate linear decreasing gene expression for inherited genes
                #inherited_expression = Simulation.simulate_distribution('normal', mean_expression, dispersion, (cur_n_cells, len(inherited_genes)))
                inherited_expression = np.array(mean_expression).reshape(1, -1) * np.ones((cur_n_cells, len(inherited_genes)))
                cur_branch_expression = np.concatenate([inherited_expression, cur_branch_expression], axis=1)
                cur_branch_genes = inherited_genes + cur_branch_genes

            # Create an AnnData object for the current branch
            cur_branch_expression_wt_noise = Simulation.simulate_distribution(distribution, cur_branch_expression, dispersion)
            adata = sc.AnnData(cur_branch_expression_wt_noise)
            adata.obs_names = cur_branch_cells
            adata.var_names = cur_branch_genes
            adata.obs['branch'] = branch_str
            adata.obs['depth'] = depth
            adata.obs['time'] = np.arange(cur_n_cells) + start_time
            end_time = start_time + cur_n_cells
            adata.layers['no_noise'] = cur_branch_expression

            # Base case: if depth is 0, return the adata
            if depth == 0:
                return adata
            
            #expression_matrix[cell_idx, gene_idx] = np.random.normal(mean_expression, dispersion, (n_cells_per_branch, num_gene_per_branch))[0]
            # Recursively simulate sub-branches
            for i in range(branching_factor):
                new_branch_path = branch_path + [i]
                new_adata = simulate_branch(depth - 1, new_branch_path, inherited_genes=cur_branch_genes, start_time=end_time)
                adata = sc.concat([adata, new_adata], join='outer')

            return adata
            
        adata = simulate_branch(depth, branch_path=[])

        adata.X = np.nan_to_num(adata.X, nan=0.0)
        for key in adata.layers.keys():
            adata.layers[key] = np.nan_to_num(adata.layers[key], nan=0.0)
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
    def simulate_distribution(distribution, mean, dispersion, size=None):
        if size is None:
            size = mean.shape

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

        logger.info(f"Percent 0s: {np.mean(dropout_indicator):.2f}")
        # Apply dropout to the UMI counts
        final_umi_mtx = mtx * (1 - dropout_indicator)
        
        return final_umi_mtx


    # Simulation in Gatto et al., 2023
    def simulate_gatto(self, n_cells=1000, n_genes=1000, t1=3/5, t2=4/5, scale=5, distribution='normal', dispersion=1.0, background_shift=0, seed=42):

        # Set random seed for reproducibility
        np.random.seed(seed)

        ## SIMULATE DATA

        # simulate base spiral structure
        n = n_cells
        z = np.arange(1, n + 1) / 40  # Differentiation progression component Z
        r = 5
        x = r * np.cos(z)  # Cycling process X
        y = r * np.sin(z)  # Cycling process Y

        # For points after t1, simulate two branches, corresponding to two cell fates
        t1 = round(t1 * n)
        # Randomly assign cells to two branches
        w = np.random.choice([0, 1], size=n)
        # Simulate two branches
        x0 = x[t1]
        y0 = y[t1]
        w_t1 = w[t1:]
        x[t1:][w_t1 == 0] = x0 + np.arange(1, sum(w_t1 == 0) + 1) / scale
        y[t1:][w_t1 == 0] = y0

        x[t1:][w_t1 == 1] = x0
        y[t1:][w_t1 == 1] = y0 + np.arange(1, sum(w_t1 == 1) + 1) / scale

        # After t2, simulate a homogenoues state for each branch
        t2 = round(t2 * n)
        w_t2 = w[t2:]
        x[t2:][w_t2 == 0] = x[t2:][w_t2 == 0][0]
        y[t2:][w_t2 == 0] = y[t2:][w_t2 == 0][0]
        x[t2:][w_t2 == 1] = x[t2:][w_t2 == 1][0]
        y[t2:][w_t2 == 1] = y[t2:][w_t2 == 1][0]

        # x = x + np.random.normal(0, dispersion, n)
        # y = y + np.random.normal(0, dispersion, n)
        # z = z + np.random.normal(0, dispersion, n)

        # Scale to [0, 1]
        x = (x - min(x)) / (max(x) - min(x))
        y = (y - min(y)) / (max(y) - min(y))
        z = (z - min(z)) / (max(z) - min(z))


        # Convert w to a state label
        state = w.astype(str)
        state[state == '0'] = 'state1'
        state[state == '1'] = 'state2'
        state[:t1] = 'state0'

        # Increase number of features 
        np_repeats = n_genes // 3
        a = np.arange(1, np_repeats + 1)  # Sequence [1, 2, ..., np_repeats]

        # Protein abundance matrices for each component
        pz = np.outer(a, z) / np.max(z)
        px = np.outer(a, x) / np.max(x)
        py = np.outer(a, y) / np.max(y)

        # Combine the matrices
        dat = np.vstack([pz, px, py])

        # add constant value
        dat = dat + background_shift
    
        # add additional expression block)
        print(dat.shape)
        print(np.random.normal(0, 1, (n_genes//5, n)).shape)
        dat = np.vstack([dat, np.random.normal(20, 1, (n_genes//5, n))])
        dat = np.vstack([dat, np.repeat(np.linspace(100, 0, n).reshape(1, -1), n_genes//5, axis=0)])

        # Add noise to the expression matrix
        expression_matrix_wt_noise = Simulation.simulate_distribution(distribution, dat.T, dispersion)
        #expression_matrix = dat.T
        # Convert to anndata object
        adata = sc.AnnData(expression_matrix_wt_noise)
        adata.obs['state'] = state
        adata.obs['state'] = adata.obs['state'].astype('category')
        adata.obs['time'] = np.arange(1, n + 1)
        adata.layers['no_noise'] = dat.T

        return adata
    

    # Simulate S curve using sklearn.datasets.make_s_curve
    def simulate_s_curve(self, n_cells=1000, n_genes=1000, mean_expression=1, distribution='normal', dispersion=1.0, seed=42, universal_gene_fraction=0.0):
        from sklearn.datasets import make_s_curve
        np.random.seed(seed)
        X, t = make_s_curve(n_cells, random_state=seed)
        X = X.T + mean_expression # Mean is used to shift the expression values

        # Increase number of features by repeating the original features multiplied by a random number sampled from a normal distribution
        np_repeats = n_genes // 3
        a = np.random.normal(1, 1, np_repeats)
        expression_matrix = np.vstack([np.outer(a, X[0]), np.outer(a, X[1]), np.outer(a, X[2])]).T

        # Add a uniform gene block if universal_gene_fraction > 0
        if universal_gene_fraction > 0:
            uniform_genes = mean_expression * np.ones((n_cells, int(universal_gene_fraction * n_genes)))
            expression_matrix = np.hstack([expression_matrix, uniform_genes])
        
        # Add noise to the expression matrix
        expression_matrix_wt_noise = Simulation.simulate_distribution(distribution, expression_matrix, dispersion)
        # Convert to anndata object
        adata = sc.AnnData(expression_matrix_wt_noise)
        adata.obs['time'] = t
        adata.layers['no_noise'] = expression_matrix
        return adata
    

    def simulate_swiss_roll(self, n_cells=1000, n_genes=1000, mean_expression=1, distribution='normal', dispersion=1.0, hole=False, seed=42, universal_gene_fraction=0.0):
        # Set random seed for reproducibility
        from sklearn.datasets import make_swiss_roll
        np.random.seed(seed)
        
        # Generate Swiss roll coordinates
        X, t = make_swiss_roll(n_samples=n_cells, noise=0.0, hole=hole, random_state=seed)
        X = X + mean_expression  # Scale down to control for expression range and shift with mean
        
        # Expand to n_genes by repeating the coordinates with random multipliers
        np_repeats = n_genes // 3
        gene_multipliers = np.random.normal(1, 0.1, np_repeats)  # Slight variance in each dimension
        expression_matrix = np.vstack([
            np.outer(gene_multipliers, X[:, 0]),
            np.outer(gene_multipliers, X[:, 1]),
            np.outer(gene_multipliers, X[:, 2])
        ]).T
        
        # Add universal gene block if specified
        if universal_gene_fraction > 0:
            uniform_genes = mean_expression * np.ones((n_cells, int(universal_gene_fraction * n_genes)))
            expression_matrix = np.hstack([expression_matrix, uniform_genes])
        
        # Add noise to the expression matrix based on the specified distribution
        expression_matrix_wt_noise = self.simulate_distribution(distribution, expression_matrix, dispersion)
        
        # Convert to an AnnData object
        adata = sc.AnnData(expression_matrix_wt_noise)
        adata.obs['time'] = t
        adata.layers['no_noise'] = expression_matrix
        
        return adata



    def simulate_intersecting_circles(self, n_cells=1000, n_genes=100, radius=1.0, mean_expression=1, distribution='normal', dispersion=0.1, seed=42, universal_gene_fraction=0.0):
        import numpy as np
        np.random.seed(seed)

        # Parameters
        num_points = n_cells // 2  # Number of points in each circle
        circle_radius = 1  # Radius of both circles

        # Generate the first circle in the x-y plane
        angles_circle1 = np.linspace(0, 2 * np.pi, num_points, endpoint=False)
        circle1_x = circle_radius * np.cos(angles_circle1)
        circle1_y = circle_radius * np.sin(angles_circle1)
        circle1_z = np.zeros(num_points)  # Z = 0 for the x-y plane

        # Generate the second circle in the y-z plane, shifted to pass through the center of the first circle
        angles_circle2 = np.linspace(0, 2 * np.pi, num_points, endpoint=False)
        circle2_x = np.zeros(num_points)  # X = 0 for the y-z plane
        circle2_y = circle_radius * np.cos(angles_circle2)
        circle2_z = circle_radius * np.sin(angles_circle2)

        # Shift the second circle along the y-axis to interlock with the first circle
        circle2_y += circle_radius  # Shift by radius so it intersects the first circle's center

        X = np.vstack((np.hstack((circle1_x, circle2_x)), np.hstack((circle1_y, circle2_y)), np.hstack((circle1_z, circle2_z)))).T
        
        # Combine points from both circles
        X = X + mean_expression

        np_repeats = n_genes // 3
        gene_multipliers = np.random.normal(1, 0.1, np_repeats)  # Slight variance in each dimension
        expression_matrix = np.vstack([
            np.outer(gene_multipliers, X[:, 0]),
            np.outer(gene_multipliers, X[:, 1]),
            np.outer(gene_multipliers, X[:, 2])
        ]).T
                
        # Add a block of universal genes if specified
        if universal_gene_fraction > 0:
            n_universal_genes = int(universal_gene_fraction * n_genes)
            universal_genes = mean_expression * np.ones((n_cells, n_universal_genes))
            expression_matrix = np.hstack([expression_matrix, universal_genes])
        
        # Add noise to the expression matrix based on the specified distribution
        expression_matrix_wt_noise = self.simulate_distribution(distribution, expression_matrix, dispersion)
        
        # Create an AnnData object
        adata = sc.AnnData(expression_matrix_wt_noise)
        adata.obs['circle'] = ['circle1'] * num_points + ['circle2'] * num_points
        adata.layers['no_noise'] = expression_matrix
        
        return adata

    def simulate_nonintersecting_loops(self, n_cells=1000, n_genes=100, mean_expression=1, distribution='normal', dispersion=0.1, seed=42, universal_gene_fraction=0.0):
        import numpy as np
        np.random.seed(seed)

        # Parameters
        num_points = n_cells // 2

        # Simulate loop 1 using the simulate trajectory function
        loop1 = self.simulate_trajectory(n_cells=num_points, n_genes=n_genes, program_num=4, program_structure="linear_bidirectional", cell_block_size_ratio=0.5, program_on_time_fraction=0.0, loop_to=0,
                                         distribution=distribution, mean_expression=mean_expression, dispersion=dispersion, seed=seed)
        
        loop1.var_names = loop1.var_names + '_loop1'
        loop1.obs_names = loop1.obs_names + '_loop1'
        loop1.obs['loop'] = 'loop1'

        # Simulate loop 2 using the simulate trajectory function
        loop2 = self.simulate_trajectory(n_cells=num_points, n_genes=n_genes, program_num=4, program_structure="linear_bidirectional", cell_block_size_ratio=0.5, program_on_time_fraction=0.0, loop_to=0,
                                         distribution=distribution, mean_expression=mean_expression, dispersion=dispersion, seed=seed)
        loop2.var_names = loop2.var_names + '_loop2'
        loop2.obs_names = loop2.obs_names + '_loop2'
        loop2.obs['loop'] = 'loop2'
        
        # Concatenate the two loops
        adata = ad.concat([loop1, loop2], join='outer', axis=0)
        adata.X = np.nan_to_num(adata.X, nan=0.0)
        for key in adata.layers.keys():
            adata.layers[key] = np.nan_to_num(adata.layers[key], nan=0.0)

        return adata
    
