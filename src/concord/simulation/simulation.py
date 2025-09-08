from __future__ import annotations
import numpy as np
import pandas as pd
import anndata as ad
import scanpy as sc
from scipy import sparse as sp
import logging
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any, Union, Tuple
from ..utils.anndata_utils import ordered_concat

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------- #
# Configuration Data Classes
# ---------------------------------------------------------------------------- #

@dataclass
class SimConfig:
    """General simulation parameters."""
    n_cells: int = 1000
    n_genes: int = 1000
    seed: int = 0
    non_neg: bool = False
    to_int: bool = False

@dataclass
class StateConfig:
    """Base configuration for cell state simulation."""
    state_type: str = 'cluster'
    distribution: str = 'normal'
    level: float = 1.0
    min_level: float = 0.0
    dispersion: float = 0.1

@dataclass
class ClusterConfig(StateConfig):
    """Parameters for cluster-based simulation."""
    n_states: int = 3
    program_structure: str = "linear"
    program_on_time_fraction: float = 0.3
    global_non_specific_gene_fraction: float = 0.1
    pairwise_non_specific_gene_fraction: Optional[Dict[Tuple[int, int], float]] = None

@dataclass
class TrajectoryConfig(StateConfig):
    """Parameters for trajectory-based simulation."""
    state_type: str = "trajectory"
    program_num: int = 3
    program_structure: str = "linear"
    program_on_time_fraction: float = 0.3
    cell_block_size_ratio: float = 0.3
    loop_to: Optional[Union[int, List[int]]] = None
    global_non_specific_gene_fraction: float = 0.0

@dataclass
class TreeConfig(StateConfig):
    """Parameters for tree-based simulation."""
    state_type: str = "tree"
    branching_factor: Union[int, List[int]] = 2
    depth: int = 3
    program_structure: str = "linear_increasing"
    program_on_time_fraction: float = 0.3
    program_gap_size: int = 1
    program_decay: float = 0.5
    cellcount_decay: float = 1.0
    noise_in_block: bool = True
    initial_inherited_genes: Optional[List[str]] = None

@dataclass
class BatchConfig:
    """Configuration for batch effects."""
    n_batches: int = 2
    effect_type: Union[str, List[str]] = 'batch_specific_features'
    distribution: Union[str, List[str]] = 'normal'
    level: Union[float, List[float]] = 1.0
    dispersion: Union[float, List[float]] = 0.1
    feature_frac: Union[float, List[float]] = 0.1
    cell_proportion: Optional[List[float]] = None

    def __post_init__(self):
        """Ensure all batch parameters are lists of the correct length."""
        self.effect_type = self._to_list(self.effect_type, self.n_batches)
        self.distribution = self._to_list(self.distribution, self.n_batches)
        self.level = self._to_list(self.level, self.n_batches)
        self.dispersion = self._to_list(self.dispersion, self.n_batches)
        self.feature_frac = self._to_list(self.feature_frac, self.n_batches)
        if self.cell_proportion is None:
            self.cell_proportion = [1 / self.n_batches] * self.n_batches
        elif len(self.cell_proportion) != self.n_batches:
            raise ValueError("Length of batch_cell_proportion must match n_batches.")

    @staticmethod
    def _to_list(value: Any, n: int) -> List:
        """Normalize a scalar or sequence to a list of length n."""
        if isinstance(value, list) and len(value) == n:
            return value
        if isinstance(value, (str, int, float)):
            return [value] * n
        raise ValueError(f"Cannot broadcast value to length {n}.")
    


class Simulation:
    """
    A refactored class for simulating single-cell gene expression data.
    """
    def __init__(self, sim_config: SimConfig, state_config: StateConfig, batch_config: BatchConfig):
        self.sim_config = sim_config
        self.state_config = state_config
        self.batch_config = batch_config
        self.rng = np.random.default_rng(self.sim_config.seed)

        # Dispatch maps for simulation logic
        self._state_sim_map = {
            "cluster": self._sim_state_cluster,
            "trajectory": self._sim_state_trajectory,
            "tree": self._sim_state_tree,
        }
        self._batch_effect_map = {
            "variance_inflation": self._be_variance_inflation,
            "batch_specific_distribution": self._be_batch_specific_distribution,
            "uniform_dropout": self._be_uniform_dropout,
            "value_dependent_dropout": self._be_value_dependent_dropout,
            "downsampling": self._be_downsampling,
            "scaling_factor": self._be_scaling_factor,
            "batch_specific_expression": self._be_batch_specific_expression,
            "batch_specific_features": self._be_batch_specific_features,
        }

    def simulate_data(self) -> Tuple[ad.AnnData, ad.AnnData]:
        """
        Simulates data by generating cell states and then applying batch effects.

        Returns:
            A tuple containing the final AnnData object and the pre-batch-effect AnnData object.
        """
        adata_state = self.simulate_state()

        batch_list, state_list = [], []
        for i in range(self.batch_config.n_batches):
            rng = np.random.default_rng(self.sim_config.seed + i)
        
            # Determine cells for this batch
            cell_proportion = self.batch_config.cell_proportion[i]
            n_cells = int(adata_state.n_obs * cell_proportion)
            #cell_indices = rng.choice(adata.n_obs, n_cells, replace=False)
            cell_indices = np.sort(rng.choice(adata_state.n_obs, n_cells, replace=False))
            batch_adata_pre = adata_state[cell_indices].copy()
            batch_adata = self.simulate_batch(
                batch_adata_pre,
                batch_idx=i
            )
            batch_list.append(batch_adata)
            state_list.append(batch_adata_pre)

        adata = self._finalize_anndata(batch_list, join='outer')
        adata_pre = self._finalize_anndata(state_list, join='outer')

        return adata, adata_pre

    def _finalize_anndata(self, adatas: List[ad.AnnData], join: str) -> ad.AnnData:
        """Concatenate, sort, and clean a list of AnnData objects."""
        from ..utils.other_util import sort_string_list # Assuming this utility exists

        adata = ad.concat(adatas, join=join, label='batch_id')
        adata.X = adata.X.toarray() if sp.issparse(adata.X) else adata.X
        adata.X = np.nan_to_num(adata.X, nan=0.0)

        # Sort genes for consistent ordering
        sorted_var_names = sort_string_list(adata.var_names)
        adata = adata[:, sorted_var_names].copy()

        # Make observation names unique
        adata.obs_names = [f"{b}_{c}" for b, c in zip(adata.obs['batch'], adata.obs_names)]
        adata.obs_names_make_unique()
        
        # Clean up layers and add counts
        for layer in adata.layers:
            adata.layers[layer] = np.nan_to_num(adata.layers[layer], nan=0.0)
        adata.layers['counts'] = adata.X.copy()
        
        return adata


    # ───────────────── STATE SIMULATION ───────────────────    
    def simulate_state(self) -> ad.AnnData:
        """Simulates the ground truth cell states based on the state_type."""
        try:
            sim_func = self._state_sim_map[self.state_config.state_type]
            adata = sim_func()
        except KeyError:
            raise ValueError(f"Unknown state_type '{self.state_config.state_type}'.")
        
        adata.X = np.nan_to_num(adata.X, nan=0.0)
        if self.sim_config.non_neg:
            adata.X[adata.X < 0] = 0
        if self.sim_config.to_int:
            adata.X = adata.X.astype(int)
        
        adata.layers['wt_noise'] = adata.X.copy()
        return adata

    def _sim_state_cluster(self) -> ad.AnnData:
        """Dispatcher for cluster simulation."""
        if not isinstance(self.state_config, ClusterConfig):
            raise TypeError("state_config must be of type ClusterConfig for clusters.")
        return self.simulate_clusters(self.state_config)

    def _sim_state_trajectory(self) -> ad.AnnData:
        """Dispatcher for trajectory simulation."""
        if not isinstance(self.state_config, TrajectoryConfig):
            raise TypeError("state_config must be of type TrajectoryConfig for trajectories.")
        return self.simulate_trajectory(self.state_config)

    def _sim_state_tree(self) -> ad.AnnData:
        """Dispatcher for tree simulation."""
        if not isinstance(self.state_config, TreeConfig):
            raise TypeError("state_config must be of type TreeConfig for trees.")
        return self.simulate_tree(self.state_config)



    def simulate_clusters(self, cfg: ClusterConfig) -> ad.AnnData:
        rng = self.rng
        n_cells  = self.sim_config.n_cells
        n_genes  = self.sim_config.n_genes
        k        = cfg.n_states
        
        # Check if n_genes is a list or integer
        if isinstance(n_genes, list):
            if len(n_genes) != k:
                raise ValueError("Length of n_genes list must match num_clusters.")
            genes_per_cluster_list = n_genes
        else:
            genes_per_cluster = n_genes // k
            genes_per_cluster_list = [genes_per_cluster] * k
        
        # Check if n_cells is a list or integer
        if isinstance(n_cells, list):
            if len(n_cells) != k:
                raise ValueError("Length of n_cells list must match num_clusters.")
            cells_per_cluster_list = n_cells
        else:
            cells_per_cluster = n_cells // k
            cells_per_cluster_list = [cells_per_cluster] * k
        
        total_n_genes = sum(genes_per_cluster_list)
        total_n_cells = sum(cells_per_cluster_list)
        X = np.zeros((total_n_cells, total_n_genes))
        cell_clusters = []
        gene_offset = 0  # Tracks the starting gene index for each cluster
        cell_offset = 0  # Tracks the starting cell index for each cluster

        cluster_gene_indices = {}
        cluster_cell_indices = {}
        for cluster in range(k):
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
            X = self.simulate_expression_block(
                X, cfg.program_structure, gene_indices, cell_indices, 
                cfg.level, 
                cfg.min_level,
                cfg.program_on_time_fraction,
            )
            
            cell_clusters.extend([f"cluster_{cluster+1}"] * len(cell_indices))

        # Add non-specific genes to the expression matrix

        if cfg.pairwise_non_specific_gene_fraction is not None:
            logger.info("Adding non-specific genes to the expression matrix. Note this will increase gene count compared to the specified value.")
            for (c1, c2), frac in cfg.pairwise_non_specific_gene_fraction.items():
                union_genes = np.union1d(cluster_gene_indices[c1],
                                        cluster_gene_indices[c2])
                n_extra = int(len(union_genes) * frac)
                extra_g = rng.choice(union_genes, n_extra, replace=False)
                union_c = np.union1d(cluster_cell_indices[c1],
                                    cluster_cell_indices[c2])
                X = self.simulate_expression_block(
                    X, cfg.program_structure, extra_g, union_c,
                    cfg.level, cfg.min_level, cfg.program_on_time_fraction
                )
            
        if cfg.global_non_specific_gene_fraction:
            n_extra = int(X.shape[1] * cfg.global_non_specific_gene_fraction)
            extra_g = rng.choice(X.shape[1], n_extra, replace=False)
            X = self.simulate_expression_block(
                X, cfg.program_structure, extra_g, np.arange(total_n_cells),
                cfg.level, cfg.min_level, cfg.program_on_time_fraction
            )
                
        # Apply distribution to simulate realistic expression values
        # ------------------ add noise -------------------
        if np.isscalar(cfg.dispersion):
            X_w = self.simulate_distribution(cfg.distribution, X, cfg.dispersion)
        else:
            X_w = X.copy()
            for cluster, disp in enumerate(cfg.dispersion):
                X_w[cluster_cell_indices[cluster], :] = self.simulate_distribution(
                    cfg.distribution,
                    X[cluster_cell_indices[cluster], :],
                    disp,
                )

        obs = pd.DataFrame(
            {"cluster": cell_clusters},
            index=[f"Cell_{i+1}" for i in range(X.shape[0])]
        )
        var = pd.DataFrame(index=[f"Gene_{i+1}" for i in range(X.shape[1])])

        adata = ad.AnnData(X=X_w, obs=obs, var=var)
        adata.layers["no_noise"] = X

        return adata
        



    def simulate_trajectory(self, cfg: TrajectoryConfig) -> ad.AnnData:
        """
        Simulate a 1-D pseudotime trajectory with `program_num` expression
        programs that successively turn on (or loop back, if `cfg.loop_to`
        is given).  The code is a direct transliteration of your earlier
        implementation, but all free parameters now come from

            * self.sim_config … global simulation settings
            * cfg              … trajectory-specific settings
        """
        rng       = self.rng
        n_cells   = self.sim_config.n_cells
        n_genes   = self.sim_config.n_genes
        prog_num  = cfg.program_num

        # sizing helpers -------------------------------------------------
        blk_size  = int(n_cells * cfg.cell_block_size_ratio)
        prog_size = n_genes // prog_num + (n_genes % prog_num > 0)
        nc_sim    = n_cells + blk_size             # padding on both ends

        X = np.zeros((nc_sim, n_genes))

        # how many cell groups along the trajectory?
        if cfg.loop_to is None:
            cell_group_num = prog_num
            loop_map = {}
        else:
            loop_vec = [cfg.loop_to] if isinstance(cfg.loop_to, int) else cfg.loop_to
            assert max(loop_vec) < prog_num - 1, "`loop_to` index out of range."
            cell_group_num = prog_num + len(loop_vec)
            # map suffix groups → which program they reuse
            loop_map = {prog_num + i: loop_vec[i] for i in range(len(loop_vec))}

        gap = n_cells / (cell_group_num - 1)

        # build expression block by block --------------------------------
        for g in range(cell_group_num):
            # which gene slice is active for this block?
            p_idx = loop_map.get(g, g)             # loop groups reuse program
            g_idx = np.arange(p_idx * prog_size,
                            min((p_idx + 1) * prog_size, n_genes))

            if g_idx.size == 0:
                continue

            c_start = int(g * gap)
            c_end   = min(c_start + blk_size, nc_sim)
            c_idx   = np.arange(c_start, c_end)

            X = self.simulate_expression_block(
                X,
                cfg.program_structure,
                g_idx,
                c_idx,
                cfg.level,
                cfg.min_level,
                cfg.program_on_time_fraction,
            )

        # centre crop back to `n_cells`
        X = X[blk_size // 2 : blk_size // 2 + n_cells, :]

        if cfg.global_non_specific_gene_fraction:
            n_extra = int(X.shape[1] * cfg.global_non_specific_gene_fraction)
            extra_g = rng.choice(X.shape[1], n_extra, replace=False)
            X = self.simulate_expression_block(
                X, cfg.program_structure, extra_g, np.arange(X.shape[0]),
                cfg.level, cfg.min_level, 1.0,  # global genes are always on
            )

        # add stochastic noise ------------------------------------------
        X_w = self.simulate_distribution(cfg.distribution, X, cfg.dispersion)

        # build AnnData --------------------------------------------------
        obs = pd.DataFrame(
            {"time": np.arange(n_cells)},
            index=[f"Cell_{i+1}" for i in range(n_cells)],
        )
        var = pd.DataFrame(index=[f"Gene_{i+1}" for i in range(n_genes)])

        adata = ad.AnnData(X=X_w, obs=obs, var=var)
        adata.layers["no_noise"] = X

        return adata



    def simulate_tree(self, cfg: TreeConfig) -> ad.AnnData:
        """
        Simulate a hierarchical branching process (gene-expression tree).

        All numeric parameters are taken from `self.sim_config`
        (n_cells, n_genes, global seed) and from `cfg` (tree settings).
        The core algorithm is the same as in the original implementation.
        """
        # -------- shorthand handles ----------------------------------
        rng        = self.rng                          # already seeded
        n_cells    = self.sim_config.n_cells
        n_genes    = self.sim_config.n_genes
        depth      = cfg.depth
        bf         = cfg.branching_factor              # branching factor(s)

        # -------- normalise branching-factor vector ------------------
        if isinstance(bf, list):
            if len(bf) != depth:
                raise ValueError("Length of branching_factor must equal depth")
            bf_vec = bf
        else:
            bf_vec = [bf] * depth

        # -------- helper vectors for #genes / #cells per level -------
        # programme/gene decay
        base_g  = n_genes // (bf_vec[0] ** depth)
        g_per   = [max(int(base_g  * cfg.program_decay  ** d), 1) for d in range(depth + 1)]
        # cell-count decay
        base_c  = n_cells // (bf_vec[0] ** depth)
        c_per   = [max(int(base_c  * cfg.cellcount_decay ** d), 1) for d in range(depth + 1)]

        # -------- counters so that gene / cell names are unique ------
        gene_ctr = 0
        cell_ctr = 0

        def _branch(level: int,
                    path: list[int],
                    inherited_genes: list[str] | None,
                    start_time: int) -> ad.AnnData:
            nonlocal gene_ctr, cell_ctr

            branch_tag = "_".join(map(str, path)) if path else "root"
            n_g = g_per[depth - level]
            n_c = c_per[depth - level]

            # ---- gene & cell labels for this branch -----------------
            genes = [f"gene_{gene_ctr + i}" for i in range(n_g)]
            cells = [f"cell_{cell_ctr + i}" for i in range(n_c)]
            gene_ctr += n_g
            cell_ctr += n_c

            # ---- simulate expression for branch-specific genes ------
            X = self.simulate_expression_block(
                np.zeros((n_c, n_g)),
                cfg.program_structure,
                np.arange(n_g),
                np.arange(n_c),
                cfg.level,
                cfg.min_level,
                cfg.program_on_time_fraction,
                gap_size=cfg.program_gap_size,
            )

            # ---- append inherited genes (constant expression) -------
            if inherited_genes:
                inh_block = np.full((n_c, len(inherited_genes)), cfg.level)
                X = np.hstack([inh_block, X])
                genes = inherited_genes + genes

            # ---- build AnnData for this branch ----------------------
            adata = sc.AnnData(X)
            adata.obs_names = cells
            adata.var_names = genes
            adata.obs["branch"] = branch_tag
            adata.obs["depth"]  = level
            adata.obs["time"]   = np.arange(n_c) + start_time
            adata.layers["no_noise"] = X.copy()

            # optional within-block noise
            if cfg.noise_in_block:
                adata.X = self.simulate_distribution(
                    cfg.distribution, X, cfg.dispersion
                )

            # ---- recurse if we’re not at a leaf ---------------------
            if level > 0:
                next_start = start_time + n_c
                for i in range(bf_vec[depth - level]):
                    child = _branch(
                        level - 1,
                        path + [i],
                        inherited_genes=genes,
                        start_time=next_start,
                    )
                    adata = ordered_concat([adata, child], join="outer")

            return adata

        # -------- kick off recursion from the root -------------------
        adata = _branch(depth, [], cfg.initial_inherited_genes, start_time=0)

        # -------- post-processing ------------------------------------
        adata.X = np.nan_to_num(adata.X, nan=0.0)
        for key in adata.layers:
            adata.layers[key] = np.nan_to_num(adata.layers[key], nan=0.0)

        if not cfg.noise_in_block:
            adata.X = self.simulate_distribution(
                cfg.distribution, adata.layers["no_noise"], cfg.dispersion
            )

        return adata



    # ────────────────── BATCH SIMULATION ───────────────────

    def simulate_batch(self, adata: ad.AnnData, batch_idx: int) -> Tuple[ad.AnnData, ad.AnnData]:
        """Applies a batch-specific effect to a subset of data."""

        adata.obs['batch'] = f"batch_{batch_idx+1}"
        batch_adata = adata.copy()
        
        # Apply the corresponding batch effect
        effect_type = self.batch_config.effect_type[batch_idx]
        try:
            effect_fn = self._batch_effect_map[effect_type]
            effect_params = {
                "distribution": self.batch_config.distribution[batch_idx],
                "level": self.batch_config.level[batch_idx],
                "dispersion": self.batch_config.dispersion[batch_idx],
                "batch_feature_frac": self.batch_config.feature_frac[batch_idx],
                "batch_name": f"batch_{batch_idx+1}",
                "rng": self.rng,
            }
            result = effect_fn(batch_adata, **effect_params)
            if isinstance(result, ad.AnnData):
                batch_adata = result # Use the new object if one was returned
        except KeyError:
            raise ValueError(f"Unknown batch effect type '{effect_type}'")

        if self.sim_config.non_neg:
            batch_adata.X[batch_adata.X < 0] = 0
        if self.sim_config.to_int:
            batch_adata.X = batch_adata.X.astype(int)
            
        return batch_adata

    def _be_variance_inflation(self, adata, *, level, rng, **_):
        """Multiply each entry by 1 + N(0,σ²)."""
        scale = 1 + rng.normal(0, level, adata.shape).reshape(adata.n_obs, adata.n_vars)
        adata.X = adata.X.toarray() * scale if sp.issparse(adata.X) else adata.X * scale

    def _be_batch_specific_distribution(self, adata, *, distribution, level, dispersion, rng, **_):
        """Add (or otherwise combine) a noise matrix drawn from `distribution`."""
        adata.X = adata.X.astype(float)
        adata.X += self.simulate_distribution(distribution, level, dispersion, adata.X.shape)

    def _be_uniform_dropout(self, adata, *, level, rng, **_):
        """Randomly zero out a fixed fraction (`level`) of values."""
        mask = rng.random(adata.shape) < level
        adata.X = adata.X.toarray() if sp.issparse(adata.X) else adata.X
        adata.X[mask] = 0

    def _be_value_dependent_dropout(self, adata, *, level, rng, **_):
        """Probability 1‑exp(−λ·x²) for each value x (λ=`level`)."""
        adata.X[adata.X < 0] = 0
        mtx = adata.X.toarray() if sp.issparse(adata.X) else adata.X
        probs = 1 - np.exp(-level * np.square(mtx))
        mask = rng.random(mtx.shape) < probs
        mtx[mask] = 0
        adata.X = mtx

    def _be_downsampling(self, adata, *, level, rng, **_):
        """Randomly subsample counts to a fraction `level`."""
        adata.X[adata.X < 0] = 0
        dense = adata.X.toarray() if sp.issparse(adata.X) else adata.X
        adata.X = self.downsample_mtx_umi(dense.astype(int), ratio=level, seed=rng.integers(1e9))


    def _be_scaling_factor(self, adata, *, level, **_):
        """Multiply the whole matrix by a scalar `level`."""
        adata.X = adata.X @ sparse.diags(level) if sp.issparse(adata.X) else adata.X * level


    def _be_batch_specific_expression(self, adata, *, distribution, level, dispersion,
                                    batch_feature_frac, rng, **_):
        """Add noise to a random subset of genes (in‑place)."""
        n_genes = int(batch_feature_frac * adata.n_vars)
        idx = rng.choice(adata.n_vars, n_genes, replace=False)
        adata.X = adata.X.astype(float)
        adata.X[:, idx] += self.simulate_distribution(distribution, level, dispersion,
                                                    (adata.n_obs, n_genes))


    def _be_batch_specific_features(
        self,
        adata,
        *,
        distribution,
        level,
        dispersion,
        batch_feature_frac,
        batch_name,
        rng,
        **_,
    ):
        """Append brand‑new genes that exist only in this batch.

        Returns a fresh AnnData; callers must replace the original object.
        """
        import pandas as pd
        import numpy as np
        import scipy.sparse as sp

        # how many new features?
        n_new = int(batch_feature_frac * adata.n_vars)
        if n_new == 0:
            return adata            # nothing to do

        # new expression block and gene names
        new_X   = self.simulate_distribution(
            distribution, level, dispersion, (adata.n_obs, n_new)
        )
        new_var = pd.DataFrame(
            index=[f"{batch_name}_Gene_{i+1}" for i in range(n_new)]
        )

        base_zeros = np.zeros_like(new_X)
        lay_no_noise = np.hstack([adata.layers['no_noise'], base_zeros])
        lay_wt_noise = np.hstack([adata.layers['wt_noise'], base_zeros])

        if sp.issparse(adata.X):
            full_X = sp.hstack([adata.X, sp.csr_matrix(new_X)])
        else:
            full_X = np.hstack([np.asarray(adata.X), new_X])

        return ad.AnnData(
            X      = full_X,
            obs    = adata.obs.copy(),
            var    = pd.concat([adata.var, new_var]),
            layers = {"no_noise": lay_no_noise, "wt_noise": lay_wt_noise},
        )




    # ────────────────── HELPER METHODS ───────────────────
    def simulate_expression_block(self, expression_matrix, structure, gene_idx, cell_idx, mean_expression, min_expression, on_time_fraction = 0.3, gap_size=None):
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
            if gap_size is None:
                gap_size = max((ncells - program_on_time) // len(gene_idx), 1)
            #print("ncells", ncells, "len(gene_idx)", len(gene_idx), "gap_size", gap_size)
            # Simulate a gene program that has each of its genes gradually turning on
            for i, gene in enumerate(gene_idx):
                cur_gene_start = min(cell_start + i * gap_size, cell_end)
                #print("cur_gene_start", cur_gene_start, "transition_end", transition_end, "cell_end", cell_end, "gene", gene)
                if cur_gene_start < transition_end:
                    expression_matrix[cur_gene_start:transition_end, gene] = np.linspace(min_expression, mean_expression, transition_end-cur_gene_start)
                expression_matrix[transition_end:cell_end, gene] = mean_expression
        elif structure == "uniform":
            expression_matrix[cell_start:cell_end, gene_idx] = mean_expression
        else:
            raise ValueError(f"Unknown structure '{structure}'.")        

        return expression_matrix
    

    @staticmethod
    def downsample_mtx_umi(mtx, ratio=0.1, seed=1):
        """
        Simulates downsampling of a gene expression matrix (UMI counts) by a given ratio.

        Args:
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
    def simulate_distribution(distribution, mean, dispersion, size=None, nonzero_only=False):
        """
        Samples values from a specified distribution.

        Args:
            distribution (str): The type of distribution ('normal', 'poisson', etc.).
            mean (np.ndarray): The mean values for the distribution.
            dispersion (float): The dispersion or standard deviation.
            size (tuple, optional): The shape of the output matrix. Defaults to mean.shape.
            nonzero_only (bool, optional): If True, noise/dispersion is only applied
                                           to non-zero elements of the mean matrix.
                                           Defaults to False.
        Returns:
            np.ndarray: The matrix with simulated values.
        """
        if size is None:
            size = mean.shape

        if not nonzero_only:
            # --- ORIGINAL BEHAVIOR: Apply noise to the entire matrix ---
            if distribution == "normal":
                return mean + np.random.normal(0, dispersion, size)
            elif distribution == 'poisson':
                # Ensure lambda for Poisson is non-negative
                return np.random.poisson(np.maximum(0, mean), size)
            elif distribution == 'negative_binomial':
                return Simulation.rnegbin(np.maximum(0, mean), dispersion, size)
            elif distribution == 'lognormal':
                 # Ensure base for lognormal is positive
                return np.random.lognormal(np.log(np.maximum(1e-9, mean)), dispersion, size)
            else:
                raise ValueError(f"Unknown distribution '{distribution}'.")

        else:
            # --- NEW BEHAVIOR: Apply noise only to non-zero mean values ---
            # Create a boolean mask of non-zero elements
            mask = mean > 0
            
            if distribution == "normal":
                # For additive noise, start with the original mean matrix
                result = mean.copy().astype(float)
                # Generate noise for only the non-zero elements
                noise = np.random.normal(0, dispersion, size=np.sum(mask))
                # Add the noise to the corresponding non-zero elements
                result[mask] += noise
                return result

            elif distribution in ['poisson', 'negative_binomial', 'lognormal']:
                # For generative distributions, start with a zero matrix
                result = np.zeros_like(mean, dtype=float)
                # Get the mean values for only the non-zero elements
                nonzero_means = mean[mask]

                if distribution == 'poisson':
                    sampled_values = np.random.poisson(np.maximum(0, nonzero_means))
                elif distribution == 'negative_binomial':
                    sampled_values = Simulation.rnegbin(np.maximum(0, nonzero_means), dispersion)
                elif distribution == 'lognormal':
                    sampled_values = np.random.lognormal(np.log(np.maximum(1e-9, nonzero_means)), dispersion)
                
                # Place the newly sampled values back into the result matrix at the correct positions
                result[mask] = sampled_values
                return result
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


