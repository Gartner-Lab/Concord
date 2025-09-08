
from __future__ import annotations
import numpy as np
import anndata as ad
from dataclasses import dataclass
from typing import Sequence, List, Dict, Any
from .simulation import Simulation, SimConfig, ClusterConfig, BatchConfig
from ..utils.anndata_utils import ordered_concat

def _random_pairwise_shared(
    n_clusters: int,
    rng: np.random.Generator,
    pair_prob: float       = 0.5,     # chance that a pair shares genes
    frac_low:  float       = 0.0,     # min shared‑gene fraction
    frac_high: float       = 0.9,     # max shared‑gene fraction
) -> Dict[tuple[int, int], float]:
    """
    For every unordered cluster pair (i<j) draw:
        with prob = pair_prob   →   fraction ~ U(frac_low, frac_high)
        else                    →   0   (pair omitted from dict)
    """
    d: Dict[tuple[int, int], float] = {}
    for i in range(n_clusters):
        for j in range(i + 1, n_clusters):
            if rng.random() < pair_prob:
                d[(i, j)] = float(rng.uniform(frac_low, frac_high))
    # ensure at least *one* pair shares genes
    if not d:
        d[(0, 1)] = float((frac_low + frac_high) / 2)
    return d


def simulate_from_state(
    *,
    # -------- state‑level settings --------
    n_cells_per_cluster: Sequence[int],
    n_genes_per_cluster: Sequence[int],
    state_dispersion: Sequence[float],
    program_structure: str = "uniform",
    mean_level: float = 10.0,
    distribution: str = "normal",
    global_nonspecific: float = 0.05,
    pairwise_non_specific: Dict[tuple[int,int], float] | None = None,
    seed_state: int = 0,
    # -------- batch‑level settings --------
    n_batches: int,
    clusters_per_batch: int | Sequence[int] = 3,
    batch_cell_fraction: float | Sequence[float] = 0.30,
    batch_effect_type: str = "batch_specific_features",
    batch_feature_frac: float = 0.05,
    batch_dispersion: float = 3.0,
    seed_batches: int | None = None,
) -> Dict[str, ad.AnnData]:
    """
    1.  Simulate a cluster manifold once.
    2.  Sample cells to create `n_batches` partially‑overlapping batches.
    3.  Apply a batch effect to each subset.
    4.  Return both the post‑batch (`adata`) and pre‑batch (`adata_pre`) data.

    Returns
    -------
    dict with keys "adata", "adata_pre", "adata_state"
    """
    rng_state   = np.random.default_rng(seed_state)
    rng_batches = np.random.default_rng(seed_batches or (seed_state + 12345))

    n_clusters  = len(n_cells_per_cluster)
    assert len(n_genes_per_cluster) == n_clusters
    assert len(state_dispersion)    == n_clusters

    # ----- 1) state simulation --------------------------------------
    sim_cfg = SimConfig(
        n_cells = n_cells_per_cluster,
        n_genes = n_genes_per_cluster,
        seed    = seed_state,
        non_neg = True,
        to_int  = True,
    )
    state_cfg = ClusterConfig(
        n_states                       = n_clusters,
        program_structure              = program_structure,
        program_on_time_fraction       = 0.3,
        global_non_specific_gene_fraction = global_nonspecific,
        pairwise_non_specific_gene_fraction = pairwise_non_specific,
        distribution                   = distribution,
        dispersion                     = state_dispersion,
        level                          = mean_level,
    )
    dummy_batch = BatchConfig(n_batches=1)
    sim_state   = Simulation(sim_cfg, state_cfg, dummy_batch)
    adata_state = sim_state.simulate_state()

    # numeric cluster id for quick masking
    adata_state.obs["clust_id"] = (
        adata_state.obs["cluster"].str.extract(r"(\d+)").astype(int).squeeze() - 1
    )

    # ----- 2) decide clusters and cells per batch -------------------
    # clusters_per_batch → list[int]
    if isinstance(clusters_per_batch, int):
        clusters_per_batch = [clusters_per_batch] * n_batches
    # cell fractions   → list[float]
    if isinstance(batch_cell_fraction, float):
        batch_cell_fraction = [batch_cell_fraction] * n_batches

    batch_indices: List[np.ndarray] = []
    for b in range(n_batches):
        # choose K clusters for this batch (1‑based label)
        pick_K = clusters_per_batch[b]
        clusters_here = rng_batches.choice(
            n_clusters, size=pick_K, replace=False
        )
        # candidate pool
        pool = np.where(adata_state.obs["clust_id"].isin(clusters_here))[0]
        take = int(round(batch_cell_fraction[b] * adata_state.n_obs))
        take = min(take, len(pool))          # guard against oversampling
        cells = rng_batches.choice(pool, size=take, replace=False)
        batch_indices.append(np.sort(cells))

    # ----- 3) simulate batch effect on each subset ------------------
    batch_cfg = BatchConfig(
        n_batches     = n_batches,
        effect_type   = [batch_effect_type] * n_batches,
        distribution  = distribution,
        level         = mean_level,
        dispersion    = batch_dispersion,
        feature_frac  = batch_feature_frac,
        cell_proportion = batch_cell_fraction,
    )
    sim_full = Simulation(sim_cfg, state_cfg, batch_cfg)   # reuse configs

    batch_list, pre_list = [], []
    for b, idx in enumerate(batch_indices):
        ad_pre = adata_state[idx].copy()
        ad_pre.obs["batch"] = f"batch_{b+1}"
        ad_post = sim_full.simulate_batch(ad_pre.copy(), batch_idx=b)
        batch_list.append(ad_post)
        pre_list.append(ad_pre)

    # ----- 4) concatenate & finish ----------------------------------
    adata     = sim_full._finalize_anndata(batch_list, join="outer")
    adata_pre = sim_full._finalize_anndata(pre_list,  join="outer")

    return dict(adata=adata, adata_pre=adata_pre, adata_state=adata_state)



# ───────────────────── level specification ──────────────────────
@dataclass
class LevelSpec:
    n_batches:        int
    alpha_dirichlet:  float
    sigma_cluster:    float          # log‑normal σ for cluster‑size skew
    clusters_per_batch: int | None = None  # None → auto (≤10 → 3 else 2)



# ────────────────── imbalanced dataset simulation ───────────────────
def simulate_imbalanced_datasets(
    *,
    total_cells:       int = 10_000,
    total_genes:       int = 4_000,
    n_clusters:        int = 6,
    levels: List[LevelSpec] | None = None,
    mean_level:        float = 10.0,
    distribution:      str   = "normal",
    state_dispersion:  float = 5.0,
    batch_feature_frac: float = 0.10,
    batch_dispersion:  float = 3.0,
    pair_prob: float = 0.5,
    pair_frac_low:  float = 0.0,
    pair_frac_high: float = 0.9,
    seed:              int   = 123,
) -> Dict[str, ad.AnnData]:
    """
    Generate increasingly imbalanced datasets (phase 2) in one call.
    Returns a dict { '<n>_batches': AnnData }  (post‑batch, with counts layer).
    """
    # default 3‑step ladder
    if levels is None:
        levels = [
            LevelSpec( 2, 5.0, 0.3),
            LevelSpec(10, 1.0, 0.8),
            LevelSpec(30, 0.3, 1.2),
        ]

    rng            = np.random.default_rng(seed)
    datasets       = {}

    # reusable per‑cluster gene counts (equal split)
    genes_per_clust = [total_genes // n_clusters] * n_clusters
    genes_per_clust[-1] += total_genes - sum(genes_per_clust)

    # helper: draw imbalanced cluster sizes
    def _draw_cluster_sizes(sigma: float) -> List[int]:
        raw = rng.lognormal(0.0, sigma, n_clusters)
        sizes = np.floor(raw / raw.sum() * total_cells).astype(int)
        sizes[-1] += total_cells - sizes.sum()
        return sizes.tolist()

    for lvl_idx, lvl in enumerate(levels):
        # --- 1) sample cluster sizes for this level -----------------
        n_cells = _draw_cluster_sizes(lvl.sigma_cluster)

        # --- 2) sample batch proportions ----------------------------
        props = rng.dirichlet([lvl.alpha_dirichlet] * lvl.n_batches).tolist()

        # --- 3) decide #clusters each batch will cover --------------
        if lvl.clusters_per_batch is None:                # ← use your new rule
            k_per_list = rng.integers(1, n_clusters + 1, size=lvl.n_batches).tolist()
        elif isinstance(lvl.clusters_per_batch, int):     # fixed K for all batches
            k_per_list = [lvl.clusters_per_batch] * lvl.n_batches
        else:                                             # tuple (min,max) → uniform
            k_min, k_max = lvl.clusters_per_batch
            k_per_list = rng.integers(k_min, k_max + 1, size=lvl.n_batches).tolist()


        pairwise_dict = _random_pairwise_shared(n_clusters, rng, 
            pair_prob=pair_prob, frac_low=pair_frac_low, frac_high=pair_frac_high)
        # --- 4) call simulate_from_state ----------------------------
        sim_out = simulate_from_state(
            n_cells_per_cluster = n_cells,
            n_genes_per_cluster = genes_per_clust,
            state_dispersion    = [state_dispersion] * n_clusters,
            program_structure   = "uniform",
            mean_level          = mean_level,
            distribution        = distribution,
            global_nonspecific  = 0.05,
            pairwise_non_specific = pairwise_dict,
            seed_state          = seed + 10 + lvl_idx,
            # batch settings
            n_batches           = lvl.n_batches,
            clusters_per_batch  = k_per_list,
            batch_cell_fraction = props,
            batch_effect_type   = "batch_specific_features",
            batch_feature_frac  = batch_feature_frac,
            batch_dispersion    = batch_dispersion,
            seed_batches        = seed + 1000 + lvl_idx,
        )

        adata = sim_out["adata"]
        label = f"{lvl.n_batches}_batches"
        adata.obs["imbalanced_level"] = label
        datasets[label] = adata

        print(f"{label:>12} | cells={adata.n_obs:,}  genes={adata.n_vars:,}")

    return datasets
