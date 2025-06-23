"""
Neighbour-based evaluation for frozen embeddings (no scaling).

* Computes k-NN accuracy / R² / MAE on the **raw latent space**.
* Supports classification or regression (auto-detected).
* `ignore_values` lets you drop cells whose label is NA/unknown/-1, etc.
* Optionally stores per-cell predictions.

Example
-------
from concord.benchmarking.knn_probe import KNNProbeEvaluator

knn_eval = KNNProbeEvaluator(
    adata         = adata,
    emb_keys      = ["Concord", "scVI_latent", "X_pca"],
    target_key    = "cell_type",
    k             = 15,
    metric        = "euclidean",   # or "cosine"
    ignore_values = ["NA", "unknown"],
    return_preds  = True,
    seed          = 0,
)
metrics_df, preds_bank = knn_eval.run()
print(metrics_df)
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import List, Dict, Any, Literal, Sequence

import numpy as np
import pandas as pd
from anndata import AnnData

@dataclass
class KNNProbeEvaluator:
    adata: AnnData
    emb_keys: List[str]
    target_key: str

    # evaluation hyper-params
    k: int = 20
    metric: Literal["euclidean", "cosine"] = "euclidean"
    task: Literal["classification", "regression", "auto"] = "auto"
    val_frac: float = 0.2
    ignore_values: Sequence[Any] | None = None

    # extras
    return_preds: bool = False
    seed: int = 0

    # internal bookkeeping
    _history: List[Dict[str, Any]] = field(default_factory=list, init=False)
    _pred_bank: Dict[str, pd.DataFrame] = field(default_factory=dict, init=False)

    # ------------------------------------------------------------------
    def run(self):
        from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
        from sklearn.preprocessing import LabelEncoder
        from sklearn.metrics import accuracy_score, r2_score, mean_absolute_error
        from sklearn.model_selection import train_test_split


        rng = np.random.RandomState(self.seed)

        y_raw = self.adata.obs[self.target_key].to_numpy()
        obs_names = np.asarray(self.adata.obs_names)

        # ----- drop unwanted labels -----------------------------------
        if self.ignore_values is not None:
            mask_keep = ~np.isin(y_raw, list(self.ignore_values))
            if any(isinstance(v, float) and np.isnan(v) for v in self.ignore_values):
                mask_keep &= ~np.isnan(y_raw.astype("float64", copy=False))
            if mask_keep.sum() == 0:
                raise ValueError("All samples were filtered out by ignore_values.")
            y_raw = y_raw[mask_keep]
            obs_names = obs_names[mask_keep]

        # determine task type ------------------------------------------
        if self.task == "auto":
            self.task = "regression" if np.issubdtype(y_raw.dtype, np.number) else "classification"

        # label processing ---------------------------------------------
        if self.task == "classification":
            enc = LabelEncoder().fit(y_raw)
            y_all = enc.transform(y_raw)
        else:
            y_all = y_raw.astype(np.float32)
            mu, sigma = y_all.mean(), y_all.std()
            if sigma == 0:
                raise ValueError("Target has zero variance; R² undefined.")
            y_all_std = (y_all - mu) / sigma  # only used internally

        # -------- evaluate each embedding -----------------------------
        for key in self.emb_keys:
            X_all = self.adata.obsm[key]
            if self.ignore_values is not None:
                X_all = X_all[mask_keep]

            # train/val split (raw coordinates)
            X_tr, X_val, y_tr, y_val, idx_tr, idx_val = train_test_split(
                X_all,
                y_all_std if self.task == "regression" else y_all,
                np.arange(len(y_all)),
                test_size=self.val_frac,
                random_state=rng,
                stratify=None,
            )

            if self.task == "classification":
                model = KNeighborsClassifier(
                    n_neighbors=self.k, metric=self.metric, weights="distance"
                )
                model.fit(X_tr, y_tr)
                y_pred = model.predict(X_val)
                metric_dict = {
                    "embedding": key,
                    "accuracy": accuracy_score(y_val, y_pred),
                }
                y_pred_store = y_pred
                y_val_store = y_val
            else:
                model = KNeighborsRegressor(
                    n_neighbors=self.k, metric=self.metric, weights="distance"
                )
                model.fit(X_tr, y_tr)
                y_pred = model.predict(X_val)
                # de-standardise for metrics
                y_pred_orig = y_pred * sigma + mu
                y_val_orig = y_val * sigma + mu
                metric_dict = {
                    "embedding": key,
                    "r2":  r2_score(y_val_orig, y_pred_orig),
                    "mae": mean_absolute_error(y_val_orig, y_pred_orig),
                }
                y_pred_store = y_pred_orig
                y_val_store = y_val_orig

            self._history.append(metric_dict)

            if self.return_preds:
                self._pred_bank[key] = pd.DataFrame(
                    {"y_true": y_val_store, "y_pred": y_pred_store},
                    index=obs_names[idx_val],
                )

        metrics_df = pd.DataFrame(self._history).set_index("embedding")
        if self.return_preds:
            return metrics_df, self._pred_bank
        return metrics_df

    # ------------------------------------------------------------------
    def get_preds(self, key: str) -> pd.DataFrame:
        if key not in self._pred_bank:
            raise KeyError("Run .run(return_preds=True) first.")
        return self._pred_bank[key]

