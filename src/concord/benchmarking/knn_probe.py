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
    predict_all: bool = False  # whether to predict on all cells or only validation set
    seed: int = 0

    # bookkeeping
    _history: List[Dict[str, Any]] = field(default_factory=list, init=False)
    _pred_bank: Dict[str, pd.DataFrame] = field(default_factory=dict, init=False)

    # ------------------------------------------------------------------
    def run(self):
        from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
        from sklearn.preprocessing import LabelEncoder
        from sklearn.metrics import (
            accuracy_score,
            r2_score,
            mean_absolute_error,
        )
        from sklearn.model_selection import train_test_split
        from pandas.api.types import is_numeric_dtype

        rng = np.random.RandomState(self.seed)

        # --------------- 0) filter unwanted rows ---------------------
        y_raw_full = self.adata.obs[self.target_key].to_numpy()  # keep native dtype
        obs_names  = np.asarray(self.adata.obs_names)

        # --- drop unwanted / missing labels ---------------------------------
        mask_keep = ~pd.isna(y_raw_full)

        if self.ignore_values is not None:
            # case-insensitive match for strings but works for numbers too
            ignore_vals = [
                v.lower() if isinstance(v, str) else v
                for v in self.ignore_values
            ]
            y_cmp = np.array(
                [v.lower() if isinstance(v, str) else v for v in y_raw_full],
                dtype=object,
            )
            mask_keep &= ~np.isin(y_cmp, ignore_vals)

        if mask_keep.sum() == 0:
            raise ValueError("All samples were filtered out by ignore_values / NaNs.")

        y_raw = y_raw_full[mask_keep]
        obs_names = obs_names[mask_keep]

        # --------------- 1) decide task ------------------------------
        if self.task == "auto":
            self.task = "regression" if is_numeric_dtype(y_raw) else "classification"

        print(f"[KNN] detected task: {self.task}")

        # --------------- 2) target preprocessing ---------------------
        if self.task == "classification":
            enc = LabelEncoder().fit(y_raw)
            y_all = enc.transform(y_raw).astype(np.int64)
            mu, sigma = 0.0, 1.0  # placeholders, never used
        else:
            y_all = y_raw.astype(np.float32)
            mu, sigma = y_all.mean(), y_all.std()
            if sigma == 0:
                raise ValueError("Target has zero variance; R² undefined.")
            y_all = (y_all - mu) / sigma  # standardised for fitting

        # --------------- 3) evaluate each embedding -----------------
        for key in self.emb_keys:
            X_all = self.adata.obsm[key][mask_keep]

            # train / val split
            X_tr, X_val, y_tr, y_val, idx_tr, idx_val = train_test_split(
                X_all,
                y_all,
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

                # y_pred_store = y_pred
                # y_val_store = y_val
                y_pred_store = enc.inverse_transform(y_pred)
                y_val_store  = enc.inverse_transform(y_val)

            else:  # regression
                model = KNeighborsRegressor(
                    n_neighbors=self.k, metric=self.metric, weights="distance"
                )
                model.fit(X_tr, y_tr)
                y_pred_std = model.predict(X_val)

                # back-transform for metrics
                y_pred_orig = y_pred_std * sigma + mu
                y_val_orig  = y_val * sigma + mu

                metric_dict = {
                    "embedding": key,
                    "r2":  r2_score(y_val_orig, y_pred_orig),
                    "mae": mean_absolute_error(y_val_orig, y_pred_orig),
                }

                y_pred_store = y_pred_orig
                y_val_store  = y_val_orig

            self._history.append(metric_dict)

            if self.return_preds:
                if self.predict_all:
                    y_pred_all = model.predict(X_all)
                    if self.task == "classification":
                        y_pred_all_store = enc.inverse_transform(y_pred_all)
                        y_true_all_store = enc.inverse_transform(y_all)
                    else:
                        y_pred_all_store = y_pred_all * sigma + mu
                        y_true_all_store = y_all * sigma + mu

                    self._pred_bank[key] = pd.DataFrame(
                        {"y_true": y_true_all_store, "y_pred": y_pred_all_store},
                        index=obs_names,                     # ← all kept cells
                    )
                else: 
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

