"""
Linear-probe evaluation for AnnData embeddings
=============================================

* Trains a *single* linear layer (“probe”) on each embedding.
* Z-scores regression targets during training, converts predictions
  back to the original scale for metrics.
* Auto-detects **classification** vs **regression**.
* Supports **early stopping** (patience) or a fixed epoch budget.
* Can **ignore** rows whose label is in `ignore_values`
  (e.g. "NA", "unknown") and drops NaNs for regression.
* Returns a metric table plus optional per-cell predictions.

Example
-------
from concord.benchmarking.linear_probe import LinearProbeEvaluator

evalr = LinearProbeEvaluator(
    adata        = adata,
    emb_keys     = ["Concord", "scVI_latent", "X_pca"],
    target_key   = "pseudotime",
    ignore_values = ["NA", "unknown"],   # skip these labels
    epochs       = 100,
    batch_size   = 512,
    device       = "cuda",
    print_every  = 10,
    early_stop_patience = 10,            # 0 → disable early stop
    return_preds = True,
)
metrics_df, preds_bank = evalr.run()
print(metrics_df)
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import List, Any, Dict, Literal

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader, random_split, Subset
from sklearn.metrics import accuracy_score, r2_score, mean_absolute_error
from sklearn.preprocessing import LabelEncoder
from anndata import AnnData


# --------------------------------------------------------------------- #
#                           tiny linear head                            #
# --------------------------------------------------------------------- #
class _Linear(nn.Module):
    """One-layer linear probe."""
    def __init__(self, in_dim: int, out_dim: int, bias: bool = True) -> None:
        super().__init__()
        self.fc = nn.Linear(in_dim, out_dim, bias=bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:     # type: ignore
        return self.fc(x)


# --------------------------------------------------------------------- #
#                             main evaluator                            #
# --------------------------------------------------------------------- #
@dataclass
class LinearProbeEvaluator:
    """Linear-probe evaluation over pre-computed embeddings."""

    # ---- required ----------------------------------------------------
    adata: AnnData
    emb_keys: List[str]
    target_key: str
    task: Literal["classification", "regression", "auto"] = "auto"

    # ---- optional ----------------------------------------------------
    ignore_values: List[Any] | None = None      # labels to skip entirely

    # training / eval hyper-params
    val_frac: float = 0.2
    batch_size: int = 1024
    epochs: int = 30
    lr: float = 1e-2
    weight_decay: float = 1e-3
    device: str = "cpu"
    seed: int = 0

    # monitoring
    print_every: int = 0          # 0 = silent
    early_stop_patience: int = 5  # 0 = no early stopping

    # extras
    return_preds: bool = False
    predict_all: bool = False

    # internal bookkeeping
    _history: List[Dict[str, Any]] = field(default_factory=list, init=False)
    _pred_bank: Dict[str, pd.DataFrame] = field(default_factory=dict, init=False)

    # ----------------------------------------------------------------- #
    #                              PUBLIC                               #
    # ----------------------------------------------------------------- #
    def run(self):
        """Train a probe for every embedding and return metrics (+ preds)."""
        torch.manual_seed(self.seed)
        np.random.seed(self.seed)

        # ------------------ 0) filter rows --------------------------- #
        y_raw_full = self.adata.obs[self.target_key].to_numpy()
        mask = np.ones_like(y_raw_full, dtype=bool)

        if self.ignore_values is not None:
            mask &= ~np.isin(y_raw_full, self.ignore_values)

        y_tmp = y_raw_full[mask]

        # ------------------ 1) determine task ------------------------ #
        if self.task == "auto":
            self.task = (
                "regression"
                if np.issubdtype(y_tmp.dtype, np.number)
                else "classification"
            )

            print(f"Detected task: {self.task}")

        # ------------------ 2) drop NaN for regression --------------- #
        if self.task == "regression":
            nan_mask = ~np.isnan(y_tmp)
            mask[mask] &= nan_mask          # refine global mask
            y_raw = y_tmp[nan_mask]
        else:
            y_raw = y_tmp

        if mask.sum() == 0:
            raise ValueError("No rows left after ignore/N-aN filtering.")

        adata_filt = self.adata[mask].copy()

        # ------------------ 3) encode / scale targets ---------------- #
        if self.task == "classification":
            enc = LabelEncoder().fit(y_raw)
            y_proc = enc.transform(y_raw).astype(np.int64)
            out_dim = len(enc.classes_)
            mu, sigma = 0.0, 1.0            # dummies (unused)
        else:  # regression
            y_raw = y_raw.astype(np.float32)
            mu, sigma = y_raw.mean(), y_raw.std()
            if sigma == 0:
                raise ValueError("Target has zero variance; R² undefined.")
            y_proc = ((y_raw - mu) / sigma).astype(np.float32)
            out_dim = 1

        y_tensor = torch.tensor(y_proc)

        # ------------------ 4) run probe on each emb ----------------- #
        for key in self.emb_keys:
            X = torch.tensor(adata_filt.obsm[key], dtype=torch.float32)
            metrics, pred_df_val, pred_df_all = self._evaluate_single_rep(
                X, y_tensor, out_dim, key, mu, sigma, adata_filt.obs_names,
                predict_all=self.predict_all,
            )
            self._history.append(metrics)

            if self.return_preds:
                # choose which to store: ALL if requested, else VAL only
                df_to_store = pred_df_all if (self.predict_all and pred_df_all is not None) else pred_df_val

                if self.task == "classification":
                    df_to_store["y_true"] = enc.inverse_transform(df_to_store["y_true"].astype(int))
                    df_to_store["y_pred"] = enc.inverse_transform(df_to_store["y_pred"].astype(int))
                # regression already back-scaled in helper
                
                self._pred_bank[key] = df_to_store

        metrics_df = pd.DataFrame(self._history).set_index("embedding")
        if self.return_preds:
            return metrics_df, self._pred_bank
        return metrics_df

    def get_preds(self, key: str) -> pd.DataFrame:
        if key not in self._pred_bank:
            raise KeyError("Run .run(return_preds=True) first.")
        return self._pred_bank[key]

    # ----------------------------------------------------------------- #
    #                             HELPERS                               #
    # ----------------------------------------------------------------- #
    def _evaluate_single_rep(
        self,
        X: torch.Tensor,
        y: torch.Tensor,
        out_dim: int,
        emb_key: str,
        mu: float,
        sigma: float,
        obs_names,
        predict_all: bool = False,
    ):
        ds = TensorDataset(X, y)
        n_val = int(len(ds) * self.val_frac)

        # deterministic split
        g = torch.Generator().manual_seed(self.seed)
        train_ds, val_ds = random_split(ds, [len(ds) - n_val, n_val], generator=g)

        train_loader = DataLoader(train_ds, batch_size=self.batch_size, shuffle=True)
        val_loader   = DataLoader(val_ds, batch_size=self.batch_size, shuffle=False)

        probe = _Linear(in_dim=X.shape[1], out_dim=out_dim).to(self.device)
        criterion = nn.CrossEntropyLoss() if self.task == "classification" else nn.MSELoss()
        opt = torch.optim.AdamW(probe.parameters(), lr=self.lr, weight_decay=self.weight_decay)

        best_val = float("inf"); best_state = None; bad_epochs = 0

        # ---------------- training loop ------------------------------ #
        for epoch in range(self.epochs):
            # ---- train --------------------------------------------- #
            probe.train(); train_loss = 0.0
            for xb, yb in train_loader:
                xb, yb = xb.to(self.device), yb.to(self.device)
                loss = criterion(probe(xb).squeeze(), yb)
                opt.zero_grad(); loss.backward(); opt.step()
                train_loss += loss.item() * len(xb)
            train_loss /= len(train_ds)

            # ---- validate ------------------------------------------ #
            probe.eval(); val_loss = 0.0
            with torch.no_grad():
                for xb, yb in val_loader:
                    xb, yb = xb.to(self.device), yb.to(self.device)
                    val_loss += criterion(probe(xb).squeeze(), yb).item() * len(xb)
            val_loss /= len(val_ds)

            if self.print_every and (epoch + 1) % self.print_every == 0:
                print(f"[{emb_key}] epoch {epoch+1:>3}/{self.epochs} "
                      f"train {train_loss:.4f}  val {val_loss:.4f}")

            # ---- early stopping bookkeeping ----------------------- #
            if val_loss < best_val - 1e-6:
                best_val, bad_epochs = val_loss, 0
                best_state = {k: v.detach().clone() for k, v in probe.state_dict().items()}
            else:
                bad_epochs += 1
                if self.early_stop_patience and bad_epochs >= self.early_stop_patience:
                    if self.print_every:
                        print(f"[{emb_key}] early-stopping at epoch {epoch+1}")
                    break

        # restore best weights
        if best_state is not None:
            probe.load_state_dict(best_state)

        # ---------------- final predictions ------------------------- #
        probe.eval(); preds, trues, idxs = [], [], []
        with torch.no_grad():
            for (xb, yb), subset_idxs in _yield_with_indices(val_loader, val_ds):
                out = probe(xb.to(self.device)).squeeze().cpu()
                preds.append(out); trues.append(yb); idxs.extend(subset_idxs)

        y_pred_scaled = torch.cat(preds).numpy()
        y_true_scaled = torch.cat(trues).numpy()

        # ---------------- metrics & dataframe ----------------------- #
        if self.task == "classification":
            y_pred_final = y_pred_scaled.argmax(1)
            metric_dict = {
                "embedding": emb_key,
                "accuracy": accuracy_score(y_true_scaled, y_pred_final),
            }
            val_df = pd.DataFrame(
                {"y_true": y_true_scaled, "y_pred": y_pred_final},
                index=np.array(obs_names)[idxs],
            )
        else:
            y_pred_final = y_pred_scaled * sigma + mu
            y_true_final = y_true_scaled * sigma + mu
            metric_dict = {
                "embedding": emb_key,
                "r2":  r2_score(y_true_final, y_pred_final),
                "mae": mean_absolute_error(y_true_final, y_pred_final),
            }
            val_df = pd.DataFrame(
                {"y_true": y_true_final, "y_pred": y_pred_final},
                index=np.array(obs_names)[idxs],
            )

        all_df = None
        if predict_all:
            probe.eval(); preds_all, trues_all = [], []
            full_loader = DataLoader(TensorDataset(X, y), batch_size=self.batch_size, shuffle=False)
            with torch.no_grad():
                for xb, yb in full_loader:
                    out = probe(xb.to(self.device)).squeeze().cpu()
                    preds_all.append(out); trues_all.append(yb.cpu())

            y_pred_all_scaled = torch.cat(preds_all).numpy()
            y_true_all_scaled = torch.cat(trues_all).numpy()

            if self.task == "classification":
                y_pred_all = y_pred_all_scaled.argmax(1)
                all_df = pd.DataFrame(
                    {"y_true": y_true_all_scaled, "y_pred": y_pred_all},
                    index=np.array(obs_names),   # ALL kept rows
                )
            else:
                y_pred_all = y_pred_all_scaled * sigma + mu
                y_true_all = y_true_all_scaled * sigma + mu
                all_df = pd.DataFrame(
                    {"y_true": y_true_all, "y_pred": y_pred_all},
                    index=np.array(obs_names),
                )

        return metric_dict, val_df, all_df


# --------------------------------------------------------------------- #
#                         utility: index mapping                        #
# --------------------------------------------------------------------- #
def _yield_with_indices(loader: DataLoader, subset: Subset):
    """Yield ((xb, yb), original_subset_indices) for each batch."""
    offset = 0
    for xb, yb in loader:
        bs = len(xb)
        idxs = subset.indices[offset : offset + bs]
        offset += bs
        yield (xb, yb), idxs

