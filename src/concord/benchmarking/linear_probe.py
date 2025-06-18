


# file: concord/eval/linear_probe.py
"""
Drop-in linear-probe evaluation for AnnData objects.

Example
-------
from concord.eval.linear_probe import LinearProbeEvaluator

evaluator = LinearProbeEvaluator(
    adata=my_adata,
    emb_keys=["X_pca", "Concord", "scVI_latent"],
    target_key="cell_type",          # or "pseudotime"
    task="auto",                     # "auto" | "classification" | "regression"
    batch_size=1024,
    epochs=1,                        # default replicates HCL
    device="cuda"                    # or "cpu"
)
results_df = evaluator.run()
print(results_df)
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Literal, List, Dict, Any

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader, random_split
from sklearn.metrics import accuracy_score, r2_score, mean_absolute_error
from sklearn.preprocessing import LabelEncoder
from anndata import AnnData


# ----------  minimal linear head ------------------------------------------------
class _Linear(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, bias: bool = True) -> None:
        super().__init__()
        self.fc = nn.Linear(in_dim, out_dim, bias=bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:          # type: ignore
        return self.fc(x)


# ----------  main evaluator -----------------------------------------------------
@dataclass
class LinearProbeEvaluator:
    adata: AnnData
    emb_keys: List[str]
    target_key: str
    task: Literal["classification", "regression", "auto"] = "auto"
    val_frac: float = 0.2
    batch_size: int = 1024
    epochs: int = 1
    lr: float = 1e-3
    weight_decay: float = 1e-4
    device: str = "cpu"
    seed: int = 0
    _history: List[Dict[str, Any]] = field(default_factory=list, init=False)

    # ------------- public API ---------------------------------------------------
    def run(self) -> pd.DataFrame:
        torch.manual_seed(self.seed)
        np.random.seed(self.seed)

        y_raw = self.adata.obs[self.target_key].to_numpy()

        if self.task == "auto":
            self.task = "regression" if np.issubdtype(y_raw.dtype, np.number) else "classification"

        if self.task == "classification":
            enc = LabelEncoder().fit(y_raw)
            y_all = torch.tensor(enc.transform(y_raw), dtype=torch.long)
            out_dim = len(enc.classes_)
        else:  # regression
            y_all = torch.tensor(y_raw, dtype=torch.float32)
            out_dim = 1

        for key in self.emb_keys:
            X = torch.tensor(self.adata.obsm[key], dtype=torch.float32)
            metrics = self._evaluate_single_rep(X, y_all, out_dim, key)
            self._history.append(metrics)

        return pd.DataFrame(self._history).set_index("embedding")

    # ------------- helpers ------------------------------------------------------
    def _evaluate_single_rep(self, X: torch.Tensor, y: torch.Tensor,
                             out_dim: int, key: str) -> Dict[str, Any]:

        # split
        ds = TensorDataset(X, y)
        n_val = int(len(ds) * self.val_frac)
        train_ds, val_ds = random_split(ds, [len(ds) - n_val, n_val])

        train_loader = DataLoader(train_ds, batch_size=self.batch_size, shuffle=True)
        val_loader = DataLoader(val_ds, batch_size=self.batch_size)

        # probe
        probe = _Linear(in_dim=X.shape[1], out_dim=out_dim).to(self.device)
        criterion = nn.CrossEntropyLoss() if self.task == "classification" else nn.MSELoss()
        opt = torch.optim.Adam(probe.parameters(), lr=self.lr, weight_decay=self.weight_decay)

        # training loop (epochs default = 1 to mirror HCL)
        for _ in range(self.epochs):
            probe.train()
            for xb, yb in train_loader:
                xb, yb = xb.to(self.device), yb.to(self.device)
                loss = criterion(probe(xb).squeeze(), yb)
                opt.zero_grad()
                loss.backward()
                opt.step()

        # evaluation
        probe.eval()
        preds, trues = [], []
        with torch.no_grad():
            for xb, yb in val_loader:
                out = probe(xb.to(self.device)).squeeze().cpu()
                preds.append(out)
                trues.append(yb)

        y_pred = torch.cat(preds).numpy()
        y_true = torch.cat(trues).numpy()

        if self.task == "classification":
            metric_dict = {"accuracy": accuracy_score(y_true, y_pred.argmax(1))}
        else:
            metric_dict = {
                "r2": r2_score(y_true, y_pred),
                "mae": mean_absolute_error(y_true, y_pred),
            }

        metric_dict.update({"embedding": key})
        return metric_dict

