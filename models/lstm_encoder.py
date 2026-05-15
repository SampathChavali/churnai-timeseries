"""
models/lstm_encoder.py
----------------------
LSTM-based encoder for 12-month customer behavioral sequences.

Architecture:
    Input        : (batch, 12, 6)   12 months × 6 features per month
    BiLSTM       : 2 layers, hidden=128, bidirectional, dropout=0.3
    Attention    : weighted average over time-steps (a learned query over
                   the 256-dim BiLSTM outputs)
    Projection   : Linear(256 -> 128) + LayerNorm
    Classifier   : Linear(128 -> 1)   (used only during training)

Public API:
    build_sequences(ts_df)          -> (ids, X) where X is (N, 12, 6)
    fit_scaler(X)                   -> sklearn StandardScaler fit on flat features
    apply_scaler(X, scaler)         -> scaled (N, 12, 6) array
    TimeSeriesDataset               -> torch Dataset
    LSTMEncoder                     -> the nn.Module
    train_lstm(ts_df, customers_df) -> (model, scaler, history)
    extract_embeddings(model, X)    -> (N, 128) numpy array
    predict_proba(model, X)         -> (N,) numpy array of churn probabilities
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dataclasses import dataclass, field
from typing import Iterable

import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, Dataset

import config as C


# ---------------------------------------------------------------------------
# Sequence construction
# ---------------------------------------------------------------------------
def build_sequences(ts_df: pd.DataFrame) -> tuple[list[str], np.ndarray, np.ndarray]:
    """Group the long time-series frame into (N, SEQ_LEN, n_features) tensor.

    Returns
    -------
    ids    : list[str]                customer IDs in row order
    X      : np.ndarray (N, 12, 6)    feature sequences
    labels : np.ndarray (N,)          0/1 churn label per customer (from first month)
    """
    ts_df = ts_df.sort_values(["customerID", "month"])
    feature_cols = C.TS_FEATURES

    ids: list[str] = []
    X_rows: list[np.ndarray] = []
    y_rows: list[int] = []

    for cid, group in ts_df.groupby("customerID", sort=False):
        seq = group[feature_cols].to_numpy(dtype=np.float32)
        if seq.shape[0] < C.SEQ_LEN:
            pad = np.zeros((C.SEQ_LEN - seq.shape[0], seq.shape[1]), dtype=np.float32)
            seq = np.concatenate([pad, seq], axis=0)
        elif seq.shape[0] > C.SEQ_LEN:
            seq = seq[-C.SEQ_LEN:]
        ids.append(cid)
        X_rows.append(seq)
        y_rows.append(int(group["churn_label"].iloc[0]))

    X = np.stack(X_rows, axis=0)
    y = np.asarray(y_rows, dtype=np.float32)
    return ids, X, y


def fit_scaler(X: np.ndarray) -> StandardScaler:
    """Fit a StandardScaler on the flattened (N*T, F) view, then save to disk."""
    n, t, f = X.shape
    scaler = StandardScaler().fit(X.reshape(n * t, f))
    joblib.dump(scaler, C.TS_SCALER_PATH)
    return scaler


def apply_scaler(X: np.ndarray, scaler: StandardScaler) -> np.ndarray:
    n, t, f = X.shape
    return scaler.transform(X.reshape(n * t, f)).reshape(n, t, f).astype(np.float32)


def load_scaler() -> StandardScaler:
    return joblib.load(C.TS_SCALER_PATH)


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------
class TimeSeriesDataset(Dataset):
    def __init__(self, X: np.ndarray, y: np.ndarray):
        self.X = torch.from_numpy(X.astype(np.float32))
        self.y = torch.from_numpy(y.astype(np.float32))

    def __len__(self) -> int:
        return self.X.shape[0]

    def __getitem__(self, idx: int):
        return self.X[idx], self.y[idx]


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------
class LSTMEncoder(nn.Module):
    def __init__(
        self,
        input_dim: int = C.TS_INPUT_DIM,
        hidden_dim: int = C.LSTM_HIDDEN,
        num_layers: int = C.LSTM_LAYERS,
        embed_dim: int = C.LSTM_EMBED_DIM,
        dropout: float = C.LSTM_DROPOUT,
    ):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        bi_dim = hidden_dim * 2

        # Additive attention over time-steps with a learned query
        self.attn = nn.Sequential(
            nn.Linear(bi_dim, bi_dim),
            nn.Tanh(),
            nn.Linear(bi_dim, 1),
        )

        self.proj = nn.Sequential(
            nn.Linear(bi_dim, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.Dropout(dropout),
        )
        self.classifier = nn.Linear(embed_dim, 1)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        x: (batch, T, F)
        Returns:
            logit       : (batch,)
            embedding   : (batch, embed_dim)
            attn_weights: (batch, T)
        """
        h, _ = self.lstm(x)                       # (B, T, 2H)
        scores = self.attn(h).squeeze(-1)         # (B, T)
        weights = torch.softmax(scores, dim=1)    # (B, T)
        context = torch.sum(h * weights.unsqueeze(-1), dim=1)  # (B, 2H)

        emb = self.proj(context)                  # (B, embed_dim)
        logit = self.classifier(emb).squeeze(-1)  # (B,)
        return logit, emb, weights


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------
@dataclass
class TrainHistory:
    train_loss: list[float] = field(default_factory=list)
    val_loss:   list[float] = field(default_factory=list)
    val_auc:    list[float] = field(default_factory=list)


def _split_indices(n: int, seed: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    idx = rng.permutation(n)
    n_test = int(n * C.TEST_SIZE)
    n_val  = int(n * C.VAL_SIZE)
    test_idx  = idx[:n_test]
    val_idx   = idx[n_test:n_test + n_val]
    train_idx = idx[n_test + n_val:]
    return train_idx, val_idx, test_idx


def train_lstm(ts_df: pd.DataFrame, verbose: bool = True) -> dict:
    """Train the LSTM encoder end-to-end and save a checkpoint to disk."""
    torch.manual_seed(C.RANDOM_SEED)
    np.random.seed(C.RANDOM_SEED)

    ids, X_raw, y = build_sequences(ts_df)
    if verbose:
        print(f"[lstm] sequences: X={X_raw.shape}  positives={int(y.sum())} / {len(y)}")

    train_idx, val_idx, test_idx = _split_indices(len(y), seed=C.RANDOM_SEED)

    scaler = fit_scaler(X_raw[train_idx])
    X = apply_scaler(X_raw, scaler)

    train_ds = TimeSeriesDataset(X[train_idx], y[train_idx])
    val_ds   = TimeSeriesDataset(X[val_idx],   y[val_idx])
    test_ds  = TimeSeriesDataset(X[test_idx],  y[test_idx])

    train_dl = DataLoader(train_ds, batch_size=C.LSTM_BATCH, shuffle=True)
    val_dl   = DataLoader(val_ds,   batch_size=C.LSTM_BATCH)
    test_dl  = DataLoader(test_ds,  batch_size=C.LSTM_BATCH)

    device = torch.device("cuda" if torch.cuda.is_available()
                          else "mps" if torch.backends.mps.is_available()
                          else "cpu")
    model = LSTMEncoder().to(device)

    pos_weight = torch.tensor([C.CHURN_POS_WEIGHT], device=device)
    criterion  = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer  = torch.optim.AdamW(model.parameters(), lr=C.LSTM_LR, weight_decay=1e-5)
    scheduler  = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=2
    )

    history = TrainHistory()
    best_val = float("inf")

    for epoch in range(1, C.LSTM_EPOCHS + 1):
        model.train()
        train_losses = []
        for xb, yb in train_dl:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            logit, _, _ = model(xb)
            loss = criterion(logit, yb)
            loss.backward()
            optimizer.step()
            train_losses.append(loss.item())

        model.eval()
        val_losses, ys, ps = [], [], []
        with torch.no_grad():
            for xb, yb in val_dl:
                xb, yb = xb.to(device), yb.to(device)
                logit, _, _ = model(xb)
                val_losses.append(criterion(logit, yb).item())
                ps.append(torch.sigmoid(logit).cpu().numpy())
                ys.append(yb.cpu().numpy())
        val_loss = float(np.mean(val_losses))
        val_auc  = float(roc_auc_score(np.concatenate(ys), np.concatenate(ps)))

        history.train_loss.append(float(np.mean(train_losses)))
        history.val_loss.append(val_loss)
        history.val_auc.append(val_auc)

        scheduler.step(val_loss)

        if val_loss < best_val:
            best_val = val_loss
            torch.save(model.state_dict(), C.LSTM_MODEL_PATH)

        if verbose:
            print(f"  epoch {epoch:>2d}  "
                  f"train_loss={history.train_loss[-1]:.4f}  "
                  f"val_loss={val_loss:.4f}  val_auc={val_auc:.4f}")

    # ---------------- Test evaluation (best checkpoint) ----------------
    model.load_state_dict(torch.load(C.LSTM_MODEL_PATH, map_location=device))
    model.eval()
    ys, ps = [], []
    with torch.no_grad():
        for xb, yb in test_dl:
            xb = xb.to(device)
            logit, _, _ = model(xb)
            ps.append(torch.sigmoid(logit).cpu().numpy())
            ys.append(yb.numpy())
    y_true = np.concatenate(ys)
    y_prob = np.concatenate(ps)
    y_pred = (y_prob >= 0.5).astype(int)

    metrics = {
        "test_auc":      float(roc_auc_score(y_true, y_prob)),
        "test_accuracy": float((y_pred == y_true).mean()),
        "test_recall":   float(((y_pred == 1) & (y_true == 1)).sum() /
                               max(int(y_true.sum()), 1)),
        "test_precision": float(((y_pred == 1) & (y_true == 1)).sum() /
                                max(int((y_pred == 1).sum()), 1)),
        "n_train": int(len(train_idx)),
        "n_val":   int(len(val_idx)),
        "n_test":  int(len(test_idx)),
        "history": {
            "train_loss": history.train_loss,
            "val_loss":   history.val_loss,
            "val_auc":    history.val_auc,
        },
    }

    return {
        "model":   model,
        "scaler":  scaler,
        "ids":     ids,
        "metrics": metrics,
    }


# ---------------------------------------------------------------------------
# Inference helpers
# ---------------------------------------------------------------------------
def _device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available()
                        else "mps" if torch.backends.mps.is_available()
                        else "cpu")


def load_model() -> LSTMEncoder:
    model = LSTMEncoder().to(_device())
    model.load_state_dict(torch.load(C.LSTM_MODEL_PATH, map_location=_device()))
    model.eval()
    return model


@torch.no_grad()
def predict_proba(model: LSTMEncoder, X: np.ndarray, batch_size: int = 128) -> np.ndarray:
    """Run inference on a pre-scaled (N, T, F) array."""
    model.eval()
    device = _device()
    out: list[np.ndarray] = []
    for i in range(0, len(X), batch_size):
        xb = torch.from_numpy(X[i:i + batch_size].astype(np.float32)).to(device)
        logit, _, _ = model(xb)
        out.append(torch.sigmoid(logit).cpu().numpy())
    return np.concatenate(out)


@torch.no_grad()
def predict_with_attention(
    model: LSTMEncoder, X: np.ndarray, batch_size: int = 128
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (probabilities, embeddings, attention weights) for X."""
    model.eval()
    device = _device()
    probs, embs, attns = [], [], []
    for i in range(0, len(X), batch_size):
        xb = torch.from_numpy(X[i:i + batch_size].astype(np.float32)).to(device)
        logit, emb, attn = model(xb)
        probs.append(torch.sigmoid(logit).cpu().numpy())
        embs.append(emb.cpu().numpy())
        attns.append(attn.cpu().numpy())
    return np.concatenate(probs), np.concatenate(embs), np.concatenate(attns)
