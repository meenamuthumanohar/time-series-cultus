"""
Advanced Time Series Forecasting
Full End-to-End Script (Transformer Encoder-Decoder + Attention + LSTM Baseline)

Save as: forecast_project.py
Requires: numpy, pandas, scikit-learn, matplotlib, torch

Author: Generated for you
Date: 2025-11-19
"""

import os
import math
import random
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

# -------------------------
# Config / Reproducibility
# -------------------------
SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)
random.seed(SEED)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# -------------------------
# Hyperparameters
# -------------------------
INPUT_SEQ = 30       # historical window
OUTPUT_SEQ = 10      # forecast horizon
FEATURES = 5         # number of variables
BATCH_SIZE = 32
D_MODEL = 64
NHEAD = 4
ENC_LAYERS = 3
DEC_LAYERS = 3
LR = 1e-3
EPOCHS = 20

# -------------------------
# Optional local files (user uploaded screenshots)
# -------------------------
# Developer instruction told me to return local paths as URLs. Here they are:
screenshot_urls = [
    "file:///mnt/data/Screenshot%20(15).png",
    "file:///mnt/data/Screenshot%20(16).png"
]
print("Screenshot URLs (local):", screenshot_urls)

# -------------------------
# Synthetic data generator
# -------------------------
def synthetic_series(n_steps=1500, features=FEATURES):
    """
    Create a multivariate synthetic timeseries with:
     - seasonal components
     - trend component
     - heteroscedastic noise
    Returns pandas.DataFrame with `features` columns named f1..f{features}
    """
    time = np.arange(n_steps)
    f = []

    # feature 1: sinusoid + small noise (seasonal)
    f1 = np.sin(time * 0.02) + 0.1 * np.random.randn(n_steps)
    f.append(f1)

    # feature 2: cosine with different freq + noise
    f2 = np.cos(time * 0.015) + 0.1 * np.random.randn(n_steps)
    f.append(f2)

    # feature 3: trend + seasonality + heteroscedastic noise
    trend = 0.0008 * time
    seasonal = 0.2 * np.sin(time * 0.04)
    noise = 0.05 * (1 + 0.5 * np.sin(time * 0.01)) * np.random.randn(n_steps)
    f3 = trend + seasonal + noise
    f.append(f3)

    # feature 4: slowly increasing log-like + noise
    f4 = np.log(time + 1) * 0.03 + 0.05 * np.random.randn(n_steps)
    f.append(f4)

    # feature 5: decaying exponential + noise
    f5 = np.exp(-time / (2000.0)) + 0.05 * np.random.randn(n_steps)
    f.append(f5)

    data = np.vstack(f).T
    colnames = [f"f{i+1}" for i in range(data.shape[1])]
    return pd.DataFrame(data, columns=colnames)

# create and scale dataset
n_steps = 1800
df = synthetic_series(n_steps=n_steps)
scaler = MinMaxScaler()
scaled = scaler.fit_transform(df.values)  # shape (n_steps, FEATURES)

# -------------------------
# Sequence creation
# -------------------------
def create_sequences(data, input_seq=INPUT_SEQ, output_seq=OUTPUT_SEQ):
    X, Y = [], []
    L = len(data)
    for i in range(L - input_seq - output_seq + 1):
        X.append(data[i:i+input_seq])
        Y.append(data[i+input_seq:i+input_seq+output_seq])
    return np.array(X), np.array(Y)

X_all, Y_all = create_sequences(scaled, INPUT_SEQ, OUTPUT_SEQ)
print("Generated sequences:", X_all.shape, Y_all.shape)

# Train/test split (80/20)
split = int(0.8 * len(X_all))
X_train, Y_train = X_all[:split], Y_all[:split]
X_test,  Y_test  = X_all[split:], Y_all[split:]

class TimeSeriesDataset(Dataset):
    def __init__(self, X, Y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.Y = torch.tensor(Y, dtype=torch.float32)
    def __len__(self):
        return len(self.X)
    def __getitem__(self, idx):
        return self.X[idx], self.Y[idx]

train_ds = TimeSeriesDataset(X_train, Y_train)
test_ds  = TimeSeriesDataset(X_test, Y_test)
train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, drop_last=False)
test_loader  = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False)

# -------------------------
# Helper: Positional encoding (learned)
# -------------------------
class LearnedPositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        self.pe = nn.Embedding(max_len, d_model)
    def forward(self, x):
        # x: (B, T, D)
        T = x.size(1)
        positions = torch.arange(T, device=x.device).unsqueeze(0)
        pos_emb = self.pe(positions)  # (1, T, D)
        return pos_emb

# -------------------------
# Transformer Encoder-Decoder Model
# -------------------------
class TransformerTS(nn.Module):
    def __init__(self, feature_size=FEATURES, d_model=D_MODEL, nhead=NHEAD,
                 enc_layers=ENC_LAYERS, dec_layers=DEC_LAYERS, out_seq=OUTPUT_SEQ, max_len=1000):
        super().__init__()
        self.feature_size = feature_size
        self.d_model = d_model
        self.input_fc = nn.Linear(feature_size, d_model)
        self.output_fc = nn.Linear(d_model, feature_size)
        self.pos_enc = LearnedPositionalEncoding(d_model, max_len=max_len)

        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dim_feedforward=4*d_model, batch_first=True)
        decoder_layer = nn.TransformerDecoderLayer(d_model=d_model, nhead=nhead, dim_feedforward=4*d_model, batch_first=True)

        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=enc_layers)
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=dec_layers)

        self.out_seq = out_seq

        # For extracting attention weights we will register hooks later
        self._saved_attn_weights = []

    def forward(self, src):
        """
        src: (B, T_src, feature_size)
        returns: (B, T_tgt=out_seq, feature_size)
        """
        B, T, _ = src.shape
        # map to d_model
        src_d = self.input_fc(src)  # (B, T, d_model)
        # add pos enc
        src_d = src_d + self.pos_enc(src_d)

        # Encoder
        memory = self.encoder(src_d)  # (B, T, d_model)

        # Autoregressive decode: start with last encoder token as initial decoder input
        # We'll generate one step at a time (teacher forcing not used here)
        # Start token: zeros (or last memory token)
        tgt = memory[:, -1:, :].clone()  # (B, 1, d_model)
        outputs = []

        # causal mask for decoder self-attention: prevent attending to future in tgt
        # But because we generate one token at a time, no mask needed for stepwise generation
        for step in range(self.out_seq):
            tgt_pos = tgt + self.pos_enc(tgt)  # add pos enc (positions start at 0 always but that's fine)
            out = self.decoder(tgt_pos, memory)  # (B, t, d_model)
            # take last token
            next_token = out[:, -1:, :]  # (B,1,d_model)
            outputs.append(next_token)
            # append for next step
            tgt = torch.cat([tgt, next_token], dim=1)

        outputs = torch.cat(outputs, dim=1)  # (B, out_seq, d_model)
        final = self.output_fc(outputs)      # (B, out_seq, feature_size)
        return final

    def save_attn_weights_hook(self, module, input, output):
        # For compatibility — not used by default. Keeping extension point.
        pass

# -------------------------
# Train / Evaluate utilities
# -------------------------
def mse_loss(pred, target):
    return ((pred - target)**2).mean()

def compute_metrics(preds, actuals, eps=1e-8):
    """
    preds, actuals: numpy arrays shaped (N, out_seq, features)
    returns rmse, mae, mape
    """
    # flatten
    preds_f = preds.reshape(-1)
    actual_f = actuals.reshape(-1)
    rmse = math.sqrt(mean_squared_error(actual_f, preds_f))
    mae = mean_absolute_error(actual_f, preds_f)
    # MAPE: avoid divide by zero
    mape = np.mean(np.abs((actuals - preds) / (np.abs(actuals) + eps))) * 100
    return rmse, mae, mape

def inverse_transform_batches(arr, scaler):
    """
    arr: (N, out_seq, features) scaled in original scaler domain (0-1)
    returns same shape with inverse transform applied per feature
    """
    N, T, F = arr.shape
    flat = arr.reshape(-1, F)
    inv = scaler.inverse_transform(flat)
    return inv.reshape(N, T, F)

# -------------------------
# Instantiate model, optimizer, loss
# -------------------------
model = TransformerTS(feature_size=FEATURES, d_model=D_MODEL, nhead=NHEAD,
                      enc_layers=ENC_LAYERS, dec_layers=DEC_LAYERS, out_seq=OUTPUT_SEQ,
                      max_len=1000).to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=LR)
criterion = nn.MSELoss()

# -------------------------
# Training loop
# -------------------------
def train_transformer(model, train_loader, epochs=EPOCHS, validate_loader=None, save_path="transformer.pth"):
    model.train()
    history = {"train_loss": [], "val_loss": []}
    for ep in range(1, epochs+1):
        total_loss = 0.0
        for xb, yb in train_loader:
            xb = xb.to(device); yb = yb.to(device)
            optimizer.zero_grad()
            ypred = model(xb)  # (B, out_seq, features)
            loss = criterion(ypred, yb)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * xb.size(0)
        avg_loss = total_loss / len(train_loader.dataset)
        history["train_loss"].append(avg_loss)

        # optional validation
        if validate_loader is not None:
            model.eval()
            val_loss = 0.0
            with torch.no_grad():
                for xb, yb in validate_loader:
                    xb = xb.to(device); yb = yb.to(device)
                    ypred = model(xb)
                    val_loss += criterion(ypred, yb).item() * xb.size(0)
            val_avg = val_loss / len(validate_loader.dataset)
            history["val_loss"].append(val_avg)
            print(f"Epoch {ep}/{epochs} - Train Loss: {avg_loss:.6f} - Val Loss: {val_avg:.6f}")
            model.train()
        else:
            print(f"Epoch {ep}/{epochs} - Train Loss: {avg_loss:.6f}")

    # Save model
    torch.save({
        "model_state": model.state_dict(),
        "scaler": scaler,
        "config": {
            "INPUT_SEQ": INPUT_SEQ,
            "OUTPUT_SEQ": OUTPUT_SEQ,
            "FEATURES": FEATURES,
            "D_MODEL": D_MODEL
        }
    }, save_path)
    print("Saved transformer to", save_path)
    return history

# Train transformer
history = train_transformer(model, train_loader, epochs=EPOCHS, validate_loader=test_loader, save_path="transformer_full.pth")

# -------------------------
# Evaluation: Transformer
# -------------------------
def predict_on_loader(model, loader):
    model.eval()
    preds = []
    acts = []
    with torch.no_grad():
        for xb, yb in loader:
            xb = xb.to(device)
            yhat = model(xb).cpu().numpy()  # (B, out_seq, features)
            preds.append(yhat)
            acts.append(yb.numpy())
    preds = np.vstack(preds)
    acts = np.vstack(acts)
    return preds, acts

preds_scaled, acts_scaled = predict_on_loader(model, test_loader)
preds = inverse_transform_batches(preds_scaled, scaler)
acts  = inverse_transform_batches(acts_scaled, scaler)
rmse, mae, mape = compute_metrics(preds, acts)
print("\nTransformer Evaluation (test):")
print(f"RMSE: {rmse:.6f}, MAE: {mae:.6f}, MAPE: {mape:.3f}%")

# -------------------------
# Visualize example predictions for feature 1
# -------------------------
def plot_prediction_example(preds, acts, sample_index=0, feature_idx=0):
    # preds, acts shape: (N, out_seq, features)
    plt.figure(figsize=(10,4))
    plt.plot(range(len(acts[sample_index,:,feature_idx])), acts[sample_index,:,feature_idx], marker='o', label="Actual")
    plt.plot(range(len(preds[sample_index,:,feature_idx])), preds[sample_index,:,feature_idx], marker='x', label="Predicted")
    plt.title(f"Sample {sample_index} - Feature f{feature_idx+1}")
    plt.legend()
    plt.tight_layout()
    plt.show()

plot_prediction_example(preds, acts, sample_index=0, feature_idx=0)

# -------------------------
# Baseline: Simple LSTM model (sequence-to-sequence via repeating final hidden)
# -------------------------
class LSTMBaseline(nn.Module):
    def __init__(self, feature_size=FEATURES, hidden=64, num_layers=1):
        super().__init__()
        self.lstm = nn.LSTM(feature_size, hidden, num_layers=num_layers, batch_first=True)
        self.fc = nn.Linear(hidden, feature_size)
        self.hidden = hidden

    def forward(self, x):
        # x: (B, T, feature)
        out, (h_n, c_n) = self.lstm(x)  # out: (B, T, hidden)
        h = h_n[-1]  # (B, hidden)
        # repeat hidden vector across OUTPUT_SEQ and apply linear
        h_rep = h.unsqueeze(1).repeat(1, OUTPUT_SEQ, 1)  # (B, out_seq, hidden)
        y = self.fc(h_rep)  # (B, out_seq, feature)
        return y

lstm_model = LSTMBaseline().to(device)
lstm_opt = torch.optim.Adam(lstm_model.parameters(), lr=LR)

def train_lstm(model, loader, epochs=15):
    model.train()
    for ep in range(1, epochs+1):
        tot = 0.0
        for xb, yb in loader:
            xb = xb.to(device); yb = yb.to(device)
            lstm_opt.zero_grad()
            yhat = model(xb)
            loss = criterion(yhat, yb)
            loss.backward()
            lstm_opt.step()
            tot += loss.item() * xb.size(0)
        avg = tot / len(loader.dataset)
        print(f"LSTM Epoch {ep}/{epochs} - Loss: {avg:.6f}")

train_lstm(lstm_model, train_loader, epochs=15)

l_preds_s, l_acts_s = predict_on_loader(lstm_model, test_loader)
l_preds = inverse_transform_batches(l_preds_s, scaler)
l_acts  = inverse_transform_batches(l_acts_s, scaler)
l_rmse, l_mae, l_mape = compute_metrics(l_preds, l_acts)
print("\nLSTM Baseline (test):")
print(f"RMSE: {l_rmse:.6f}, MAE: {l_mae:.6f}, MAPE: {l_mape:.3f}%")

# -------------------------
# Save both models for future use
# -------------------------
torch.save({"model_state": lstm_model.state_dict()}, "lstm_baseline.pth")
print("Saved LSTM to lstm_baseline.pth")

# -------------------------
# Attention introspection (method outline)
# -------------------------
# Note: The PyTorch built-in Transformer modules do not expose attention weights by default in
# the high-level nn.TransformerEncoder/Decoder unless you modify layers to return them.
# For deep introspection you'd implement custom MultiheadAttention layers that return weights,
# or use hooks inside nn.MultiheadAttention modules. Below is an example of how to capture
# attention weights from decoder/encoder layers if you replace TransformerEncoderLayer/DecoderLayer
# to expose attn weights. For simplicity and reliability in a single script we keep an explanation
# and a minimal stub.

print("\nAttention extraction note:")
print("If you need attention weights, implement a custom MultiheadAttention module that returns "
      "the attention matrices or register forward hooks on the built-in MultiheadAttention layers. "
      "This script uses the standard layers for reliability; extract attention by instrumenting "
      "the encoder/decoder layers when required.")

# -------------------------
# Quick plots: training curves
# -------------------------
plt.figure(figsize=(8,4))
plt.plot(history["train_loss"], label="train_loss")
if len(history["val_loss"]) > 0:
    plt.plot(history["val_loss"], label="val_loss")
plt.title("Training Loss")
plt.xlabel("Epoch")
plt.ylabel("MSE Loss")
plt.legend()
plt.tight_layout()
plt.show()

# -------------------------
# Example: Save prediction CSV for inspection
# -------------------------
out_dir = Path("results")
out_dir.mkdir(exist_ok=True)
# store first 50 predictions and actuals for first feature
N_export = min(50, preds.shape[0])
export_df = pd.DataFrame({
    "pred_f1_t{}".format(i+1): preds[i, 0, 0] if i==0 else None for i in range(1) # placeholder line
})
# Instead create a structured CSV with flattened predictions:
flat_preds = preds.reshape(preds.shape[0], -1)
flat_acts  = acts.reshape(acts.shape[0], -1)
preds_df = pd.DataFrame(flat_preds)
acts_df  = pd.DataFrame(flat_acts)
preds_df.to_csv(out_dir / "transformer_preds_flat.csv", index=False)
acts_df.to_csv(out_dir / "transformer_actuals_flat.csv", index=False)
print("Exported predictions to", out_dir)

# -------------------------
# Print final summary
# -------------------------
print("\n=== SUMMARY ===")
print(f"Data steps: {n_steps}, Features: {FEATURES}, Input seq: {INPUT_SEQ}, Output seq: {OUTPUT_SEQ}")
print("Transformer test RMSE, MAE, MAPE:", rmse, mae, mape)
print("LSTM test RMSE, MAE, MAPE       :", l_rmse, l_mae, l_mape)
print("Screenshot local URLs (for report):")
for u in screenshot_urls:
    print("  ", u)

# End of script
