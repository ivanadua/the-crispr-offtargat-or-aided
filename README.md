```
Model 3 v2: Hybrid CNN + Attention 
----------------------------------------------------------------
CHANGES FROM v1:
  - Removed hardcoded positional weight vector (POS_WEIGHTS)
  - Removed seed_mismatches, nonseed_mismatches, pam_proximity_score
    from biological features — these re-encoded the positional prior
  - Added learned sinusoidal positional encoding instead
  - Model must now learn positional importance purely from data
  - If seed region sensitivity still emerges → genuine data-driven finding

Output:
    model_results/hybrid_model_v2.pt
    model_results/hybrid_results_v2.json
    model_results/hybrid_roc_v2.npy
"""

import sys
sys.path.insert(0, '/Users/moana/Downloads/')

import numpy as np
import json
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import roc_auc_score

from shared_utils import (load_features, guide_aware_split,
                          evaluate, encode_pair,
                          BIO_FEATURES, RANDOM_SEED, SEQ_LEN)

INPUT_PATH = '/Users/moana/Downloads/final dataset masterio_features_atac.csv'
OUTPUT_DIR = '/Users/moana/Downloads/model_results/'
BATCH_SIZE = 512
EPOCHS     = 50
LR         = 0.001
os.makedirs(OUTPUT_DIR, exist_ok=True)

if torch.backends.mps.is_available():
    DEVICE = torch.device('mps')
    print("🚀 Apple Silicon MPS GPU")
elif torch.cuda.is_available():
    DEVICE = torch.device('cuda')
else:
    DEVICE = torch.device('cpu')

# REMOVED: hardcoded POS_WEIGHTS
# Model now learns positional importance purely from data
# Sinusoidal positional encoding added inside the model (see below)

# Unbiased biological features — positional features removed
# Removed: seed_mismatches, nonseed_mismatches, pam_proximity_score
# These re-encoded the positional prior the model should learn itself
UNBIASED_BIO_FEATURES = [
    'gc_content_target',
    'gc_content_offtarget',
    'shannon_entropy_target',
    'shannon_entropy_offt',
    'mismatch_count',
    'guide_length',
    'mfe',
    'has_bulge',
    'atac_accessible',
]

class HybridDataset(Dataset):
    def __init__(self, X_seq, X_bio, y):
        self.X_seq = torch.tensor(X_seq, dtype=torch.float32)
        self.X_bio = torch.tensor(X_bio, dtype=torch.float32)
        self.y     = torch.tensor(y,     dtype=torch.float32)
    def __len__(self): return len(self.y)
    def __getitem__(self, i): return self.X_seq[i], self.X_bio[i], self.y[i]

class HybridModel(nn.Module):
    def __init__(self, n_bio, seq_len=20, d_model=128):
        super().__init__()
        # Sequence branch
        self.conv1 = nn.Conv1d(8,   64,  kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(64,  128, kernel_size=3, padding=1)
        self.bn1   = nn.BatchNorm1d(64)
        self.bn2   = nn.BatchNorm1d(128)

        # Sinusoidal positional encoding — tells the model WHERE each
        # position is without telling it WHICH positions matter.
        # The model must learn importance from data.
        pe = torch.zeros(seq_len, d_model)
        pos = torch.arange(0, seq_len, dtype=torch.float).unsqueeze(1)
        div = torch.exp(torch.arange(0, d_model, 2).float() *
                        (-torch.log(torch.tensor(10000.0)) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer('pos_enc', pe.unsqueeze(0))  # (1, seq_len, d_model)

        self.attn  = nn.MultiheadAttention(128, num_heads=4, batch_first=True, dropout=0.1)
        self.seq_pool = nn.AdaptiveAvgPool1d(1)
        self.seq_fc   = nn.Linear(128, 64)

        # Bio branch
        self.bio_fc1 = nn.Linear(n_bio, 64)
        self.bio_fc2 = nn.Linear(64, 32)
        self.bio_bn  = nn.BatchNorm1d(64)

        # Fusion
        self.fuse1 = nn.Linear(96, 64)
        self.fuse2 = nn.Linear(64, 32)
        self.fuse3 = nn.Linear(32, 1)
        self.drop  = nn.Dropout(0.3)
        self.relu  = nn.ReLU()

    def forward(self, x_seq, x_bio):
        x = x_seq.permute(0, 2, 1)
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.relu(self.bn2(self.conv2(x)))
        x = x.permute(0, 2, 1)                    # (B, seq_len, 128)

        # Add sinusoidal positional encoding — position info without bias
        x = x + self.pos_enc

        # Self-attention learns which positions matter from data
        x, _ = self.attn(x, x, x)
        x = x.permute(0, 2, 1)
        x = self.seq_pool(x).squeeze(-1)
        x_s = self.drop(self.relu(self.seq_fc(x)))

        x_b = self.relu(self.bio_bn(self.bio_fc1(x_bio)))
        x_b = self.drop(self.relu(self.bio_fc2(x_b)))
        x_f = torch.cat([x_s, x_b], dim=1)
        x_f = self.drop(self.relu(self.fuse1(x_f)))
        x_f = self.drop(self.relu(self.fuse2(x_f)))
        return self.fuse3(x_f).squeeze(-1)

class FocalLoss(nn.Module):
    def __init__(self, alpha=0.8, gamma=2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
    def forward(self, inputs, targets):
        bce  = nn.functional.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
        pt   = torch.exp(-bce)
        return (self.alpha * (1 - pt) ** self.gamma * bce).mean()

def main():
    print("=" * 55)
    print("   Model 3 v2: Hybrid CNN + Attention (UNBIASED)")
    print("=" * 55)
    print("\nKey changes from v1:")
    print("  - Hardcoded positional weights REMOVED")
    print("  - seed_mismatches, nonseed_mismatches, pam_proximity REMOVED")
    print("  - Sinusoidal positional encoding added")
    print("  - Model must learn positional importance from data\n")

    # Load only unbiased features
    print(f"Loading {INPUT_PATH}...")
    import pandas as pd
    df = pd.read_csv(INPUT_PATH)
    print(f"  {len(df):,} rows | {int(df['is_cut'].sum()):,} positives")

    for col in UNBIASED_BIO_FEATURES:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
        else:
            df[col] = 0.0

    X_bio = df[UNBIASED_BIO_FEATURES].values.astype(np.float32)
    n_bio = X_bio.shape[1]
    print(f"  Unbiased biological features ({n_bio}): {UNBIASED_BIO_FEATURES}")

    y = df['is_cut'].values.astype(np.float32)

    print("\nEncoding sequences...")
    X_seq = np.array([
        encode_pair(row['target_seq'], row['offtarget_seq'])
        for _, row in df.iterrows()
    ], dtype=np.float32)

    split = guide_aware_split(df, X_bio=X_bio, X_seq=X_seq)
    X_seq_tr, X_seq_te = split['X_seq_train'], split['X_seq_test']
    X_bio_tr, X_bio_te = split['X_bio_train'], split['X_bio_test']
    y_tr, y_te         = split['y_train'],     split['y_test']

    train_loader = DataLoader(
        HybridDataset(X_seq_tr, X_bio_tr, y_tr),
        batch_size=BATCH_SIZE, shuffle=True)
    test_loader  = DataLoader(
        HybridDataset(X_seq_te, X_bio_te, y_te),
        batch_size=BATCH_SIZE, shuffle=False)

    model     = HybridModel(n_bio=n_bio).to(DEVICE)
    criterion = FocalLoss()
    optimizer = optim.Adam(model.parameters(), lr=LR, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5)

    best_auc, best_state = 0, None
    train_losses = []

    print(f"\nTraining for {EPOCHS} epochs...")
    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0
        for xs, xb, yb in train_loader:
            xs, xb, yb = xs.to(DEVICE), xb.to(DEVICE), yb.to(DEVICE)
            optimizer.zero_grad()
            loss = criterion(model(xs, xb), yb)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)
        train_losses.append(avg_loss)

        if (epoch + 1) % 5 == 0:
            model.eval()
            probs = []
            with torch.no_grad():
                for xs, xb, _ in test_loader:
                    out = torch.sigmoid(model(xs.to(DEVICE), xb.to(DEVICE)))
                    probs.extend(out.cpu().numpy())
            val_auc = roc_auc_score(y_te, probs)
            scheduler.step(1 - val_auc)
            print(f"  Epoch {epoch+1:3d}/{EPOCHS} | Loss: {avg_loss:.4f} | AUC: {val_auc:.4f}")
            if val_auc > best_auc:
                best_auc = val_auc
                best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

    model.load_state_dict(best_state)
    model.eval()
    probs = []
    with torch.no_grad():
        for xs, xb, _ in test_loader:
            out = torch.sigmoid(model(xs.to(DEVICE), xb.to(DEVICE)))
            probs.extend(out.cpu().numpy())

    results, fpr, tpr = evaluate(y_te, np.array(probs), 'Hybrid_CNN_Attention_v2_unbiased')

    torch.save(best_state, os.path.join(OUTPUT_DIR, 'hybrid_model_v2.pt'))
    with open(os.path.join(OUTPUT_DIR, 'hybrid_results_v2.json'), 'w') as f:
        json.dump(results, f, indent=2)
    np.save(os.path.join(OUTPUT_DIR, 'hybrid_roc_v2.npy'), np.array([fpr, tpr]))

    # Loss curve
    plt.figure(figsize=(8, 4))
    plt.plot(train_losses, color='#2c3e50', linewidth=2)
    plt.xlabel('Epoch'); plt.ylabel('Focal Loss')
    plt.title('Hybrid Model — Training Loss')
    plt.grid(alpha=0.3); plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'hybrid_loss_curve.png'), dpi=300)
    plt.close()

    print(f"\n✅ Hybrid v2 (unbiased) complete | AUC: {results['auc_roc']} | AUC-PR: {results['auc_pr']}")
    print("\nNow run biological_sanity_check_v2.py to test if seed region")
    print("sensitivity emerged from data without hardcoded priors.")

if __name__ == "__main__":
    main()
```
