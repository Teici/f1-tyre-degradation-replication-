from __future__ import annotations
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

class StintSequenceDataset(Dataset):
    def __init__(self, df: pd.DataFrame, feature_cols, target_col: str, seq_len: int):
        self.seq_len = seq_len
        self.feature_cols = list(feature_cols)
        self.target_col = target_col

        df = df.sort_values(["stint_id", "lap"]).copy()

        self.groups = []
        for _, g in df.groupby("stint_id", sort=False):
            X = g[self.feature_cols].to_numpy(dtype=np.float32)
            y = g[target_col].to_numpy(dtype=np.float32)
            self.groups.append((X, y))

        self.index = []
        for gi, (X, y) in enumerate(self.groups):
            for t in range(self.seq_len - 1, len(y)):
                self.index.append((gi, t))

    def __len__(self):
        return len(self.index)

    def __getitem__(self, idx):
        gi, t = self.index[idx]
        X, y = self.groups[gi]
        start = t - self.seq_len + 1
        x_seq = X[start:t+1]
        y_t = y[t]
        return torch.from_numpy(x_seq), torch.tensor([y_t], dtype=torch.float32)
