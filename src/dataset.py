import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


class PhoenixCSLTDataset(Dataset):
    """
    Dataset for PHOENIX CSL → Text

    Returns:
      feats:      (max_T, 225)
      feat_mask: (max_T,)
      text:      string
    """

    def __init__(self, manifest_csv, max_frames=200):
        self.df = pd.read_csv(manifest_csv)
        self.max_frames = max_frames

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        feats = np.load(row["feature_path"])  # (T, 225)
        T, D = feats.shape

        if T >= self.max_frames:
            feats = feats[: self.max_frames]
            mask = np.ones(self.max_frames, dtype=np.float32)
        else:
            pad = self.max_frames - T
            feats = np.concatenate(
                [feats, np.zeros((pad, D), dtype=np.float32)], axis=0
            )
            mask = np.concatenate(
                [np.ones(T, dtype=np.float32), np.zeros(pad, dtype=np.float32)]
            )

        return {
            "feats": torch.from_numpy(feats).float(),
            "feat_mask": torch.from_numpy(mask).float(),
            "text": row["translation"],
        }
