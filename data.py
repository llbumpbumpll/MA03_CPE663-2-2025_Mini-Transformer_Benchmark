from pathlib import Path

import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset

TOKEN_COLS = [f"token_{i:02d}" for i in range(1, 21)]
MASK_COLS = [f"mask_{i:02d}" for i in range(1, 21)]
PAD_ID = 0
VOCAB = {"PAD": 0, "A": 1, "B": 2, "C": 3, "D": 4}
MAX_LEN = 20


class CSVDataset(Dataset):
    def __init__(self, csv_path):
        self.csv_path = Path(csv_path)
        self.df = pd.read_csv(self.csv_path)
        missing = {"label", *TOKEN_COLS, *MASK_COLS} - set(self.df.columns)
        if missing:
            raise ValueError(f"{self.csv_path} is missing columns: {sorted(missing)}")

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        tokens = row[TOKEN_COLS].astype(int).values
        mask = row[MASK_COLS].astype(int).values
        label = int(row["label"])

        return {
            "tokens": torch.tensor(tokens, dtype=torch.long),
            "mask": torch.tensor(mask, dtype=torch.long),
            "label": torch.tensor(label, dtype=torch.long),
        }


def create_dataloaders(
    train_csv="data/train.csv",
    validation_csv="data/validation.csv",
    test_csv="data/test.csv",
    batch_size=32,
    num_workers=0,
):
    train_dataset = CSVDataset(train_csv)
    validation_dataset = CSVDataset(validation_csv)
    test_dataset = CSVDataset(test_csv)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
    )
    validation_loader = DataLoader(
        validation_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
    )
    return train_loader, validation_loader, test_loader


def describe_split(csv_path):
    df = pd.read_csv(csv_path)
    return {
        "rows": int(len(df)),
        "positive": int((df["label"] == 1).sum()),
        "negative": int((df["label"] == 0).sum()),
        "min_length": int(df["seq_len"].min()),
        "max_length": int(df["seq_len"].max()),
        "mean_length": float(df["seq_len"].mean()),
    }
