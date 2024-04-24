import numpy as np
import pandas as pd
import torch

from pathlib import Path
from torch.utils.data import Dataset

from .preprocess import Preprocessor

class BinaryRankDataset(Dataset):
    def __init__(self,
        data_path: Path,
        threshold: int | None = None,
        preprocessor: Preprocessor | None = None
    ):
        self.data = pd.read_csv(data_path)

        text_columns = set(self.data.columns).intersection({"title", "text", "query"})

        self.data.dropna(subset=["query"], inplace=True)
        self.data[text_columns].fillna("", inplace=True)

        self.threshold = threshold
        self.preprocessor = preprocessor
        if self.threshold is not None:
            self.data.label = np.where(self.data.label >= self.threshold, 1., 0.)
        else:
            if self.data.label.dtype != bool:
                if not np.all(self.data.label.isin([0,1])):
                    raise ValueError("Labels must be bool or binary")
                self.data.label = self.data.label.astype(float)
            else:
                self.data.label = np.where(self.data.label, 1., 0.)

    def __getitem__(self, index):
        item = self.data.iloc[index].to_dict()
        item = self.preprocessor.process(item)
        return item

    def __len__(self):
        return len(self.data)
    
class RankDataset(Dataset):
    def __init__(self,
        data_path: Path,
        preprocessor: Preprocessor | None = None
    ):
        self.data = pd.read_csv(data_path)

        text_columns = list(set(self.data.columns).intersection({"title", "text", "query"}))

        self.data.dropna(subset=["query"], inplace=True)
        self.data[text_columns] = self.data[text_columns].fillna("")
        self.data["label"] = self.data["label"] / self.data["label"].max()

        self.preprocessor = preprocessor
        
    def __getitem__(self, index):
        item = self.data.iloc[index].to_dict()
        item = self.preprocessor.process(item)
        return item

    def __len__(self):
        return len(self.data)