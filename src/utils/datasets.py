import pandas as pd
import torch
from typing import List


class Dataset(torch.utils.data.Dataset):
    """Keeps data in format accepted by the torch Dataloader."""

    def __init__(self, encodings, labels=None):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        if self.labels:
            item["labels"] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.encodings["input_ids"])


class ClassificationDataset:
    """
    Implements methods to do all the needed operations on a dataset, like:
        preprocessing, tokenizing, creation of torch dataset and dataloader
    and stores the data in all these formats derived from the original data for one dataset.
    """

    def __init__(
        self,
        X: List[str],
        y: List = None,
    ):
        self.X = X
        self.y = y
        self.X_tok = None
        self.X_preprocessed = None
        self.y_pred = None

    def preprocess(self, preprocessor):
        """
        This will be implemented later when I have the emails
        The main functionality will be to get the meat from the emails.
        """
        self.X_preprocessed = preprocessor.preprocess(self.X)
        return self.X_preprocessed

    def tokenize(
        self, tokenizer, padding="max_length", max_length=512, truncation=True
    ):
        """
        Returns tokenized inputs and stores them at the same time.
        """
        self.X_tok = tokenizer.model(
            self.X_preprocessed,
            padding=padding,
            max_length=max_length,
            truncation=truncation,
        )
        return self.X_tok

    def create_dataset(self):
        self.torch_dataset = Dataset(self.X_tok, self.y)
        pass

    def create_dataloader(self, batch_size, shuffle, num_workers):
        self.torch_dataloader = DataLoader(
            self.torch_dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
        )
        pass

    def get_predictions(self, include_inputs=True):
        if include_inputs:
            predictions = pd.DataFrame({"X": self.X, "y_pred": self.y_pred})
        else:
            predictions = pd.DataFrame({"y_pred": self.y_pred})
        return predictions

    def save_predictions(self):
        raise NotImplementedError
