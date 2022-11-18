import pandas as pd
import torch
from typing import List
from sklearn.model_selection import train_test_split

from src.config.parameters import RANDOM_STATE


def drop_undefined_classes(dataset: pd.DataFrame):
    dataset = dataset.loc[dataset.label.isin([0, 1, 2])]
    return dataset


def get_finetuning_datasets(source_dataset: pd.DataFrame):
    """
    Getting the dataset for finetuning is not dependant on the target dataset.
    Basically just splits the source dataset to train and val.
    """
    source_train_X, source_val_X, source_train_y, source_val_y = train_test_split(
        source_dataset.text,
        source_dataset.label,
        test_size=0.2,
        random_state=RANDOM_STATE,
    )
    source_train = pd.DataFrame({"text": source_train_X, "label": source_train_y})
    source_val = pd.DataFrame({"text": source_val_X, "label": source_val_y})
    return source_train, source_val


def get_adaptation_datasets(
    source_train_dataset_with_predictions: pd.DataFrame,
    source_val_dataset: pd.DataFrame,
    target_dataset: pd.DataFrame,
):
    """
    After the source encoder and classifier are trained,
    get the source train dataset with predictions to train on it in step 2b
    with the predicted probabilities for the distilation.
    Keep the source validation dataset with original labels to track
    how the target encoder and classier are performing on that.
    Sample the source train dataset based on the length of target dataset
    to get as much information as possible from the target while adapting.

    Returns:
        adaptation_source_train, that will be used in both adaptation steps, but with different labels
            - in 2a with artificial domain labels and in 2b with sentiment predicted probabilities labels
        adaptation_source_val, that will be used in 2b to check how the target encoder is performing on the
            original task
        adaptation_target, that will be used in 2a to train the discriminator and target encoder
            - both with artificial labels, for the discriminator the label is "target domain" and for
            target encoder the label is "source domain"
            - this dataset is train and test at the same time, just (of course) using different labels
    """
    replace_samples = len(target_dataset) > len(source_train_dataset_with_predictions)
    adaptation_source_train = source_train_dataset_with_predictions.sample(
        len(target_dataset), replace=replace_samples
    )
    adaptation_source_val = source_val_dataset
    adaptation_target = target_dataset
    return adaptation_source_train, adaptation_source_val, adaptation_target


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
        X: pd.Series,
        y: pd.Series,
    ):
        """
        Stores the texts and labels.
        Drops rows where label is not in (0, 1, 2).
        """
        self.X = X
        self.y = y
        self.X_tok = None
        self.X_preprocessed = None
        self.y_pred = None

    def transform_labels_to_probs(self, drop_neutral=True):
        """
        Based on how I want to finetune the source classifier, I need to transform the labels.
        I can either drop the neutral class to have binary classification or make it 0.5 and finetune in distilationlike settings.
        """
        if drop_neutral:
            logger.info(
                f"Dropped neutral class, data transformed to binary classification problem."
            )
            self.X = self.X.loc[self.y.isin([0, 1])]
            self.y = self.y.loc[self.y.isin([0, 1])]
        else:
            logger.info(
                f"Neutral class labelled as 0.5, data transformed to distilationlike settings."
            )
            self.y.loc[self.y == 2] = 0.5
        pass

    def report_metric(self, metric):
        """
        Return the selected metric based on the true and predicted labels.
        """
        pass

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
