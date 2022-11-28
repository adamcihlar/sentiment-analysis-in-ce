import pandas as pd
import torch
from typing import List
from sklearn.model_selection import train_test_split
from loguru import logger

from src.config.parameters import RANDOM_STATE


def drop_undefined_classes(dataset: pd.DataFrame):
    dataset = dataset.loc[dataset.label.isin([0, 1, 2])]
    return dataset


def transform_labels_to_probs(dataset, drop_neutral=True):
    """
    Based on how I want to finetune the source classifier, I need to transform the labels.
    I can either drop the neutral class to have binary classification or make it 0.5 and finetune in distilationlike settings.
    """
    if drop_neutral:
        sum_neutral = sum(dataset.label == 2)
        logger.info(
            f"Dropped neutral class - {sum_neutral} samples, data transformed to binary classification problem."
        )
        dataset = dataset.loc[dataset.label.isin([0, 1])]
    else:
        logger.info(
            f"Neutral class labelled as 0.5, data transformed to distilationlike settings."
        )
        dataset.label.loc[dataset.label == 2] = 0.5
    return dataset


def get_finetuning_datasets(source_dataset: pd.DataFrame):
    """
    Getting the dataset for finetuning is not dependant on the target dataset.
    Basically just splits the source dataset to train and val.
    """
    source_train_X, source_val_X, source_train_y, source_val_y = train_test_split(
        source_dataset.loc[:, source_dataset.columns != "label"],
        source_dataset.label,
        test_size=0.2,
        random_state=RANDOM_STATE,
    )
    source_train = pd.concat([source_train_X, source_train_y], axis=1)
    source_val = pd.concat([source_val_X, source_val_y], axis=1)
    return source_train, source_val


def get_adaptation_datasets(
    source_train_df: pd.DataFrame,
    source_val_df: pd.DataFrame,
    target_df: pd.DataFrame,
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

    def __init__(self, encodings, labels=None, source=None):
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
        y: pd.Series = None,
        source: pd.Series = None,
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
        self.source = source

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
        pass

    def tokenize(
        self, tokenizer, padding="max_length", max_length=512, truncation=True
    ):
        """
        Returns tokenized inputs and stores them at the same time.
        """
        self.X_tok = tokenizer.tokenize(
            list(self.X_preprocessed),
            max_length=max_length,
        )
        pass

    def create_dataset(self):
        torch_y = list(self.y) if self.y is not None else None
        self.torch_dataset = Dataset(self.X_tok, torch_y)
        pass

    def create_dataloader(self, batch_size=64, shuffle=True, num_workers=0):
        self.torch_dataloader = torch.utils.data.DataLoader(
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

    def save_data(self, save_path):
        df = pd.DataFrame({"text": self.X, "label": self.y, "source": self.source})
        df.to_csv(save_path)
        pass


def get_datasets_ready_for_finetuning(
    datasets: List[pd.DataFrame],
    drop_neutral,
    preprocessor,
    tokenizer,
    batch_size,
    shuffle,
    num_workers,
):
    """
    Just wrapping many functions and methods that are common for preparing
    source finetuning train and validation datasets.
    Takes list of source datasets, performs basic preprocessing operations,
    splits them to train and validation parts and ClassificationDataset
    instances from all of them with torch dataset and dataloader ready for
    training/evaluation.
    """
    datasets = [drop_undefined_classes(ds) for ds in datasets]
    datasets = [
        transform_labels_to_probs(ds, drop_neutral=drop_neutral) for ds in datasets
    ]
    datasets = [get_finetuning_datasets(ds) for ds in datasets]
    train_datasets, val_datasets = list(zip(*datasets))

    train_datasets = {
        ds.source.iloc[0]: ClassificationDataset(ds.text, ds.label, ds.source)
        for ds in train_datasets
    }
    val_datasets = {
        ds.source.iloc[0]: ClassificationDataset(ds.text, ds.label, ds.source)
        for ds in val_datasets
    }

    [ds.preprocess(preprocessor) for ds in train_datasets.values()]
    [ds.tokenize(tokenizer) for ds in train_datasets.values()]
    [ds.create_dataset() for ds in train_datasets.values()]
    [
        ds.create_dataloader(
            batch_size=batch_size, shuffle=shuffle, num_workers=num_workers
        )
        for ds in train_datasets.values()
    ]

    [ds.preprocess(preprocessor) for ds in val_datasets.values()]
    [ds.tokenize(tokenizer) for ds in val_datasets.values()]
    [ds.create_dataset() for ds in val_datasets.values()]
    [
        ds.create_dataloader(
            batch_size=batch_size, shuffle=False, num_workers=num_workers
        )
        for ds in val_datasets.values()
    ]
    return train_datasets, val_datasets


def get_datasets_ready_for_adaptation(
    source_train_df: pd.DataFrame,
    source_val_df: pd.DataFrame,
    target_df: pd.DataFrame,
    preprocessor,
    tokenizer,
    batch_size,
    shuffle,
    num_workers,
):
    source_train_df, source_val_df, target_df = get_adaptation_datasets(
        source_train_df, source_val_df, target_df
    )
    source_train = ClassificationDataset(
        source_train_df.text, source_train_df.label, source_train_df.source
    )
    source_val = ClassificationDataset(
        source_val_df.text, source_val_df.label, source_val_df.source
    )
    target = ClassificationDataset(target_df.text, target_df.label, target_df.source)

    [ds.preprocess(preprocessor) for ds in [source_train, source_val, target]]
    [ds.tokenize(tokenizer) for ds in [source_train, source_val, target]]
    [ds.create_dataset() for ds in [source_train, source_val, target]]
    [
        ds.create_dataloader(batch_size, shuffle, num_workers)
        for ds in [source_train, target]
    ]
    source_val.create_dataloader(batch_size, shuffle=False, num_workers=num_workers)
    return source_train, source_val, target
