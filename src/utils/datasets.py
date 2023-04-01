import json
import math
import os
from typing import List

import numpy as np
import pandas as pd
import torch
from loguru import logger
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from src.config import parameters, paths
from src.config.parameters import RANDOM_STATE, TokenizerParams
from src.reading.readers import read_adaptation_target, read_finetuning_train_val


def drop_undefined_classes(dataset: pd.DataFrame):
    dataset = dataset.loc[dataset.label.isin([0, 1, 2])]
    return dataset


def transform_labels(dataset, transformation="ordinal_regression"):
    """
    Based on how I want to finetune the source classifier, I need to transform the labels.
    I can either drop the neutral class to have binary classification or make it 0.5 and finetune in distilationlike settings.
    """
    if transformation == "drop_neutral":
        sum_neutral = sum(dataset.label == 1)
        logger.info(
            f"Dropped neutral class - {sum_neutral} samples, data transformed to binary classification problem."
        )
        dataset = dataset.loc[dataset.label.isin([0, 2])]
        dataset.label.loc[dataset.label == 2] = 1
    elif transformation == "ordinal_regression":
        logger.info(
            f"Labels kept in order negative - neutral - positive to allow for ordinal regression."
        )
    elif transformation == "multiclass_classification":
        logger.info(f"Labels kept in original settings.")
    else:
        raise NotImplementedError
    return dataset


def random_undersampling(dataset, majority_ratio=1, random_state=42):
    label_counts = dataset.label.value_counts()
    minor_samples_count = np.min(label_counts)
    minor_class = label_counts.index.values[np.argmin(label_counts)]
    major_samples_count = minor_samples_count * majority_ratio
    sampled_majority = dataset.loc[dataset.label != minor_class].sample(
        major_samples_count, replace=False, random_state=random_state
    )
    minority = dataset.loc[dataset.label == minor_class]
    balanced_dataset = pd.concat([minority, sampled_majority])
    return balanced_dataset


def filter_min_query_len(dataset, min_query_len):
    mask = dataset.text.str.findall(r"[\w]+").str.len() >= min_query_len
    return dataset.loc[mask]


def get_finetuning_datasets(source_dataset: pd.DataFrame):
    """
    Getting the dataset for finetuning is not dependant on the target dataset.
    Basically just splits the source dataset to train and val and saves the
    splits to
    """
    source_train_X, source_val_X, source_train_y, source_val_y = train_test_split(
        source_dataset.loc[:, source_dataset.columns != "label"],
        source_dataset.label,
        test_size=0.2,
        random_state=RANDOM_STATE,
    )
    source_train = pd.concat([source_train_X, source_train_y], axis=1)
    source_val = pd.concat([source_val_X, source_val_y], axis=1)

    source_val_X, source_test_X, source_val_y, source_test_y = train_test_split(
        source_val.loc[:, source_val.columns != "label"],
        source_val.label,
        test_size=0.5,
        random_state=RANDOM_STATE,
    )
    source_val = pd.concat([source_val_X, source_val_y], axis=1)
    source_test = pd.concat([source_test_X, source_test_y], axis=1)

    train_path = os.path.join(
        paths.DATA_FINAL_FINETUNING_TRAIN,
        source_train.source.iloc[0] + ".csv",
    )
    os.makedirs(os.path.split(train_path)[0], exist_ok=True)

    val_path = os.path.join(
        paths.DATA_FINAL_FINETUNING_VAL,
        source_val.source.iloc[0] + ".csv",
    )
    os.makedirs(os.path.split(val_path)[0], exist_ok=True)

    test_path = os.path.join(
        paths.DATA_FINAL_FINETUNING_TEST,
        source_test.source.iloc[0] + ".csv",
    )
    os.makedirs(os.path.split(test_path)[0], exist_ok=True)

    source_train.to_csv(train_path)
    source_val.to_csv(val_path)
    source_test.to_csv(test_path)
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
    replace_samples = len(target_df) > len(source_train_df)
    # take same amount of positive and negative samples from the source train
    unique_labels = source_train_df.label.unique()
    source_train_dfs = []
    for label in unique_labels:
        source_train_dfs.append(
            source_train_df.loc[source_train_df.label == label].sample(
                math.ceil(len(target_df) / len(unique_labels)),
                replace=replace_samples,
                random_state=parameters.RANDOM_STATE,
            )
        )
    adaptation_source_train = pd.concat(source_train_dfs, axis=0).iloc[
        0 : len(target_df)
    ]

    replace_samples = len(target_df) > len(source_val_df)
    if replace_samples:
        adaptation_source_val = source_val_df
    else:
        # take same amount of positive and negative samples from the source val
        unique_labels = source_val_df.label.unique()
        source_val_dfs = []
        for label in unique_labels:
            source_val_dfs.append(
                source_val_df.loc[source_val_df.label == label].sample(
                    math.ceil(len(target_df) / len(unique_labels)),
                    replace=replace_samples,
                    random_state=parameters.RANDOM_STATE,
                )
            )
        adaptation_source_val = pd.concat(source_val_dfs, axis=0).iloc[
            0 : len(target_df)
        ]

    adaptation_target = target_df
    return adaptation_source_train, adaptation_source_val, adaptation_target


def save_train_info(file, path):
    os.makedirs(os.path.split(path)[0], exist_ok=True)
    saved = 0
    while saved != 1:
        try:
            with open(path, "w+") as fp:
                json.dump(file, fp)
        except IOError:
            pass
        else:
            saved = 1
            logger.info(f"Train info saved at {path}.")
    pass


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

    def evaluation_report(self, save_path):
        """
        Return confusion matrix and other metrics.
        """
        print(classification_report(self.y, self.y_pred))
        report = classification_report(self.y, self.y_pred, output_dict=True)
        print(confusion_matrix(self.y, self.y_pred))
        json_report = json.dumps(report, indent=4)
        with open(save_path, "w") as outfile:
            outfile.write(json_report)
        pass

    def preprocess(self, preprocessor):
        """
        This will be implemented later when I have the emails
        The main functionality will be to get the meat from the emails.
        """
        self.X_preprocessed = preprocessor.preprocess(self.X)
        pass

    def tokenize(
        self,
        tokenizer,
        padding="max_length",
        max_length=TokenizerParams.MAX_LENGTH,
        truncation=TokenizerParams.COMBINED_TRUNCATION,
    ):
        """
        Returns tokenized inputs and stores them at the same time.
        """
        self.X_tok = tokenizer.tokenize(
            list(self.X_preprocessed),
            max_length=max_length,
            combined_truncation=truncation,
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

    def save_predictions(self, path):
        df = pd.DataFrame({"text": self.X, "label": self.y, "label_pred": self.y_pred})
        os.makedirs(os.path.split(path)[0], exist_ok=True)
        saved = 0
        while saved != 1:
            try:
                df.to_csv(path)
            except IOError:
                pass
            else:
                saved = 1
                logger.info(f"Dataset saved at {path}.")
        pass

    def save_data(self, path):
        df = pd.DataFrame({"text": self.X, "label": self.y, "source": self.source})
        os.makedirs(os.path.split(path)[0], exist_ok=True)
        saved = 0
        while saved != 1:
            try:
                df.to_csv(path)
            except IOError:
                pass
            else:
                saved = 1
                logger.info(f"Dataset saved at {path}.")
        pass


def get_datasets_ready_for_finetuning(
    datasets: List[pd.DataFrame],
    transformation,
    balance_data,
    majority_ratio,
    preprocessor,
    tokenizer,
    batch_size,
    shuffle,
    num_workers,
    skip_validation,
    min_query_len,
    share_classifier,
):
    """
    Just wrapping many functions and methods that are common for preparing
    source finetuning train and validation datasets.
    Takes list of source datasets, performs basic preprocessing operations,
    splits them to train and validation parts and ClassificationDataset
    instances from all of them with torch dataset and dataloader ready for
    training/evaluation.
    """
    # try to find the already created train and val datasets
    datasets = [read_finetuning_train_val(ds) for ds in datasets]
    # if you cannot find it, create a new split and save it
    datasets = [
        get_finetuning_datasets(ds) if isinstance(ds, pd.DataFrame) else ds
        for ds in datasets
    ]

    # get trains and vals in lists
    train_datasets, val_datasets = list(zip(*datasets))

    # do this for both train and val
    train_datasets = [drop_undefined_classes(ds) for ds in train_datasets]
    val_datasets = [drop_undefined_classes(ds) for ds in val_datasets]

    # both
    train_datasets = [
        transform_labels(ds, transformation=transformation) for ds in train_datasets
    ]
    val_datasets = [
        transform_labels(ds, transformation=transformation) for ds in val_datasets
    ]

    # only train
    if balance_data:
        train_datasets = [
            random_undersampling(ds, majority_ratio, RANDOM_STATE)
            for ds in train_datasets
        ]
        majority_ratio = "N/A"

    if skip_validation:
        val_datasets = [ds.iloc[0] for ds in val_datasets]

    train_datasets = [filter_min_query_len(ds, min_query_len) for ds in train_datasets]

    if share_classifier:
        name = "_".join([ds.source.iloc[0] for ds in train_datasets])
        ds = pd.concat(train_datasets, axis=0)
        ds["source"] = name
        train_datasets = {name: ClassificationDataset(ds.text, ds.label, ds.source)}
    else:
        train_datasets = {
            ds.source.iloc[0]: ClassificationDataset(ds.text, ds.label, ds.source)
            for ds in train_datasets
        }

    val_datasets = {
        ds.source.iloc[0]: ClassificationDataset(ds.text, ds.label, ds.source)
        for ds in val_datasets
    }

    [ds.preprocess(preprocessor) for ds in train_datasets.values()]
    if tokenize_during_training:
        pass
    else:
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
    [
        logger.info(
            f"Resulting train dataset {ds_name} has {len(train_datasets[ds_name].X)} rows and classes in ratio {majority_ratio}:1."
        )
        for ds_name in train_datasets
    ]
    [
        logger.info(
            f"Resulting val dataset {ds_name} has {len(val_datasets[ds_name].X)} rows and classes in ratio {majority_ratio}:1."
        )
        for ds_name in val_datasets
    ]
    return train_datasets, val_datasets


def get_datasets_ready_for_adaptation(
    source_train_df: pd.DataFrame,
    source_val_df: pd.DataFrame,
    target_df: pd.DataFrame,
    transformation,
    preprocessor,
    tokenizer,
    batch_size,
    shuffle,
    num_workers,
    skip_validation,
):
    # check if target df is from training sets
    if "source" in target_df:
        orig_len = len(target_df)
        target_df = read_adaptation_target(target_df)
        loaded_len = len(target_df)
        if orig_len != loaded_len:
            logger.warning(
                f"Lengths of provided and loaded target datasets differ. Please delete the target datasets stored at {paths.DATA_FINAL_ADAPTATION_TARGET} to provide the model with your latest data"
            )
    if "label" in target_df and target_df.label is not None:
        target_df = drop_undefined_classes(target_df)
        target_df = transform_labels(target_df, transformation=transformation)

    source_train_df, source_val_df, target_df = get_adaptation_datasets(
        source_train_df, source_val_df, target_df
    )

    source_train = ClassificationDataset(
        source_train_df.text, source_train_df.label, source_train_df.source
    )

    if skip_validation:
        source_val = souce_val.iloc[0]
    source_val = ClassificationDataset(
        source_val_df.text, source_val_df.label, source_val_df.source
    )
    if "source" in target_df and "label" in target_df:
        target = ClassificationDataset(
            target_df.text, target_df.label, target_df.source
        )
    else:
        target = ClassificationDataset(target_df.text, None, "emails")

    [ds.preprocess(preprocessor) for ds in [source_train, source_val, target]]
    logger.info("Preprocessing completed.")
    [ds.tokenize(tokenizer) for ds in [source_train, source_val, target]]
    logger.info("Tokenization completed.")
    [ds.create_dataset() for ds in [source_train, source_val, target]]
    [
        ds.create_dataloader(batch_size, shuffle, num_workers)
        for ds in [source_train, target]
    ]
    source_val.create_dataloader(batch_size, shuffle=False, num_workers=num_workers)
    logger.info("Dataloaders ready.")
    return source_train, source_val, target
