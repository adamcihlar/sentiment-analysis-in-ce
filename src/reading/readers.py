import os
import re

import pyreadr
import pandas as pd
from loguru import logger
from src.config import parameters, paths


def _check_file(path_to_file, warn=False):
    if not os.path.isfile(path_to_file):
        if warn:
            logger.warning(f"File not found at {path_to_file}")
        else:
            logger.error(f"File not found at {path_to_file}")
        return False
    return True


def read_csfd(path=paths.DATA_PROCESSED_CSFD):
    if _check_file(path):
        df = pd.read_csv(path, index_col=0)
        return df
    return None


def read_facebook(path=paths.DATA_PROCESSED_FACEBOOK):
    if _check_file(path):
        df = pd.read_csv(path, index_col=0)
        return df
    return None


def read_mall(path=paths.DATA_PROCESSED_MALL):
    if _check_file(path):
        df = pd.read_csv(path, index_col=0)
        return df
    return None


def read_all_source(path=paths.DATA_PROCESSED_CONCAT, merge_source=False):
    if _check_file(path):
        df = pd.read_csv(path, index_col=0)
        if merge_source:
            df["source"] = "all"
        return df
    return None


def read_finetuning_train_val(dataset: pd.DataFrame):
    dataset_name = dataset.source.iloc[0]
    train_path = os.path.join(
        paths.DATA_FINAL_FINETUNING_TRAIN,
        dataset_name + ".csv",
    )
    val_path = os.path.join(
        paths.DATA_FINAL_FINETUNING_VAL,
        dataset_name + ".csv",
    )
    if _check_file(train_path, warn=True) and _check_file(val_path, warn=True):
        train_ds = pd.read_csv(train_path, index_col=0)
        val_ds = pd.read_csv(val_path, index_col=0)
        logger.info(f"Dataset {dataset_name} found at {train_path} and {val_path}.")
        return train_ds, val_ds
    else:
        return dataset


def read_finetuning_source(
    selected_model=parameters.FINETUNED_CHECKPOINT,
    selected_dataset=parameters.FINETUNED_DATASET,
):
    # train_path = os.path.join(
    #     paths.DATA_FINAL_SOURCE_TRAIN,
    #     "_".join([selected_model, selected_dataset]) + ".csv",
    # )
    train_path = os.path.join(
        paths.DATA_FINAL_SOURCE_TRAIN,
        selected_dataset + ".csv",
    )
    train_ds = pd.read_csv(train_path, index_col=0)

    if re.search("_", selected_dataset):
        val_names = selected_dataset.split("_")
        val_paths = [
            os.path.join(
                paths.DATA_FINAL_SOURCE_VAL,
                "_".join([selected_model, selected_dataset]) + ".csv",
            )
            for selected_dataset in val_names
        ]
        val_datasets = [pd.read_csv(val_path, index_col=0) for val_path in val_paths]
        val_ds = pd.concat(val_datasets, axis=0)
        val_ds.source = selected_dataset
    else:
        val_path = os.path.join(
            paths.DATA_FINAL_SOURCE_VAL,
            "_".join([selected_model, selected_dataset]) + ".csv",
        )
        val_ds = pd.read_csv(val_path, index_col=0)
    return train_ds, val_ds


def read_adaptation_target(target_df: pd.DataFrame):
    source = target_df.source.iloc[0]
    target_path = os.path.join(paths.DATA_FINAL_ADAPTATION_TARGET, source + ".csv")
    if _check_file(target_path, warn=True):
        target_df = pd.read_csv(target_path, index_col=0)
    else:
        os.makedirs(os.path.split(target_path)[0], exist_ok=True)
        target_df.to_csv(target_path)
        logger.info(f"New dataset saved at {target_path}")
    return target_df


def read_raw_responses(path=paths.DATA_RAW_RESPONSES):
    if _check_file(path):
        responses = pyreadr.read_r(path)
        res_df = responses["responses"]
        return res_df
    return None


def read_raw_sent(path=paths.DATA_RAW_SENT):
    if _check_file(path):
        sent = pyreadr.read_r(path)
        res_df = sent["master_schedule"]
        return res_df
    return None


def read_preprocessed_emails(path=paths.DATA_PROCESSED_RESPONSES_CONFIRMED):
    if _check_file(path):
        res_df = pd.read_csv(path, index_col=0)
        return res_df
    return None
