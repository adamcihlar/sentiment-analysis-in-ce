import pandas as pd
import os
from loguru import logger

from src.config import paths
from src.config import parameters


def _check_file(path_to_file):
    if not os.path.isfile(path_to_file):
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


def read_all_source(path=paths.DATA_PROCESSED_CONCAT):
    if _check_file(path):
        df = pd.read_csv(path, index_col=0)
        return df
    return None


def read_finetuning_source(
    selected_model=parameters.FINETUNED_CHECKPOINT,
    selected_dataset=parameters.FINETUNED_DATASET,
):
    train_path = os.path.join(
        paths.DATA_FINAL_SOURCE_TRAIN,
        "_".join([selected_model, selected_dataset] + ".csv"),
    )
    val_path = os.path.join(
        paths.DATA_FINAL_SOURCE_VAL,
        "_".join([selected_model, selected_dataset] + ".csv"),
    )
    train_ds = pd.read_csv(train_path, index_col=0)
    val_ds = pd.read_csv(val_path, index_col=0)
    return train_ds, val_ds
