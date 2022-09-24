import pandas as pd
import os
import fire
from loguru import logger

from src.config import paths, dictionaries


def _preprocess():
    texts = pd.read_table(
        os.path.join(paths.DATA_RAW_DIR_FACEBOOK, "gold-posts.txt"),
        header=None,
        names=["text"],
    )
    labels = pd.read_table(
        os.path.join(paths.DATA_RAW_DIR_FACEBOOK, "gold-labels.txt"),
        header=None,
        names=["label"],
    )
    labels = labels.replace({"label": dictionaries.LABEL_FROM_TEXT_TO_VALUE})

    df = pd.concat([texts, labels], axis=1).reset_index(drop=True)
    df["source"] = "facebook"
    return df


def _save(df):
    df.to_csv(paths.DATA_PROCESSED_FACEBOOK)
    logger.info(f"Preprocessed dataset saved at {paths.DATA_PROCESSED_FACEBOOK}")
    pass


def main():
    df = _preprocess()
    _save(df)
    pass


if __name__ == "__main__":
    fire.Fire(main)
