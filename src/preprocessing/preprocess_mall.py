import pandas as pd
import os
import fire
from loguru import logger

from src.config import paths


def _preprocess():
    negative = pd.read_table(
        os.path.join(paths.DATA_RAW_DIR_MALL, "negative.txt"),
        header=None,
        names=["text"],
    )
    negative["label"] = 0
    positive = pd.read_table(
        os.path.join(paths.DATA_RAW_DIR_MALL, "positive.txt"),
        header=None,
        names=["text"],
    )
    positive["label"] = 2
    neutral = pd.read_table(
        os.path.join(paths.DATA_RAW_DIR_MALL, "neutral.txt"),
        header=None,
        names=["text"],
    )
    neutral["label"] = 1

    df = pd.concat([negative, positive, neutral], axis=0).reset_index(drop=True)
    df["source"] = "mall"
    return df


def _save(df):
    df.to_csv(paths.DATA_PROCESSED_MALL)
    logger.info(f"Preprocessed dataset saved at {paths.DATA_PROCESSED_MALL}")
    pass


def main():
    df = _preprocess()
    _save(df)
    pass


if __name__ == "__main__":
    fire.Fire(main)
