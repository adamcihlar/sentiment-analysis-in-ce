import pandas as pd
import os
import fire
from loguru import logger

from src.config import paths


def _preprocess():
    df = pd.read_table(
        os.path.join(paths.DATA_RAW_DIR_EMAILS, "mails.txt"),
        header=None,
        names=["text"],
    )

    df["source"] = "emails"
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
