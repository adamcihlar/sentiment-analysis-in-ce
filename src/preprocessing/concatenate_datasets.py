import pandas as pd
import fire
from loguru import logger

from src.config import paths
from src.reading.readers import read_csfd, read_facebook, read_mall


def concatenate_datasets():
    csfd = read_csfd()
    facebook = read_facebook()
    mall = read_mall()

    df = pd.concat([csfd, facebook, mall], axis=0)
    return df


def _save(df):
    df.to_csv(paths.DATA_PROCESSED_CONCAT)
    logger.info(f"Concatenated dataset saved at {paths.DATA_PROCESSED_CONCAT}")
    pass


def main():
    df = concatenate_datasets()
    _save(df)
    pass


if __name__ == "__main__":
    fire.Fire(main)
