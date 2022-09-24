import pandas as pd
import os

from src.config import paths


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
