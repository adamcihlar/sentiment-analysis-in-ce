import subprocess
import fire
import os
from loguru import logger
from typing import List

from src.config import paths
from src.ingestion.download_data import download_csfd, download_mall, download_facebook


def _check_zip_file(path_to_file):
    if os.path.isfile(path_to_file):
        logger.info(f"Zipped file found at {path_to_file}")
        return True
    else:
        logger.info(f"Zipped file not found at {path_to_file}")
        return False


def unzip_facebook():
    """
    Unzip facebook data to separate folder, moves all the extracted files to
    the root of the specified folder and deletes empty subfolders.
    """
    if not _check_zip_file(paths.DATA_RAW_ZIP_FACEBOOK):
        download_facebook()

    command = (
        "unzip -o " + paths.DATA_RAW_ZIP_FACEBOOK + " -d " + paths.DATA_RAW_DIR_FACEBOOK
    )
    run = subprocess.run(command, shell=True)
    if run.returncode == 0:
        logger.info(
            f"Successfully unzipped Facebook dataset from {paths.DATA_RAW_ZIP_FACEBOOK}."
        )
        for path, subdirs, files in os.walk(paths.DATA_RAW_DIR_FACEBOOK):
            for f in files:
                os.rename(
                    os.path.join(path, f), os.path.join(paths.DATA_RAW_DIR_FACEBOOK, f)
                )
    subprocess.run(
        f"find {paths.DATA_RAW_DIR_FACEBOOK} -type d -empty -delete", shell=True
    )
    pass


def unzip_csfd():
    """
    Unzip csfd data to separate folder, moves all the extracted files to
    the root of the specified folder and deletes empty subfolders.
    """
    if not _check_zip_file(paths.DATA_RAW_ZIP_CSFD):
        download_csfd()

    command = "unzip -o " + paths.DATA_RAW_ZIP_CSFD + " -d " + paths.DATA_RAW_DIR_CSFD
    run = subprocess.run(command, shell=True)
    if run.returncode == 0:
        logger.info(
            f"Successfully unzipped CSFD dataset from {paths.DATA_RAW_ZIP_CSFD}."
        )
        for path, subdirs, files in os.walk(paths.DATA_RAW_DIR_CSFD):
            for f in files:
                os.rename(
                    os.path.join(path, f), os.path.join(paths.DATA_RAW_DIR_CSFD, f)
                )
    subprocess.run(f"find {paths.DATA_RAW_DIR_CSFD} -type d -empty -delete", shell=True)
    pass


def unzip_mall():
    """
    Unzip mall data to separate folder, moves all the extracted files to
    the root of the specified folder and deletes empty subfolders.
    """
    if not _check_zip_file(paths.DATA_RAW_ZIP_MALL):
        download_mall()

    command = "unzip -o " + paths.DATA_RAW_ZIP_MALL + " -d " + paths.DATA_RAW_DIR_MALL
    run = subprocess.run(command, shell=True)
    if run.returncode == 0:
        logger.info(
            f"Successfully unzipped Mall dataset from {paths.DATA_RAW_ZIP_MALL}."
        )
        for path, subdirs, files in os.walk(paths.DATA_RAW_DIR_MALL):
            for f in files:
                os.rename(
                    os.path.join(path, f), os.path.join(paths.DATA_RAW_DIR_MALL, f)
                )
    subprocess.run(f"find {paths.DATA_RAW_DIR_MALL} -type d -empty -delete", shell=True)
    pass


def unzip_emails():
    """
    Unzip emails to separate folder, moves all the extracted files to
    the root of the specified folder and deletes empty subfolders.
    """
    if not _check_zip_file(paths.DATA_RAW_ZIP_EMAILS):
        logger.error(f"File {paths.DATA_RAW_ZIP_EMAILS} not found.")

    command = (
        "unzip -o " + paths.DATA_RAW_ZIP_EMAILS + " -d " + paths.DATA_RAW_DIR_EMAILS
    )
    run = subprocess.run(command, shell=True)
    if run.returncode == 0:
        logger.info(
            f"Successfully unzipped emails dataset from {paths.DATA_RAW_ZIP_EMAILS}."
        )
        for path, subdirs, files in os.walk(paths.DATA_RAW_DIR_EMAILS):
            for f in files:
                os.rename(
                    os.path.join(path, f), os.path.join(paths.DATA_RAW_DIR_EMAILS, f)
                )
    subprocess.run(
        f"find {paths.DATA_RAW_DIR_EMAILS} -type d -empty -delete", shell=True
    )
    pass


def main(datasets: List = []):
    if datasets == []:
        unzip_facebook()
        unzip_csfd()
        unzip_mall()
    else:
        for data in datasets:
            if data == "facebook":
                unzip_facebook()
            elif data == "csfd":
                unzip_csfd()
            elif data == "mall":
                unzip_mall()
            else:
                logger.error(f"No method to unzip dataset: {data}")
    pass


if __name__ == "__main__":
    fire.Fire(main)
