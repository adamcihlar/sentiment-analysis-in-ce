import subprocess
import fire
from loguru import logger
from typing import List

from src.config import urls, paths


def download_facebook():
    command = (
        "curl --remote-name-all "
        + urls.FACEBOOK_DATA
        + " --output "
        + paths.DATA_RAW_ZIP_FACEBOOK
    )
    run = subprocess.run(command, shell=True)
    if run.returncode == 0:
        logger.info(
            f"Successfully downloaded Facebook dataset from {urls.FACEBOOK_DATA}. \nFile located at: {paths.DATA_RAW_ZIP_FACEBOOK}"
        )
    pass


def download_csfd():
    command = (
        "curl --remote-name-all "
        + urls.CSFD_DATA
        + " --output "
        + paths.DATA_RAW_ZIP_CSFD
    )
    run = subprocess.run(command, shell=True)
    if run.returncode == 0:
        logger.info(
            f"Successfully downloaded CSFD dataset from {urls.CSFD_DATA}.  \nFile located at: {paths.DATA_RAW_ZIP_CSFD}"
        )
    pass


def download_mall():
    command = (
        "curl --remote-name-all "
        + urls.MALL_DATA
        + " --output "
        + paths.DATA_RAW_ZIP_MALL
    )
    run = subprocess.run(command, shell=True)
    if run.returncode == 0:
        logger.info(
            f"Successfully downloaded Mall dataset from {urls.MALL_DATA}. \nFile located at: {paths.DATA_RAW_ZIP_MALL}"
        )
    pass


def main(datasets: List = []):
    if datasets == []:
        download_facebook()
        download_csfd()
        download_mall()
    else:
        for data in datasets:
            if data == "facebook":
                download_facebook()
            elif data == "csfd":
                download_csfd()
            elif data == "mall":
                download_mall()
            else:
                logger.error(f"No method to download dataset: {data}")
    pass


if __name__ == "__main__":
    fire.Fire(main)
