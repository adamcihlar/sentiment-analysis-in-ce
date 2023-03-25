import subprocess
import fire
from loguru import logger
from typing import List

from src.config import paths

https://ucnmuni-my.sharepoint.com/:u:/g/personal/468087_muni_cz/EbOMD975TUlJiuUI1vuGgZABD4qigN-t8DcN-JbZAI5S-Q?e=DA27OX



command = ("""curl https://ucnmuni-my.sharepoint.com/personal/468087_muni_cz/_layouts/15/download.aspx?SourceUrl=%2Fpersonal%2F468087%5Fmuni%5Fcz%2FDocuments%2Fsentiment%2Danalysis%2Din%2Dce%2Fmodels%2Fseznamsmall%2De%2Dczech%5F20221204%2D221123%5F3"""
    + " --output "
    + paths.OUTPUT_MODELS_ADAPTED_ENCODER_FINAL
)
run = subprocess.run(command, shell=True)
