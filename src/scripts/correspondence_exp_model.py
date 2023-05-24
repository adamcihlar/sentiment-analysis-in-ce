import os
import pandas as pd
from src.config import paths
from src.reading.readers import read_raw_sent, read_raw_responses

responses_df = read_preprocessed_emails()
sent_df = read_raw_sent()
