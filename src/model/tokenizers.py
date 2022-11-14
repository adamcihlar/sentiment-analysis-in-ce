from transformers import AutoTokenizer
from src.config.parameters import TOKENIZER_MODEL


class Tokenizer:
    """
    Easy initialization of the correct tokenizer that is stored in a way so it
    can be passed to the ClassificationDataset's tokenize method.
    """

    def __init__(self, tokenizer=TOKENIZER_MODEL):
        self.model = AutoTokenizer.from_pretrained(tokenizer)
