from transformers import AutoTokenizer
from typing import List, Tuple

from src.config.parameters import TokenizerParams


class Tokenizer:
    """
    Easy initialization of the correct tokenizer that is stored in a way so it
    can be passed to the ClassificationDataset's tokenize method.
    """

    def __init__(self, tokenizer=TokenizerParams.MODEL):
        self.model = AutoTokenizer.from_pretrained(tokenizer)

    def __truncate_tokenized_sequence__(self, tok: Tuple):
        len_sentence = sum(tok[1])
        if len_sentence > 512:
            input_ids = tok[0][0:129] + tok[0][0:len_sentence][-383:]
            attention_mask = [0] * 512
        else:
            input_ids = tok[0][0:512]
            attention_mask = tok[1][0:512]
        return input_ids, attention_mask

    def tokenize(self, texts: List[str], max_length=512, combined_truncation=True):
        """
        As per finetuning BERT paper, I am taking first 128 and last 382 tokens from the sequence if it is longer than 512.
        The max_length parameter only applies when the combined_truncation is False
        - classical left truncation is implemented instead.
        """
        if combined_truncation:
            tok_texts_long = self.model(texts, max_length=2048, padding="max_length")
            tok_texts_long = [
                self.__truncate_tokenized_sequence__(tok)
                for tok in zip(
                    tok_texts_long["input_ids"], tok_texts_long["attention_mask"]
                )
            ]
            tok_texts_long = list(zip(*tok_texts_long))
            tok_texts = dict()
            tok_texts["input_ids"] = list(tok_texts_long[0])
            tok_texts["attention_mask"] = list(tok_texts_long[1])
        else:
            tok_texts = self.model(
                texts, max_length=max_length, padding="max_length", truncation=True
            )
        return tok_texts
