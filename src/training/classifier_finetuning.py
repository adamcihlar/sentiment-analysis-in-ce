import pandas as pd

from transformers import RobertaTokenizer


tokenizer = RobertaTokenizer.from_pretrained("ufal/robeczech-base")

tokenizer(":D")
tokenizer("D")
tokenizer(":")
tokenizer("Dobry den")

list(tokenizer.get_vocab().keys())[list(tokenizer.get_vocab().values()).index(75)]
list(tokenizer.get_vocab().keys())[list(tokenizer.get_vocab().values()).index(385)]

list(tokenizer.get_vocab().keys())[list(tokenizer.get_vocab().values()).index(2)]

