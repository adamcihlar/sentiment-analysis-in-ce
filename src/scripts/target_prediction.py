import os
import pandas as pd
from src.config import paths
from src.utils.datasets import ClassificationDataset
from src.model.classifiers import (
    AdaptiveSentimentClassifier,
    ClassificationHead,
    Discriminator,
)
from src.model.encoders import Encoder
from src.model.tokenizers import Tokenizer
from src.utils.text_preprocessing import Preprocessor
from src.reading.readers import read_raw_sent

model = ("seznamsmall-e-czech_20230218-071534", "5", "csfd_facebook_mall", "ordinal")
enc = "_".join([model[0], model[1]])
enc_pth = os.path.join(paths.OUTPUT_MODELS_FINETUNED_ENCODER, enc)
cls = "_".join([model[0], model[2], model[1]])
cls_pth = os.path.join(paths.OUTPUT_MODELS_FINETUNED_CLASSIFIER, cls)

test_df = read_preprocessed_emails()

test_df = pd.read_csv("data/preprocessed/responses_confirmed_full.csv", index_col=0)
test_df = test_df.loc[~test_df.invalid]

orig_emails_count = 311
foreign = 29
sum(test_df.invalid) - orig_emails_count - foreign

sum(~test_df.invalid)


asc = AdaptiveSentimentClassifier(
    Preprocessor(),
    Tokenizer(),
    Encoder(path_to_finetuned=enc_pth),
    ClassificationHead,
    Discriminator(),
    Encoder(path_to_finetuned=enc_pth),
    classifier_checkpoint_path=cls_pth,
    inference_mode=True,
    task_settings=model[3],
)

test = ClassificationDataset(test_df.text, None, None)
test.preprocess(asc.preprocessor)
test.tokenize(asc.tokenizer)
test.create_dataset()
test.create_dataloader(16, False)

y_pred = asc.bulk_predict(test, predict_scale=False)

test_df.y_pred = y_pred

save_pth = os.path.join(paths.OUTPUT_PREDICTIONS, "emails.csv")
test_df.to_csv(save_pth)

test_df = pd.read_csv(save_pth)
sent_df = read_raw_sent()

merged = pd.merge(test_df, sent_df, how="left", left_on="id_2", right_on="id")

pd.read_csv("output/train_info/finetuning/test/csfd.csv")
