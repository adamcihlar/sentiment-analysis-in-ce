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
from src.reading.readers import read_raw_sent, read_preprocessed_emails
from sklearn.decomposition import PCA
import plotly.express as px

model = ("seznamsmall-e-czech_20230218-071534", "5", "csfd_facebook_mall", "ordinal")
enc = "_".join([model[0], model[1]])
enc_pth = os.path.join(paths.OUTPUT_MODELS_FINETUNED_ENCODER, enc)
cls = "_".join([model[0], model[2], model[1]])
cls_pth = os.path.join(paths.OUTPUT_MODELS_FINETUNED_CLASSIFIER, cls)

test_df = read_preprocessed_emails()

test_df = pd.read_csv("data/preprocessed/responses_confirmed_v1.csv", index_col=0)
test_df = test_df.loc[~test_df.invalid]

layer = -1
dim_size = 0.999
anchor_set_size = 150

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

y_pred, _ = asc.bulk_predict(test, predict_scale=True, output_hidden=layer)
# y_pred = asc.bulk_predict(test, predict_scale=False)
anch_set = asc.suggest_anchor_set(
    test, layer, dim_size=dim_size, anchor_set_size=anchor_set_size
)

dim_size = 2
pca = PCA(dim_size)
hiddens = pca.fit_transform(asc.hiddens_full)
sum(pca.explained_variance_ratio_)

hiddens_df = pd.DataFrame(hiddens, index=test.X.index)
hiddens_df = pd.concat(
    [
        hiddens_df,
        pd.DataFrame({"cls": list(y_pred)}, index=test.X.index),
        # pd.DataFrame({"cls": list(y_pred)}, index=test.X.index).astype(str),
    ],
    axis=1,
)
# hiddens_df = pd.concat([hiddens_df, test.y], axis=1)
hiddens_df = pd.concat([hiddens_df, test.X], axis=1)
is_anch = pd.Series(0, index=test.X.index)
is_anch.loc[anch_set.index] = 1
is_anch.name = "anchor"
hiddens_df = pd.concat([hiddens_df, is_anch.astype(str)], axis=1)

fig = px.scatter(hiddens_df, x=0, y=1, color="cls", hover_data=["text"])
fig.write_html("output/assets/emails_target_pred.html")

fig = px.scatter(hiddens_df, x=0, y=1, color="anchor", hover_data=['text'])
fig.write_html("output/assets/emails_target_anchor.html")
