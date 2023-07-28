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


def get_df(new=False):
    """
    Load the dfs, merge them and save.
    Or just load the merged DataFrame.
    """
    if new:
        responses_df = read_preprocessed_emails()
        responses_df = responses_df[~responses_df.invalid]

        sent_df = read_raw_sent()

        ico_first_date = pd.DataFrame(sent_df.groupby("ico").min().date).reset_index()
        ico_first_date["join"] = ico_first_date.ico.astype(
            str
        ) + ico_first_date.date.astype(str)

        sent_df["join"] = sent_df.ico.astype(str) + sent_df.date.astype(str)

        merged = sent_df.merge(ico_first_date, how="left", on="join")
        merged["sent_first"] = False
        merged["sent_first"].loc[~merged.ico_y.isna()] = True

        sent_df["sent_first"] = merged["sent_first"]

        predictions_df = pd.read_csv("output/predictions/emails.csv", index_col=0)
        responses_df = responses_df.reset_index(drop=True)

        responses = predictions_df.merge(
            responses_df, how="inner", left_index=True, right_index=True
        )
        responses = responses[["y_pred", "id_2", "X"]]

        df = sent_df.merge(responses, how="inner", left_on="id", right_on="id_2")

        df.to_csv("data/final/emails_merged.csv")
    else:
        df = pd.read_csv("data/final/emails_merged.csv", index_col=0)
    return df


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

fig = px.scatter(hiddens_df, x=0, y=1, color="anchor", hover_data=["text"])
fig.write_html("output/assets/emails_target_anchor.html")


##### PLOT EMAILS REPRESENTATION WITH PREDICTIONS #####

import os
import pandas as pd
from src.config import paths
from src.config.parameters import SupportInferenceParams as sip
from src.utils.datasets import ClassificationDataset, drop_undefined_classes
from src.model.classifiers import (
    AdaptiveSentimentClassifier,
    ClassificationHead,
    Discriminator,
)
from src.model.encoders import Encoder
from src.model.tokenizers import Tokenizer
from src.utils.text_preprocessing import Preprocessor
from loguru import logger
from sklearn.decomposition import PCA
import plotly.express as px
import numpy as np

scale = True
external_anchor_set = False
layer = -1

model = AdaptiveSentimentClassifier(
    Preprocessor(),
    Tokenizer(),
    Encoder(),
    ClassificationHead,
    Discriminator(),
    Encoder(path_to_finetuned=paths.OUTPUT_PROD_ENCODER),
    paths.OUTPUT_PROD_CLASSIFIER,
)

target_ds = ClassificationDataset(None)
target_ds.read_user_input()

target_ds.read_anchor_set(external_anchor_set)

target_ds.preprocess(model.preprocessor)
target_ds.tokenize(model.tokenizer)
target_ds.create_dataset()
target_ds.create_dataloader(4, False)

y_pred_base = model.bulk_predict(target_ds, predict_scale=True, output_hidden=layer)
y_pred_base = y_pred_base[0]

y_pred = model.mix_bulk_predict(
    target_ds,
    knn=1,
    layer=sip.LAYER,
    dim_size=sip.DIM_SIZE,
    k=sip.K,
    scale=scale,
    emp_prob=sip.EMP_CONF,
    radius=True,
)
anchor_df = pd.read_csv("input/anchor/anchor_set.csv", index_col=0)

dim_size = 2
pca = PCA(dim_size)
hiddens = pca.fit_transform(model.hiddens_full)
sum(pca.explained_variance_ratio_)

hiddens_df = pd.DataFrame(hiddens, index=target_ds.X.index)
hiddens_df = pd.concat(
    [
        hiddens_df,
        pd.DataFrame({"cls_base": list(y_pred_base)}, index=target_ds.X.index),
        pd.DataFrame({"cls": list(y_pred)}, index=target_ds.X.index),
        pd.DataFrame({"text": list(target_ds.X)}, index=target_ds.X.index),
        # pd.DataFrame({"cls": list(y_pred)}, index=test.X.index).astype(str),
    ],
    axis=1,
)
is_anch = pd.Series(0, index=target_ds.X.index)
is_anch.loc[anchor_df.index] = 1
is_anch.name = "anchor"
anch_size = pd.Series(1, index=target_ds.X.index)
anch_size.loc[anchor_df.index] = 1
anch_size.name = "anch_size"
hiddens_df = pd.concat([hiddens_df, is_anch.astype(str), anch_size], axis=1)
hiddens_df = hiddens_df.reset_index()
hiddens_df["size_max"] = 0.6
hiddens_df.cls_base.iloc[np.argmax(hiddens_df.cls_base)] = 1
hiddens_df.cls_base.iloc[np.argmin(hiddens_df.cls_base)] = 0

fig = px.scatter(
    hiddens_df,
    x=0,
    y=1,
    color="cls_base",
    labels={"cls_base": "Predicted<br>sentiment"},
    color_continuous_scale="Portland",
    # hover_data=["index", "text"],
    symbol_sequence=["circle-open"],
    size="size_max",
    # size_max=100000
    template="plotly_white",
)
fig.update_layout(yaxis_title="Component 2")
fig.update_layout(xaxis_title="Component 1")
fig.update_layout(legend=dict(font=dict(size=200)))
fig.write_html("output/assets/emails_target_pred_base_axis.html")

hiddens_df.iloc[2419]
is_anch.iloc[2419]

hiddens_df.iloc[1760].text
is_anch.iloc[1760]

fig = px.scatter(
    hiddens_df,
    x=0,
    y=1,
    color="cls",
    labels={"cls": "Predicted<br>sentiment"},
    color_continuous_scale="Portland",
    size="anch_size",
    # size_max=1,
    symbol="anchor",
    symbol_sequence=["circle-open", "circle"],
    template="plotly_white",
)
fig.update_layout(yaxis_title="Component 2")
fig.update_layout(xaxis_title="Component 1")
fig.update_layout(showlegend=False)
fig.write_html("output/assets/emails_target_pred_ensemble_axis.html")

cls = hiddens_df[[0, 1, "cls", "anchor"]]
cls["wrap"] = "ensemble"
cls_base = hiddens_df[[0, 1, "cls_base", "anchor"]]
cls_base["wrap"] = "base"
cls_base["anchor"] = "0"

hist_df = pd.DataFrame(
    np.vstack([cls, cls_base]),
    columns=cls.columns,
)


sum((hiddens_df.cls >= 0.00) & (hiddens_df.cls < 0.20)) / len(hiddens_df.cls)
sum((hiddens_df.cls >= 0.20) & (hiddens_df.cls < 0.40)) / len(hiddens_df.cls)
sum((hiddens_df.cls >= 0.40) & (hiddens_df.cls < 0.60)) / len(hiddens_df.cls)
sum((hiddens_df.cls >= 0.60) & (hiddens_df.cls < 0.80)) / len(hiddens_df.cls)
sum((hiddens_df.cls >= 0.80) & (hiddens_df.cls < 1.80)) / len(hiddens_df.cls)

sum((hiddens_df.cls >= 0.00) & (hiddens_df.cls < 0.20))
sum((hiddens_df.cls >= 0.20) & (hiddens_df.cls < 0.40))
sum((hiddens_df.cls >= 0.40) & (hiddens_df.cls < 0.60))
sum((hiddens_df.cls >= 0.60) & (hiddens_df.cls < 0.80))
sum((hiddens_df.cls >= 0.80) & (hiddens_df.cls < 1.80))

sum((hiddens_df.cls >= 0.45) & (hiddens_df.cls < 0.55)) / len(hiddens_df.cls)
sum((hiddens_df.cls > 0.75)) / len(hiddens_df.cls)
sum((hiddens_df.cls < 0.45)) / len(hiddens_df.cls)

fig = px.histogram(
    hist_df,
    x="cls",
    template="plotly_white",
    color="anchor",
    facet_row="wrap",
    color_discrete_sequence=["#4A6274", "#E2725A"],  # "#79AEB2"],
)
fig.update_layout(yaxis_title=None)
fig.update_layout(xaxis_title=None)
fig.write_html("output/assets/emails_target_pred_dists.html")

fig = px.histogram(
    hiddens_df,
    x="cls",
    template="plotly_white",
    color="anchor",
    color_discrete_sequence=["#4A6274", "#E2725A"],  # "#79AEB2"],
)
fig.update_layout(yaxis_title=None)
fig.update_layout(xaxis_title=None)
fig.update_layout(showlegend=False)
fig.write_html("output/assets/emails_target_pred_dist.html")

fig = px.histogram(
    hiddens_df,
    x="cls_base",
    template="plotly_white",
    # color="anchor",
    color_discrete_sequence=["#4A6274"],  # , "#E2725A"]# "#79AEB2"],
)
fig.update_layout(yaxis_title=None)
fig.update_layout(xaxis_title=None)
fig.write_html("output/assets/emails_target_pred_dist_base.html")

# New histograms
df = get_df()
anchor_df = pd.read_csv("input/anchor/anchor_set.csv", index_col=0)
df["anchor"] = 0
found_anch = anchor_df.loc[anchor_df.index.isin(df.index)]
df.anchor.loc[found_anch.index] = 1

# restrict the sample?
icos = df.groupby(["ico"]).id_2.nunique() > 1
sel_icos = icos.loc[icos].index
df = df.loc[df.ico.isin(sel_icos)]

df_nat = df.loc[~df.foreigner]
df_for = df.loc[df.foreigner]

fig = px.histogram(
    df,
    x="y_pred",
    template="plotly_white",
    color="foreigner",
    color_discrete_sequence=["#4A6274", "#E2725A"],  # "#79AEB2"],
    barmode="overlay",
    histnorm="probability density",
)
fig.update_layout(yaxis_title=None)
fig.update_layout(xaxis_title=None)
fig.update_layout(showlegend=False)
fig.write_html("output/assets/emails_target_pred_dist_nat_for_full.html")
