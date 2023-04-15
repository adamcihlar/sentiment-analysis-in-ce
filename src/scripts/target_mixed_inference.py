from src.reading.readers import read_csfd
import os
import pandas as pd
from src.config import paths, parameters
from src.utils.datasets import ClassificationDataset
from src.model.classifiers import (
    AdaptiveSentimentClassifier,
    ClassificationHead,
    Discriminator,
)
from src.reading.readers import (
    read_csfd,
    read_facebook,
    read_mall,
    read_preprocessed_emails,
)
from src.model.encoders import Encoder
from src.model.tokenizers import Tokenizer
from src.utils.text_preprocessing import Preprocessor

# setup
model = (
    "seznamsmall-e-czech_20230218-142950",
    "5",
    "mall_facebook",
    "ordinal",
)
dataset = read_csfd
dim_size = 0.98
layer = -1
samples_labelled = 50

enc = "_".join([model[0], model[1]])
enc_pth = os.path.join(paths.OUTPUT_MODELS_FINETUNED_ENCODER, enc)
cls = "_".join([model[0], model[2], model[1]])
cls_pth = os.path.join(paths.OUTPUT_MODELS_FINETUNED_CLASSIFIER, cls)

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

target_df = dataset().sample(parameters.AdaptationOptimizationParams.N_EMAILS)
target_df.text.to_csv(os.path.join(paths.INPUT, "hello.csv"), index=False)
y_true = target_df.label.copy().reset_index(drop=True)

target_ds = ClassificationDataset(None)
target_ds.read_user_input()

target_ds.preprocess(asc.preprocessor)
target_ds.tokenize(asc.tokenizer)
target_ds.create_dataset()
target_ds.create_dataloader(16, False)

# output samples for labelling
asc.suggest_anchor_set(target_ds, layer=layer, dim_size=dim_size)

### THIS WILL BE DONE BY USER
anch = pd.read_csv(os.path.join(paths.INPUT_ANCHOR, "anchor_set.csv"), index_col=0)
anch.label.iloc[0:samples_labelled] = y_true.loc[
    anch.label.iloc[0:samples_labelled].index
]
anch.to_csv(os.path.join(paths.INPUT_ANCHOR, "anchor_set.csv"))
###

# read labelled subset
target_ds.read_anchor_set()

# bulk inference
y_pred = asc.mix_bulk_predict(target_ds, scale=False)

target_ds.y = y_true

target_ds.evaluation_report()
