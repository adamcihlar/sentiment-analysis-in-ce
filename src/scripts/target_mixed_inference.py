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

target_df = read_csfd().sample(parameters.AdaptationOptimizationParams.N_EMAILS)
y_true = target_df.label.copy()
target_ds = ClassificationDataset(target_df.text, target_df.label, None)
target_ds.preprocess(asc.preprocessor)
target_ds.tokenize(asc.tokenizer)
target_ds.create_dataset()
target_ds.create_dataloader(16, False)
target_ds.y.iloc[40:] = None

# bulk inference
y_pred = asc.mix_bulk_predict(target_ds, dim_size=0.999, layer=-2)

y_pred = np.floor(y_pred * 3)
target_ds.y_pred = y_pred
target_ds.y

# one sample
asc.get_nn_sim_distribution(target_ds)
asc.predict(["necum", "hruza"])

asc.mix_predict(["super", "hruza"], scale=False)

# TODO scale to labels to get report !!!!!

target_ds.y = y_true
target_ds

target_ds.evaluation_report()
