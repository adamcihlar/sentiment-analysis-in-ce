from src.reading.readers import read_csfd
import os
import pandas as pd
from src.config import paths, parameters
from src.utils.datasets import ClassificationDataset, drop_undefined_classes
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

models = [
    ("seznamsmall-e-czech_20230218-142950", "5", "mall_facebook", "ordinal"),
    ("seznamsmall-e-czech_20230218-183829", "5", "facebook_csfd", "ordinal"),
    ("seznamsmall-e-czech_20230218-213739", "5", "mall_csfd", "ordinal"),
]

datasets = [read_csfd, read_mall, read_facebook]
datasets_names = ["csfd", "mall", "facebook"]

emb_layers = [-1, -2]

sim_dist_types = [1, 5]

dim_sizes = [None, 0.999, 0.99, 0.98, 0.95, 0.9, 0.8]

labelled_sizes = [30, 50, 100]

for i, model in enumerate(models):

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

    target_df = datasets[i]()
    target_df = drop_undefined_classes(target_df)
    target_df = target_df.sample(
        parameters.AdaptationOptimizationParams.N_EMAILS,
        random_state=parameters.RANDOM_STATE,
    )

    target_df.text.to_csv(os.path.join(paths.INPUT, "hello.csv"), index=False)
    y_true = target_df.label.copy().reset_index(drop=True)

    for layer in emb_layers:
        target_ds = ClassificationDataset(None)
        target_ds.read_user_input()

        target_ds.preprocess(asc.preprocessor)
        target_ds.tokenize(asc.tokenizer)
        target_ds.create_dataset()
        target_ds.create_dataloader(16, False)
        asc.bulk_predict(target_ds, predict_scale=False, output_hidden=layer)
        # asc.bulk_predict(target_ds, predict_scale=True, output_hidden=layer)
        y_pred_copy = pd.Series(target_ds.y_pred, index=target_ds.y.index).copy()

        for knn in sim_dist_types:
            for dim_size in dim_sizes:
                #                 target_ds = ClassificationDataset(None)
                #                 target_ds.read_user_input()

                #                 target_ds.preprocess(asc.preprocessor)
                #                 target_ds.tokenize(asc.tokenizer)
                #                 target_ds.create_dataset()
                #                 target_ds.create_dataloader(16, False)

                # output samples for labelling
                #                 asc.suggest_anchor_set(target_ds, layer=layer, dim_size=dim_size)
                # creates target_ds.y_pred

                for samples_labelled in labelled_sizes:
                    target_ds = ClassificationDataset(None)
                    target_ds.read_user_input()

                    target_ds.preprocess(asc.preprocessor)
                    target_ds.tokenize(asc.tokenizer)
                    target_ds.create_dataset()
                    target_ds.create_dataloader(16, False)

                    asc.suggest_anchor_set(
                        target_ds,
                        layer=layer,
                        dim_size=dim_size,
                        anchor_set_size=samples_labelled,
                    )

                    target_ds.y_pred = y_pred_copy.copy()

                    ### THIS WILL BE DONE BY USER
                    anch = pd.read_csv(
                        os.path.join(paths.INPUT_ANCHOR, "anchor_set.csv"), index_col=0
                    )
                    anch.label.iloc[0:samples_labelled] = y_true.loc[
                        anch.label.iloc[0:samples_labelled].index
                    ]
                    anch.to_csv(os.path.join(paths.INPUT_ANCHOR, "anchor_set.csv"))
                    ###

                    # read labelled subset
                    target_ds.read_anchor_set()

                    # evaluate without the knn extension to have comparable
                    # results
                    anchor = target_ds.y.loc[~target_ds.y.isna()].copy()
                    target_ds.y_pred.loc[anchor.index] = anchor

                    # bulk inference
                    # y_pred = asc.mix_bulk_predict(target_ds, knn=knn, scale=False)

                    target_ds.y = y_true

                    file_name = (
                        "kmeans"
                        + "_"
                        + datasets_names[i]
                        + "_"
                        + str(layer)
                        + "_"
                        + str(knn)
                        + "_"
                        + str(dim_size)
                        + "_"
                        + str(samples_labelled)
                        + ".json"
                    )
                    save_results_path = os.path.join(
                        paths.OUTPUT_INFO_INFERENCE, file_name
                    )
                    target_ds.evaluation_report(save_results_path)


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

target_df = dataset().sample(
    parameters.AdaptationOptimizationParams.N_EMAILS, random_state=42
)
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
