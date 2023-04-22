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

layer = -2

dim_size = 0.999

labelled_sizes = [60, 100, 150, 200, 300]

balanced_anchor = False

ks = [5, 7, 11, 15, 20]

emp_probs = [False, True]

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

    target_ds = ClassificationDataset(None)
    target_ds.read_user_input()

    target_ds.preprocess(asc.preprocessor)
    target_ds.tokenize(asc.tokenizer)
    target_ds.create_dataset()
    target_ds.create_dataloader(16, False)
    #         asc.bulk_predict(target_ds, predict_scale=False, output_hidden=layer)
    asc.bulk_predict(target_ds, predict_scale=True, output_hidden=layer)
    y_pred_copy = pd.Series(target_ds.y_pred, index=target_ds.y.index).copy()

    for emp_prob in emp_probs:
        for anchor_size in labelled_sizes:
            for k in ks:
                target_ds = ClassificationDataset(None)
                target_ds.read_user_input()

                target_ds.preprocess(asc.preprocessor)
                target_ds.tokenize(asc.tokenizer)
                target_ds.create_dataset()
                target_ds.create_dataloader(16, False)

                # I want balanced dataset - oversize the suggested set and label
                # only subset
                if balanced_anchor:
                    anch_s = anchor_size * 3
                else:
                    anch_s = anchor_size

                asc.suggest_anchor_set(
                    target_ds,
                    layer=layer,
                    dim_size=dim_size,
                    anchor_set_size=anch_s,
                )

                target_ds.y_pred = y_pred_copy.copy()

                ### THIS WILL BE DONE BY USER
                anch = pd.read_csv(
                    os.path.join(paths.INPUT_ANCHOR, "anchor_set.csv"), index_col=0
                )
                if balanced_anchor:
                    for lab in y_true.unique():
                        y_true_subs = (
                            y_true.loc[(anch.label.iloc[0:anchor_size].index)]
                            .loc[y_true == lab]
                            .iloc[0 : (round(anchor_size / 3))]
                        )
                        anch.loc[y_true_subs.index, "label"] = y_true_subs
                else:
                    anch.label.iloc[0:anchor_size] = y_true.loc[
                        anch.label.iloc[0:anchor_size].index
                    ]

                anch.to_csv(os.path.join(paths.INPUT_ANCHOR, "anchor_set.csv"))
                ###

                # read labelled subset
                target_ds.read_anchor_set()

                # bulk inference
                # y_pred, _ = asc.knn_bulk_predict(
                #     target_ds, knn=1, layer=-1, dim_size=dim_size, k=5, predict_scale=False
                # )

                y_pred = asc.mix_bulk_predict(
                    target_ds,
                    knn=1,
                    layer=layer,
                    dim_size=dim_size,
                    k=k,
                    scale=False,
                    emp_prob=emp_prob,
                    radius=True,
                )

                target_ds.y = y_true

                file_name = (
                    "knn_radius_mix"
                    + "_"
                    + datasets_names[i]
                    + "_"
                    + str(layer)
                    + "_"
                    + str(k)
                    + "_"
                    + str(dim_size)
                    + "_"
                    + str(anchor_size)
                    + "_"
                    + str(emp_prob)
                    + ".json"
                )
                save_results_path = os.path.join(paths.OUTPUT_INFO_INFERENCE, file_name)
                target_ds.evaluation_report(save_results_path)
#                 target_ds.evaluation_report()
