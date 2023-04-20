import os
import pandas as pd
import torch
from src.config import paths, parameters
from src.utils.datasets import ClassificationDataset, drop_undefined_classes
from src.model.classifiers import (
    AdaptiveSentimentClassifier,
    ClassificationHead,
    Discriminator,
)
from src.model.encoders import Encoder
from src.model.tokenizers import Tokenizer
from src.reading.readers import read_csfd, read_facebook, read_mall
from src.utils.datasets import get_target_datasets_ready_for_finetuning
from src.utils.text_preprocessing import Preprocessor
from torch.optim import AdamW
from transformers import get_linear_schedule_with_warmup
from transformers import get_constant_schedule
from src.config.parameters import (
    ClassifierParams,
    DataLoaderParams,
    DatasetParams,
    FinetuningOptimizationParams,
)

models = [
    ("seznamsmall-e-czech_20230218-142950", "5", "mall_facebook", "ordinal"),
    ("seznamsmall-e-czech_20230218-183829", "5", "facebook_csfd", "ordinal"),
    ("seznamsmall-e-czech_20230218-213739", "5", "mall_csfd", "ordinal"),
]

datasets = [read_csfd, read_mall, read_facebook]
# datasets = [read_mall]
datasets_names = ["target_csfd", "target_mall", "target_facebook"]
# datasets_names = ["target_mall"]

layer = -1

dim_size = 0.999

# labelled_sizes = [30, 50, 100]
labelled_sizes = [100]

balanced_anchor = True

for i, model in enumerate(models):
    for anchor_size in labelled_sizes:

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
            # 200,
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

        train_df, val_df = target_ds.get_target_train_test_splits()
        val_df.label = y_true.loc[val_df.index]

        # weights for the corn loss
        n_classes = len(train_df.label.unique())
        weights = torch.tensor(
            len(train_df.label) / train_df.label.value_counts().sort_index()
        )
        weights = weights / n_classes

        train_df["source"] = datasets_names[i]
        val_df["source"] = datasets_names[i]

        train_datasets, val_datasets = get_target_datasets_ready_for_finetuning(
            [train_df, val_df],
            transformation=DatasetParams.TRANSFORMATION,
            preprocessor=asc.preprocessor,
            tokenizer=asc.tokenizer,
            # batch_size=DataLoaderParams.BATCH_SIZE,
            batch_size=8,
            shuffle=DataLoaderParams.SHUFFLE,
            num_workers=DataLoaderParams.NUM_WORKERS,
            share_classifier=ClassifierParams.SHARE_CLASSIFIER,
        )

        asc.finetune(
            train_datasets=train_datasets,
            val_datasets=val_datasets,
            optimizer=AdamW,
            # optimizer_params=FinetuningOptimizationParams.OPTIMIZATION,
            optimizer_params={"lr": 1e-05, "betas": (0.9, 0.999)},
            lr_decay=FinetuningOptimizationParams.LR_DECAY,
            lr_scheduler_call=get_linear_schedule_with_warmup,
            # lr_scheduler_call=get_constant_schedule,
            warmup_steps_proportion=FinetuningOptimizationParams.WARM_UP_STEPS_PROPORTION,
            num_epochs=6,
            metrics=["f1"],
            task=ClassifierParams.TASK,
            share_classifier=True,
            save_models=False,
            # weights=weights,
        )
