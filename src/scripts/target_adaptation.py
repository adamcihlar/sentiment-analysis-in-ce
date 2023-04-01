from src.config import parameters, paths
from src.config.parameters import AdaptationOptimizationParams
from src.model.classifiers import (
    AdaptiveSentimentClassifier,
    ClassificationHead,
    Discriminator,
)
from src.model.encoders import Encoder
from src.model.tokenizers import Tokenizer
from src.reading.readers import (
    read_csfd,
    read_facebook,
    read_finetuning_source,
    read_mall,
    read_preprocessed_emails,
)
from src.utils.datasets import get_datasets_ready_for_adaptation
from src.utils.text_preprocessing import Preprocessor
from torch.optim import AdamW
from transformers import get_linear_schedule_with_warmup

if __name__ == "__main__":

    import os

    # mall
    mall_models = [
        ("seznamsmall-e-czech_20230123-173904", "5", "facebook"),
        ("seznamsmall-e-czech_20230123-181226", "5", "csfd"),
        ("seznamsmall-e-czech_20230128-144144", "4", "facebook"),
        ("seznamsmall-e-czech_20230128-144144", "5", "csfd"),
        ("seznamsmall-e-czech_20230218-183829", "5", "facebook_csfd"),
    ]

    # csfd
    csfd_models = [
        ("seznamsmall-e-czech_20230123-173904", "5", "facebook"),
        ("seznamsmall-e-czech_20230123-213412", "5", "mall"),
        ("seznamsmall-e-czech_20230128-171700", "5", "facebook"),
        ("seznamsmall-e-czech_20230128-171700", "5", "mall"),
        ("seznamsmall-e-czech_20230218-142950", "5", "mall_facebook"),
    ]

    # facebook
    facebook_models = [
        ("seznamsmall-e-czech_20230123-181226", "5", "csfd"),
        ("seznamsmall-e-czech_20230123-213412", "5", "mall"),
        ("seznamsmall-e-czech_20230128-055746", "5", "csfd"),
        ("seznamsmall-e-czech_20230128-055746", "5", "mall"),
        ("seznamsmall-e-czech_20230218-213739", "5", "mall_csfd"),
    ]

    models = mall_models + csfd_models + facebook_models

    datasets = (
        [read_mall for i in range(5)]
        + [read_csfd for i in range(5)]
        + [read_facebook for i in range(5)]
    )

    for i, model in enumerate(models):
        source_train_df, source_val_df = read_finetuning_source(
            # selected_model=parameters.FINETUNED_CHECKPOINT,
            selected_model=model[0],
            # selected_dataset=parameters.FINETUNED_DATASET,
            selected_dataset=model[2],
        )
        target_df = datasets[i]().sample(AdaptationOptimizationParams.N_EMAILS)

        enc = "_".join([model[0], model[1]])
        enc_pth = os.path.join(paths.OUTPUT_MODELS_FINETUNED_ENCODER, enc)
        cls = "_".join([model[0], model[2], model[1]])
        cls_pth = os.path.join(paths.OUTPUT_MODELS_FINETUNED_CLASSIFIER, cls)

        temperatures = [1, 2, 5, 10, 20]
        loss_combination_params_list = [
            (0.2, 0.8),
            (0.3, 0.7),
            (0.4, 0.6),
            (0.5, 0.5),
        ]

        for temp in temperatures:
            for loss_comb in loss_combination_params_list:

                asc = AdaptiveSentimentClassifier(
                    Preprocessor(),
                    Tokenizer(),
                    Encoder(path_to_finetuned=enc_pth),
                    ClassificationHead,
                    Discriminator(),
                    Encoder(path_to_finetuned=enc_pth),
                    cls_pth,
                )

                source_train, source_val, target = get_datasets_ready_for_adaptation(
                    source_train_df,
                    source_val_df,
                    target_df,
                    transformation="ordinal_regression",
                    preprocessor=asc.preprocessor,
                    tokenizer=asc.tokenizer,
                    batch_size=16,
                    shuffle=True,
                    num_workers=0,
                    skip_validation=AdaptationOptimizationParams.SKIP_VALIDATION,
                )

                asc.adapt(
                    source_train,
                    source_val,
                    target,
                    optimizer=AdamW,
                    # optimizer_params=AdaptationOptimizationParams.OPTIMIZATION,
                    optimizer_params={"lr": 1e-05, "betas": (0.9, 0.999)},
                    lr_decay=AdaptationOptimizationParams.LR_DECAY,
                    lr_scheduler_call=get_linear_schedule_with_warmup,
                    warmup_steps_proportion=AdaptationOptimizationParams.WARM_UP_STEPS_PROPORTION,
                    # num_epochs=AdaptationOptimizationParams.NUM_EPOCHS,
                    num_epochs=3,
                    # temperature=AdaptationOptimizationParams.TEMPERATURE,
                    temperature=temp,
                    # loss_combination_params=AdaptationOptimizationParams.LOSS_COMBINATION_PARAMS,
                    loss_combination_params=loss_comb,
                    metrics=["f1"],
                )

    # emails
    # I need to copy the source train split to the desired location
    # or adjust the read_finetuning_source to simply read from data/final/finetuning_train
    source_train_df, source_val_df = read_finetuning_source(
        selected_model=parameters.FINETUNED_CHECKPOINT,
        selected_dataset=parameters.FINETUNED_DATASET,
    )
    target_df = read_preprocessed_emails()
    test_df = pd.read_csv("data/preprocessed/responses_confirmed_full.csv", index_col=0)
    test_df.loc[~test_df.invalid]

    asc = AdaptiveSentimentClassifier(
        Preprocessor(),
        Tokenizer(),
        Encoder(path_to_finetuned=paths.OUTPUT_MODELS_FINETUNED_ENCODER_FINAL),
        ClassificationHead,
        Discriminator(),
        Encoder(path_to_finetuned=paths.OUTPUT_MODELS_FINETUNED_ENCODER_FINAL),
        paths.OUTPUT_MODELS_FINETUNED_CLASSIFIER_FINAL,
    )

    source_train, source_val, target = get_datasets_ready_for_adaptation(
        source_train_df,
        source_val_df,
        target_df,
        transformation="ordinal_regression",
        preprocessor=asc.preprocessor,
        tokenizer=asc.tokenizer,
        batch_size=16,
        shuffle=True,
        num_workers=0,
        skip_validation=AdaptationOptimizationParams.SKIP_VALIDATION,
    )

    asc.adapt(
        source_train,
        source_val,
        target,
        optimizer=AdamW,
        optimizer_params=AdaptationOptimizationParams.OPTIMIZATION,
        lr_decay=AdaptationOptimizationParams.LR_DECAY,
        lr_scheduler_call=get_linear_schedule_with_warmup,
        warmup_steps_proportion=AdaptationOptimizationParams.WARM_UP_STEPS_PROPORTION,
        num_epochs=AdaptationOptimizationParams.NUM_EPOCHS,
        temperature=AdaptationOptimizationParams.TEMPERATURE,
        loss_combination_params=AdaptationOptimizationParams.LOSS_COMBINATION_PARAMS,
        metrics=["f1"],
    )
