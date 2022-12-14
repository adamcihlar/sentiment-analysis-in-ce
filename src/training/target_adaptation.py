from torch.optim import AdamW
from transformers import get_linear_schedule_with_warmup

from src.config import paths
from src.config.parameters import AdaptationOptimizationParams
from src.reading.readers import (
    read_csfd,
    read_finetuning_source,
)
from src.utils.datasets import get_datasets_ready_for_adaptation
from src.utils.text_preprocessing import Preprocessor
from src.model.tokenizers import Tokenizer
from src.model.encoders import Encoder
from src.model.classifiers import (
    AdaptiveSentimentClassifier,
    ClassificationHead,
    Discriminator,
)


if __name__ == "__main__":
    source_train_df, source_val_df = read_finetuning_source()
    target_df = read_csfd()

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
        drop_neutral=True,
        preprocessor=asc.preprocessor,
        tokenizer=asc.tokenizer,
        batch_size=16,
        shuffle=True,
        num_workers=0,
        skip_validation=paths.AdaptationOptimizationParams.SKIP_VALIDATION,
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
        metrics=["f1", "accuracy", "precision", "recall"],
    )
