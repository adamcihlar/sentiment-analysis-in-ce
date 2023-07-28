from src.config.parameters import (
    ClassifierParams,
    DataLoaderParams,
    DatasetParams,
    FinetuningOptimizationParams,
)
from src.model.classifiers import (
    AdaptiveSentimentClassifier,
    ClassificationHead,
    Discriminator,
)
from src.model.encoders import Encoder
from src.model.tokenizers import Tokenizer
from src.reading.readers import read_csfd, read_facebook, read_mall
from src.utils.datasets import get_datasets_ready_for_finetuning
from src.utils.text_preprocessing import Preprocessor
from torch.optim import AdamW
from transformers import get_linear_schedule_with_warmup

if __name__ == "__main__":
    source_mall = read_mall()
    source_facebook = read_facebook()
    source_csfd = read_csfd()
    datasets = [source_csfd, source_facebook, source_mall]

    asc = AdaptiveSentimentClassifier(
        Preprocessor(),
        Tokenizer(),
        Encoder(),
        ClassificationHead,
        Discriminator(),
        Encoder(),
    )

    train_datasets, val_datasets = get_datasets_ready_for_finetuning(
        datasets,
        transformation=DatasetParams.TRANSFORMATION,
        balance_data=False,
        majority_ratio=1,
        preprocessor=asc.preprocessor,
        tokenizer=asc.tokenizer,
        batch_size=DataLoaderParams.BATCH_SIZE,
        shuffle=DataLoaderParams.SHUFFLE,
        num_workers=DataLoaderParams.NUM_WORKERS,
        skip_validation=FinetuningOptimizationParams.SKIP_VALIDATION,
        min_query_len=FinetuningOptimizationParams.MIN_QUERY_LEN,
        share_classifier=ClassifierParams.SHARE_CLASSIFIER,
    )

    asc.finetune(
        train_datasets=train_datasets,
        val_datasets=val_datasets,
        optimizer=AdamW,
        optimizer_params=FinetuningOptimizationParams.OPTIMIZATION,
        lr_decay=FinetuningOptimizationParams.LR_DECAY,
        lr_scheduler_call=get_linear_schedule_with_warmup,
        warmup_steps_proportion=FinetuningOptimizationParams.WARM_UP_STEPS_PROPORTION,
        num_epochs=FinetuningOptimizationParams.NUM_EPOCHS,
        metrics=["f1"],
        task=ClassifierParams.TASK,
        share_classifier=ClassifierParams.SHARE_CLASSIFIER,
    )
