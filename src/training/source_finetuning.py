from torch.optim import AdamW
from transformers import get_linear_schedule_with_warmup

from src.config.parameters import (
    DatasetParams,
    DataLoaderParams,
    FinetuningOptimizationParams,
)
from src.reading.readers import read_facebook, read_mall, read_csfd
from src.utils.datasets import get_datasets_ready_for_finetuning
from src.utils.text_preprocessing import Preprocessor
from src.model.tokenizers import Tokenizer
from src.model.encoders import Encoder
from src.model.classifiers import (
    AdaptiveSentimentClassifier,
    ClassificationHead,
    Discriminator,
)

if __name__ == "__main__":
    source_mall = read_mall().sample(20)
    source_mall = read_mall()
    source_mall = pd.concat(
        [
            source_mall.loc[source_mall.label == 1].sample(20),
            source_mall.loc[source_mall.label == 0].sample(20),
        ]
    )
    source_facebook = read_facebook().sample(20)
    source_facebook = read_facebook()
    source_facebook = pd.concat(
        [
            source_facebook.loc[source_facebook.label == 1].sample(20),
            source_facebook.loc[source_facebook.label == 0].sample(20),
        ]
    )
    datasets = [source_facebook, source_mall]

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
        drop_neutral=True,
        balance_data=DatasetParams.BALANCE_DATA,
        majority_ratio=DatasetParams.MAJORITY_RATIO,
        preprocessor=asc.preprocessor,
        tokenizer=asc.tokenizer,
        batch_size=4,  # DataLoaderParams.BATCH_SIZE
        shuffle=True,  # DataLoaderParams.SHUFFLE
        num_workers=0,  # DataLoaderParams.NUM_WORKERS
    )

    asc.finetune(
        train_datasets=train_datasets,
        val_datasets=val_datasets,
        optimizer=AdamW,
        # optimizer_params=FinetuningOptimizationParams.OPTIMIZATION,
        optimizer_params={"lr": 5e-5, "betas": (0.9, 0.999)},
        lr_decay=FinetuningOptimizationParams.LR_DECAY,
        lr_scheduler_call=get_linear_schedule_with_warmup,
        warmup_steps_proportion=FinetuningOptimizationParams.WARM_UP_STEPS_PROPORTION,
        # num_epochs=FinetuningOptimizationParams.NUM_EPOCHS,
        num_epochs=20,
        metrics=["f1", "accuracy", "precision", "recall"],
    )
